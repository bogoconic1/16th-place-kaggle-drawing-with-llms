import os
import io
import re
import random
import base64
from io import BytesIO
import time
from datetime import timedelta
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import cairosvg
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler
from transformers import AutoProcessor, AutoModel
from scour.scour import scourString, parse_args
import cupy as cp
from cuml.cluster import KMeans
import statistics
import metric

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion
pipe = StableDiffusionXLPipeline.from_pretrained("/root/cache/models/sd-community/sdxl-flash", torch_dtype=torch.float16).to(device)
pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
seed_generator = torch.Generator(device=device).manual_seed(42)

# Load SigLIP
siglip1_model = AutoModel.from_pretrained("/root/cache/models/google/siglip-so400m-patch14-384").to(device)
siglip1_processor = AutoProcessor.from_pretrained("/root/cache/models/google/siglip-so400m-patch14-384")

def generate_bitmap(prompt, negative_prompt="", num_inference_steps=6, guidance_scale=2.5):
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        generator=seed_generator,
        width=640, 
        height=640
    ).images[0]
    
    return image

@torch.inference_mode()
def cosine_batch_siglip(siglip_model, siglip_processor, prompts, pil_images, is_siglip2=False):
    siglip_scores = []            
    for pil_images_list in pil_images:
        if not is_siglip2:
            inputs = siglip_processor(text=prompts, images=pil_images_list, padding="max_length", return_tensors="pt").to(device)
        else:
            inputs = siglip_processor(text=prompts, images=pil_images_list, padding="max_length", max_length=64, return_tensors="pt").to(device)
        outputs = siglip_model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        siglip_scores.append(probs.ravel().cpu().tolist())

    return siglip_scores

def kmeans_gpu_quantize(img_np, k=12, max_iter=100, tol=0.2):
    """
    img_np : H×W×3 uint8 (CPU)
    returns: labels (H×W int), palette (k×3 uint8)
    """
    h, w, _ = img_np.shape
    # ── 1. copy pixels to GPU ──────────────────────────
    pixels_gpu = cp.asarray(img_np.reshape(-1, 3), dtype=cp.float32)

    # ── 2. fit KMeans on GPU ───────────────────────────
    km = KMeans(n_clusters=k, init="k-means++",
                max_iter=max_iter, tol=tol, verbose=0)
    km.fit(pixels_gpu)

    labels_gpu  = km.labels_               # (H·W,) on GPU
    centers_gpu = km.cluster_centers_      # (k,3) float32 on GPU

    # ── 3. bring results back to CPU in one shot ───────
    labels  = cp.asnumpy(labels_gpu).reshape(h, w).astype(np.int32)
    palette = cp.asnumpy(centers_gpu).round().astype(np.uint8)

    return labels, palette

def scour_svg_string(svg_text: str, precision: int = 1) -> str:
    # Build an options object just like CLI flags
    opts = parse_args([
        '--no-line-breaks', '--indent=none', '--strip-xml-prolog',
        '--enable-comment-stripping', '--remove-metadata',
        '--enable-id-stripping', '--shorten-ids',
        f'--set-precision={precision}',
    ])
    return scourString(svg_text, opts)

def compress_hex_color(hex_color):
    """Convert hex color to shortest possible representation"""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    return hex_color

def extract_features_by_scale(img_np, num_colors=16):
    """
    Extract image features hierarchically by scale
    
    Args:
        img_np (np.ndarray): Input image
        num_colors (int): Number of colors to quantize
    
    Returns:
        list: Hierarchical features sorted by importance
    """
    # Convert to RGB if needed
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    # Perform color quantization
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    try:
        labels, palette = kmeans_gpu_quantize(img_rgb, num_colors)
    except Exception as e:
        print(e)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
        # Quantized image
        palette = centers.astype(np.uint8)
        
    quantized = palette[labels.flatten()].reshape(img_rgb.shape)
    
    # Hierarchical feature extraction
    hierarchical_features = []
    
    # Sort colors by frequency
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_colors = [palette[i] for i in sorted_indices]
    
    # Center point for importance calculations
    center_x, center_y = width/2, height/2
    
    for color in sorted_colors:
        # Create color mask
        color_mask = cv2.inRange(quantized, color, color)
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Convert RGB to compressed hex
        hex_color = compress_hex_color(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
        
        color_features = []
        for contour in contours:
            # Skip tiny contours
            area = cv2.contourArea(contour)
            
            # Calculate contour center
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            
            # Distance from image center (normalized)
            dist_from_center = np.sqrt(((cx - center_x) / width)**2 + ((cy - center_y) / height)**2)
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Generate points string
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])
            
            # Calculate importance (area, proximity to center, complexity)
            importance = (
                area * 
                (1 - dist_from_center) * 
                (1 / (len(approx) + 1))
            )
            
            color_features.append({
                'points': points,
                'color': hex_color,
                'area': area,
                'importance': importance,
                'point_count': len(approx),
                'original_contour': approx
            })
        
        # Sort features by importance within this color
        color_features.sort(key=lambda x: x['importance'], reverse=True)
        hierarchical_features.extend(color_features)
    
    # Final sorting by overall importance
    hierarchical_features.sort(key=lambda x: x['importance'], reverse=True)
    
    return hierarchical_features

def simplify_polygon(points_str, simplification_level):
    """
    Simplify a polygon by reducing coordinate precision or number of points
    
    Args:
        points_str (str): Space-separated "x,y" coordinates
        simplification_level (int): Level of simplification (0-3)
    
    Returns:
        str: Simplified points string
    """
    if simplification_level == 0:
        return points_str
    
    points = points_str.split()
    
    # Level 1: Round to 1 decimal place
    if simplification_level == 1:
        return " ".join([f"{float(p.split(',')[0]):.1f},{float(p.split(',')[1]):.1f}" for p in points])
    
    # Level 2: Round to integer
    if simplification_level == 2:
        return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in points])
    
    # Level 3: Reduce number of points (keep every other point, but ensure at least 3 points)
    if simplification_level == 3:
        if len(points) <= 4:
            # If 4 or fewer points, just round to integer
            return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in points])
        else:
            # Keep approximately half the points, but maintain at least 3
            step = min(2, len(points) // 3)
            reduced_points = [points[i] for i in range(0, len(points), step)]
            # Ensure we keep at least 3 points and the last point
            if len(reduced_points) < 3:
                reduced_points = points[:3]
            if points[-1] not in reduced_points:
                reduced_points.append(points[-1])
            return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in reduced_points])
    
    return points_str

def add_ocr_decoy_svg(svg_code: str) -> str:
    """
    Adds nested circles with second darkest and second brightest colors from the existing SVG,
    positioned in one of the four corners (randomly selected) but positioned to avoid being
    cropped out during image processing.
    
    Parameters:
    -----------
    svg_code : str
        The original SVG string
    
    Returns:
    --------
    str
        Modified SVG with the nested circles added
    """
    import random
    import re
    from colorsys import rgb_to_hls, hls_to_rgb
    
    # Check if SVG has a closing tag
    if "</svg>" not in svg_code:
        return svg_code
    
    # Extract viewBox if it exists to understand the dimensions
    viewbox_match = re.search(r'viewBox=["\'](.*?)["\']', svg_code)
    if viewbox_match:
        viewbox = viewbox_match.group(1).split()
        try:
            x, y, width, height = map(float, viewbox)
        except ValueError:
            # Default dimensions if we can't parse viewBox
            width, height = 384, 384
    else:
        # Default dimensions if viewBox not found
        width, height = 384, 384
    
    # Function to convert hex color to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Function to convert RGB to hex
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
    
    # Function to calculate color lightness
    def get_lightness(color):
        # Handle different color formats
        if color.startswith('#'):
            rgb = hex_to_rgb(color)
            return rgb_to_hls(*rgb)[1]  # Lightness is the second value in HLS
        elif color.startswith('rgb'):
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if rgb_match:
                r, g, b = map(lambda x: int(x)/255, rgb_match.groups())
                return rgb_to_hls(r, g, b)[1]
        return 0.5  # Default lightness if we can't parse
    
    # Extract all colors from the SVG
    color_matches = re.findall(r'(?:fill|stroke)="(#[0-9A-Fa-f]{3,6}|rgb\(\d+,\s*\d+,\s*\d+\))"', svg_code)
    
    # Default colors in case we don't find enough
    second_darkest_color = "#333333"  # Default to dark gray
    second_brightest_color = "#CCCCCC"  # Default to light gray
    
    if color_matches:
        # Remove duplicates and get unique colors
        unique_colors = list(set(color_matches))
        
        # Calculate lightness for each unique color
        colors_with_lightness = [(color, get_lightness(color)) for color in unique_colors]
        
        # Sort by lightness (brightness)
        sorted_colors = sorted(colors_with_lightness, key=lambda x: x[1])
        
        # Handle different scenarios based on number of unique colors
        if len(sorted_colors) >= 4:
            # We have at least 4 unique colors - use 2nd darkest and 2nd brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[-2][0]
        elif len(sorted_colors) == 3:
            # We have 3 unique colors - use 2nd darkest and brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[2][0]
        elif len(sorted_colors) == 2:
            # We have only 2 unique colors - use the darkest and brightest
            second_darkest_color = sorted_colors[0][0]
            second_brightest_color = sorted_colors[1][0]
        elif len(sorted_colors) == 1:
            # Only one color - use it for second_darkest and a derived lighter version
            base_color = sorted_colors[0][0]
            base_lightness = sorted_colors[0][1]
            second_darkest_color = base_color
            
            # Create a lighter color variant if the base is dark, or darker if base is light
            if base_lightness < 0.5:
                # Base is dark, create lighter variant
                second_brightest_color = "#CCCCCC"
            else:
                # Base is light, create darker variant
                second_darkest_color = "#333333"
    
    # Ensure the colors are different
    if second_darkest_color == second_brightest_color:
        # If they ended up the same, modify one of them
        if get_lightness(second_darkest_color) < 0.5:
            # It's a dark color, make the bright one lighter
            second_brightest_color = "#CCCCCC"
        else:
            # It's a light color, make the dark one darker
            second_darkest_color = "#333333"
    
    # Base size for the outer circle
    base_outer_radius = width * 0.023
    
    # Randomize size by ±10%
    size_variation = base_outer_radius * 0.1
    outer_radius = base_outer_radius + random.uniform(-size_variation, size_variation)
    
    # Define radii for inner circles based on outer radius
    middle_radius = outer_radius * 0.80
    inner_radius = middle_radius * 0.65
    
    # Calculate the maximum crop margin based on the image processing (5% of dimensions)
    # Add 20% extra margin for safety
    crop_margin_w = int(width * 0.05 * 1.2)
    crop_margin_h = int(height * 0.05 * 1.2)
    
    # Calculate center point based on the outer radius to ensure the entire circle stays visible
    safe_offset = outer_radius + max(crop_margin_w, crop_margin_h)
    
    # Choose a random corner (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
    corner = random.randint(0, 3)
    
    # Position the circle in the chosen corner, accounting for crop margin
    if corner == 0:  # Top-left
        center_x = safe_offset
        center_y = safe_offset
    elif corner == 1:  # Top-right
        center_x = width - safe_offset
        center_y = safe_offset
    elif corner == 2:  # Bottom-left
        center_x = safe_offset
        center_y = height - safe_offset
    else:  # Bottom-right
        center_x = width - safe_offset
        center_y = height - safe_offset
    
    # Add a small random offset (±10% of safe_offset) to make positioning less predictable
    random_offset = safe_offset * 0.1
    center_x += random.uniform(-random_offset, random_offset)
    center_y += random.uniform(-random_offset, random_offset)
    
    # Round to 1 decimal place to keep file size down
    outer_radius = round(outer_radius, 1)
    middle_radius = round(middle_radius, 1)
    inner_radius = round(inner_radius, 1)
    center_x = round(center_x, 1)
    center_y = round(center_y, 1)
    
    # Create the nested circles
    outer_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{outer_radius}" fill="{second_darkest_color}" />'
    middle_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{middle_radius}" fill="{second_brightest_color}" />'
    inner_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{inner_radius}" fill="{second_darkest_color}" />'
    
    # Create a group element that contains all three circles
    group_element = f'{outer_circle}{middle_circle}{inner_circle}'
    
    # Insert the group element just before the closing SVG tag
    modified_svg = svg_code.replace("</svg>", f"{group_element}</svg>")
    
    return modified_svg

def bitmap_to_svg_layered(image, max_size_bytes=9810, resize=True, target_size=(384, 384), 
                         adaptive_fill=True, num_colors=None):
    """
    Convert bitmap to SVG using layered feature extraction with optimized space usage
    
    Args:
        image: Input image (PIL.Image)
        max_size_bytes (int): Maximum SVG size
        resize (bool): Whether to resize the image before processing
        target_size (tuple): Target size for resizing (width, height)
        adaptive_fill (bool): Whether to adaptively fill available space
        num_colors (int): Number of colors to quantize, if None uses adaptive selection
    
    Returns:
        str: SVG representation
    """
    # Adaptive color selection based on image complexity
    if num_colors is None:
        # Simple heuristic: more colors for complex images
        if resize:
            pixel_count = target_size[0] * target_size[1]
        else:
            pixel_count = image.size[0] * image.size[1]
        
        if pixel_count < 65536:  # 256x256
            num_colors = 8
        elif pixel_count < 262144:  # 512x512
            num_colors = 12
        else:
            num_colors = 16
    
    # Resize the image if requested
    if resize:
        original_size = image.size
        image = image.resize(target_size, Image.LANCZOS)
    else:
        original_size = image.size
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Get image dimensions
    height, width = img_np.shape[:2]
    
    # Calculate average background color
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        avg_bg_color = np.mean(img_np, axis=(0,1)).astype(int)
        bg_hex_color = compress_hex_color(f'#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}')
    else:
        bg_hex_color = '#fff'
    
    # Start building SVG
    # Use original dimensions in viewBox for proper scaling when displayed
    orig_width, orig_height = original_size
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">'
    svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>'
    svg_base = svg_header + svg_bg
    svg_footer = '</svg>'
    
    # Calculate base size
    base_size = len((svg_base + svg_footer).encode('utf-8'))
    available_bytes = max_size_bytes - base_size
    
    # Extract hierarchical features
    features = extract_features_by_scale(img_np, num_colors=num_colors)
    
    
    # For adaptive fill, use binary search to find optimal simplification level
    
    # First attempt: calculate size of all features at different simplification levels
    feature_sizes = []
    for feature in features:
        feature_sizes.append({
            'original': len(f'<path d="M{feature["points"]}z"/>'.encode('utf-8')),
            'level1': len(f'<path d="M{simplify_polygon(feature["points"], 1)}z"/>'.encode('utf-8')),
            'level2': len(f'<path d="M{simplify_polygon(feature["points"], 2)}z"/>'.encode('utf-8')),
            'level3': len(f'<path d="M{simplify_polygon(feature["points"], 3)}z"/>'.encode('utf-8'))
        })

    # Two-pass approach: first add most important features, then fill remaining space
    svg, extended_svg = [svg_base], [svg_base]
    bytes_used, extended_bytes_used = base_size, base_size
    added_features, extended_added_features = set(), set()
    features_list, extended_features_list = defaultdict(list), defaultdict(list)
    
    # Pass 2: Try to add remaining features with progressive simplification
    for level in range(2, 4):  # Try simplification levels 1-3
        for i, feature in enumerate(features):
            if i in extended_added_features:
                continue

            feature_size = feature_sizes[i][f'level{level}']
            if feature['color'] not in features_list:
                feature_size += 22

            if bytes_used + feature_size <= max_size_bytes:
                feature_svg = f'<path d="M{simplify_polygon(feature["points"], level)}z"/>'
                features_list[feature['color']].append(feature_svg)
                bytes_used += feature_size
                added_features.add(i)

            if extended_bytes_used + feature_size <= 13000:
                feature_svg = f'<path d="M{simplify_polygon(feature["points"], level)}z"/>'
                extended_features_list[feature['color']].append(feature_svg)
                extended_bytes_used += feature_size
                extended_added_features.add(i)
    
    # Finalize SVG
    for color in extended_features_list:
        extended_svg.append(f'<g fill="{color}">')
        extended_svg.extend(extended_features_list[color])
        extended_svg.append('</g>')
    extended_svg.append(svg_footer)
    extended_svg = "".join(extended_svg)
    
    extended_svg = scour_svg_string(add_ocr_decoy_svg(extended_svg))
    final_size = len(extended_svg.encode('utf-8'))
    if final_size <= 9995:
        return extended_svg
    else:
        # Fall back to other SVG
        for color in features_list:
            svg.append(f'<g fill="{color}">')
            svg.extend(features_list[color])
            svg.append('</g>')
        svg.append(svg_footer)
        svg = "".join(svg)

        svg = add_ocr_decoy_svg(svg)
        
        # Double check we didn't exceed limit
        final_size = len(svg.encode('utf-8'))
        
        if final_size > 10000:
            # If we somehow went over, return basic SVG
            return add_ocr_decoy_svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>')

        return svg

def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)

class FasterImageProcessor(metric.ImageProcessor):

    def __init__(self, image: Image.Image, seed=None):
        super().__init__(image, seed)

    def apply_median_filter(self, size=9):
        """Fast median filter using OpenCV (≈10-20× PIL speed)."""
        # Pillow → NumPy (BGR order is okay; cv2 works on uint8 directly)
        img_array = np.asarray(self.image)
        # OpenCV expects odd kernel sizes ≥3
        if size % 2 == 0:
            size += 1
        filtered = cv2.medianBlur(img_array, ksize=size)
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency=0.5, device=0):
        x = cp.asarray(self.image, dtype=cp.float32)      # H×W×3
        f = cp.fft.fftshift(cp.fft.fft2(x, axes=(0,1)), axes=(0,1))
    
        rows, cols = x.shape[:2]
        crow, ccol = rows // 2, cols // 2
        r = int(min(crow, ccol) * cutoff_frequency)
    
        y, xg = cp.ogrid[:rows, :cols]
        mask = ((y - crow) ** 2 + (xg - ccol) ** 2) <= r * r
        f *= mask[..., None]                              # broadcast over channels
    
        img_back = cp.fft.ifft2(cp.fft.ifftshift(f, axes=(0,1)), axes=(0,1)).real
        img_back = cp.clip(img_back, 0, 255).astype(cp.uint8).get()
        self.image = Image.fromarray(img_back)
        return self


def get_aes_and_ocr_score(svg_content, prompt, aesthetic_only=False, compute_aesthetic=True):
    image_processor = FasterImageProcessor(image=svg_to_png(svg_content), seed=33).apply()
    image = image_processor.image.copy()
    
    if not aesthetic_only:
        query_template = 'Does <image> portray "{}"? Answer yes or no.'
        query = query_template.format(prompt)
        vqa_score = metric.vqa_evaluator.score_yes_no(query, image)
        
    if compute_aesthetic:
        aesthetic_score = metric.aesthetic_evaluator.score(image)
    else:
        aesthetic_score = 1
        
    if not aesthetic_only:
        image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
        ocr_score = metric.vqa_evaluator.ocr(image_processor.image)
        
    if aesthetic_only:
        return aesthetic_score
    else:
        return vqa_score, aesthetic_score, ocr_score

class Model:
    def __init__(self):
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.negative_prompt = "lines, framing, hatching, background, textures, patterns, details, outlines"

        self.num_inference_steps = 6
        self.guidance_scale = 2.5
        self.num_attempt = 10

    def gen_bitmap(self, description):
        prompt = f'Cartoon style of "{description}".'
        bitmap = generate_bitmap(prompt, self.negative_prompt, self.num_inference_steps, self.guidance_scale)
        return bitmap

    def predict_impl(self, prompt: str) -> str:
        best_score = 0.0
        best_svg = None
        best_img = None
        start_time = time.time()
        attempt = 0

        results = []

        # Generate 20 SVG attempts with aesthetic scoring
        for attempt in range(self.num_attempt):
            print(f"===== Attempt {attempt + 1} =====")
            bitmap = self.gen_bitmap(prompt)
            svg = bitmap_to_svg_layered(bitmap)

            score = get_aes_and_ocr_score(svg, prompt, aesthetic_only=True)
            print("AES score: ", score)
            results.append((score, bitmap, svg, svg_to_png(svg)))
            print(f"Attempt {attempt + 1} finished")

        svgs = [x[3] for x in results]

        aesthetic_scores = np.array([x[0] for x in results])
        siglip1_scores = np.array(cosine_batch_siglip(siglip1_model, siglip1_processor, prompt, [svgs], is_siglip2=False)[0])
        ranked_scores = aesthetic_scores * siglip1_scores

        print(f"aesthetic scores: {aesthetic_scores}")
        print(f"siglip1 scores: {siglip1_scores}")
        print(f"ranked scores: {ranked_scores}")

        sorted_indices = np.argsort(ranked_scores)[::-1]

        print(f"sorted indices: {sorted_indices}")

        # Evaluate top 5 with VQA
        for i, current_index in enumerate(sorted_indices[:5]):
            aesthetic_score, bitmap, svg = results[current_index][0], results[current_index][1], results[current_index][2]
            print(f"index {current_index}, aesthetic score: {aesthetic_score}")
            
            vqa_score, _, ocr_score = get_aes_and_ocr_score(svg, prompt, compute_aesthetic=False)
            score = metric.harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
            print(vqa_score, aesthetic_score, ocr_score)
            print("Proxy score: ", score)
            if score >= best_score:
                best_score = score
                best_svg = svg
                best_img = bitmap
                print('update score:', best_score)
            print(f"Top-{i+1} evaluation finished")
        
        if best_svg is None:
            best_svg = self.default_svg

        return best_svg, best_img

    def predict(self, prompt: str) -> str:
        svg, img = self.predict_impl(prompt)
        return svg, img

def generate_svg(prompt: str):
    """Main function to generate SVG from text prompt"""
    model = Model()
    return model.predict(prompt)