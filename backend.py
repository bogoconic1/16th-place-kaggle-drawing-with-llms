import shlex
import subprocess
from pathlib import Path
import yaml
import modal
from modal import Image, App, asgi_app, Volume
from fastapi.staticfiles import StaticFiles
import sys, os

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["libcairo2-dev", "pkg-config", "python3-dev", "libgl1", "libpango-1.0-0", "libpangocairo-1.0-0", "gdk-pixbuf2.0-0"])
    .pip_install_private_repos(
        "github.com/bogoconic1/CLIP",
        git_user='bogoconic1',
        secrets=[modal.Secret.from_name("github-secret-2")],
    )
    .pip_install([
        "torch",
        "torchvision", 
        "diffusers",
        "scikit-image",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "pillow",
        "opencv-python",
        "numpy",
        "cupy-cuda12x",
        "cuml-cu12",
        "scour",
        "cairosvg",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "gradio",
        "matplotlib",
        "sentencepiece",
        "more_itertools",
    ])
    .add_local_file(Path(__file__).parent / "conf/config.yaml", "/root/config.yaml")
    .add_local_file(Path(__file__).parent / "code/svg_generator.py", "/root/svg_generator.py")
    .add_local_file(Path(__file__).parent / "code/metric.py", "/root/metric.py")
)

models_volume = Volume.from_name("drawing-with-llms", create_if_missing=True)
app = App("text-to-svg-generator", image=image)

@app.function(
    image=image, 
    volumes={"/root/cache": models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-40GB",
    min_containers=0, # the deployment can continue with 0 containers (in a period with no users, I won't be charged)
    scaledown_window=15*60, # if the app is idle for >= 15 minutes, the container will be stopped to save costs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def run():
    # Check if model exists in cache, if not download it
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    for model in config['models']:
        model_path = f"/root/cache/models/{model}"
        if not os.path.exists(model_path):
            print(f"Downloading {model_path} for first time use...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
            )
            print("Model downloaded successfully!")
    
    import gradio as gr
    from gradio.routes import mount_gradio_app
    import numpy as np
    from PIL import Image
    import base64
    from io import BytesIO
    from svg_generator import generate_svg, svg_to_png
    from fastapi import FastAPI
    
    def generate_svg_interface(prompt):
        if not prompt.strip():
            return None, None, "⚠️ Please enter a prompt to generate an SVG."
        
        try:
            # Generate SVG and bitmap
            svg_content, bitmap_image = generate_svg(prompt.strip())
            svg_png = svg_to_png(svg_content)
            
            return bitmap_image, svg_png, svg_content
            
        except Exception as e:
            error_msg = f"❌ An error occurred: {str(e)}\nPlease try again with a different prompt or check the logs for more details."
            return None, None, error_msg
    
    # Create Gradio interface
    with gr.Blocks(title="Text-to-SVG Generator") as demo:
        gr.Markdown("# 16th place solution for Kaggle's Drawing with LLMs competition")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Enter your prompt:",
                    placeholder="Describe what you want to generate as an SVG image",
                    value="a purple forest at dusk"
                )
                generate_btn = gr.Button("🚀 Generate SVG", variant="primary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📱 Original Bitmap")
                bitmap_output = gr.Image(label="Generated bitmap image")
            
            with gr.Column():
                gr.Markdown("### 🖼️ Final SVG (Best Result)")
                svg_output = gr.Image(label="SVG converted to PNG for display")
        
        with gr.Row():
            svg_code_output = gr.Code(label="📄 SVG Code", language="markdown")
        
        generate_btn.click(
            fn=generate_svg_interface,
            inputs=[prompt_input],
            outputs=[bitmap_output, svg_output, svg_code_output]
        )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")

