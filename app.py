import streamlit as st
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from svg_generator import generate_svg, svg_to_png

def main():
    st.set_page_config(page_title="Text-to-SVG Generator", layout="wide")
    
    st.markdown("**Text-to-SVG Generator using the 16th place solution approach**")
    st.markdown("This app generates 20 SVG attempts, scores them with aesthetic * SigLIP, evaluates the top 5 with VQA, and returns the image with the highest harmonic mean.")
    
    # User input
    prompt = st.text_input(
        "Enter your prompt:", 
        value="a purple forest at dusk",
        help="Describe what you want to generate as an SVG image"
    )
    
    # Generate button
    if st.button("🚀 Generate SVG", type="primary"):
        if prompt.strip():
            with st.spinner("Generating SVG... This may take a few minutes as we generate 20 attempts and evaluate the best ones."):
                try:
                    # Generate SVG and bitmap
                    svg_content, bitmap_image = generate_svg(prompt.strip())
                    
                    # Convert SVG to PNG for display
                    svg_png = svg_to_png(svg_content)
                    
                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📱 Original Bitmap")
                        st.image(bitmap_image, caption="Generated bitmap image", use_column_width=True)
                    
                    with col2:
                        st.subheader("🖼️ Final SVG (Best Result)")
                        st.image(svg_png, caption="SVG converted to PNG for display", use_column_width=True)
                    
                    # Show SVG details
                    st.subheader("📄 SVG Code")
                    with st.expander("Click to view SVG code"):
                        st.code(svg_content, language="xml")
                    
                    # Download options
                    st.subheader("💾 Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # SVG download
                        svg_bytes = svg_content.encode('utf-8')
                        st.download_button(
                            label="📁 Download SVG",
                            data=svg_bytes,
                            file_name=f"{prompt.replace(' ', '_')}.svg",
                            mime="image/svg+xml"
                        )
                    
                    with col2:
                        # PNG download
                        png_buffer = BytesIO()
                        svg_png.save(png_buffer, format="PNG")
                        png_bytes = png_buffer.getvalue()
                        st.download_button(
                            label="🖼️ Download PNG",
                            data=png_bytes,
                            file_name=f"{prompt.replace(' ', '_')}.png",
                            mime="image/png"
                        )
                    
                    # Display SVG file size
                    svg_size = len(svg_content.encode('utf-8'))
                    st.info(f"📊 SVG file size: {svg_size:,} bytes (limit: 10,000 bytes)")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.error("Please try again with a different prompt or check the logs for more details.")
        else:
            st.warning("⚠️ Please enter a prompt to generate an SVG.")
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### 16th Place Solution Approach
        
        This application implements the 16th place solution from the Kaggle "Drawing with LLMs" competition:
        
        1. **Generate 20 attempts**: Creates 20 different bitmap images using Stable Diffusion XL
        2. **Convert to SVG**: Each bitmap is converted to SVG using hierarchical feature extraction
        3. **Aesthetic scoring**: Each SVG is scored for aesthetic quality
        4. **SigLIP ranking**: Uses SigLIP model to rank images by text-image similarity
        5. **Combined ranking**: Multiplies aesthetic score × SigLIP score
        6. **VQA evaluation**: Evaluates top 5 candidates using Visual Question Answering
        7. **Final selection**: Returns the image with highest harmonic mean of VQA + aesthetic scores
        
        ### Technical Details
        - **Model**: Stable Diffusion XL Flash
        - **SVG Conversion**: GPU-accelerated K-means clustering + contour detection
        - **Scoring**: Aesthetic evaluator + SigLIP text-image similarity + VQA
        - **Final Metric**: Harmonic mean(VQA, Aesthetic, β=0.5) × OCR score
        """)
    
    # Example prompts
    with st.expander("💡 Example prompts to try"):
        example_prompts = [
            "a purple forest at dusk",
            "a lighthouse overlooking the ocean", 
            "gray wool coat with a faux fur collar",
            "orange corduroy overalls",
            "a snowy plain",
            "crimson rectangles forming a chaotic grid",
            "purple pyramids spiraling around a bronze cone",
            "a starlit night over snow-covered peaks",
            "black and white checkered pants",
            "a green lagoon under a cloudy sky"
        ]
        
        for prompt_example in example_prompts:
            if st.button(f"🎯 {prompt_example}", key=prompt_example):
                st.rerun()

if __name__ == "__main__":
    main()