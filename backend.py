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
        secrets=[modal.Secret.from_name("github-secret")],
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
        "streamlit",
        "matplotlib",
        "more_itertools",
        "sentencepiece",
    ])
    .add_local_file(Path(__file__).parent / "config.yaml", "/root/config.yaml")
    .add_local_file(Path(__file__).parent / "app.py", "/root/app.py")
    .add_local_file(Path(__file__).parent / "metric.py", "/root/metric.py")
    .add_local_file(Path(__file__).parent / "svg_generator.py", "/root/svg_generator.py")
)

models_volume = Volume.from_name("drawing-with-llms", create_if_missing=True)
app = App("text-to-svg-generator-streamlit", image=image)

@app.function(
    image=image, 
    volumes={"/root/cache": models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-40GB"
)
@modal.web_server(8000)
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
    
    target = shlex.quote("/root/app.py")
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)

