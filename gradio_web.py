
import gradio as gr
import subprocess
import os
import shutil
import time
import glob

def predict(image):
    # Ensure data directory exists
    os.makedirs("/app/data", exist_ok=True)
    
    input_path = "/app/data/input.jpg"
    
    # Save/Copy input image
    # image provided by gradio (type='filepath') is a temp path
    shutil.copy(image, input_path)

    # Run sharp command
    # sharp predict -i /app/data/input.jpg -o /app/data/output --render
    cmd = [
        "sharp", "predict",
        "-i", input_path,
        "-o", "/app/data/output",
        "--render"
    ]
    
    # Execute command
    try:
        t = time.time()
        print("Sharp started")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Sharp command took {round(time.time() - t, 3)} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Error running sharp: {e}")
        print(f"Stdout: {e.stdout.decode()}")
        print(f"Stderr: {e.stderr.decode()}")
        return None

    # Find output videos
    rgb_video = "/app/data/output/input.mp4"
    depth_video = "/app/data/output/input.depth.mp4"
    
    if os.path.exists(rgb_video) and os.path.exists(depth_video):
        return rgb_video, depth_video
    elif os.path.exists(rgb_video):
        return rgb_video, None
        
    return None, None

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Input Image"),
    outputs=[gr.Video(label="RGB Video"), gr.Video(label="Depth Video")],
    title="Sharp 3D View Synthesis",
    description="Upload an image to generate a 3D view synthesis video."
)

if __name__ == "__main__":
    print("Sharp Monocular View Synthesis in Less Than a Second (https://github.com/apple/ml-sharp)")
    demo.launch(server_name="0.0.0.0", server_port=7860)
