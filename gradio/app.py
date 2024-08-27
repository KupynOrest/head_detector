import uuid

from fire import Fire
from head_detector import HeadDetector
import gradio as gr

detector = HeadDetector()
INVERSE_MAPPING = {
    'Full': 'full',
    'Head Boxes': 'bbox',
    'Face Landmarks': 'points',
    'Head Mesh': 'landmarks',
    'Head Pose': 'pose'
}


def detect_faces(image, method):
    image = detector(image).draw(method=INVERSE_MAPPING[method])
    return image


iface = gr.Interface(
    fn=detect_faces,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Radio(['Full', 'Head Boxes', 'Face Landmarks', 'Head Mesh', 'Head Pose'], label="Face Detection Method", value='Full')
    ],
    outputs=gr.Image(type="numpy", label="Image with Faces"),
    title="Demo of: VGGHeads: A Large-Scale Synthetic Dataset for 3D Human Heads",
    description="Upload an image and get all predictions at once! The model performs simultaneous heads detection and head meshes reconstruction from a single image in a single step. This demo showcases the capabilities of VGGHeads, a large-scale synthetic dataset designed to advance the field of 3D human head modeling. The model trained on fully synthetic data can accurately detect human heads and reconstruct their 3D meshes with high fidelity in real world. Users can choose from five visualization methods: 'Full', 'Head Boxes', 'Face Landmarks', 'Head Mesh', and 'Head Pose', demonstrating the model's versatility in predicting various aspects of head detection and reconstruction. Ideal for applications in virtual reality, gaming, and animation, VGGHeads aims to bridge the gap between 2D image inputs and 3D head outputs.",
    allow_flagging='never',
)


def main():
    iface.launch()


if __name__ == '__main__':
    Fire(main)
