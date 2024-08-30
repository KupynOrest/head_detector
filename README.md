<div align="center">

# VGGHeads: A Large-Scale Synthetic Dataset for 3D Human Heads

[**Orest Kupyn**](https://github.com/KupynOrest)<sup>13</sup> · [**Eugene Khvedchenia**](https://github.com/BloodAxe)<sup>2</sup> · [**Christian Rupprecht**](https://chrirupp.github.io/)<sup>1</sup> ·

<sup>1</sup>University of Oxford · <sup>2</sup>Ukrainian Catholic University · <sup>3</sup>PiñataFarms AI

<a href='https://www.robots.ox.ac.uk/~vgg/research/vgg-heads/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2407.18245'><img src='https://img.shields.io/badge/arXiv Paper-red'></a>
<a href='https://huggingface.co/spaces/okupyn/vgg_heads'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href='https://huggingface.co/okupyn/head-mesh-controlnet-xl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20ControlNet%20XL-blue'></a>
[![Model](https://img.shields.io/badge/Model-Weights-blue)](https://huggingface.co/okupyn/vgg_heads)

</div>

VGGHeads is a large-scale fully synthetic dataset for human head detection and 3D mesh estimation with over 1 million images generated with diffusion models. A model trained only on synthetic data generalizes well to real-world and is capable of simultaneous heads detection and head meshes reconstruction from a single image in a single step.

![banner](./images/banner.jpg)

## News
- [2024/08/29] 🔥🔥 We release the dataset, training instructions and ONNX weights!!
- [2024/08/09] 🔥 We release VGGHeads_L Checkpoint and [Mesh ControlNet](https://huggingface.co/okupyn/head-mesh-controlnet-xl)
- [2024/07/26] 🔥 We release the initial version of the codebase, the paper, project webpage and an image demo!!

## VGGHeads Dataset Download Instructions

### 1. Download the Dataset

To download the VGGHeads dataset, you have two options:

1. Torrent download (preferred method):
 <a href='https://academictorrents.com/docs/downloading.html'>How To Download</a>
```bash
pip install academictorrents
at-get 1ac36f16386061685ed303dea6f0d6179d2e2121
```
or use  <a href='https://aria2.github.io/'>aria2c</a>

```bash
aria2c --seed-time=0 --max-overall-download-limit=10M --file-allocation=none https://academictorrents.com/download/1ac36f16386061685ed303dea6f0d6179d2e2121.torrent
```



<a href='https://academictorrents.com/download/1ac36f16386061685ed303dea6f0d6179d2e2121.torrent'>Full Torrent Link</a>

We recommend using the torrent method as it's typically faster and helps reduce the load on our servers.

2. Direct download:
   
```bash
wget https://thor.robots.ox.ac.uk/vgg-heads/VGGHeads.tar
```

This will download a file named `VGGHeads.tar` to your current directory.

### 2. Download the MD5 Checksums

To verify the integrity of the downloaded file, we'll need the MD5 checksums. Download them using:

```bash
wget https://thor.robots.ox.ac.uk/vgg-heads/MD5SUMS
```

### 3. Verify the Download

After both files are downloaded, verify the integrity of the `VGGHeads.tar` file:

```bash
md5sum -c MD5SUMS
```

If the download was successful and the file is intact, you should see an "OK" message.

### 4. Extract the Dataset

If the verification was successful, extract the contents of the tar file:

```bash
tar -xvf VGGHeads.tar
```

This will extract the contents of the archive into your current directory.

Notes:

- The size of the dataset is approximately 187 GB. Ensure you have sufficient disk space before downloading and extracting.
- The download and extraction process may take some time depending on your internet connection and computer speed.
- If you encounter any issues during the download or extraction process, try the download again or check your system's tar utility.

## Installation

#### Create a Conda virtual environment

```bash
conda create --name vgg_heads python=3.10
conda activate vgg_heads
```

#### Clone the project and install the package

```bash
git clone https://github.com/KupynOrest/head_detector.git
cd head_detector

pip install -e ./
```

Or simply install

```bash
pip install git+https://github.com/KupynOrest/head_detector.git
```

## Usage

To test VGGHeads model on your own images simply use this code:

```python
from head_detector import HeadDetector
import cv2
detector = HeadDetector()
image_path = "your_image.jpg"
predictions = detector(image_path)
# predictions.heads contain a list of heads with .bbox, .vertices_3d, .head_pose params
result_image = predictions.draw() # draw heads on the image
cv2.imwrite("result.png",result_image) # save result image to preview it.
```

### Exporting Head Meshes

You can export head meshes as OBJ files using the `save_meshes` method:

```python
# After getting predictions
save_folder = "path/to/save/folder"
predictions.save_meshes(save_folder)
```

This will save individual OBJ files for each detected head in the specified folder.

### Getting Aligned Head Crops

To obtain aligned head crops, use the `get_aligned_heads` method:

```python
# After getting predictions
aligned_heads = predictions.get_aligned_heads()

# Process or save aligned head crops
for i, head in enumerate(aligned_heads):
    cv2.imwrite(f"aligned_head_{i}.png", head)
```

This returns a list of aligned head crops that you can further process or save.

### Extended Example

Here's a complete example incorporating all features:

```python
from head_detector import HeadDetector
import cv2
import os

# Initialize the detector
detector = HeadDetector()

# Specify the path to your image
image_path = "your_image.jpg"

# Get predictions
predictions = detector(image_path)

# Draw heads on the image
result_image = predictions.draw()
cv2.imwrite("result.png", result_image)

# Save head meshes
save_folder = "head_meshes"
os.makedirs(save_folder, exist_ok=True)
predictions.save_meshes(save_folder)

# Get and save aligned head crops
aligned_heads = predictions.get_aligned_heads()
for i, head in enumerate(aligned_heads):
    cv2.imwrite(f"aligned_head_{i}.png", head)

print(f"Detected {len(predictions.heads)} heads.")
print(f"Result image saved as 'result.png'")
print(f"Head meshes saved in '{save_folder}' folder")
print(f"Aligned head crops saved as 'aligned_head_*.png'")
```

This extended example demonstrates how to use all the features of the VGGHeads model, including basic head detection, drawing results, exporting head meshes, and obtaining aligned head crops.

Additionally, the ONNX weights are available at <a href='https://huggingface.co/okupyn/vgg_heads/tree/main'>HuggingFace</a>. The example of the inference can be found at: <a href='https://colab.research.google.com/drive/1EJn9dPdlX2qIWrZok9LF185ZJwAGOr9Y'>Colab</a>

## Gradio Demo

We also provide a Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> demo, which you can run locally:

```bash
cd gradio
pip install -r requirements.txt
python app.py
```
You can specify the `--server_port`, `--share`, `--server_name` arguments to satisfy your needs!

## Training

Check `yolo_head_training/Makefile` for examples of train scripts.

To run the training on all data with Distributed Data Parallel (DDP), use the following command:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS train.py --config-name=yolo_heads_l \
    dataset_params.train_dataset_params.data_dir=DATA_FOLDER/large \
    dataset_params.val_dataset_params.data_dir=DATA_FOLDER/large \
    num_gpus=NUM_GPUS multi_gpu=DDP
```

Replace the following placeholders:
- `NUM_GPUS`: The number of GPUs you want to use for training.
- `DATA_FOLDER`: The path to the directory containing your extracted dataset.

### Additional Training Options

1. Single GPU Training:
   If you're using a single GPU, you can simplify the command:
   ```bash
   python train.py --config-name=yolo_heads_l \
       dataset_params.train_dataset_params.data_dir=DATA_FOLDER/large \
       dataset_params.val_dataset_params.data_dir=DATA_FOLDER/large
   ```

2. Custom Configuration:
   You can modify the `--config-name` parameter to use different model configurations. Check the configuration files in the project directory for available options.

3. Adjusting Hyperparameters:
   You can adjust various hyperparameters by adding them to the command line. For example:
   ```bash
   python train.py --config-name=yolo_heads_l \
       dataset_params.train_dataset_params.data_dir=DATA_FOLDER/large \
       dataset_params.val_dataset_params.data_dir=DATA_FOLDER/large \
       training_hyperparams.initial_lr=0.001 \
       training_hyperparams.max_epochs=100
   ```

4. Resuming Training:
   If you need to resume training from a checkpoint, you can use the `training_hyperparams.resume` flag:
   ```bash
   python train.py --config-name=yolo_heads_l \
       dataset_params.train_dataset_params.data_dir=DATA_FOLDER/large \
       dataset_params.val_dataset_params.data_dir=DATA_FOLDER/large \
       training_hyperparams.resume=True
   ```

### Monitoring Training

You can monitor the training progress through the console output. Consider using tools like TensorBoard for more detailed monitoring and visualization of training metrics.

## Cite

If you find VGGHeads useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{vggheads,
      title={VGGHeads: A Large-Scale Synthetic Dataset for 3D Human Heads},
      author={Orest Kupyn and Eugene Khvedchenia and Christian Rupprecht},
      year={2024},
      eprint={2407.18245},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.18245},
}
```

 [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
