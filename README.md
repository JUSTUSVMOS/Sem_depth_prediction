# ResNet18-UNet Depth Estimation

This repository provides an implementation for training a depth estimation model using a ResNet18-UNet architecture. The project is built on Python 3.9, CUDA 11.8, and PyTorch 2.0.0, and it includes tools for both training and fine-tuning the model.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Fine-tuning](#fine-tuning)
- [Dataset Preparation](#dataset-preparation)
  - [NYU Depth V2 Format](#nyu-depth-v2-format)
  - [Nebula SEM Data Preparation](#nebula-sem-data-preparation)
  - [Blender File Preparation](#blender-file-preparation)
- [Results](#results)
- [License](#license)

## Features

- **Model Architecture:** ResNet18-UNet for depth estimation.
- **Training Notebook:** `resnet18-unet.ipynb` for model training.
- **Fine-Tuning Script:** `finetune.py` for further model fine-tuning.
- **Environment Setup:** Easy environment installation using Conda with the provided `environment.yaml`.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Setup the Conda Environment:**

   Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed. Then, create the environment by running:

   ```bash
   conda env create -f environment.yaml
   conda activate your-environment-name
   ```

   *Note: Replace `your-environment-name` with the actual environment name specified in `environment.yaml` if different.*

3. **Verify CUDA and PyTorch Setup:**

   The project uses CUDA 11.8 and PyTorch 2.0.0. Make sure your system supports these configurations.

## Usage

### Training

To train the model, open the Jupyter Notebook `resnet18-unet.ipynb`:

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook resnet18-unet.ipynb
   ```

2. Follow the instructions within the notebook to train the depth estimation model on your dataset.

### Fine-tuning

For fine-tuning an already trained model, use the `finetune.py` script:

```bash
python finetune.py --model_path path/to/your/model.pth --other_options value
```

*Customize the command-line arguments in `finetune.py` as needed for your use case.*

## Dataset Preparation

### NYU Depth V2 Format

The dataset should be organized following the NYU Depth V2 format:

- **Original Images:** Use JPG files.
- **Depth Maps:** Each depth map should be named as the original image filename appended with `_depth.png`.
- **Directory Structure:** Place both the original images and their corresponding depth maps in a folder named `data`.

For example:

```
data/
├── image1.jpg
├── image1_depth.png
├── image2.jpg
├── image2_depth.png
└── ...
```

Compress the dataset folder into a ZIP file to provide it for training.

### Nebula SEM Data Preparation

- Follow the setup instructions in the `Nebula-Installation` folder to configure Nebula SEM.
- Ensure that the simulated SEM images are correctly formatted and stored in the dataset.
- The processed SEM images should be saved in the `data/` directory, following the same naming convention as NYU Depth V2.

### Blender File Preparation

- Refer to the `Blender_code` folder for scripts and configurations used to generate synthetic depth maps.
- Ensure that the Blender-generated images and their depth maps follow the required dataset format before use in training.

## Results

After training or fine-tuning the model, save your results (such as sample predictions, comparison images, and evaluation metrics) in the `results/` folder. Example directory structure:

```
![image](https://github.com/user-attachments/assets/2639d61c-29c2-4f34-a99c-097e478f8196)
![image](https://github.com/user-attachments/assets/f84599e9-094b-47fc-bc98-5f71ed9661e6)
```


## Acknowledgements

- Based on the NYU Depth V2 dataset format.
- Built with PyTorch and CUDA for high-performance depth estimation.
- Utilizes Nebula SEM for synthetic SEM image generation.
- Uses Blender for simulated wafer surface modeling and depth map generation.

For any issues or feature requests, please open an issue on GitHub.
