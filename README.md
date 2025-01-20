# Weapon Detection Using YOLO

This repository uses [Git Large File Storage (Git LFS)](https://git-lfs.com/) to handle large files.

## Prerequisites

### 1. Install Git LFS
Make sure you have Git LFS installed:
```bash
git lfs install
```

### 2. Dataset Setup
The dataset used for training is stored in `weapon_detection.zip`. Due to filename length limitations, it cannot be directly stored in Git. Please unzip the file before running the training:
```bash
unzip weapon_detection.zip
```

### 3. Environment Setup
Set up a Python environment with **PyTorch** and **YOLO** installed. You can find the installation instructions for [PyTorch](https://pytorch.org/get-started/locally/) and [Ultralytics YOLO](https://docs.ultralytics.com/).

## Training the Model

Once the environment is ready, you can train a YOLO model using the command-line interface (CLI). Example:
```bash
yolo detect train data=Path_To_data.yaml model=yolo11n.pt epochs=15 imgsz=640 batch=16
```

- `data=Path_To_data.yaml`: Path to the YAML file specifying the dataset.
- `model=yolo11n.pt`: The model architecture to use for training.
- `epochs=15`: Number of training epochs.
- `imgsz=640`: Image size for training.
- `batch=16`: Batch size for training.

## Output

The latest trained model is located in:
```
runs/detect/train13/weights/best.pt
```

You can use this model for inference or further fine-tuning.

## Notes

- Ensure the dataset is correctly prepared and split into training and validation sets before starting the training process.
- For more information about YOLO and its configuration, refer to the [Ultralytics YOLO Documentation](https://docs.ultralytics.com/).

---
