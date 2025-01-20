This git repository is using git-lfs for storing large files
install with git-lfs install

The dataset used for training is stored in weapon_detection.zip due to big filenames that can't be stored on git, unzip it before running the training.

After setting an environment with PyTorch and YOLO, you can also train a model in the CLI, example:
yolo detect train data=Path_To_data.yaml model=yolo11n.pt epochs=15 imgsz=640 batch=16

Latest model found in runs/detect/train13/weights/best.pt
