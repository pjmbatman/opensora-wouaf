# Training with Video and Image Datasets

This repository provides a training pipeline that can be used for both image and video datasets. The dataset and training parameters can be easily modified through the provided arguments.

## Dataset

By changing the `--dataset_name` argument in `train.sh`, you can select between the following datasets:
- `"HuggingFaceM4/COCO"` for COCO images.
- `"webvid"` for WebVid videos.

You can use any video dataset as long as they follow the path format:
