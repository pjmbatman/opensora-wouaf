# Training with Video and Image Datasets
This repository provides a training pipeline that can be used for both image and video datasets. The dataset and training parameters can be easily modified through the provided arguments.

## Dataset

By changing the `--dataset_name` argument in `train.sh`, you can select between the following datasets:
- `"HuggingFaceM4/COCO"` for COCO images.
- `"webvid"` for WebVid videos.

You can use any video dataset as long as they follow the path format:

### Additional Dataset Arguments:
- `--num_train_data` and `--num_val_data`: These arguments are only relevant for WebVid datasets. They specify the number of data samples to use for training and validation, respectively.
- `--num_result`: Specifies the number of results to visualize during each validation step.

### Example:

In the `train.sh` file, adjust the dataset argument like this to use the COCO dataset:
```bash
--dataset_name "HuggingFaceM4/COCO"

Training Options:
--num_train_data: Number of training samples (for WebVid datasets).
--num_val_data: Number of validation samples (for WebVid datasets).
--num_result: Number of validation results to visualize.
--exp_name: Name of the experiment, useful for organizing multiple runs.
--lr_mult: Learning rate multiplier for affine layers.
--train_batch_size: Batch size for training.

The rest of the arguments remain consistent with the Wouaf framework.

`````
### Training:

Run train.sh
