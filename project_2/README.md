# Project 2: Car Image Classification

## Instructions
### Training the model
Model can trained using the script `scripts/train_model.sh`. It supports the following options:
- `--model`: Model to train. See below for supported models.
- `--epochs`: Number of epochs to train the model.
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for training.
- `--data_dir`: Directory containing the training and validation data.

Supported models:
- `resnet50`
- `resnet18`
- `efficientnet_b0`
- `efficientnet_v2_s`
- `vit_b_16`
- `regnet_y_32gf`

Example usage:
```bash
./scripts/train_model.sh --model resnet18 --epochs 10 --batch_size 32 --learning_rate 0.001 --data_dir data/
```

### Running the API
API can be run using the script `scripts/serve.sh`. It supports the following options:
- `--model`: Model to use. Pretrained models are available for `resnet18` and `efficientnet_b0`.

Example usage:
```bash
./scripts/serve.sh --model efficientnet_b0
```
