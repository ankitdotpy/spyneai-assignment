#!/bin/bash

# Default value
MODEL="efficientnet_b0"

# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --model)
    MODEL="$2"
    shift
    shift
    ;;
    *)
    echo "Unknown option: $1"
    exit 1
    ;;
esac
done

# if models/model_name directory does not exist or is empty, download the models from gdrive
if [ ! -d "models/$MODEL" ] || [ -z "$(ls -A models/$MODEL)" ]; then
    echo "Models directory does not exist or is empty. Downloading models..."
    mkdir -p models/$MODEL
    if [ "$MODEL" == "resnet18" ]; then
        gdown https://drive.google.com/uc?id=1AgSKsDDh9qnYZKjPc6zxGN1RfUfpI-PB
        mv best_model_resnet18.pth models/$MODEL/
    elif [ "$MODEL" == "efficientnet_b0" ]; then
        gdown https://drive.google.com/file/d/1mpEiJt6CU2ZbRxe4yG7YM8bz_jz4bj4C
        mv best_model_efficientnet_b0.pth models/$MODEL/
    else
        echo "Pretrained model $MODEL is not available. Please train the model first."
        exit 1
    fi
fi

# Run the API
CMD="python api.py --model $MODEL"
echo "Running: $CMD"
$CMD

