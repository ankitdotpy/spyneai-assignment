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
        gdown https://drive.google.com/uc?id=15F2IyaRAa4nGydRaf8Ny9NfoqCwwx0Pr
        mv best_model_efficientnet_b0.pth models/$MODEL/
    fi
fi

# Run the API
CMD="python api.py --model $MODEL"
echo "Running: $CMD"
$CMD

