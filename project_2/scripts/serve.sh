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
        gdown https://drive.google.com/file/d/1RYUm5eAbLlpXsKRAsaz2ZxLCD4UyinXQ
        mv final_model_resnet18.onnx models/$MODEL/
    elif [ "$MODEL" == "efficientnet_b0" ]; then
        gdown https://drive.google.com/file/d/1V85vUrItWk9FLr2UkTnVXFYcFqW1Q22F
        mv final_model_efficientnet_b0.onnx models/$MODEL/
    else
        echo "Pretrained model $MODEL is not available. Please train the model first."
        exit 1
    fi
fi

# Run the API
CMD="python api.py --model $MODEL"
echo "Running: $CMD"
$CMD

