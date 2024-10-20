#!/bin/bash

# Default value
MODEL="resnet18"

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

# Run the API
CMD="python api.py --model $MODEL"
echo "Running: $CMD"
$CMD

