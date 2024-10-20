#!/bin/bash

# Default values
MODEL="efficientnet_b0"
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=0.001
DATA_DIR="data/"
FREEZE_BASE=false

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
    --epochs)
    EPOCHS="$2"
    shift
    shift
    ;;
    --batch_size)
    BATCH_SIZE="$2"
    shift
    shift
    ;;
    --learning_rate)
    LEARNING_RATE="$2"
    shift
    shift
    ;;
    --data_dir)
    DATA_DIR="$2"
    shift
    shift
    ;;
    --freeze_base)
    FREEZE_BASE=true
    shift
    ;;
    *)
    echo "Unknown option: $1"
    exit 1
    ;;
esac
done

# Construct the command
CMD="python src/train.py --model $MODEL --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --data_dir $DATA_DIR"

if $FREEZE_BASE; then
    CMD="$CMD --freeze_base"
fi

# Run the command
echo "Running: $CMD"
$CMD
