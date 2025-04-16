#!/bin/bash
# Trains each model and creates files in /Models:
# binary-sequential.spm
# binary-transformer.spm
# linear.spm
# sequential.spm
# transformer.spm
#
# Command line arguments:
#   -t 1: Sets stock training/testing split to 1
#   -o: Overwrites existing models, not used
#   -s: Specifies stock list file

# set -e

# Activate virtual environment
source venv/bin/activate;

# Make sure Models directory exists
mkdir -p Models

# Skip model if already trained

OVERWRITE=false
for arg in "$@"; do
  if [ "$arg" = "-o" ]; then
    OVERWRITE=true
    break
  fi
done

# Train Linear model if needed
if [ "$OVERWRITE" = true ] || [ ! -f "Models/linear.spm" ]; then
  echo "Training Linear model..."
  python3 Train.py Models/linear.spm Linear -t 1 -s StockList/train.csv
else
  echo "Skipping Linear model (already exists)"
fi

# Train Sequential model if needed
if [ "$OVERWRITE" = true ] || [ ! -f "Models/sequential.spm" ]; then
  echo "Training Sequential model..."
  python3 Train.py Models/sequential.spm Sequential -t 1 -s StockList/train.csv
else
  echo "Skipping Sequential model (already exists)"
fi

# Train Binary Sequential model if needed
if [ "$OVERWRITE" = true ] || [ ! -f "Models/binary-sequential.spm" ]; then
  echo "Training Binary Sequential model..."
  python3 Train.py Models/binary-sequential.spm BinarySequential -t 1 -s StockList/train.csv
else
  echo "Skipping Binary Sequential model (already exists)"
fi

# Train Binary Transformer model if needed
if [ "$OVERWRITE" = true ] || [ ! -f "Models/binary-transformer.spm" ]; then
  echo "Training Binary Transformer model..."
  python3 Train.py Models/binary-transformer.spm BinaryTransformer -t 1 -s StockList/train.csv
else
  echo "Skipping Binary Transformer model (already exists)"
fi

# Train Transformer model if needed
if [ "$OVERWRITE" = true ] || [ ! -f "Models/transformer.spm" ]; then
  echo "Training Transformer model..."
  python3 Train.py Models/transformer.spm Transformer -t 1 -s StockList/train.csv
else
  echo "Skipping Transformer model (already exists)"
fi

echo "All models trained successfully!"

