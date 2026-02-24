#!/bin/bash

# Quick start script for CS234 RL Question Answering project

echo "================================================"
echo "CS234 RL Question Answering - Quick Start"
echo "================================================"
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Virtual environment not activated. Activating..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
    fi
fi

# Install dependencies
echo "Checking dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Dependencies already installed."
fi

echo ""
echo "================================================"
echo "Choose training mode:"
echo "================================================"
echo "1. Quick demo (small dataset, 10 epochs, 20 iterations)"
echo "2. Full pipeline (500 questions, 50 epochs, 250 iterations)"
echo "3. Supervised training only"
echo "4. PPO training only (requires pretrained model)"
echo "5. Evaluation only (requires trained model)"
echo "6. Interactive demo"
echo ""

read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Running quick demo..."
        python main.py --mode full \
            --num_questions 50 \
            --supervised_epochs 5 \
            --ppo_iterations 10 \
            --batch_size 8
        ;;
    2)
        echo ""
        echo "Running full pipeline (this will take several hours)..."
        python main.py --mode full
        ;;
    3)
        echo ""
        echo "Running supervised training..."
        python main.py --mode supervised
        ;;
    4)
        echo ""
        echo "Running PPO training..."
        if [ -d "checkpoints/supervised/best_model" ]; then
            python main.py --mode ppo --model_path checkpoints/supervised/best_model
        else
            echo "ERROR: Pretrained supervised model not found!"
            echo "Please run supervised training first (option 3)"
            exit 1
        fi
        ;;
    5)
        echo ""
        read -p "Enter model path: " model_path
        if [ -d "$model_path" ]; then
            python main.py --mode eval --model_path "$model_path"
        else
            echo "ERROR: Model not found at $model_path"
            exit 1
        fi
        ;;
    6)
        echo ""
        read -p "Enter model path: " model_path
        if [ -d "$model_path" ]; then
            python demo.py --model_path "$model_path" --mode sample
        else
            echo "ERROR: Model not found at $model_path"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Done!"
echo "================================================"
