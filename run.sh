#!/bin/bash

export PYTHONPATH="your own path to src directory"




MODEL_TYPE=$1 
MODEL_CARD=$2

case $MODEL_TYPE in 
    "gemini")
        python -m scripts.run_gemini --model_card $MODEL_CARD 
        ;;
    "llava")
        python -m scripts.run_llava_onevision --model_card $MODEL_CARD 
        ;;
    "openai")
        python -m scripts.run_openai --model_card $MODEL_CARD 
        ;;
    "qwen")
        python -m scripts.run_qwen --model_card $MODEL_CARD 
        ;;
    "gemma")
        python -m scripts.run_gemma --model_card $MODEL_CARD 
        ;;
    *)
        echo "Invalid model type: $MODEL_TYPE. Must be one of 'gemini', 'llava', 'openai', 'gemma', or 'qwen'."
        exit 1
esac


