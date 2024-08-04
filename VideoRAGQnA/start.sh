#!/bin/bash

# Check if video-llama model exists
if [ ! -e "./meta-llama/Llama-2-7b-chat-hf" ]; then
    echo "Downloading llama-2-7b-chat-hf model weights ..."
    mkdir -p meta-llama/Llama-2-7b-chat-hf
    cd meta-llama/Llama-2-7b-chat-hf
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/config.json
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/generation_config.json
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/pytorch_model-00001-of-00002.bin
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/pytorch_model-00002-of-00002.bin
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/pytorch_model.bin.index.json
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/special_tokens_map.json
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/tokenizer.json
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/tokenizer.model
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/llama-2-7b-chat-hf/tokenizer_config.json
    cd ../..
else
    echo "Found llama weights already downloaded."
fi
if [ ! -e "./embedding/video_llama_weights" ]; then
    echo "Downloading VL_LLaMA_2_7B_Finetuned weights ..."
    mkdir -p embedding/video_llama_weights
    cd embedding/video_llama_weights
    wget https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/VL_LLaMA_2_7B_Finetuned.pth
    cd ../..
else
    echo "Found video-llama VL branch weights already downloaded."
fi

# Function to check if the container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Eq "^$1\$"
}

# Container name
CONTAINER_NAME="vdms-rag"

# Check if the container exists
if container_exists $CONTAINER_NAME; then
    echo "Container $CONTAINER_NAME exists."

    # Stop the container
    echo "Stopping container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME

    # Start the container in detached mode
    echo "Starting container $CONTAINER_NAME in detached mode..."
    docker run --rm -d --name vdms-rag -p 55555:55555 intellabs/vdms:latest
else
    echo "Starting container $CONTAINER_NAME in detached mode..."
    docker run --rm -d --name vdms-rag -p 55555:55555 intellabs/vdms:latest
fi

# Remove the folder ./video/video_metadata/
if [ -d "./video_ingest/video_metadata" ]; then
    echo "Removing folder ./video_ingest/video_metadata ..."
    rm -rf ./video_ingest/video_metadata
else
    echo ""
fi

# Run the Python script for UI
streamlit run video-rag-ui.py docs/config.yaml --server.address 0.0.0.0 --server.port 50055
