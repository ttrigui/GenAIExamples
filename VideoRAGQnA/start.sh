#!/bin/bash

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

# Remove the folder ./video/del/
if [ -d "./video_ingest/video_metadata" ]; then
    echo "Removing folder ./video_ingest/video_metadata ..."
    rm -rf ./video_ingest/video_metadata
else
    echo ""
fi

# Run the Python script for UI
streamlit run video-rag-ui.py docs/config.yaml --server.address 0.0.0.0 --server.port 50055
