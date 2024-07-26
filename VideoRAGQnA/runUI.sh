#!/bin/bash

# Run the Python script for UI
streamlit run video-rag-ui.py docs/config.yaml --server.address 0.0.0.0 --server.port 50055
