#!/bin/bash

# Fix all service Dockerfiles to have correct COPY commands

# Core - already correct, no changes needed

# LLM - already fixed above

# TTS
sed -i '/# Copy application code/,/COPY . ./c\# Copy shared utilities first\nCOPY shared/ ./shared/\n\n# Copy TTS service code\nCOPY services/tts/ ./' services/tts/Dockerfile

# STT  
sed -i '/# Copy application code/,/COPY . ./c\# Copy shared utilities first\nCOPY shared/ ./shared/\n\n# Copy STT service code\nCOPY services/stt/ ./' services/stt/Dockerfile

echo "Dockerfiles fixed"
