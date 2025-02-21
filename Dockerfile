# Use an official CUDA-enabled base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies (including OpenCV missing dependencies)
# Install system dependencies (including OpenCV missing dependencies)
RUN apt-get update && apt-get install -y \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set Conda environment variables
ENV PATH="/opt/conda/bin:$PATH"

# Set the working directory inside the container
WORKDIR /workspace

# Copy the Conda environment file and files
COPY environment_fastapi.yaml /workspace/environment_fastapi.yaml
COPY . /workspace

# Create the Conda environment
RUN conda env create -f /workspace/environment_fastapi.yaml && conda clean --all -y

# Check if Conda environment exists
RUN conda info --envs 

# Activate Conda environment when the container starts
SHELL ["conda", "run", "--no-capture-output", "-n", "controlNet", "/bin/bash", "-c"]

# Set the working directory to app/
WORKDIR /workspace/app

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI using Conda
CMD ["conda", "run", "--no-capture-output", "-n", "controlNet", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]