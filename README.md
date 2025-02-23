# ControlNet-FastAPI 🚀  

A FastAPI-based backend for generating images using ControlNet, enabling edge detection-based image generation with AI.  

## Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [API Endpoints](#api-endpoints)  
- [Docker Support](#docker-support)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview  

This project integrates **ControlNet** with **FastAPI**, allowing users to generate AI-driven images using edge detection. The API processes input images and generates corresponding output images based on the ControlNet model.  

---

## Features  

✅ FastAPI-based backend for handling input in the form of image and prompts requests  
✅ Integration with **ControlNet** for image generation using Denoising Diffusion Implicit Models(DDIM)  
✅ Uses **Conda environment** for dependency management  
✅ Supports all applications originally supported by ControlNet.
✅ **Dockerized** for easy deployment
---

## Installation  

### 1. Clone the Repository  

```bash
git clone https://github.com/shayari21/ControlNet-fastapi.git
cd ControlNet-fastapi
```

### 2. Setting up environment and dependecies 

Using environmemt-fastapi.yaml(This is the enhanced version of the original environment.yaml):
```bash
conda env create -f environment_fastapi.yaml
conda activate controlNet

```

### 3. Run the FastAPI Server
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```
## 4. Usage
Once the server is running, open `http://127.0.0.1:8000/docs` 
in your browser to access the interactive API documentation.

![image](https://github.com/user-attachments/assets/fb7382d3-a3fe-4bb4-b2f4-9d523840daec)

![image](https://github.com/user-attachments/assets/d4fd80cc-3d9a-4473-8e50-9c19482d200f)

## Docker Support
### 1. Build Docker Image (From `ControlNet-fastapi main directory`)
Run the `Dockerfile` in `ControlNet-fastapi\` directory. The `Dockerfile` creates a conda environment inside the docker image using `environment_fastapi.yaml`.

```bash
cd ..
docker build -t controlnet-fastapi .
```
![image](https://github.com/user-attachments/assets/d6833adb-d839-49c5-9322-c4a062308efa)

### 2. Run Container

```bash
docker run -gpus all -p 8000:8000 controlnet-fastapi
```
![image](https://github.com/user-attachments/assets/b6051afe-e4be-434c-a89c-13054d9031d2)

### 3. Test API 
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate_image/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@bird.png;type=image/png' \
  -F 'prompt=bird'
```
or open `http://127.0.0.1:8000/docs` in your browser to access the interactive API documentation similar to local usage.
![image](https://github.com/user-attachments/assets/211aa8b9-2ff5-4f53-ba0f-5ed960b5470e)

![image](https://github.com/user-attachments/assets/da802e45-2249-4e4f-b2b0-90c585e3e3f6)

---

## 📖 References  

Here are some useful references related to **ControlNet** and **FastAPI**:  
🔗 **[ControlNet Repository](https://github.com/lllyasviel/ControlNet)**  
🔗 **ControlNet Paper**: [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)  
🔗 **FastAPI Documentation**: [FastAPI Official Docs](https://fastapi.tiangolo.com/)  
🔗 **Uvicorn ASGI Server**: [Uvicorn GitHub](https://github.com/encode/uvicorn)  
🔗 **Docker Documentation**: [Docker Docs](https://docs.docker.com/)  

