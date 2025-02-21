# ControlNet-FastAPI  

A FastAPI-based backend for generating images using ControlNet, enabling edge detection-based image generation with AI.  

## Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Docker Support](#docker-support)  

---

## 📝 Overview  

This project integrates **ControlNet** with **FastAPI**, allowing users to generate AI-driven images using edge detection. The API processes input images and generates corresponding output images based on the ControlNet model.  

---

## ✨ Features  

✅ **FastAPI-based backend** for handling image and prompt-based input requests  
✅ **Integration with ControlNet** for AI-powered image generation using **Denoising Diffusion Implicit Models (DDIM)**  
✅ **Dockerized** for easy deployment  
✅ Uses **Conda environment** for dependency management  
✅ Supports all applications originally supported by **ControlNet**  

📥 **Download ControlNet models from:**  
🔗 [Hugging Face - ControlNet Models](https://huggingface.co/lllyasviel/ControlNet/tree/main/models)  
📁 **Place the downloaded models in:** `ControlNet-fastapi/Models/`  

---

## Installation  

### 1. Clone the Repository  

```bash
git clone https://github.com/shayari21/ControlNet-fastapi.git
cd ControlNet-fastapi
```
### 2. Clone the Repository  

Using environmemt-fastapi.yaml(This is the enhanced version of the original environment.yaml):
```bash
git clone https://github.com/shayari21/ControlNet-fastapi.git
cd ControlNet-fastapi
```
Using pip:
```bash
pip install -r requirements.txt
```
### 3. Run the FastAPI Server
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```
## 4. Usage
Once the server is running, open `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)` in your browser to access the interactive API documentation.

## Docker Support
### 1. Build Docker Image (From ControlNet-fastapi main directory)

```bash
docker build -t controlnet-fastapi .
```
### 2. Run Container

```bash
docker run -gpus all -p 8000:8000 controlnet-fastapi
```
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




