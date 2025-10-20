# Cuvolve

**Cuvolve** is a high-performance CUDA-based convolution project that demonstrates parallel image filtering using shared and constant memory for maximum efficiency. It explores GPU acceleration, memory hierarchy optimization, and thread-level parallelism to enhance convolution performance — a fundamental operation in computer vision and deep learning.

---

## Overview

This project implements 2D image convolution on NVIDIA GPUs using CUDA.  
It focuses on comparing performance between:
- Global memory convolution
- Shared memory convolution
- Constant memory convolution

Through these implementations, Cuvolve highlights how CUDA memory models and thread block optimizations impact the speed and efficiency of convolution operations in real-world GPU-accelerated applications.

---

## Features

- Parallelized 2D convolution using CUDA kernels  
- Shared memory optimization for performance boost  
- Constant memory for filter storage  
- Timing and performance benchmarking  
- Scalable design for different image sizes and filters  
- Supports both grayscale and RGB images (depending on configuration)  

---

## Key Learning Objectives

- Understand CUDA memory hierarchy (global, shared, constant memory)  
- Learn kernel design and thread mapping for 2D convolution  
- Analyze performance trade-offs between memory models  
- Explore synchronization and coalesced memory access for efficient GPU usage  

---

## Project Structure

| File / Folder | Description |
|----------------|-------------|
| `cuvolve.cu` | Main CUDA source file implementing convolution kernels |
| `Makefile` | Build configuration file for compiling CUDA code |
| `input/` | Directory containing sample input images |
| `output/` | Directory where processed images are saved |
| `README.md` | Project documentation |
| `report.pdf` | Optional report describing implementation and results |

---

## Requirements

| Requirement | Version / Details |
|--------------|-------------------|
| CUDA Toolkit | ≥ 11.0 |
| NVIDIA GPU | CUDA-enabled GPU with compute capability ≥ 5.0 |
| C++ Compiler | GCC or NVCC-compatible compiler |
| Operating System | Linux (Ubuntu preferred) or Windows with WSL2 |
| Dependencies | OpenCV (optional, for image loading and display) |

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/Cuvolve.git
   cd Cuvolve
Ensure CUDA Toolkit is Installed

bash
Copy code
nvcc --version
Build the Project

bash
Copy code
nvcc cuvolve.cu -o cuvolve
(Alternatively, use make if a Makefile is provided)

Run the Program

bash
Copy code
./cuvolve input/image.jpg output/result.jpg
View Results
The convolved image will be saved inside the output/ directory.

Example Usage
Command:

bash
Copy code
./cuvolve input/lena.jpg output/lena_convolved.jpg
Output Example:

vbnet
Copy code
Convolution using global memory: 12.34 ms
Convolution using shared memory: 3.27 ms
Speedup: 3.77x
