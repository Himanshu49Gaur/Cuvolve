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
