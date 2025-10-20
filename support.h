/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Header guard to prevent multiple inclusions of this header file
#ifndef SUPPORT_H
#define SUPPORT_H

#include <cuda_runtime.h>   // Include CUDA runtime API for GPU programming

// Define a Timer structure for measuring execution time
struct Timer {
    cudaEvent_t start;      // CUDA event to mark the start time
    cudaEvent_t stop;       // CUDA event to mark the stop time
};

// Define a Matrix structure to represent 2D matrices
struct Matrix {
    int width;             // Width of the matrix in elements
    int height;            // Height of the matrix in elements
    float* elements;       // Pointer to the matrix data in memory
};

// Function to start timing using the Timer structure
void startTime(Timer* timer);

// Function to stop timing using the Timer structure
void stopTime(Timer* timer);

// Function to calculate elapsed time between start and stop events
// Returns time in milliseconds
float elapsedTime(Timer timer);

// Function to allocate a matrix on host (CPU) memory
// Takes height and width as parameters, returns initialized Matrix struct
Matrix allocateMatrix(int height, int width);

// Function to allocate a matrix on device (GPU) memory
// Takes height and width as parameters, returns initialized Matrix struct
Matrix allocateDeviceMatrix(int height, int width);

// Function to copy matrix data from host to device
// d_mat: destination matrix on device
// h_mat: source matrix on host
void copyToDeviceMatrix(Matrix d_mat, Matrix h_mat);

// Function to copy matrix data from device to host
// h_mat: destination matrix on host
// d_mat: source matrix on device
void copyFromDeviceMatrix(Matrix h_mat, Matrix d_mat);

// Function to free host memory allocated for a matrix
// Takes Matrix struct to be freed
void freeMatrix(Matrix mat);

// Function to free device memory allocated for a matrix
// Takes Matrix struct to be freed
void freeDeviceMatrix(Matrix mat);

// Function to initialize matrix elements with values
// Takes Matrix struct to be initialized
void initMatrix(Matrix mat);

// Function to verify matrix operation results
// Compares output against expected results using input
void verify(Matrix input, Matrix output, Matrix expected);

// End of header guard
#endif // SUPPORT_H