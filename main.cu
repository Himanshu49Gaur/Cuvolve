#include "support.h"        // Include custom header file with utility functions
#include <cuda_runtime.h>   // Include CUDA runtime API for GPU programming
#include <stdio.h>          // Include standard I/O library for printf

// Declaration of the CUDA kernel function that performs convolution
extern __global__ void convolutionKernel(Matrix d_input, Matrix d_output);

int main() {
    // Declare Timer structure for logging execution time
    Timer timer;

    // Declare host matrices (h_ prefix indicates host/CPU memory)
    Matrix h_input, h_output;
    // Declare device matrices (d_ prefix indicates device/GPU memory)
    Matrix d_input, d_output;

    // Step 1: Setting up the problem (Allocating memory for matrices)
    startTime(&timer);  // Start the timer for setup
    h_input = allocateMatrix(512, 512);  // Allocate memory for input matrix on host (512x512 elements)
    h_output = allocateMatrix(512, 512); // Allocate memory for output matrix on host (512x512 elements)
    initMatrix(h_input);  // Initialize input matrix with values
    stopTime(&timer);  // Stop the timer after setup
    printf("Setting up the problem...%f s\n", elapsedTime(timer) / 1000.0f);  // Log time taken

    // Step 2: Allocating memory on the device (GPU)
    startTime(&timer);  // Start the timer for device memory allocation
    d_input = allocateDeviceMatrix(h_input.height, h_input.width); // Allocate memory for input matrix on device/GPU
    d_output = allocateDeviceMatrix(h_output.height, h_output.width); // Allocate memory for output matrix on device/GPU
    stopTime(&timer);  // Stop the timer after allocation
    printf("Allocating device variables...%f s\n", elapsedTime(timer) / 1000.0f);  // Log time taken

    // Step 3: Copying data from host to device
    startTime(&timer);  // Start the timer for data copy
    copyToDeviceMatrix(d_input, h_input);  // Copy input matrix from host (CPU) to device (GPU)
    stopTime(&timer);  // Stop the timer after copying
    printf("Copying data from host to device...%f s\n", elapsedTime(timer) / 1000.0f);  // Log time taken

    // Step 4: Launching the kernel
    startTime(&timer);  // Start the timer for kernel launch
    dim3 blockDim(16, 16);  // Define block dimensions for CUDA kernel (16x16 threads per block)
    dim3 gridDim((h_input.width + blockDim.x - 1) / blockDim.x,  // Number of blocks in x direction
                 (h_input.height + blockDim.y - 1) / blockDim.y); // Number of blocks in y direction
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output);  // Launch CUDA kernel
    cudaDeviceSynchronize();  // Wait for GPU to finish execution before proceeding
    stopTime(&timer);  // Stop the timer after kernel execution
    printf("Launching kernel...%f s\n", elapsedTime(timer) / 1000.0f);  // Log time taken

    // Step 5: Copying results from device to host
    startTime(&timer);  // Start the timer for copying data back
    copyFromDeviceMatrix(h_output, d_output);  // Copy results from device (GPU) back to host (CPU)
    stopTime(&timer);  // Stop the timer after copying
    printf("Copying data from device to host...%f s\n", elapsedTime(timer) / 1000.0f);  // Log time taken

    // Step 6: Verifying results
    startTime(&timer);  // Start the timer for verification
    verify(h_input, h_output, h_output);  // Verify the convolution results
    stopTime(&timer);  // Stop the timer after verification
    printf("Verifying results...TEST PASSED\n");

    // Step 7: Free memory
    freeMatrix(h_input);  // Free host memory allocated for input matrix
    freeMatrix(h_output); // Free host memory allocated for output matrix
    freeDeviceMatrix(d_input);  // Free device memory allocated for input matrix
    freeDeviceMatrix(d_output); // Free device memory allocated for output matrix

    // Return 0 to indicate successful program completion
    return 0;
}
