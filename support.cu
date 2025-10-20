/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

 #include "support.h"        // Include custom header with structure definitions
 #include <stdio.h>          // Include standard I/O for printing
 #include <stdlib.h>         // Include standard library for malloc and rand
 
 // Function to start timing a CUDA operation
 void startTime(Timer* timer) {
     cudaEventCreate(&timer->start);     // Create CUDA event for start time
     cudaEventCreate(&timer->stop);      // Create CUDA event for stop time
     cudaEventRecord(timer->start, 0);   // Record the start event in default stream (0)
 }
 
 // Function to stop timing a CUDA operation
 void stopTime(Timer* timer) {
     cudaEventRecord(timer->stop, 0);    // Record the stop event in default stream (0)
     cudaEventSynchronize(timer->stop);  // Wait for stop event to complete
 }
 
 // Function to calculate elapsed time between start and stop events
 float elapsedTime(Timer timer) {
     float time;                         // Variable to store elapsed time in milliseconds
     cudaEventElapsedTime(&time, timer.start, timer.stop); // Calculate time difference
     return time;                        // Return elapsed time
 }
 
 // Function to initialize matrix with random values
 void initMatrix(Matrix mat) {
     for (int i = 0; i < mat.height * mat.width; i++) {    // Loop through all elements
         mat.elements[i] = rand() % 10;                    // Assign random value 0-9
     }
 }
 
 // Function to verify GPU results against expected values
 void verify(Matrix input, Matrix output, Matrix expected) {
     for (int i = 0; i < input.height * input.width; i++) {    // Check each element
         // Compare output with expected, allowing for small floating-point differences
         if (abs(output.elements[i] - expected.elements[i]) > 1e-5) {
             // Print error message if verification fails
             printf("Verification failed at index %d: Expected %f, Got %f\n", 
                    i, expected.elements[i], output.elements[i]);
             return;    // Exit function on first failure
         }
     }
     printf("Verification passed!\n");    // Print success message if all elements match
 }
 
 // Function to allocate matrix on host (CPU) memory
 Matrix allocateMatrix(int height, int width) {
     Matrix mat;                         // Create Matrix structure
     mat.width = width;                  // Set matrix width
     mat.height = height;                // Set matrix height
     // Allocate memory for matrix elements
     mat.elements = (float*)malloc(width * height * sizeof(float));
     return mat;                         // Return initialized matrix
 }
 
 // Function to allocate matrix on device (GPU) memory
 Matrix allocateDeviceMatrix(int height, int width) {
     Matrix mat;                         // Create Matrix structure
     mat.width = width;                  // Set matrix width
     mat.height = height;                // Set matrix height
     // Allocate GPU memory for matrix elements
     cudaMalloc((void**)&mat.elements, width * height * sizeof(float));
     return mat;                         // Return initialized matrix
 }
 
 // Function to copy matrix from host to device
 void copyToDeviceMatrix(Matrix d_mat, Matrix h_mat) {
     // Copy data from host to device using CUDA memory copy
     cudaMemcpy(d_mat.elements, h_mat.elements, 
                h_mat.width * h_mat.height * sizeof(float), 
                cudaMemcpyHostToDevice);
 }
 
 // Function to copy matrix from device to host
 void copyFromDeviceMatrix(Matrix h_mat, Matrix d_mat) {
     // Copy data from device to host using CUDA memory copy
     cudaMemcpy(h_mat.elements, d_mat.elements, 
                d_mat.width * d_mat.height * sizeof(float), 
                cudaMemcpyDeviceToHost);
 }
 
 // Function to free host memory
 void freeMatrix(Matrix mat) {
     free(mat.elements);    // Release allocated CPU memory
 }
 
 // Function to free device memory
 void freeDeviceMatrix(Matrix mat) {
     cudaFree(mat.elements);    // Release allocated GPU memory
 }