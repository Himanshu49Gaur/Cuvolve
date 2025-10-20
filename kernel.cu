#include "support.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define MASK_SIZE 3  // Define the mask size explicitly

__constant__ float d_mask[MASK_SIZE * MASK_SIZE];

__global__ void convolutionKernel(Matrix d_input, Matrix d_output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < d_output.height && col < d_output.width) {
        float sum = 0.0f;
        int maskOffset = MASK_SIZE / 2;
        
        for (int i = -maskOffset; i <= maskOffset; i++) {
            for (int j = -maskOffset; j <= maskOffset; j++) {
                int curRow = row + i;
                int curCol = col + j;
                if (curRow >= 0 && curRow < d_input.height && curCol >= 0 && curCol < d_input.width) {
                    sum += d_input.elements[curRow * d_input.width + curCol] * 
                           d_mask[(i + maskOffset) * MASK_SIZE + (j + maskOffset)];
                }
            }
        }
        d_output.elements[row * d_output.width + col] = sum;
    }
}
