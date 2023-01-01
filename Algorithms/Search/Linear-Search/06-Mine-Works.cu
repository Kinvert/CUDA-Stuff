// Written by me
#include <stdio.h>

// CUDA Search Kernel
__global__ void gpuLinearSearchKernel(int* arr, int numVals, int target, int* index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numVals) {
        if (arr[i] == target) {
            printf("FOUND %d AT %d\n", target, i);
            *index = i;
        }
    }
    __syncthreads();
}

int main()
{
    // Array to GPU
    int numVals = 8;
    int h_arr[numVals] = { 4, 2, 6, 1, 3, 7, 8, 5 };
    int target = 5;
    int* g_arr;    
    cudaMalloc(&g_arr, numVals * sizeof(int));
    cudaMemcpy(g_arr, h_arr, numVals * sizeof(int), cudaMemcpyHostToDevice);

    // Index to GPU
    int idx = -1;
    int* h_index;
    h_index = &idx;
    int* g_index;
    cudaMalloc(&g_index, sizeof(int));
    cudaMemcpy(g_index, h_index, sizeof(int), cudaMemcpyHostToDevice);

    // Do the Search
    gpuLinearSearchKernel<<<1,numVals>>>(g_arr, numVals, target, g_index);

    // Index from GPU to CPU
    cudaMemcpy(h_index, g_index, sizeof(int), cudaMemcpyDeviceToHost);

    // Free Memory
    cudaFree(g_arr);
    cudaFree(g_index);

    // Print Results
    if (*h_index != -1) {
        printf("Target found at index: %d\n", *h_index);
    }
    else {
        printf("Target not found\n");
    }
    
    return 0;
}
