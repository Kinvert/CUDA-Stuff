// Written by ChatGPT
#include <cuda_runtime.h>
#include <iostream>

const int BLOCK_SIZE = 128;

// Function to perform insertion sort
__global__ void insertionSort(int *data, int size) {
    // Each thread block sorts a block of elements
    __shared__ int block[256];

    // Load block of elements into shared memory
    int i = threadIdx.x;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < size) {
        block[i] = data[j];
    }
    __syncthreads();

    // Sort block of elements in shared memory
    for (int k = 1; k < blockDim.x; k *= 2) {
        i = (threadIdx.x + k) % blockDim.x;
        if (i > 0 && block[i - 1] > block[i]) {
            int temp = block[i - 1];
            block[i - 1] = block[i];
            block[i] = temp;
        }
        __syncthreads();
    }

    // Store sorted block of elements in global memory
    i = threadIdx.x;
    j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < size) {
        data[j] = block[i];
    }
}
int main() {
    // Initialize data
    const int size = 1000;
    int data[size];
    for (int i = 0; i < size; i++) {
        data[i] = rand();
    }

    // Allocate memory on GPU
    int *deviceData;
    cudaMalloc((void **)&deviceData, size * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(deviceData, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch insertion sort kernel
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    insertionSort<<<numBlocks, BLOCK_SIZE>>>(deviceData, size);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    // Copy data back to CPU
    cudaMemcpy(data, deviceData, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(deviceData);

    // Print sorted data
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Time taken: " << elapsed_time / 1000.0 << " seconds" << std::endl;

    return 0;
}
