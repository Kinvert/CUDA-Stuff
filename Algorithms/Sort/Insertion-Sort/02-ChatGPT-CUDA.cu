// Written by ChatGPT
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

__global__ void insertionSort(int *data, int size) {
    // Each thread sorts one element
    int i = threadIdx.x;

    // Insertion sort
    int key = data[i];
    int j = i - 1;
    while (j >= 0 && data[j] > key) {
        data[j + 1] = data[j];
        j--;
    }
    data[j + 1] = key;
}

int main() {
    // Initialize data
    const int size = 100;
    int data[size];
    for (int i = 0; i < size; i++) {
        data[i] = rand();
    }

    // Allocate memory on GPU
    int *deviceData;
    cudaMalloc((void **)&deviceData, size * sizeof(int));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Copy data to GPU
    cudaMemcpy(deviceData, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch insertion sort kernel
    insertionSort<<<1, size>>>(deviceData, size);

    // Copy data back to CPU
    cudaMemcpy(data, deviceData, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();

    // Free GPU memory
    cudaFree(deviceData);

    // Print sorted data
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    // Calculate elapsed time
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print elapsed time
    std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;

    return 0;
}

