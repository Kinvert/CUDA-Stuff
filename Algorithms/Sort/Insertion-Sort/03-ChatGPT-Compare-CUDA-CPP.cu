// Written by ChatGPT
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to perform insertion sort (C++ version)
void insertionSortCpp(int arr[], int n)
{
    for (int i = 1; i < n; i++)
    {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Define the kernel function for insertion sort (CUDA version)
__global__ void insertionSortKernel(int* arr, int n)
{
    // Each thread sorts one element
    int i = threadIdx.x;

    // Insertion sort
    int key = arr[i];
    int j = i - 1;
    while (j >= 0 && arr[j] > key)
    {
        arr[j + 1] = arr[j];
        j--;
    }
    arr[j + 1] = key;
}

// Function to print the array
void printArray(int arr[], int n)
{
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main()
{
    // Generate a random array of size 1000
    const int n = 1000;
    int arr[n];
    srand(time(0));
    for (int i = 0; i < n; i++)
        arr[i] = rand();

    // Make a copy of the array for the CUDA version
    int arr_cuda[n];
    for (int i = 0; i < n; i++)
        arr_cuda[i] = arr[i];

    // Sort the array using C++ insertion sort
    auto start_cpp = std::chrono::high_resolution_clock::now();
    insertionSortCpp(arr, n);
    auto end_cpp = std::chrono::high_resolution_clock::now();
    auto elapsedTimeCpp = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpp - start_cpp).count();

    // Allocate memory on the device for the array for the CUDA version
    int* dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));

    // Copy the array from host to device for the CUDA version
    cudaMemcpy(dev_arr, arr_cuda, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set the number of threads and blocks for the kernel
    const int num_threads = 256;
    const int num_blocks = (n + num_threads - 1) / num_threads;

    // Sort the array using CUDA insertion sort
    cudaEvent_t start_cuda, end_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&end_cuda);
    cudaEventRecord(start_cuda, 0);
    insertionSortKernel<<<num_blocks, num_threads>>>(dev_arr, n);
    cudaEventRecord(end_cuda, 0);
    cudaEventSynchronize(end_cuda);

    // Copy the sorted array from device to host for the CUDA version
    cudaMemcpy(arr_cuda, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_arr);

    // Calculate elapsed time for CUDA version
    float elapsedTimeCuda;
    cudaEventElapsedTime(&elapsedTimeCuda, start_cuda, end_cuda);

    // Print elapsed time for C++ and CUDA versions
    std::cout << "Elapsed time (C++): " << elapsedTimeCpp << " milliseconds" << std::endl;
    std::cout << "Elapsed time (CUDA): " << elapsedTimeCuda << " milliseconds" << std::endl;

    return 0;
}
