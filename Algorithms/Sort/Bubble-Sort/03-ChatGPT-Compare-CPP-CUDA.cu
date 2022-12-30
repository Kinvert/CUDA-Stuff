// Written by ChatGPT
#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to perform bubble sort (C++ version)
void bubbleSortCpp(int arr[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Define the kernel function for bubble sort (CUDA version)
__global__ void bubbleSortKernel(int* arr, int n)
{
    // Determine the position of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread is within the bounds of the array
    if (i < n)
    {
        // Perform bubble sort on the current element
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
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

    // Sort the array using C++ bubble sort
    clock_t start_cpp = clock();
    bubbleSortCpp(arr, n);
    clock_t end_cpp = clock();

    // Allocate memory on the device for the array for the CUDA version
    int* dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));

    // Copy the array from host to device for the CUDA version
    cudaMemcpy(dev_arr, arr_cuda, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set the number of threads and blocks for the kernel
    const int num_threads = 256;
    const int num_blocks = (n + num_threads - 1) / num_threads;

    cudaEvent_t start_cuda, end_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&end_cuda);
    cudaEventRecord(start_cuda, 0);
    bubbleSortKernel<<<num_blocks, num_threads>>>(dev_arr, n);
    cudaEventRecord(end_cuda, 0);
    cudaEventSynchronize(end_cuda);

    // Copy the sorted array from device to host for the CUDA version
    cudaMemcpy(arr_cuda, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(dev_arr);

    // Print the sorted array and the time taken for both C++ and CUDA versions
    printArray(arr, n);
    std::cout << "Time taken (C++):  " << (end_cpp - start_cpp) / (double)CLOCKS_PER_SEC << " seconds" << std::endl;
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_cuda, end_cuda);
    std::cout << "Time taken (CUDA): " << elapsed_time / 1000.0 << " seconds" << std::endl;

    return 0;
}
