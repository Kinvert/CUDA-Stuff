// ChatGPT Dec 15 Version wrote this
#include <cstdio>
#include <ctime>
#include <iostream>

// Function to print the array
void printArray(int arr[], int n)
{
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

// Define the kernel function for bubble sort
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

int main()
{
    // Generate a random array of size 1000
    const int n = 1000;
    int arr[n];
    srand(time(0));
    for (int i = 0; i < n; i++)
        arr[i] = rand();

    // Allocate memory on the device for the array
    int* dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set the number of threads and blocks for the kernel
    const int num_threads = 256;
    const int num_blocks = (n + num_threads - 1) / num_threads;

    // Measure the time taken by the kernel
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    bubbleSortKernel<<<num_blocks, num_threads>>>(dev_arr, n);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    // Copy the sorted array from device to host
    cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(dev_arr);

    // Print the sorted array and the time taken
    printArray(arr, n);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Time taken: " << elapsed_time / 1000.0 << " seconds" << std::endl;

    return 0;
}
