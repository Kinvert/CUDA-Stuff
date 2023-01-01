//Written by ChatGPT
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernel function to search for the target on the GPU
__global__
void gpuLinearSearchKernel(int* arr, int target, int* index)
{
    // get the thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // if the thread ID is within the bounds of the array, check the element
    if (tid < arr.size())
    {
        // if target is found, set the index and return
        if (arr[tid] == target)
        {
            *index = tid;
            return;
        }
    }
}

int main()
{
    std::vector<int> arr = { 4, 2, 6, 1, 3, 7, 8, 5 };
    int target = 5;

    // get the size of the vector
    int numVals = arr.size();

    // allocate memory on the GPU
    int* gpu_arr;
    cudaMalloc((void**)&gpu_arr, numVals * sizeof(int));

    // copy data from the host to the GPU
    cudaMemcpy(gpu_arr, arr.data(), numVals * sizeof(int), cudaMemcpyHostToDevice);

    // launch the kernel to search for the target on the GPU
    int index = -1;
    gpuLinearSearchKernel<<<1,numVals>>>(gpu_arr, target, &index);

    // copy the result back to the host
    cudaMemcpy(&index, gpu_arr, sizeof(int), cudaMemcpyDeviceToHost);

    // free memory on the GPU
    cudaFree(gpu_arr);

    if (index != -1)
        std::cout << "Target found at index: " << index << std::endl;
    else
        std::cout << "Target not found" << std::endl;

    return 0;
}
