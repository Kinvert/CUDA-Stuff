// ChatGPT Dec 15 Version wrote this
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernel function to search for the target on the GPU
__global__
void gpuLinearSearchKernel(int* arr, int target, int* index)
{
    // search for target in the array
    for (int i = 0; i < arr.size(); i++)
    {
        // if target is found, set the index and return
        if (arr[i] == target)
        {
            *index = i;
            return;
        }
    }
}

int main()
{
    std::vector<int> arr = { 4, 2, 6, 1, 3, 7, 8, 5 };
    int target = 5;

    // allocate memory on the GPU
    int* gpu_arr;
    cudaMalloc((void**)&gpu_arr, arr.size() * sizeof(int));

    // copy data from the host to the GPU
    cudaMemcpy(gpu_arr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    // launch the kernel to search for the target on the GPU
    int index = -1;
    gpuLinearSearchKernel<<<1,1>>>(gpu_arr, target, &index);

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
