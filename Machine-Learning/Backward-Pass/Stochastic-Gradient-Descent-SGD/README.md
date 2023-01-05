# Stochastic Gradient Descent SGD

**Note: ChatGPT wrote much of this readme**

## Introduction

This code demonstrates how to use Stochastic Gradient Descent (SGD) to perform linear regression using CUDA. The code generates synthetic data with random noise added to the output, and then fits a line to the data using SGD. The code is implemented in C++, and makes use of the CUDA runtime API to launch kernels and manage device memory.

## Code Structure

The code is structured as follows:

- generate_synthetic_data: This function generates synthetic data with random noise added to the output. It returns a vector of Example structures, where each Example consists of an input x and an output y.
- sgd_update_kernel: This is the kernel that performs a single SGD update for a given example. It takes as input the example, the current weight and bias, and the learning rate, and updates the weight and bias according to the gradient of the loss function.
- main: This is the main entry point of the program. It generates synthetic data, allocates device memory, copies the data to the device, and then launches the SGD update kernel in a loop to perform SGD. It also shuffles the examples at the start of each epoch to decorrelate the examples. Finally, it copies the weight and bias back to the host, prints them to the console, and frees device memory.

## Code Snippets

Here are some small code snippets with explanations:

### Kernel Launch

This code launches the SGD update kernel:

```cpp
sgd_update_kernel<<<num_blocks, block_size>>>(d_examples, num_examples, d_weight, d_bias, LEARNING_RATE, block_size);
```

The kernel is launched with num_blocks blocks and block_size threads per block. The kernel takes as input the device pointers to the examples, weight, and bias, as well as the learning rate and the block size.

### Error Checking

This code checks for errors after launching the kernel:

```cpp
cudaError_t kernel_launch_status = cudaGetLastError();
if (kernel_launch_status != cudaSuccess) {
  std::cerr << "Error launching SGD update kernel: " << cudaGetErrorString(kernel_launch_status) << std::endl;
  return 1;
}
```

If an error occurred during kernel launch, it is printed to the console. This is important for debugging and ensuring that the kernel is launched correctly.

### Memory Allocation and Copying

This code allocates device memory and copies data to the device:

```cpp
float* d_weight, *d_bias;
cudaError_t weight_alloc_status = cudaMalloc(&d_weight, sizeof(float));
cudaError_t bias_alloc_status = cudaMalloc(&d_bias, sizeof(float));
if (weight_alloc_status != cudaSuccess || bias_alloc_status != cudaSuccess) {
  std::cerr << "Error allocating device memory for weight and bias: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 1;
}

cudaError_t weight_copy_status = cudaMemcpy(d_weight, &weight, sizeof(float), cudaMemcpyHostToDevice);
cudaError_t bias_copy_status = cudaMemcpy(d_bias, &bias, sizeof(float), cudaMemcpyHostToDevice);
if (weight_copy_status != cudaSuccess || bias_copy_status != cudaSuccess) {
  std::cerr << "Error copying weight and bias to device: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 1;
}
```

This code allocates device memory for the weight and bias, and then copies the weight and bias from the host to the device. It is important to check for errors after each CUDA API call to ensure that the operation was successful.

### Memory Freeing

This code frees device memory:

```cpp
cudaError_t weight_free_status = cudaFree(d_weight);
cudaError_t bias_free_status = cudaFree(d_bias);
cudaError_t examples_free_status = cudaFree(d_examples);
if (weight_free_status != cudaSuccess || bias_free_status != cudaSuccess || examples_free_status != cudaSuccess) {
  std::cerr << "Error freeing device memory: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 1;
}
```

This code frees the device memory that was allocated for the weight, bias, and examples. It is important to free device memory when it is no longer needed to avoid memory leaks.
