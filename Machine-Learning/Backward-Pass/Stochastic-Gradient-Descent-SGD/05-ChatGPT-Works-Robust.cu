// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

const int NUM_EXAMPLES = 1000;
const int NUM_EPOCHS = 100;
const float LEARNING_RATE = 0.001;

struct Example {
  float x;
  float y;
};

// Generates a synthetic dataset with random noise added to the output
std::vector<Example> generate_synthetic_data(int num_examples) {
  std::vector<Example> examples;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-10.0, 10.0);

  for (int i = 0; i < num_examples; i++) {
    float x = dis(gen);
    float y = 3.0 * x + 2.0 + dis(gen) * 0.1; // y = 3x + 2 + noise
    examples.push_back({x, y});
  }
  return examples;
}

// Performs SGD updates on the weight and bias using a block of examples
__global__ void sgd_update_kernel(const Example* examples, int num_examples,
                                  float* weight, float* bias, float learning_rate,
                                  int block_size) {
  __shared__ float s_weight, s_bias;
  s_weight = *weight;
  s_bias = *bias;

  int example_idx = blockIdx.x * block_size + threadIdx.x;
  if (example_idx >= num_examples) return;

  const Example example = examples[example_idx];

  // Make a prediction using the current weight and bias
  float prediction = s_weight * example.x + s_bias;

  // Compute the error
  float error = prediction - example.y;

  // Update the weight and bias
  s_weight -= learning_rate * error * example.x;
  s_bias -= learning_rate * error;

  *weight = s_weight;
  *bias = s_bias;
}
  
int main() {
  std::vector<Example> examples = generate_synthetic_data(NUM_EXAMPLES);
  int num_examples = examples.size();

  // Allocate device memory for the weight and bias
  float* d_weight, *d_bias;
  cudaError_t weight_alloc_status = cudaMalloc(&d_weight, sizeof(float));
  cudaError_t bias_alloc_status = cudaMalloc(&d_bias, sizeof(float));
  if (weight_alloc_status != cudaSuccess || bias_alloc_status != cudaSuccess) {
    std::cerr << "Error allocating device memory for weight and bias: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  // Initialize the weight and bias to 0 on the device
  cudaError_t weight_init_status = cudaMemset(d_weight, 0, sizeof(float));
  cudaError_t bias_init_status = cudaMemset(d_bias, 0, sizeof(float));
  if (weight_init_status != cudaSuccess || bias_init_status != cudaSuccess) {
    std::cerr << "Error initializing weight and bias on device: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  // Allocate device memory for the examples
  Example* d_examples;
  cudaError_t examples_alloc_status = cudaMalloc(&d_examples, num_examples * sizeof(Example));
  if (examples_alloc_status != cudaSuccess) {
    std::cerr << "Error allocating device memory for examples: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  // Copy the examples to the device
  cudaError_t examples_copy_status = cudaMemcpy(d_examples, examples.data(), num_examples * sizeof(Example), cudaMemcpyHostToDevice);
  if (examples_copy_status != cudaSuccess) {
    std::cerr << "Error copying examples to device: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  // Set the number of threads and blocks for the kernel
  int block_size = 256;
  int num_blocks = (num_examples + block_size - 1) / block_size;

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    std::shuffle(examples.begin(), examples.end(), std::mt19937(std::random_device()()));

    // Copy the shuffled examples to the device
    cudaError_t shuffled_examples_copy_status = cudaMemcpy(d_examples, examples.data(), num_examples * sizeof(Example), cudaMemcpyHostToDevice);
    if (shuffled_examples_copy_status != cudaSuccess) {
      std::cerr << "Error copying shuffled examples to device: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return 1;
    }

    // Launch the SGD update kernel
    sgd_update_kernel<<<num_blocks, block_size>>>(d_examples, num_examples, d_weight, d_bias, LEARNING_RATE, block_size);

    // Check for kernel launch errors
    cudaError_t kernel_launch_status = cudaGetLastError();
    if (kernel_launch_status != cudaSuccess) {
      std::cerr << "Error launching SGD update kernel: " << cudaGetErrorString(kernel_launch_status) << std::endl;
      return 1;
    }
  }

  // Copy the weight and bias back to the host
  float weight, bias;
  cudaError_t weight_copy_status = cudaMemcpy(&weight, d_weight, sizeof(float), cudaMemcpyDeviceToHost);
  cudaError_t bias_copy_status = cudaMemcpy(&bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);
  if (weight_copy_status != cudaSuccess || bias_copy_status != cudaSuccess) {
    std::cerr << "Error copying weight and bias from device: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  std::cout << "Final weight: " << weight << std::endl;
  std::cout << "Final bias: " << bias << std::endl;

  // Free device memory
  cudaError_t weight_free_status = cudaFree(d_weight);
  cudaError_t bias_free_status = cudaFree(d_bias);
  cudaError_t examples_free_status = cudaFree(d_examples);
  if (weight_free_status != cudaSuccess || bias_free_status != cudaSuccess || examples_free_status != cudaSuccess) {
    std::cerr << "Error freeing device memory: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  return 0;
}
