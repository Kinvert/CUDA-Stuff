// Written by ChatGPT Dec 15 Version

#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

// Include the SGD kernel
#include "50-Kernel-ChatGPT-SGD.cu"

// Define the Example structure
struct Example {
  float x;
  float y;
};

const int NUM_EXAMPLES = 1000;
const int NUM_EPOCHS = 100;
const float LEARNING_RATE = 0.001;

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

int main() {
  std::vector<Example> examples = generate_synthetic_data(NUM_EXAMPLES);
  float weight = 0.0, bias = 0.0;

  // Allocate device memory for the weight and bias
  float* d_weight;
  cudaMalloc((void**)&d_weight, sizeof(float));
  float* d_bias;
  cudaMalloc((void**)&d_bias, sizeof(float));

  // Allocate device memory for the examples
  Example* d_examples;
  cudaMalloc((void**)&d_examples, sizeof(Example) * NUM_EXAMPLES);

  // Copy the examples to the device
  cudaMemcpy(d_examples, examples.data(), sizeof(Example) * NUM_EXAMPLES, cudaMemcpyHostToDevice);

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    std::shuffle(examples.begin(), examples.end(), std::mt19937(std::random_device()()));

    // Set the number of threads and blocks for the kernel
    int block_size = 256;
    int num_blocks = (num_examples  + block_size - 1) / block_size;

  // Launch the kernel
  sgd_update_kernel<<<num_blocks, block_size>>>(d_examples, NUM_EXAMPLES, d_weight, d_bias, LEARNING_RATE);

  // Check for kernel launch errors
  cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    std::cerr << "Error launching kernel: " << cudaGetErrorString(launch_status) << std::endl;
    return 1;
  }
}

// Copy the weight and bias back to the host
float weight, bias;
cudaError_t weight_copy_status = cudaMemcpy(&weight, d_weight, sizeof(float), cudaMemcpyDeviceToHost);
cudaError_t bias_copy_status = cudaMemcpy(&bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);

// Check for errors in the memory copies
if (weight_copy_status != cudaSuccess || bias_copy_status != cudaSuccess) {
  std::cerr << "Error copying weight or bias from device: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 1;
}

// Free the device memory
cudaError_t weight_free_status = cudaFree(d_weight);
cudaError_t bias_free_status = cudaFree(d_bias);
cudaError_t examples_free_status = cudaFree(d_examples);

// Check for errors in the memory freeing
if (weight_free_status != cudaSuccess || bias_free_status != cudaSuccess || examples_free_status != cudaSuccess) {
  std::cerr << "Error freeing device memory: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 1;
}

std::cout << "Final weight: " << weight << std::endl;
std::cout << "Final bias: " << bias << std::endl;

  return 0;
}

