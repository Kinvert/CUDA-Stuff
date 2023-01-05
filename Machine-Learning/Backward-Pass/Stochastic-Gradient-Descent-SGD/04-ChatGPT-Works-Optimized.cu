// Written by ChatGPT Dec 15 Version First Try
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
  cudaMalloc(&d_weight, sizeof(float));
  cudaMalloc(&d_bias, sizeof(float));

  // Initialize the weight and bias to 0 on the device
  cudaMemset(d_weight, 0, sizeof(float));
  cudaMemset(d_bias, 0, sizeof(float));

  // Allocate device memory for the examples
  Example* d_examples;
  cudaMalloc(&d_examples, num_examples * sizeof(Example));

  // Copy the examples to the device
  cudaMemcpy(d_examples, examples.data(), num_examples * sizeof(Example), cudaMemcpyHostToDevice);

  // Set the number of threads and blocks for the kernel
  int block_size = 256;
  int num_blocks = (num_examples + block_size - 1) / block_size;

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    std::shuffle(examples.begin(), examples.end(), std::mt19937(std::random_device()()));

    // Copy the shuffled examples to the device
    cudaMemcpy(d_examples, examples.data(), num_examples * sizeof(Example), cudaMemcpyHostToDevice);

    // Launch the kernel to perform SGD updates on the device
    sgd_update_kernel<<<num_blocks, block_size, sizeof(float) * 2>>>(d_examples, num_examples, d_weight, d_bias, LEARNING_RATE, block_size);
  }

  // Copy the weight and bias back to the host
  float weight, bias;
  cudaMemcpy(&weight, d_weight, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Final weight: " << weight << std::endl;
  std::cout << "Final bias: " << bias << std::endl;

  // Free device memory
  cudaFree(d_examples);
  cudaFree(d_weight);
  cudaFree(d_bias);

  return 0;
}
