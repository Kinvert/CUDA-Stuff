// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand.h>

double randn() {
  static curandGenerator_t gen;
  static bool initialized = false;
  if (!initialized) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    initialized = true;
  }
  double mean = 0.0;
  double stddev = 1.0;
  double result;
  curandGenerateNormal(gen, &result, mean, stddev);
  return result;
}

__global__ void matrix_vector_product(const int m, const int n, double* d_z, const double* d_bias, const double* d_weight, const double* d_input) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    d_z[i] += d_weight[i * n + j] * d_input[j];
  }
}

// Sigmoid activation function
__global__ void sigmoid_kernel(const int n, double* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = 1.0 / (1.0 + exp(-x[i]));
  }
}

// Feedforward neural network class
class FFNN {
 public:
  // Constructor
  FFNN(const std::vector<int>& layer_sizes) :
      num_layers(layer_sizes.size()),
      biases(num_layers - 1),
      weights(num_layers - 1) {
    // Initialize biases and weights randomly
    for (int i = 0; i < num_layers - 1; i++) {
      biases[i] = std::vector<double>(layer_sizes[i + 1]);
      weights[i] = std::vector<std::vector<double>>(layer_sizes[i + 1], std::vector<double>(layer_sizes[i]));
      for (int j = 0; j < layer_sizes[i + 1]; j++) {
        biases[i][j] = randn();
        for (int k = 0; k < layer_sizes[i]; k++) {
          weights[i][j][k] = randn();
        }
      }
    }
  }

  // Feedforward function
  std::vector<double> feedforward(const std::vector<double>& input) {
    int n = input.size();
    double* d_input;
    cudaMalloc(&d_input, n * sizeof(double));
    cudaMemcpy(d_input, input.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    std::vector<double> activations = input;
    // Propagate input through the layers
    for (int i = 0; i < num_layers - 1; i++) {
      int m = biases[i].size();
      double* d_z = new double[m];
      double* d_bias = new double[m];
      double* d_weight = new double[m * n];
      double* d_output = new double[m];
      cudaMemcpy(d_bias, biases[i].data(), m * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_weight, weights[i].data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
      // Compute z = bias + input * weight
      cudaMemset(d_z, 0, m * sizeof(double));
      cudaMemset(d_output, 0, m * sizeof(double));
      cudaMemset(d_output, 0, m * sizeof(double));
      dim3 block(16, 16);
      dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
      matrix_vector_product<<<grid, block>>>(m, n, d_z, d_bias, d_weight, d_input);
      cudaDeviceSynchronize();
      // Compute output = sigmoid(z)
      sigmoid_kernel<<<m, 1>>>(m, d_output);
      cudaDeviceSynchronize();
      // Update activations for next layer
      cudaMemcpy(d_input, d_output, m * sizeof(double), cudaMemcpyDeviceToDevice);
      delete[] d_z;
      delete[] d_bias;
      delete[] d_weight;
      delete[] d_output;
    }
    cudaMemcpy(activations.data(), d_input, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    return activations;
  }

 private:
  int num_layers;
  std::vector<std::vector<double>> biases;
  std::vector<std::vector<std::vector<double>>> weights;
};

int main() {
  // Create a neural network with 2 input units, 3 hidden units, and 1 output unit
  FFNN nn({2, 3, 1});

  // Feed input through the network and print the output
  std::vector<double> input = {1.0, 2.0};
  std::vector<double> output = nn.feedforward(input);
  std::cout << output[0] << std::endl;

  return 0;
}
