// Written by ChatGPT Dec 15 Version
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Activation function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

// Matrix-vector product kernel
__global__ void matrix_vector_product(int m, int n, double* z, const double* bias, const double* weight, const double* input) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    z[i] = bias[i];
    for (int j = 0; j < n; j++) {
      z[i] += weight[i * n + j] * input[j];
    }
  }
}

// Neural network class
class NeuralNetwork {
 public:
  // Constructor
  NeuralNetwork(const std::vector<int>& layer_sizes) : num_layers(layer_sizes.size()), biases(num_layers - 1), weights(num_layers - 1) {
    // Initialize biases and weights randomly
    for (int i = 0; i < num_layers - 1; i++) {
      int m = layer_sizes[i + 1];
      int n = layer_sizes[i];
      biases[i] = std::vector<double>(m);
      weights[i] = std::vector<std::vector<double>>(m, std::vector<double>(n));
      for (int j = 0; j < m; j++) {
        biases[i][j] = rand() / static_cast<double>(RAND_MAX);
        for (int k = 0; k < n; k++) {
          weights[i][j][k] = rand() / static_cast<double>(RAND_MAX);
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
      cudaMemcpy(d_weight, weights[i].data()->data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
      // Compute z = bias + input * weight
      matrix_vector_product<<<(m + 31) / 32, 32>>>(m, n, d_z, d_bias, d_weight, d_input);
      // Compute output = sigmoid(z)
      for (int j = 0; j < m; j++) {
        d_output[j] = sigmoid(d_z[j]);
      }
      cudaMemcpy(d_output, d_z, m * sizeof(double), cudaMemcpyDeviceToHost);
      activations = std::vector<double>(d_output, d_output + m);
      delete[] d_z;
      delete[] d_bias;
      delete[] d_weight;
      delete[] d_output;
      cudaFree(d_input);
      d_input = d_z;
    }
    cudaFree(d_input);
    return activations;
  }

 private:
  int num_layers;
  std::vector<std::vector<double>> biases;
  std::vector<std::vector<std::vector<double>>> weights;
};

int main() {
  // Create neural network
  std::vector<int> layer_sizes = {2, 3, 2};
  NeuralNetwork nn(layer_sizes);

  // Test feedforward function
  std::vector<double> input = {1.0, 2.0};
  std::vector<double> output = nn.feedforward(input);
  for (double x : output) {
    std::cout << x << std::endl;
  }

  return 0;
}


