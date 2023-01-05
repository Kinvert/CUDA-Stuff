// Written by ChatGPT Dec 15 Version
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <vector>

// ChatGPT added this when I asked it to make an int main
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

// Activation function
__device__ double sigmoid(double x) {
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
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    for (int i = 0; i < num_layers - 1; i++) {
      int m = layer_sizes[i + 1];
      int n = layer_sizes[i];
      biases[i] = std::vector<double>(m);
      weights[i] = std::vector<std::vector<double>>(m, std::vector<double>(n));
      curandGenerateNormalDouble(gen, biases[i].data(), 0.0, 1.0, m, 0);
      curandGenerateNormalDouble(gen, weights[i].data()->data(), 0.0, 1.0, m * n, 0);
    }
    curandDestroyGenerator(gen);
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
      matrix_vector_product<<<(m + 15) / 16, 16>>>(m, n, d_z, d_bias, d_weight, d_input);
      // Compute output = sigmoid(z)
      sigmoid<<<(m + 15) / 16, 16>>>(m, d_z);
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
  // Generate toy dataset
  int num_samples = 1000;
  int num_features = 2;
  std::vector<std::vector<double>> X(num_samples, std::vector<double>(num_features));
  std::vector<int> y(num_samples);
  std::default_random_engine engine;
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < num_samples; i++) {
    X[i][0] = dist(engine);
    X[i][1] = dist(engine);
    y[i] = (X[i][0] * X[i][0] + X[i][1] * X[i][1] < 1.0) ? 1 : 0;
  }
  // Shuffle the dataset
  std::shuffle(X.begin(), X.end(), engine);
  std::shuffle(y.begin(), y.end(), engine);

  // Create neural network
  std::vector<int> layer_sizes = {num_features, 8, 8, 1};
  NeuralNetwork nn(layer_sizes);

  // Train the neural network
  double learning_rate = 0.01;
  int num_epochs = 10;
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int i = 0; i < num_samples; i++) {
      // Compute forward pass
      std::vector<double> output = nn.feedforward(X[i]);
      // Compute loss
      double loss = (output[0] - y[i]) * (output[0] - y[i]);
      // Backpropagate gradients
      // ...
      // Update weights and biases
      // ...
    }
  }

  // Test the neural network
  int num_correct = 0;
  for (int i = 0; i < num_samples; i++) {
    std::vector<double> output = nn.feedforward(X[i]);
    int prediction = std::round(output[0]);
    if (prediction == y[i]) {
      num_correct++;
    }
  }
  std::cout << "Accuracy: " << num_correct / static_cast<double>(num_samples) << std::endl;

  return 0;
}
