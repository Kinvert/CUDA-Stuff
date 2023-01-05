// Written by ChatGPT Dec 15 Version
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

double randn() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::normal_distribution<> dist(0.0, 1.0);
  return dist(gen);
}

// Sigmoid activation function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
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
    std::vector<double> activations = input;
    // Propagate input through the layers
    for (int i = 0; i < num_layers - 1; i++) {
      std::vector<double> zs;
      for (int j = 0; j < biases[i].size(); j++) {
        double z = biases[i][j];
        for (int k = 0; k < activations.size(); k++) {
          z += activations[k] * weights[i][j][k];
        }
        zs.push_back(z);
      }
      activations.clear();
      for (int j = 0; j < zs.size(); j++) {
        activations.push_back(sigmoid(zs[j]));
      }
    }
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
