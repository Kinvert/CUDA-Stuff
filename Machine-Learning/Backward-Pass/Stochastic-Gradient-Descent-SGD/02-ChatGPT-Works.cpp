// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>

const int NUM_EXAMPLES = 1000;
const int NUM_EPOCHS = 100;
const float LEARNING_RATE = 0.001;

struct Example {
  std::vector<float> x;
  int y;
};

// Generates a synthetic dataset with random noise added to the output
std::vector<Example> generate_synthetic_data(int num_examples, int num_features) {
  std::vector<Example> examples;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-10.0, 10.0);

  for (int i = 0; i < num_examples; i++) {
    std::vector<float> x(num_features);
    for (int j = 0; j < num_features; j++) {
      x[j] = dis(gen);
    }
    int y = (x[0] + x[1] > 0) ? 1 : 0; // y = 1 if x1 + x2 > 0, else 0
    examples.push_back({x, y});
  }
  return examples;
}

int main() {
  std::vector<Example> examples = generate_synthetic_data(NUM_EXAMPLES, 2);
  int num_features = examples[0].x.size();
  std::vector<float> weight(num_features, 0.0);
  float bias = 0.0;

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    std::shuffle(examples.begin(), examples.end(), std::mt19937(std::random_device()()));
    for (const auto& example : examples) {
      // Make a prediction using the current weight and bias
      float logit = std::inner_product(weight.begin(), weight.end(), example.x.begin(), bias);
      float prediction = 1.0 / (1.0 + std::exp(-logit));

      // Compute the error
      float error = example.y - prediction;

      // Update the weight and bias
      for (int i = 0; i < num_features; i++) {
        weight[i] += LEARNING_RATE * error * example.x[i] * prediction * (1.0 - prediction);
      }
      bias += LEARNING_RATE * error * prediction * (1.0 - prediction);
    }
  }

  std::cout << "Final weights:";
  for (int i = 0; i < num_features; i++) {
    std::cout << " " << weight[i];
  }
  std::cout << std::endl;
  std::cout << "Final bias: " << bias << std::endl;

  return 0;
}
