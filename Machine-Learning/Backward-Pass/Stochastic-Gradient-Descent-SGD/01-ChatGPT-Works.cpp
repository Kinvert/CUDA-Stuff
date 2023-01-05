// Written by ChatGPT Dec 15 Version First Try
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

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

int main() {
  std::vector<Example> examples = generate_synthetic_data(NUM_EXAMPLES);
  float weight = 0.0, bias = 0.0;

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    std::shuffle(examples.begin(), examples.end(), std::mt19937(std::random_device()()));
    for (const auto& example : examples) {
      // Make a prediction using the current weight and bias
      float prediction = weight * example.x + bias;

      // Compute the error
      float error = prediction - example.y;

      // Update the weight and bias
      weight -= LEARNING_RATE * error * example.x;
      bias -= LEARNING_RATE * error;
    }
  }

  std::cout << "Final weight: " << weight << std::endl;
  std::cout << "Final bias: " << bias << std::endl;

  return 0;
}
