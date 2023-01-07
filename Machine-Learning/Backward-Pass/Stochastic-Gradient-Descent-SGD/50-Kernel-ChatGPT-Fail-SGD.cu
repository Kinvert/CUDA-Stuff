// Written by ChatGPT Dec 15 Version

/**
 * @brief Performs a single SGD update for a given example.
 *
 * @param examples A pointer to an array of Example structures. Each Example consists of an input x and an output y.
 * @param num_examples The number of examples in the array.
 * @param weight A pointer to the current weight.
 * @param bias A pointer to the current bias.
 * @param learning_rate The learning rate to use for the update.
 *
 * @return void
 */
__global__ void sgd_update_kernel(const Example* examples, int num_examples, float* weight, float* bias, float learning_rate) {
  // Get the index of the current thread
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the index is within bounds
  if (tid >= num_examples) {
    return;
  }

  // Get the example for the current thread
  Example example = examples[tid];

  // Make a prediction using the current weight and bias
  float prediction = (*weight) * example.x + (*bias);

  // Compute the error
  float error = prediction - example.y;

  // Update the weight and bias
  (*weight) -= learning_rate * error * example.x;
  (*bias) -= learning_rate * error;
}

