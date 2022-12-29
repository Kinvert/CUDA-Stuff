// ChatGPT wrote this
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// define the number of points to generate
const int num_points = 10000000;

// define the kernel function to estimate Pi
__global__ void estimate_pi(int* num_in_circle, curandState* states) {
  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // initialize the cuRAND states
  curand_init(0, 0, 0, states);
  
  // generate a random point using the cuRAND library
  double x = curand_uniform(&states[i]);
  double y = curand_uniform(&states[i]);

  // check if the point lies within the unit circle
  if (x * x + y * y < 1.0) {
    // increment the number of points within the circle
    atomicAdd(num_in_circle, 1);
  }
}

int main() {
  // allocate memory on the host for the number of points in the circle
  int* num_in_circle_h;
  cudaMallocHost(&num_in_circle_h, sizeof(int));

  // allocate memory on the device for the number of points in the circle
  int* num_in_circle_d;
  cudaMalloc(&num_in_circle_d, sizeof(int));

  // initialize the number of points in the circle to 0
  *num_in_circle_h = 0;
  cudaMemcpy(num_in_circle_d, num_in_circle_h, sizeof(int), cudaMemcpyHostToDevice);

  // allocate memory on the device for the cuRAND states
  curandState* states_d;
  cudaMalloc(&states_d, num_points * sizeof(curandState));

  // launch the kernel to estimate Pi
  estimate_pi<<<(num_points + 255) / 256, 256>>>(num_in_circle_d, states_d);

  // copy the result from the device to the host
  cudaMemcpy(num_in_circle_h, num_in_circle_d, sizeof(int), cudaMemcpyDeviceToHost);

  // compute and print the estimated value of Pi
  double pi = 4.0 * *num_in_circle_h / num_points;
  std::cout << "Estimated value of Pi: " << pi << std::endl;

  // free the memory allocated on the host and device
  cudaFree(num_in_circle_d);
  cudaFree(states_d);

  // free the memory on the host
  delete[] num_in_circle_h;

  return 0;
}
