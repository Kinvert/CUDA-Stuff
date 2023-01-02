// ChatGPT Dec 15 Version wrote this
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

__global__ void estimate_pi(int n, curandState* states, float* pi) {
  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // initialize the cuRAND states
  curand_init(0, i, 0, states);

  // generate a random point in the unit square
  float x = curand_uniform(&states[i]);
  float y = curand_uniform(&states[i]);

  // count how many points fall within the unit circle
  if (x * x + y * y < 1) {
    atomicAdd(pi, 1);
  }
}

int main() {
  // number of points to generate
  const int n = 100000000;

  // allocate device memory
  curandState* states;
  cudaMalloc((void**)&states, n * sizeof(curandState));
  float* pi;
  cudaMalloc((void**)&pi, sizeof(float));

  // create curand states on the device
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
//  curandCreateStates(states, n, 0);

  // launch kernel to generate random points and estimate pi
  estimate_pi<<<num_blocks, block_size>>>(n, states, pi);

  // copy result from device to host
  float pi_h;
  cudaMemcpy(&pi_h, pi, sizeof(float), cudaMemcpyDeviceToHost);

  // print result
  std::cout << "Estimated value of pi: " << 4.0 * pi_h / n << std::endl;

  // free device memory
  cudaFree(states);
  cudaFree(pi);

  return 0;
}
