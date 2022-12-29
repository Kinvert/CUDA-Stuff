// ChatGPT wrote this
// This does not work due to rand() in kernel
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda.h>

// define the function to run on the GPU
__global__ void estimate_pi(int num_samples, int* num_hits) {
  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // check if the current thread is within bounds of the array
  if (i < num_samples) {
    // generate random x and y coordinates
    float x = (float)rand() / RAND_MAX;
    float y = (float)rand() / RAND_MAX;

    // check if the point lies inside the unit circle
    if (x*x + y*y < 1) {
      // increment the number of hits
      num_hits[i] = 1;
    } else {
      num_hits[i] = 0;
    }
  }
}

int main() {
  // set the number of samples to generate
  const int num_samples = 10000000;

  // allocate memory on the device for the num_hits array
  int* d_num_hits;
  cudaMalloc((void**)&d_num_hits, num_samples * sizeof(int));

  // launch the kernel to estimate pi
  estimate_pi<<<(num_samples + 255) / 256, 256>>>(num_samples, d_num_hits);

  // copy the num_hits array back to the host
  int* h_num_hits = new int[num_samples];
  cudaMemcpy(h_num_hits, d_num_hits, num_samples * sizeof(int), cudaMemcpyDeviceToHost);

  // sum up the number of hits
  int num_hits = 0;
  for (int i = 0; i < num_samples; ++i) {
    num_hits += h_num_hits[i];
  }

  // estimate the value of pi
  float pi = 4.0f * num_hits / num_samples;

  // print the result
  std::cout << "Estimated value of pi: " << pi << std::endl;

  // free memory on the device and host
  cudaFree(d_num_hits);
  delete[] h_num_hits;

  return 0;
}

