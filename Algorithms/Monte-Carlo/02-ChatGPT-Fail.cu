// ChatGPT Dec 15 Version wrote this
// Fails due to fand() in estimate_pi kernel
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// define the number of points to use in the Monte Carlo simulation
const int N = 10000000;

// define the function to run on the GPU
__global__ void estimate_pi(int n, int* in_circle, int* total) {
  // get the index of the current thread
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check if the current thread is within bounds of the array
  if (idx < n) {
    // generate random x and y values between -1 and 1
    double x = (double)rand() / RAND_MAX * 2 - 1;
    double y = (double)rand() / RAND_MAX * 2 - 1;

    // check if the point lies within the unit circle
    if (sqrt(x*x + y*y) < 1) {
      // increment the count of points in the circle
      atomicAdd(in_circle, 1);
    }
    // increment the total count of points
    atomicAdd(total, 1);
  }
}

int main() {
  // allocate memory on the host
  int* h_in_circle = new int[1];
  int* h_total = new int[1];

  // set the initial values to 0
  h_in_circle[0] = 0;
  h_total[0] = 0;

  // allocate memory on the device
  int* d_in_circle;
  int* d_total;
  cudaMalloc(&d_in_circle, sizeof(int));
  cudaMalloc(&d_total, sizeof(int));

  // copy the initial values to the device
  cudaMemcpy(d_in_circle, h_in_circle, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_total, h_total, sizeof(int), cudaMemcpyHostToDevice);

  // launch the kernel on the device
  estimate_pi<<<N/1024, 1024>>>(N, d_in_circle, d_total);

  // copy the results back to the host
  cudaMemcpy(h_in_circle, d_in_circle, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);

  // print the results
  std::cout << "Estimated value of pi: " << (double)h_in_circle[0] / h_total[0] * 4 << std::endl;

  // free the memory on the device
  cudaFree(d_in_circle);
  cudaFree(d_total);

  // free the memory on the host
  delete[] h_in_circle;
  delete[] h_total;

  return 0;
}
