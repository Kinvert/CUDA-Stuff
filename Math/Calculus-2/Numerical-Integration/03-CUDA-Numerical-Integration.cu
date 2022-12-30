// ChatGPT wasn't really working here so I had to write this
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void integrateKernel(double a, double b, int n, double *sum) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  double h = (b - a) / n;
  if (gid < n) {
    double x = a + h * gid;
    sum[gid] = x * x * h;
  }
  
  __syncthreads();
}

int main() {
  double a = 0.0; // Lower limit of integration
  double b = 1.0; // Upper limit of integration
  int n = 100; // Number of intervals

  double *d_sum;
  cudaMallocManaged((void **)&d_sum, n * sizeof(double));

  integrateKernel<<<1, n>>>(a, b, n, d_sum);
  
  double *h_sum = new double[n];
  cudaMemcpy(h_sum, d_sum, n * sizeof(double), cudaMemcpyDeviceToHost);

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += d_sum[i];
  }

  cudaFree(d_sum);

  std::cout << "The integral is: " << sum << std::endl;
  return 0;
}
