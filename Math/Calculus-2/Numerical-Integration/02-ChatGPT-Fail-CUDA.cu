// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
/*
__global__ void integrateKernel(double a, double b, int n, double *sum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double h = (b - a) / n;
  double x = a + h * idx;
  *sum += x * x * h;
  
  __syncthreads();
}

__global__ void integrateKernel(double a, double b, int n, double *sum) {
  __shared__ double s[1024];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  double h = (b - a) / n;
  double x = a + h * gid;
  s[tid] = x * x * h;
  __syncthreads();

  // Reduction in shared memory
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (tid < i) {
      s[tid] += s[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    sum[blockIdx.x] = s[0];
  }
}
*/
__global__ void integrateKernel(double a, double b, int n, double *sum) {
  __shared__ double s[1024];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  double h = (b - a) / n;
  if (gid < n) {
    double x = a + h * gid;
    s[tid] = x * x * h;
  } else {
    s[tid] = 0.0;
  }
  __syncthreads();

  // Reduction in shared memory
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (tid < i) {
      s[tid] += s[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    sum[blockIdx.x] = s[0];
  }
}


int main() {
  double a = 0.0; // Lower limit of integration
  double b = 1.0; // Upper limit of integration
  int n = 100; // Number of intervals

  double sum = 0.0;
  double *d_sum;
  cudaMalloc((void **)&d_sum, sizeof(double));
  cudaMemcpy(d_sum, &sum, sizeof(double), cudaMemcpyHostToDevice);

  integrateKernel<<<1, n>>>(a, b, n, d_sum);

  cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_sum);

  std::cout << "The integral is: " << sum << std::endl;
  return 0;
}
