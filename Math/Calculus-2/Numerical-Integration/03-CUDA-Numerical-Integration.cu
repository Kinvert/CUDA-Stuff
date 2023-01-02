// ChatGPT Dec 15 Version wasn't really working here so I had to write this
#include <stdio.h>

__global__ void integrateKernel(double a, double b, int n, double *sum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double dx = (b - a) / n;
  if (tid < n) {
    double x = a + dx * tid;
    double funcVal = x * x; // The function to numerically integrate
    sum[tid] = funcVal * dx;
  }
  __syncthreads();
}

int main() {
  double a = 0.0; // Lower limit of integration
  double b = 1.0; // Upper limit of integration
  int n = 1000; // Number of intervals

  // Allocate memory
  double *h_sum = new double[n];
  double *d_sum;
  cudaMalloc(&d_sum, n * sizeof(double));
  cudaMemcpy(d_sum, h_sum, n * sizeof(double), cudaMemcpyHostToDevice);

  // Run Kernel
  integrateKernel<<<1, n>>>(a, b, n, d_sum);
  
  // Memory from GPU to CPU
  cudaMemcpy(h_sum, d_sum, n * sizeof(double), cudaMemcpyDeviceToHost);

  // Sum values
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += h_sum[i];
  }
  
  printf("The integral is: %f\n", sum);

  cudaFree(d_sum);

  return 0;
}
