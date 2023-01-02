// Written by ChatGPT Dec 15 Version
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

const int N = 10;

__global__ void kernel(double X[N][2], double y[N], double w[2]) {
  // Coefficients
  double XtX[2][2] = {{0, 0}, {0, 0}};
  double Xty[2] = {0, 0};

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    for (int j = 0; j < 2; j++) {
      XtX[j][0] += X[i][j] * X[i][0];
      XtX[j][1] += X[i][j] * X[i][1];
      Xty[j] += X[i][j] * y[i];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    double det = XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0];
    w[0] = (XtX[1][1] * Xty[0] - XtX[0][1] * Xty[1]) / det;
    w[1] = (XtX[0][0] * Xty[1] - XtX[1][0] * Xty[0]) / det;
  }
}

int main() {
  // Number of data points
  int n = N;

  // Data points
  double X[N][2] = {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5},
                    {1, 6}, {1, 7}, {1, 8}, {1, 9}, {1, 10}};

  // Target values
  double y[N] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Coefficients
  double w[2] = {0, 0};

  // Allocate device memory
  double *d_X, *d_y, *d_w;
  cudaMalloc(&d_X, N * 2 * sizeof(double));
  cudaMalloc(&d_y, N * sizeof(double));
  cudaMalloc(&d_w, 2 * sizeof(double));

  // Copy data to device memory
  cudaMemcpy(d_X, X, N * 2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, 2 * sizeof(double), cudaMemcpyHostToDevice);

  // Launch kernel
//  kernel<<<(N + 255) / 256, 256>>>(d_X, d_y, d_w);
  kernel<<<(N + 255) / 256, 256>>>((double (*)[2]) d_X, d_y, d_w);

  // Copy result from device memory
  cudaMemcpy(w, d_w, 2 * sizeof(double), cudaMemcpyDeviceToHost);

  std::cout << "Coefficients: " << w[0] << " " << w[1] << std::endl;

  // Free device memory
  cudaFree(d_X);
  cudaFree(d_y);
  cudaFree(d_w);

  return 0;
}
