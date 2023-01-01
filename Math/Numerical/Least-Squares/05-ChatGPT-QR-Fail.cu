//Written by ChatGPT
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

const int N = 10;

__global__ void kernel(double X[N][2], double y[N], double w[2]) {
  // Transpose of X
  double Xt[2][N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < 2; j++) {
      Xt[j][i] = X[i][j];
    }
  }

  // Matrix product Xt * X
  double XtX[2][2] = {{0, 0}, {0, 0}};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < N; k++) {
        XtX[i][j] += Xt[i][k] * X[k][j];
      }
    }
  }

  // Matrix product Xt * y
  double Xty[2] = {0, 0};
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < N; k++) {
      Xty[i] += Xt[i][k] * y[k];
    }
  }

  // QR decomposition of XtX
  double R[2][2] = {{0, 0}, {0, 0}};
  double Q[2][2] = {{1, 0}, {0, 1}};

  for (int i = 0; i < 2; i++) {
    for (int j = i + 1; j < 2; j++) {
      double s = 0;
      for (int k = 0; k < 2; k++) {
        s += XtX[k][i] * XtX[k][j];
      }

      double c = s / (XtX[i][i] * XtX[i][i] + XtX[j][i] * XtX[j][i]);
      double t = XtX[j][i] / (XtX[i][i] + XtX[j][i] * XtX[j][i]);

      for (int k = 0; k < 2; k++) {
        double tmp = XtX[k][i];
        XtX[k][i] = tmp + t * XtX[k][j];
        XtX[k][j] = XtX[k][j] - c * tmp - t * XtX[k][j];
      }

      for (int k = 0; k < 2; k++) {
        double tmp = Q[k][i];
        Q[k][i] = tmp + t * Q[k][j];
        Q[k][j] = Q[k][j] - c * tmp - t * Q[k][j];
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      R[i][j] = XtX[i][j];
    }
  }

  // Compute least squares solution
  for (int i = 0; i < 2; i++) {
    double s = 0;
    for (int j = 0; j < 2; j++) {
      s += Q[j][i] * Xty[j];
    }
    w[i] = s / R[i][i];
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

  // Launch kernel
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
