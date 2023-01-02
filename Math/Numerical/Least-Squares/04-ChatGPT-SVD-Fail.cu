// Written by ChatGPT Dec 15 Version
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

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

  // SVD of XtX
  double U[2][2] = {{0, 0}, {0, 0}};
  double S[2] = {0, 0};
  double Vt[2][2] = {{0, 0}, {0, 0}};
  double tmp[2];

  // Jacobi SVD algorithm
  for (int i = 0; i < 2; i++) {
    for (int j = i + 1; j < 2; j++) {
      while (fabs(XtX[j][i]) > 1e-12) {
        double tau = (XtX[i][i] - XtX[j][j]) / (2 * XtX[j][i]);
        double t = copysign(1.0 / (fabs(tau) + sqrt(1 + tau * tau)), tau);
        double c = 1.0 / sqrt(1 + t * t);
        double s = c * t;

        for (int k = 0; k < 2; k++) {
          tmp[k] = XtX[i][k];
          XtX[i][k] = c * tmp[k] - s * XtX[j][k];
          XtX[j][k] = s * tmp[k] + c * XtX[j][k];
        }

        for (int k = 0; k < 2; k++) {
          tmp[k] = XtX[k][i];
          XtX[k][i] = c * tmp[k] - s * XtX[k][j];
          XtX[k][j] = s * tmp[k] + c * XtX[k][j];
        }

        for (int k = 0; k < 2; k++) {
          tmp[k] = U[k][i];
          U[k][i] = c * tmp[k] - s * U[k][j];
          U[k][j] = s * tmp[k] + c * U[k][j];
        }
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    S[i] = XtX[i][i];
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      Vt[i][j] = U[j][i];
    }
  }

  // Compute least squares solution
  for (int i = 0; i < 2; i++) {
    if (fabs(S[i]) > 1e-12) {
      w[i] = Xty[i] / S[i];
    } else {
      w[i] = 0;
    }
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
