// ChatGPT several tries not working
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE; \
}} while(0)

__global__ void rowReduceKernel(double* A, int m, int n) {
  int i = blockIdx.x;
  int j = threadIdx.x;

  if (i >= m || j >= n) {
    return;
  }

  // Normalize the pivot row
  double pivot = A[i*n + i];
  const double epsilon = 1e-9;
  if (fabs(pivot) > epsilon) {
    if (pivot != 0) {
      for (int k = 0; k < n; k++) {
        A[i*n + k] /= pivot;
      }
    }
  }

  // Eliminate the pivot element from other rows
  for (int k = 0; k < m; k++) {
    if (k == i) continue;
    double factor = A[k*n + i];
    for (int l = 0; l < n; l++) {
      A[k*n + l] -= factor * A[i*n + l];
    }
  }
  
  printf("i=%d j=%d\n", i, j);
  
  __syncthreads();
}

int main() {
  const int m = 3;
  const int n = 4;

  // Initialize the matrix
  double** A;
  A = new double*[m];
  for (int i = 0; i < m; i++) {
    A[i] = new double[n];
  }
  A[0][0] = 1;
  A[0][1] = 2;
  A[0][2] = 3;
  A[0][3] = 4;
  A[1][0] = 5;
  A[1][1] = 6;
  A[1][2] = 7;
  A[1][3] = 8;
  A[2][0] = 9;
  A[2][1] = 10;
  A[2][2] = 11;
  A[2][3] = 12;

  double* d_A;
  CUDA_CALL(cudaMalloc((void**)&d_A, m*n*sizeof(double)));
  CUDA_CALL(cudaMemcpy(d_A, A[0], m*n*sizeof(double), cudaMemcpyHostToDevice));

  rowReduceKernel<<<m,n>>>(d_A, m, n);

  CUDA_CALL(cudaMemcpy(A[0], d_A, m*n*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaFree(d_A));

  // Print the row-reduced matrix
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << A[i][j] << " ";
    }
    std::cout << std::endl;
  }

  // Free memory
  for (int i = 0; i < m; i++) {
    delete[] A[i];
  }
  delete[] A;

  return 0;
}

