// Written by ChatGPT Dec 15 Version

#include <cuda_runtime.h>

/**
 * @brief Add two matrices element-wise on the GPU
 *
 * This function adds two matrices element-wise and stores the result in a third matrix,
 * using CUDA to perform the computation on the GPU. All matrices must have the same size.
 *
 * @tparam T Scalar type of the matrices
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix, stores the result of A + B
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 *
 * @returns 0 on success, -1 if the matrices have different sizes, or a CUDA error code on failure
 */
template <typename T>
int matrix_addition_cuda(const T* A, const T* B, T* C, int rows, int cols) {
  if (A == nullptr || B == nullptr || C == nullptr) {
    // Invalid input: one of the matrices is a null pointer
    return -1;
  }

  // Check if the matrices have the same size
  if (rows <= 0 || cols <= 0 || rows != B[0] || cols != B[1]) {
    // Invalid input: matrices have different sizes
    return -1;
  }

  // Allocate memory on the GPU
  T* d_A, d_B, d_C;
  cudaError_t err = cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
  if (err != cudaSuccess) {
    // Error allocating memory on the GPU
    return err;
  }
  err = cudaMalloc((void**)&d_B, rows * cols * sizeof(T));
  if (err != cudaSuccess) {
    // Error allocating memory on the GPU
    cudaFree(d_A);
    return err;
  }
  err = cudaMalloc((void**)&d_C, rows * cols * sizeof(T));
  if (err != cudaSuccess) {
    // Error allocating memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    return err;
  }

  // Copy the matrices from the host to the GPU
  err = cudaMemcpy(d_A, A, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    // Error copying memory from the host to the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return err;
  }
  err = cudaMemcpy(d_B, B, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    // Error copying memory from the host to the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return err;
  }

  // Launch the kernel to perform the element-wise addition on the GPU
  matrix_addition_kernel<<<(rows * cols + 255) / 256, 256>>>(d_A, d_B, d_C, rows, cols);

  // Check for errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    // CUDA error occurred
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return err;
  }

  // Copy the result from the GPU to the host
  err = cudaMemcpy(C, d_C, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    // Error copying memory from the GPU to the host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return err;
  }

  // Free the memory allocated on the GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}

/**
 * @brief Kernel to perform element-wise matrix addition on the GPU
 *
 * This kernel adds two matrices element-wise and stores the result in a third matrix.
 * It is called by the `matrix_addition_cuda` function.
 *
 * @tparam T Scalar type of the matrices
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix, stores the result of A + B
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
template <typename T>
__global__ void matrix_addition_kernel(const T* A, const T* B, T* C, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * cols) {
    C[i] = A[i] + B[i];
  }
}

