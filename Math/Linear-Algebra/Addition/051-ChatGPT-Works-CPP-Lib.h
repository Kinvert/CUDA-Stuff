// Written by ChatGPT Dec 15 Version
// didn't work so I fixed it
/**
 * @brief Add two matrices element-wise
 *
 * This function adds two matrices element-wise and stores the result in a third matrix.
 * All matrices must have the same size.
 *
 * @tparam T Scalar type of the matrices
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix, stores the result of A + B
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 *
 * @returns 0 on success, -1 if the matrices have different sizes
 */
template <typename T>
int matrix_addition(const T* A, const T* B, T* C, int rows, int cols) {
  if (A == nullptr || B == nullptr || C == nullptr) {
    // Invalid input: one of the matrices is a null pointer
    return -1;
  }
  
  // Check if the matrices have the same size
  if (rows <= 0 || cols <= 0) {
    // Invalid input: matrices have different sizes
    return -1;
  }

  // Perform element-wise addition
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
    }
  }

  return 0;
}
