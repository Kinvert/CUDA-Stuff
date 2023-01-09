// Written by ChatGPT Dec 15 Version

#include <iostream>
#include "050-ChatGPT-CPP-Lib.h"

int main() {
  // Initialize matrices A, B, and C
  const int rows = 3, cols = 3;
  int A[rows * cols + 1] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::cout << sizeof(A) << std::endl;
  int B[rows * cols] = {9, 8, 7, 6, 5, 4, 3, 2, 2};
  std::cout << sizeof(B) << std::endl;
  int C[rows * cols];

  // Add the matrices
  int result = matrix_addition(A, B, C, rows, cols);
  if (result != 0) {
    std::cerr << "Error: matrix addition failed with error code " << result << std::endl;
    return 1;
  }

  // Print the result
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << C[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
