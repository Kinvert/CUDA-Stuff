// Written by ChatGPT, First Try!
#include <iostream>
#include <cmath>

void rowReduce(double** A, int m, int n) {
  for (int i = 0; i < m; i++) {
    // Find the pivot row
    int pivotRow = i;
    for (int j = i + 1; j < m; j++) {
      if (fabs(A[j][i]) > fabs(A[pivotRow][i])) {
        pivotRow = j;
      }
    }

    // Swap the pivot row with the current row (row i)
    if (pivotRow != i) {
      for (int j = 0; j < n; j++) {
        std::swap(A[i][j], A[pivotRow][j]);
      }
    }

    // Normalize the pivot row
    double pivot = A[i][i];
    for (int j = 0; j < n; j++) {
      A[i][j] /= pivot;
    }

    // Eliminate the pivot element from other rows
    for (int k = 0; k < m; k++) {
      if (k == i) continue;
      double factor = A[k][i];
      for (int j = 0; j < n; j++) {
        A[k][j] -= factor * A[i][j];
      }
    }
  }
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

  // Perform row reduction
  rowReduce(A, m, n);

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
