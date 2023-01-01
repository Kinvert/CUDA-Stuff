//Written by ChatGPT First Try
#include <iostream>
#include <cmath>

const int N = 10;

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

  // Solve least squares problem
  double XtX[2][2] = {{0, 0}, {0, 0}};
  double Xty[2] = {0, 0};

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      XtX[j][0] += X[i][j] * X[i][0];
      XtX[j][1] += X[i][j] * X[i][1];
      Xty[j] += X[i][j] * y[i];
    }
  }

  double det = XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0];
  w[0] = (XtX[1][1] * Xty[0] - XtX[0][1] * Xty[1]) / det;
  w[1] = (XtX[0][0] * Xty[1] - XtX[1][0] * Xty[0]) / det;

  std::cout << "Coefficients: " << w[0] << " " << w[1] << std::endl;

  return 0;
}
