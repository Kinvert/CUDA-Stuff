// Written by ChatGPT First Try
// I used: g++ 01.cpp -o 1.out -I ~/eigen-3.4.0/
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

int main() {
  // Number of data points
  int n = 10;

  // Data points
  MatrixXd X(n, 2);
  X << 1, 1, 1, 2, 1, 3, 1, 4, 1, 5,
       1, 6, 1, 7, 1, 8, 1, 9, 1, 10;

  // Target values
  VectorXd y(n);
  y << 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  // Solve least squares problem
  VectorXd w = (X.transpose() * X).ldlt().solve(X.transpose() * y);

  std::cout << "Coefficients: " << w(0) << " " << w(1) << std::endl;

  return 0;
}
