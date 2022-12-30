// Written by ChatGPT
#include <iostream>
#include <cmath>

double f(double x) {
  // This function represents the function that we want to integrate.
  // You can change this to any function you want.
  return x*x;
}

int main() {
  double a = 0.0; // Lower limit of integration
  double b = 1.0; // Upper limit of integration
  int n = 100; // Number of intervals
  double h = (b - a) / n; // Width of each interval

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    double x = a + h * i; // Left endpoint of the i-th interval
    sum += f(x) * h; // Add the area of the rectangle to the sum
  }

  std::cout << "The integral is: " << sum << std::endl;
  return 0;
}
