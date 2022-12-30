# Numerical Integration

**Note: This was written by ChatGPT**

## Introduction

Numerical integration is a method for approximating the value of a definite integral. A definite integral is a mathematical operation that involves the integration (i.e., summation) of a function over a given range. Numerical integration is often used when it is not possible or practical to obtain an exact solution to an integral using analytical methods.

There are several methods for performing numerical integration, including the rectangular rule, the trapezoidal rule, and Simpson's rule. These methods involve dividing the range of integration into smaller intervals, and approximating the integral by summing the contributions of each interval. The accuracy of the approximation depends on the number of intervals used and the method used to calculate the contribution of each interval.

The rectangular rule is a simple method for numerical integration that involves approximating the integral by summing the contributions of each interval using the value of the function at the midpoint of the interval. The rectangular rule is given by the following formula:

$$\int_a^bf(x)dx\approx\sum_{i=1}^nf\left(a+\frac{(b-a)i}{n}\right)\frac{b-a}{n}$$

where $a$ and $b$ are the lower and upper limits of integration, $n$ is the number of intervals, and $f(x)$ is the function being integrated.

## C++ Implementation

```cpp
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
```

This program approximates the definite integral of the function f(x) between the limits a and b using the rectangular rule. The rectangular rule approximates the area under the curve by dividing the integration interval into n subintervals (or rectangles) of equal width h, and summing the areas of the rectangles. The area of each rectangle is calculated by multiplying the value of the function at the left endpoint of the interval (f(x)) by the width of the interval (h).

## CUDA Implementation

**Note: ChatGPT was not writing good code for this so I had to do it instead**

**Note: ChatGPT did write this part of the README though, after I wrote the code**

Here is a breakdown of the code, with explanations for each code snippet:

```cpp
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
```

These lines include the necessary header files for the program. The iostream header file is used for input/output operations, the cmath header file is used for mathematical functions, and the cuda_runtime.h header file is used for CUDA runtime functions.

```cpp
__global__ void integrateKernel(double a, double b, int n, double *sum) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  double h = (b - a) / n;
  if (gid < n) {
    double x = a + h * gid;
    sum[gid] = x * x * h;
  }
  
  __syncthreads();
}
```

This is the kernel function that performs the numerical integration. The kernel function takes four arguments: a and b are the lower and upper limits of integration, n is the number of intervals, and sum is a pointer to an array that will store the result of the integration.

Next, the kernel function calculates the width h of each interval using the formula h = (b - a) / n.

The kernel function then checks if the global index gid is less than n, and if it is, it calculates the value of the function at the midpoint of the interval using the formula x = a + h * gid. The kernel function then stores the contribution of the interval to the integral in the global memory location sum[gid] using the formula sum[gid] = x * x * h.

Finally, the kernel function calls the __syncthreads() function to synchronize all threads in the block. This ensures that all threads have completed their calculations before the kernel function returns.

```cpp
int main() {
  double a = 0.0; // Lower limit of integration
  double b = 1.0; // Upper limit of integration
  int n = 100; // Number of intervals

  double *d_sum;
  cudaMallocManaged((void **)&d_sum, n * sizeof(double));
```

In the main function, the lower and upper limits of integration, a and b, and the number of intervals, n, are initialized.

The cudaMallocManaged function is then used to allocate managed memory on the device for the result of the integration. Managed memory is a type of memory that is automatically managed by the CUDA runtime and can be accessed by both the host (CPU) and the device (GPU).

```cpp
  integrateKernel<<<1, n>>>(a, b, n, d_sum);
  
  double *h_sum = new double[n];
  cudaMemcpy(h_sum, d_sum, n * sizeof(double), cudaMemcpyDeviceToHost);
```

After the device memory has been allocated, the kernel function is launched using the <<<1, n>>> syntax. This launches a single block of n threads on the device.

The cudaMemcpy function is then used to copy the device memory d_sum back to host memory. The host memory array h_sum is allocated using the new operator, and the cudaMemcpy function is used to copy the device memory to the host memory. The cudaMemcpy function takes four arguments: a pointer to the destination memory location, a pointer to the source memory location, the size of the memory to be copied, and the type of memory copy to be performed (in this case, cudaMemcpyDeviceToHost to copy from device to host memory).

```cpp
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += d_sum[i];
  }

  cudaFree(d_sum);

  std::cout << "The integral is: " << sum << std::endl;
  return 0;
}
```

Finally, the host memory array h_sum is iterated over, and the contributions of each interval are summed to calculate the integral. 

The device memory is freed using the cudaFree function. This is an important step to ensure that the device memory is properly deallocated and is not left hanging.

Finally, the result of the integration is printed to the console using the std::cout function, and the main function returns with a value of 0.
