// Written by Claude 2.5
#include <iostream>
#include <ctime>
#include <cstdlib>

void bubbleSortCPP(int arr[], int n) {
  bool swapped = true;
  int i = 0;
  int j = 0;
  int temp;
  
  while (swapped) {
    swapped = false;
    j++;
    for (i = 0; i < n - j; i++) {
      if (arr[i] > arr[i + 1]) {
        temp = arr[i];
        arr[i] = arr[i + 1];  
        arr[i + 1] = temp;
        swapped = true;
      }
    }
  }
}

__global__ void bubbleSortCUDA(int *arr, int n) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    bool swapped = true;
    int j = 0; 
    int temp;

    while (swapped) {
      swapped = false;
      j++;
      if (i < n - j) {
        if (arr[i] > arr[i + 1]) {
          temp = arr[i];
          arr[i] = arr[i + 1];
          arr[i + 1] = temp;
          swapped = true;
        }
      }
    }
  }
}

int main() {

  int n = 1024; // Array size

  int arr[n];
  // Generate random array

  int *d_arr;
  cudaMalloc(&d_arr, n * sizeof(int));
  cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

  // Time C++ sort
  clock_t start = clock();
  bubbleSortCPP(arr, n);
  clock_t end = clock();
  double timeTakenCPP = double(end - start) / double(CLOCKS_PER_SEC);

  // Print sorted array
  std::cout << "Sorted array (CPP): ";
  for (int i = 0; i < n; i++) {
     std::cout << arr[i] << " ";
  }

  // Copy original unsorted array back to GPU
  cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
  // Time CUDA sort
  start = clock();
  bubbleSortCUDA<<<(n+255)/256, 256>>>(d_arr, n);
  end = clock();
  double timeTakenCUDA = double(end - start) / double(CLOCKS_PER_SEC);

  // Print sorted array  
  cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "\nSorted array (CUDA): ";
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << " ";
  }

  std::cout << "\nTime taken (CPP): " << timeTakenCPP << " seconds" << std::endl;
  std::cout << "Time taken (CUDA): " << timeTakenCUDA << " seconds" << std::endl;

  cudaFree(d_arr);

  return 0;
}
