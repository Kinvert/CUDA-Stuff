// Written by Claude 2.5
#include <iostream>
#include <ctime>
#include <cstdlib>

__global__ void bubbleSort(int *arr, int n) {

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

  int n;
  std::cout << "Enter number of elements: ";
  std::cin >> n;

  int arr[n];

  // Generate random array
  srand(time(NULL));
  for(int i = 0; i < n; i++) {
    arr[i] = rand(); // Remove modulo for larger numbers
  }

  int *d_arr;
  cudaMalloc(&d_arr, n * sizeof(int));
  cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

  clock_t start = clock();

  bubbleSort<<<(n+255)/256, 256>>>(d_arr, n);

  clock_t end = clock();
  double timeTaken = double(end - start) / double(CLOCKS_PER_SEC);

  cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_arr);

  std::cout << "Sorted array: ";
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << " "; 
  }

  std::cout << "\nTime taken: " << timeTaken << " seconds" << std::endl;

  return 0;
}
