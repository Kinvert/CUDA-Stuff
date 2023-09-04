// Written by Claude 2.5
#include <iostream>
#include <ctime>

#define N 65536

__global__ void bubbleSortShared(int *arr, int n) {

  __shared__ int sArr[512];

  int threadID = threadIdx.x;
  int blockID = blockIdx.x;

  int i = blockID * blockDim.x + threadID;

  sArr[threadID] = arr[i];
  __syncthreads();

  for(int pass=0; pass<n-1; pass++) {

    if(threadID < n-pass-1) {
      if(sArr[threadID] > sArr[threadID+1]) {
        int temp = sArr[threadID];
        sArr[threadID] = sArr[threadID+1];
        sArr[threadID+1] = temp;
      }
    }

    __syncthreads();
  }

  arr[i] = sArr[threadID];

}

int main() {

  int n = N;
  int *arr = new int[n];
  int *d_arr;

  // Initialize arr with random numbers
  srand(time(NULL));
  for (int i = 0; i < n; i++) {
    arr[i] = rand(); 
  }

  cudaMalloc(&d_arr, n * sizeof(int));
  cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  bubbleSortShared<<<(n+255)/256, 256>>>(d_arr, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Print sorted array
  cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Sorted array: ";
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << " ";
  }

  std::cout << "\nTime taken: " << elapsedTime << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_arr);
  delete[] arr;

  return 0;
}
