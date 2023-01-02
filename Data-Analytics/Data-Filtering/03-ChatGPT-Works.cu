// Written by ChatGPT Dec 15 Version
#include <stdio.h>
#include <cuda_runtime.h>

// define the input data size
#define DATA_SIZE 100

// define the threshold for filtering
#define THRESHOLD 50

// host (CPU) function to generate input data
void generateData(int* data) {
  for (int i = 0; i < DATA_SIZE; i++) {
    data[i] = rand() % 100; // generate a random number between 0 and 99
  }
}

// CUDA kernel to filter the data
__global__ void filterData(int* data, int* filtered, int* numFiltered) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < DATA_SIZE && data[idx] > THRESHOLD) {
    filtered[atomicAdd(numFiltered, 1)] = data[idx];
  }
}

int main() {
  // allocate memory on the host (CPU)
  int* data = (int*)malloc(DATA_SIZE * sizeof(int));
  int* filtered = (int*)malloc(DATA_SIZE * sizeof(int));
  int numFiltered = 0;

  // generate input data on the host
  generateData(data);

  // allocate memory on the device (GPU)
  int* d_data;
  int* d_filtered;
  int* d_numFiltered;
  cudaMalloc((void**)&d_data, DATA_SIZE * sizeof(int));
  cudaMalloc((void**)&d_filtered, DATA_SIZE * sizeof(int));
  cudaMalloc((void**)&d_numFiltered, sizeof(int));

  // copy the input data and numFiltered variable from the host to the device
  cudaMemcpy(d_data, data, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_numFiltered, &numFiltered, sizeof(int), cudaMemcpyHostToDevice);

  // launch the CUDA kernel
  filterData<<<1, DATA_SIZE>>>(d_data, d_filtered, d_numFiltered);

  // copy the numFiltered variable back from the device to the host
  cudaMemcpy(&numFiltered, d_numFiltered, sizeof(int), cudaMemcpyDeviceToHost);

  // copy the filtered data back from the device to the host
  cudaMemcpy(filtered, d_filtered, numFiltered * sizeof(int), cudaMemcpyDeviceToHost);

  // print the filtered data
  for (int i = 0; i < numFiltered; i++) {
    printf("%d ", filtered[i]);
  }

  // free memory on the device
  cudaFree(d_data);
  cudaFree(d_filtered);
  cudaFree(d_numFiltered);

  // free memory on the host
  free(data);
  free(filtered);

  return 0;
}
