// Written by ChatGPT Jan 30 Version

#include <cuda_runtime.h>
#include <iostream>

#define N 100
#define M 100

__global__ void update_depth(float *h, float *hu, float *hv, float *h_prev, float *hu_prev, float *hv_prev) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= N || j >= M) {
    return;
  }

  float hl = h_prev[j * N + i];
  float hr = h_prev[j * N + i + 1];
  float hu_top = hu_prev[j * N + i];
  float hu_bottom = hu_prev[j * N + i + N];
  float hv_left = hv_prev[j * N + i];
  float hv_right = hv_prev[j * N + i + 1];

  float h_new = 0.5 * (hl + hr);
  float hu_new = 0.5 * (hu_top + hu_bottom);
  float hv_new = 0.5 * (hv_left + hv_right);

  h[j * N + i] = h_new;
  hu[j * N + i] = hu_new;
  hv[j * N + i] = hv_new;
}

int main() {
  float *h, *hu, *hv, *h_prev, *hu_prev, *hv_prev;
  cudaMalloc((void **) &h, N * M * sizeof(float));
  cudaMalloc((void **) &hu, N * M * sizeof(float));
  cudaMalloc((void **) &hv, N * M * sizeof(float));
  cudaMalloc((void **) &h_prev, N * M * sizeof(float));
  cudaMalloc((void **) &hu_prev, N * M * sizeof(float));
  cudaMalloc((void **) &hv_prev, N * M * sizeof(float));

// Initialize variables on the host
float *h_data, *hu_data, *hv_data;
h_data = (float *) malloc(N * M * sizeof(float));
hu_data = (float *) malloc(N * M * sizeof(float));
hv_data = (float *) malloc(N * M * sizeof(float));
for (int i = 0; i < N; i++) {
  for (int j = 0; j < M; j++) {
    h_data[i * M + j] = hu_data[i * M + j] = hv_data[i * M + j] = 0;
  }
}

// Copy host variables to device
cudaMemcpy(h, h_data, N * M * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(hu, hu_data, N * M * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(hv, hv_data, N * M * sizeof(float), cudaMemcpyHostToDevice);

// Perform simulation
simulate <<< 1, N * M >>> (h, hu, hv, h_prev, hu_prev, hv_prev, N, M, dt, dx, dy);

// Copy device variables to host
cudaMemcpy(h_data, h, N * M * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(hu_data, hu, N * M * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(hv_data, hv, N * M * sizeof(float), cudaMemcpyDeviceToHost);

// Clean up memory
cudaFree(h);
cudaFree(hu);
cudaFree(hv);
cudaFree(h_prev);
cudaFree(hu_prev);
cudaFree(hv_prev);
free(h_data);
free(hu_data);
free(hv_data);

return 0;
} // It failed to add this so I just added it
