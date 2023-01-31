// Written by ChatGPT Jan 9 Version

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
  float hd = h_prev[j * N + i + N];
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

  // Set initial conditions for depth
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (i >= N / 4 && i <= N / 4 * 3 && j >= M / 4 && j <= M / 4 * 3) {
        h_prev[j * N + i] = 1;
      }
    }
  }

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  update_depth<&#8203;`oaicite:{"index":0,"invalid_reason":"Malformed citation <<grid, block>>>"}`&#8203;(d_h, d_hu, d_hv, d_h_prev, d_hu_prev, d_hv_prev, N, M);

  cudaDeviceSynchronize();

  cudaMemcpy(h, d_h, N * M * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hu, d_hu, N * M * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hv, d_hv, N * M * sizeof(float), cudaMemcpyDeviceToHost);

  // Print result
  std::cout << "Depth at (100, 100): " << h[100][100] << std::endl;
  std::cout << "Depth at (150, 150): " << h[150][150] << std::endl;
  std::cout << "Depth at (200, 200): " << h[200][200] << std::endl;

  cudaFree(d_h);
  cudaFree(d_hu);
  cudaFree(d_hv);
  cudaFree(d_h_prev);
  cudaFree(d_hu_prev);
  cudaFree(d_hv_prev);

  return 0;
}
