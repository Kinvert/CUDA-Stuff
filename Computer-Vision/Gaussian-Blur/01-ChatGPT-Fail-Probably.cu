// Written by ChatGPT Dec 15 Version
// Most programs here are tested but this one is not

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define KERNEL_RADIUS 4
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define CHANNEL_COUNT 4

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, (int)result, cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__global__ void blurKernel(const float* const input, float* const output, const int width, const int height, const float* const kernel, const int kernelLength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum[CHANNEL_COUNT] = {0};
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i) {
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; ++j) {
            int xi = MIN(MAX(x + i, 0), width - 1);
            int yj = MIN(MAX(y + j, 0), height - 1);
            int index = (yi * width + xi) * CHANNEL_COUNT;
            for (int c = 0; c < CHANNEL_COUNT; ++c) {
                sum[c] += input[index + c] * kernel[(i + KERNEL_RADIUS) * KERNEL_LENGTH + (j + KERNEL_RADIUS)];
            }
        }
    }
    int index = (y * width + x) * CHANNEL_COUNT;
    for (int c = 0; c < CHANNEL_COUNT; ++c) {
        output[index + c] = sum[c];
    }
}

int main(int argc, char** argv) {
    int width, height, channelCount;
    float* h_input = (float*)stbi_loadf("image.jpg", &width, &height, &channelCount, CHANNEL_COUNT);
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, width * height * CHANNEL_COUNT * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * CHANNEL_COUNT * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * CHANNEL_COUNT * sizeof(float), cudaMemcpyHostToDevice);

    const int blockCountX = (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    const int blockCountY = (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    dim3 blockCount(blockCountX, blockCountY);
    dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT);

    // Generate Gaussian kernel
    float kernel[KERNEL_LENGTH * KERNEL_LENGTH];
    float sigma = 1.0f;
    float twoSigmaSquared = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = 0; i < KERNEL_LENGTH; ++i) {
        for (int j = 0; j < KERNEL_LENGTH; ++j) {
            int x = i - KERNEL_RADIUS;
            int y = j - KERNEL_RADIUS;
            kernel[i * KERNEL_LENGTH + j] = expf(-(x * x + y * y) / twoSigmaSquared) / (M_PI * twoSigmaSquared);
            sum += kernel[i * KERNEL_LENGTH + j];
        }
    }
    for (int i = 0; i < KERNEL_LENGTH * KERNEL_LENGTH; ++i) {
        kernel[i] /= sum;
    }

    blurKernel<<<blockCount, blockSize>>>(d_input, d_output, width, height, kernel, KERNEL_LENGTH);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    float* h_output = (float*)malloc(width * height * CHANNEL_COUNT * sizeof(float));
    cudaMemcpy(h_output, d_output, width * height * CHANNEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", width, height, CHANNEL_COUNT, h_output, width * CHANNEL_COUNT * sizeof(float));

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

