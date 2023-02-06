// Written by ChatGPT Jan 30 Version

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

const int BLOCK_SIZE = 32;

__global__ void GaussianBlurKernel(float *inputImage, float *outputImage, int width, int height, float *filter, int filterSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int center = filterSize / 2;
    float sum = 0;
    for (int x = 0; x < filterSize; ++x) {
        for (int y = 0; y < filterSize; ++y) {
            int imageX = i + x - center;
            int imageY = j + y - center;

            // Check if the pixel is outside the image boundaries
            if (imageX < 0 || imageX >= width || imageY < 0 || imageY >= height) continue;

            sum += inputImage[imageY * width + imageX] * filter[y * filterSize + x];
        }
    }

    outputImage[j * width + i] = sum;
}

void GaussianBlur(float *inputImage, float *outputImage, int width, int height, float *filter, int filterSize) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    GaussianBlurKernel<<<gridSize, blockSize>>>(inputImage, outputImage, width, height, filter, filterSize);
}

int main() {
    int width = 100;
    int height = 100;
    int filterSize = 5;
    int filterRadius = filterSize / 2;

    float *inputImage = new float[width * height];
    float *outputImage = new float[width * height];
    float *filter = new float[filterSize * filterSize];

    // Generate input image
    for (int i = 0; i < width * height; ++i) {
        inputImage[i] = i;
    }

    // Generate filter
    float sigma = 2.0f;
    float sum = 0;
    for (int y = 0; y < filterSize; ++y) {
        for (int x = 0; x < filterSize; ++x) {
            int xDistance = x - filterRadius;
            int yDistance = y - filterRadius;
            float value = exp(-(xDistance * xDistance + yDistance * yDistance) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            filter[y * filterSize + x] = value;
            sum += value;
        }
    }

// Normalize the filter
for (int i = 0; i < filterSize * filterSize; ++i) {
    filter[i] /= sum;
}
