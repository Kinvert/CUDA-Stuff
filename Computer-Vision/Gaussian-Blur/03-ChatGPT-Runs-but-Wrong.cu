// Written by ChatGPT Dec 15 Version
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "opencv2/opencv.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define KERNEL_RADIUS 4
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define CHANNEL_COUNT 3

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
            int index = (yj * width + xi) * CHANNEL_COUNT;
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
    // Load image data
    int width, height;
    unsigned char* imageData = NULL;

    // Load image data with OpenCV
    cv::Mat image = cv::imread("../../zData/image1.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }
    width = image.cols;
    height = image.rows;
    imageData = (unsigned char*)malloc(width * height * CHANNEL_COUNT * sizeof(unsigned char));
    memcpy(imageData, image.data, width * height * CHANNEL_COUNT * sizeof(unsigned char));
    
    cv::imshow("Original Image", image);
    
    std::cout << image.cols << " " << image.rows << " " << image.channels() << std::endl;

    // Convert image data to floating point
    float* h_input = (float*)malloc(width * height * CHANNEL_COUNT * sizeof(float));
    for (int i = 0; i < width * height * CHANNEL_COUNT; ++i) {
        h_input[i] = imageData[i] / 255.0f;
    }
    free(imageData);

    // Copy input data to device
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, width * height * CHANNEL_COUNT * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * CHANNEL_COUNT * sizeof(float));
    cudaMemcpy(d_input, h_input, width * height * CHANNEL_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    free(h_input);

    // Compute kernel
    float h_kernel[KERNEL_LENGTH * KERNEL_LENGTH];
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i) {
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; ++j) {
            h_kernel[(i + KERNEL_RADIUS) * KERNEL_LENGTH + (j + KERNEL_RADIUS)] = expf(-(i * i + j * j) / (2.0f * KERNEL_RADIUS * KERNEL_RADIUS));
        }
    }
    float* d_kernel;
    cudaMalloc((void**)&d_kernel, KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    // Set kernel launch parameters
    dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridSize((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    // Launch kernel
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_kernel, KERNEL_LENGTH);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy output data from device to host
    float* h_output = (float*)malloc(width * height * CHANNEL_COUNT * sizeof(float));
    cudaMemcpy(h_output, d_output, width * height * CHANNEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert output data to unsigned char
    unsigned char* outputData = (unsigned char*)malloc(width * height * CHANNEL_COUNT * sizeof(unsigned char));
    for (int i = 0; i < width * height * CHANNEL_COUNT; ++i) {
        outputData[i] = (unsigned char)(h_output[i] * 255.0f);
    }
    free(h_output);
    
    cv::Mat blurredImage(height, width, CV_8UC3, outputData);
    cv::imshow("Blurred Image", blurredImage);

    // Save output image
    // Replace this block with your own code to save the image data in the "outputData" array to a file
    {
        for (int i = 0; i < width * height * CHANNEL_COUNT; ++i) {
            //printf("%d ", outputData[i]);
        }
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    cv::waitKey(0);

    return 0;
}
