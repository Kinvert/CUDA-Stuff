// ChatGPT Dec 15 Version was unable to fix this so I finally did
// nvcc ChatGPT-Attempt-04-Works.cu -o 4.out `pkg-config opencv4 --cflags --libs`
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define CHANNEL_COUNT 3

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code= " << static_cast<unsigned int>(result)
                  << "(" << cudaGetErrorName(result) << ") \"" << func << "\"" << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel to perform Gaussian blur
__global__ void blurKernel(const float* const input, float* const output, const int width, const int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // 3x3 Gaussian blur kernel
    const float kernel[3][3] = {{1.f / 16, 2.f / 16, 1.f / 16},
                                {2.f / 16, 4.f / 16, 2.f / 16},
                                {1.f / 16, 2.f / 16, 1.f / 16}};

    float sum[3] = {0};
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int xi = x + i;
            int yj = y + j;
            if (xi < 0 || xi >= width || yj < 0 || yj >= height) {
                continue;
            }
            int index = (yj * width + xi) * 3;
            for (int c = 0; c < 3; ++c) {
                sum[c] += input[index + c] * kernel[i + 1][j + 1];
            }
        }
    }
    int index = (y * width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        output[index + c] = sum[c];
    }
}

int main() {
    // Load image data with OpenCV
    cv::Mat image = cv::imread("image1.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }
    cv::imshow("Original Image", image);
      
    // Allocate memory for input and output images on host and device
    int width = image.cols;
    int height = image.rows;
    float* h_input = (float*)malloc(width * height * CHANNEL_COUNT * sizeof(float));
    float* h_output = (float*)malloc(width * height * CHANNEL_COUNT * sizeof(float));
    float* d_input;
    float* d_output;
    checkCudaErrors(cudaMalloc(&d_input, width * height * CHANNEL_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output, width * height * CHANNEL_COUNT * sizeof(float)));

    // Convert image data to floating point
    for (int i = 0; i < width * height * CHANNEL_COUNT; ++i) {
        h_input[i] = image.data[i] / 255.0f;
    }

    // Set kernel parameters
    int kernelRadius = 1;
    int kernelLength = 2 * kernelRadius + 1;
    float* h_kernel = (float*)malloc(kernelLength * kernelLength * sizeof(float));
    for (int i = 0; i < kernelLength; ++i) {
        for (int j = 0; j < kernelLength; ++j) {
            int x = i - kernelRadius;
            int y = j - kernelRadius;
            h_kernel[i * kernelLength + j] = exp(-(x * x + y * y) / (2 * kernelRadius * kernelRadius)) / (2 * M_PI * kernelRadius * kernelRadius);
        }
    }

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(d_input, h_input, width * height * CHANNEL_COUNT * sizeof(float), cudaMemcpyHostToDevice));
    float* d_kernel;
    checkCudaErrors(cudaMalloc(&d_kernel, kernelLength * kernelLength * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, kernelLength * kernelLength * sizeof(float), cudaMemcpyHostToDevice));

    // Set grid and block dimensions
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Execute kernel
    blurKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy data from device to host
    checkCudaErrors(cudaMemcpy(h_output, d_output, width * height * CHANNEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost));
    
    unsigned char* output_int = (unsigned char*)malloc(width * height * CHANNEL_COUNT * sizeof(unsigned char));
    
    // Convert output data back to unsigned char
    for (int i = 0; i < width * height * CHANNEL_COUNT; ++i) {
        output_int[i] = (unsigned char) (h_output[i] * 255.0);
    }
    
    cv::Mat blurredImage(height, width, CV_8UC3, output_int);

    // Save output image
    cv::imwrite("image1-gaussian-blur.jpg", blurredImage);

    // Free memory
    free(h_input);
    free(h_output);
    free(h_kernel);
    cudaFree(d_input);
    cudaFree(d_output);

    // Display output image
    cv::imshow("Blurred Image", blurredImage);
    cv::waitKey(0);

    return 0;
}
