// Written by ChatGPT Dec 15 Version
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// Generates a random float between 0 and 1 on the CPU
__device__ float randomFloat(curandState_t* state)
{
    return curand_uniform(state);
}

// Interpolates between a and b by t on the GPU
__device__ float interpolate(float a, float b, float t)
{
    return a * (1 - t) + b * t;
}

// Returns the dot product of the distance and gradient vectors on the GPU
__device__ float dotGridGradient(int ix, int iy, float x, float y, float* gradient)
{
    float gradientb[2][2] = {{randomFloat(), randomFloat()}, {randomFloat(), randomFloat()}};

    // Distance vectors
    float dx = x - (float)ix;
    float dy = y - (float)iy;

    // Compute the dot-product
    return (dx * gradient[iy * 2 + ix] + dy * gradient[iy * 2 + ix + 1]);
}

// Computes Perlin noise at coordinates x, y on the GPU
__global__ void perlinKernel(float x, float y, float* noise, float* gradient)
{
    // Allocate shared memory for the gradient array
    __shared__ float s_gradient[2][2];

    // Copy the gradient values from global memory to shared memory
    s_gradient[threadIdx.y][threadIdx.x] = gradient[threadIdx.y * 2 + threadIdx.x];
    __syncthreads();

    // Determine grid cell coordinates
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    // Determine interpolation weights
    // Could also use higher order polynomial/s-curve here
    float sx = x - (float)x0;
    float sy = y - (float)y0;

    // Interpolate between grid point gradients
    float n0, n1, ix0, ix1;
    n0 = dotGridGradient(x0, y0, x, y, s_gradient[0]);
    n1 = dotGridGradient(x1, y0, x, y, s_gradient[0]);
    ix0 = interpolate(n0, n1, sx);
    n0 = dotGridGradient(x0, y1, x, y, s_gradient[1]);
    n1 = dotGridGradient(x1, y1, x, y, s_gradient[1]);
    ix1 = interpolate(n0, n1, sx);
    *noise = interpolate(ix0, ix1, sy);
    __syncthreads();
}

int main()
{
    // Seed the random number generator
    srand(time(NULL));

    // Create a 28x28 image
    cv::Mat image(28, 28, CV_8UC1);

    // Allocate memory on the GPU for the noise and gradient arrays
    float* d_noise;
    float* d_gradient;
    cudaMalloc(&d_noise, image.rows * image.cols * sizeof(float));
    cudaMalloc(&d_gradient, 2 * 2 * sizeof(float));

    // Generate random gradient values on the CPU and copy them to the GPU
    float h_gradient[2][2];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            h_gradient[i][j] = randomFloat();
        }
    }
    cudaMemcpy(d_gradient, h_gradient, 2 * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the Perlin noise kernel
    int blockSize = 32;
    int numBlocks = (image.rows * image.cols + blockSize - 1) / blockSize;
    perlinKernel<<<numBlocks, blockSize>>>(0.0f, 0.0f, d_noise, d_gradient);

    // Copy the noise values from the GPU to the image on the CPU
    cudaMemcpy(image.data, d_noise, image.rows * image.cols * sizeof(float), cudaMemcpyDeviceToHost);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            std::cout << (int)image.at<uchar>(y, x) << " ";
        }
        std::cout << std::endl;
    }

    // Scale the noise values to the range [0, 255] and cast to uchar
    cv::Mat scaledImage;
    image.convertTo(scaledImage, CV_8UC1, 255.0);

    // Save the image using OpenCV
    cv::imwrite("perlin_noise.png", scaledImage);

    // Free GPU memory
    cudaFree(d_noise);
    cudaFree(d_gradient);

    return 0;
}
