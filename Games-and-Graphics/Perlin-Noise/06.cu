// I'm the one writing this, ChatGPT isn't working for CUDA Perlin Noise
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// Interpolates between a and b by t on the GPU
__device__ double interpolate(double a, double b, double t)
{
    return a * (1.0 - t) + b * t;
}

// Returns the dot product of the distance and gradient vectors on the GPU
__device__ double dotGridGradient(int ix, int iy, int x, int y, double* gradient)
{
    int ix2 = (int)(ix / 4);
    int iy2 = (int)(iy / 4);

    // Distance vectors
    double dx = (double)x - (double)ix;
    double dy = (double)y - (double)iy;
    
    double grad = gradient[iy2 * 8 + ix2];
    
    double cosx = cos(grad);
    double sinx = sin(grad);
    
    double result = dx * cosx + dy * sinx;
    //float result = dx * cos(ang) + dy * sin(ang);
    //float result = dx * randomFloat(states, id) + dy * randomFloat(states, id);
    
    //printf("x=%.3f y=%.3f ix=%d iy=%d ix2=%d iy2=%d dx=%.3f dy=%.3f result=%.3f\n", x, y, ix, iy, ix2, iy2, dx, dy, result);
    
    //printf("1=%.3f 1=%.3f 1=%.3f 1=%.3f\n", random1, random2, random3, random4);
    
    //printf("ix=%d grad=%f result=%.3f\n", ix, gradient[iy][ix+1], result);
    
    //printf("ix=%d iy=%d x=%d y=%d dx=%.3f dy=%.3f grad=%.3f cosx=%.3f sinx=%.3f RESULT=%.3f\n", ix, iy, x, y, dx, dy, grad, cosx, sinx, result);

    return result;
}

// Computes Perlin noise at coordinates x, y on the GPU
__global__ void perlinKernel(int width, int height, uint* noise, double* gradient)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
    
        int idx = y * width + threadIdx.x;

        // Determine grid cell coordinates
        int x0 = (int)floor((double)((x / 4) * 4));
        int x1 = x0 + 4;
        int y0 = (int)floor((double)((y / 4) * 4));
        int y1 = y0 + 4;

        // Determine interpolation weights
        // Could also use higher order polynomial/s-curve here
        double sx = ((double)x - (double)x0) / 4.0;
        double sy = ((double)y - (double)y0) / 4.0;

        // Interpolate between grid point gradients
        double n0a, n0b, n1a, n1b, ix0, ix1, ix2, ix3;
        uint ix4, ix5;
        n0a = dotGridGradient(x0, y0, x, y, gradient);
        n1a = dotGridGradient(x1, y0, x, y, gradient);
        ix0 = interpolate(n0a, n1a, sx);
        n0b = dotGridGradient(x0, y1, x, y, gradient);
        n1b = dotGridGradient(x1, y1, x, y, gradient);
        ix1 = interpolate(n0b, n1b, sx);
        ix2 = interpolate(ix0, ix1, sy);
        ix3 = ((ix2 + 2.0) / 4.0) * 255.0;
        ix4 = (uint)floor(ix3);
        ix5 = min(max(0, ix4), 255);
        noise[idx] = ix5;
        //if (ix4 == 0) {
        //printf("x=%d y=%d x0=%d x1=%d y0=%d y1=%d sx=%.3f sy=%.3f n0a=%.3f n0b=%.3f n1a=%.3f n1b=%.3f ix0=%.3f ix1=%.3f ix2=%.3f ix3=%.3f ix4=%d ix5=%d\n", x, y, x0, x1, y0, y1, sx, sy, n0a, n0b, n1a, n1b, ix0, ix1, ix2, ix3, ix4, ix5);
        //}
        
    } // end if x00 < width && y00 < height
    __syncthreads();
}

int main()
{

    for (int i = 0; i < 10; i++) {
        std::cout << std::endl;
    }
    // Seed the random number generator on the CPU
    srand(time(NULL));

    // Create a 28x28 image
    cv::Mat image(28, 28, CV_8UC1);
    
    // Random angle grid
    int gridSpacing = 4; // 4 pixel spacing for grid lines makes 7 grids
    int gridCornersX = image.cols / gridSpacing + 1;
    int gridCornersY = image.rows / gridSpacing + 1;
    printf("gridCornersX=%d gridCornersY=%d\n", gridCornersX, gridCornersY);
    double h_gridRands[gridCornersY][gridCornersX];
    for (int i = 0; i < gridCornersX; i++) {
        for(int j = 0; j < gridCornersY; j++) {
            h_gridRands[j][i] = (double)(((double)rand() / (double)RAND_MAX) * (double)6.28);
        }
    }
    double* d_gridRands;
    cudaMalloc(&d_gridRands, gridCornersY * gridCornersX * sizeof(double));
    cudaMemcpy(d_gridRands, h_gridRands, gridCornersY * gridCornersX * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory on the GPU for the noise and gradient arrays
    int numVals = image.rows * image.cols;
    uint* h_noise;
    cudaMallocHost(&h_noise, numVals * sizeof(uint));
    uint* d_noise;
    cudaMalloc(&d_noise, numVals * sizeof(uint));
    cudaMemcpy(d_noise, h_noise, numVals * sizeof(uint), cudaMemcpyHostToDevice);

    // Launch the Perlin noise kernel
    int block_size = 32;
    dim3 blockSize(block_size, block_size);
    int numBlocks = (image.rows * image.cols + block_size - 1) / block_size;
    perlinKernel<<<numBlocks, blockSize>>>(image.cols, image.rows, d_noise, d_gridRands);
    cudaDeviceSynchronize();
    
    // Copy the noise values from the GPU to the image on the CPU
    std::cout << "before copy GPU to CPU" << std::endl;
    cudaMemcpy(h_noise, d_noise, numVals * sizeof(uint), cudaMemcpyDeviceToHost);
    std::cout << "after copy GPU to CPU" << std::endl;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            image.at<uchar>(y, x) = h_noise[y * image.cols + x];
            std::cout << h_noise[y * image.cols + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "after print grid of values" << std::endl;

    // Save the image using OpenCV
    cv::imwrite("perlin_noise.png", image);
    cv::namedWindow("image");
    cv::resize(image, image, cv::Size(), 10, 10);
    cv::imshow("image", image);
    cv::resizeWindow("image", 280, 280);
    cv::waitKey(0);

    // Free the GPU memory
    cudaFree(d_noise);
    cudaFree(d_gridRands);

    return 0;
}
