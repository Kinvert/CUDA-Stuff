// Doesn't work yet
// Curand version, might model this after 06 which is mostly working now.
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Generates a random float between 0 and 1
__device__ double randomFloat(curandState_t* states, int id)
{
    // Generate a random float between 0 and 1 using curand
    return curand_uniform(&states[id]);
}

__global__ void setupKernel(curandState_t* states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //curand_init(id, 0, 0, &states[id]);
    curand_init(clock64(), id, 0, &states[id]);
}

// Interpolates between a and b by t on the GPU
__device__ double interpolate(double a, double b, double t)
{
    return a * (1 - t) + b * t;
}

// Returns the dot product of the distance and gradient vectors on the GPU
__device__ double dotGridGradient(int ix, int iy, double x, double y, curandState_t* states, int id)
{
    // Precomputed (or otherwise) gradient vectors at each grid node
    double random1 = randomFloat(states, id);
    double random2 = randomFloat(states, id + 1);
    double random3 = randomFloat(states, id);
    double random4 = randomFloat(states, id);
    double gradient[2][2] = {{random1, random2}, {random3, random4}};
    
    double ang = randomFloat(states, id) * 3.14f;
    
    // Distance vectors
    double dx = x - (double)ix;
    double dy = y - (double)iy;
    double result = dx * gradient[iy][ix] + dy * gradient[iy][ix + 1];
    //float result = dx * cos(ang) + dy * sin(ang);
    //float result = dx * randomFloat(states, id) + dy * randomFloat(states, id);
    
    //printf("1=%.3f 1=%.3f 1=%.3f 1=%.3f\n", random1, random2, random3, random4);
    
    //printf("ix=%d grad=%f result=%.3f\n", ix, gradient[iy][ix+1], result);
    
    //printf("ix=%d iy=%d x=%.3f y=%.3f dx=%.3f dy=%.3f grad1=%.3f grad2=%.3f RESULT1=%.3f\n", ix, iy, x, y, dx, dy, gradient[iy][ix], gradient[iy][ix+1], result);

    return (result);
}

// Generates a random float between 0 and 1 on the GPU
__device__ double randomFloat(curandState_t* state)
{
    return curand_uniform(state);
}

// Computes Perlin noise at coordinates x, y on the GPU
__global__ void perlinKernel(int width, int height, double* noise, curandState_t* states)
{
    int x00 = blockIdx.x * blockDim.x + threadIdx.x;
    int y00 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x00 < width && y00 < height) {
    
        int idx = y00 * blockDim.x + threadIdx.x;
        
        curandState_t state = states[idx];

        double x = (double)((double)x00 / (double)width);
        double y = (double)((double)y00 / (double)height);

        // Determine grid cell coordinates
        //int x0 = (int)x;
        //int x1 = x0 + 1;
        //int y0 = (int)y;
        //int y1 = y0 + 1;
        //int x0 = 0;
        //int x1 = 1;
        //int y0 = 0;
        //int y1 = 1;
        //int x0 = x00;
        //int x1 = x00 + 1;
        //int y0 = y00;
        //int y1 = y00 + 1;
        int x0 = (int)floor((double)((x00 / 2) * 2));
        int x1 = x0 + 2;
        int y0 = (int)floor((double)((y00 / 2) * 2));
        int y1 = y0 + 2;

        // Determine interpolation weights
        // Could also use higher order polynomial/s-curve here
        double sx = (x - (double)x0) * 2.0;
        double sy = (y - (double)y0) * 2.0;

        // Interpolate between grid point gradients
        double n0a, n0b, n1a, n1b, ix0, ix1;
        n0a = dotGridGradient(x0, y0, x, y, states, idx);
        n1a = dotGridGradient(x1, y0, x, y, states, idx);
        ix0 = interpolate(n0a, n1a, sx);
        n0b = dotGridGradient(x0, y1, x, y, states, idx);
        n1b = dotGridGradient(x1, y1, x, y, states, idx);
        ix1 = interpolate(n0b, n1b, sx);
        *noise = interpolate(ix0, ix1, sy);
        
        //printf("x=%.3f y=%.3f x0=%d x1=%d y0=%d y1=%d n0a=%.3f n0b=%.3f n1a=%.3f n1b=%.3f\n", x, y, x0, x1, y0, y1, n0a, n0b, n1a, n1b);
        
    } // end if x00 < width && y00 < height
    __syncthreads();
}

int main()
{
    // Seed the random number generator on the CPU
    srand(time(NULL));

    // Create a 28x28 image
    cv::Mat image(28, 28, CV_8UC1);
    
    // Random angle grid
    int gridSpacing = 4; // 4 pixel spacing for grid lines makes 7 grids
    int gridCornersX = image.cols / gridSpacing + 1;
    int gridCornersY = image.rows / gridSpacing + 1;
    double h_gridRands[gridCornersY][gridCornersX];
    for (int i = 0; i < gridCornersX; i++) {
        for(int j = 0; j < gridCornersY; j++) {
            h_gridRands[j][i] = (double)(((double)rand() / (double)RAND_MAX) * 6.28);
        }
    }
    double* d_gridRands;
    cudaMalloc(&d_gridRands, gridCornersY * gridCornersX * sizeof(double));
    cudaMemcpy(d_gridRands, h_gridRands, gridCornersY * gridCornersX * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory on the GPU for the noise and gradient arrays
    double* d_noise;
    cudaMalloc(&d_noise, image.rows * image.cols * sizeof(double));
    cudaMemcpy(d_noise, image.data, image.rows * image.cols * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the Perlin noise kernel
    int block_size = 32;
    dim3 blockSize(block_size, block_size);
    int numBlocks = (image.rows * image.cols + block_size - 1) / block_size;
    
    curandState_t* d_states;
    int numThreads = block_size * block_size;
    cudaMalloc(&d_states, block_size * block_size * sizeof(curandState_t));
    setupKernel<<<1, numThreads>>>(d_states);
    cudaDeviceSynchronize();
    
    perlinKernel<<<numBlocks, blockSize>>>(image.cols, image.rows, d_noise, d_states);
    cudaDeviceSynchronize();
    
    // Copy the noise values from the GPU to the image on the CPU
    cudaMemcpy(image.data, d_noise, image.rows * image.cols * sizeof(double), cudaMemcpyDeviceToHost);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            std::cout << (int)image.at<uchar>(y, x) << " ";
        }
        std::cout << std::endl;
    }

    // Copy the noise values from the GPU to the image on the CPU
    cudaMemcpy(image.data, d_noise, image.rows * image.cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Save the image using OpenCV
    cv::imwrite("perlin_noise.png", image);

    // Free the GPU memory
    cudaFree(d_noise);
    cudaFree(d_states);

    return 0;
}
