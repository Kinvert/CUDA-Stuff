// Written by ChatGPT Jan 30 Version

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

const int sobel_x[3][3] = {{-1, 0, 1}, 
                           {-2, 0, 2}, 
                           {-1, 0, 1}};

const int sobel_y[3][3] = {{-1, -2, -1}, 
                           {0, 0, 0}, 
                           {1, 2, 1}};

__global__ void sobel_kernel(const unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int gradient_x = 0, gradient_y = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int x_ = x + j;
            int y_ = y + i;
            if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height) continue;
            int pixel = input[y_ * width + x_];
            gradient_x += sobel_x[i + 1][j + 1] * pixel;
            gradient_y += sobel_y[i + 1][j + 1] * pixel;
        }
    }
    int gradient = (int) round(sqrt(gradient_x * gradient_x + gradient_y * gradient_y));
    output[y * width + x] = gradient;
}

void sobel(const cv::Mat &input, cv::Mat &output) {
    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, input.rows * input.cols * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, input.rows * input.cols * sizeof(unsigned char));

    // Copy data from host to device
    cudaMemcpy(d_input, input.data, input.rows * input.cols * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((input.cols + blockDim.x - 1) / blockDim.x, (input.rows + blockDim.y - 1) / blockDim.y);
    sobel_kernel<<<gridDim, blockDim>>>(d_input, d_output, input.cols, input.rows);
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(output.data, d_output, input.rows * input.cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) {
    cv::Mat input = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Failed to open image.jpg" << std::endl;
        return 1;
    }

    cv::Mat output(input.rows, input.cols, CV_8UC1);
    sobel(input, output);

    cv::imshow("Input", input);
    cv::imshow("Output", output);
    cv::imwrite("edges.jpg", output);

    cv::waitKey();
    return 0;
}

