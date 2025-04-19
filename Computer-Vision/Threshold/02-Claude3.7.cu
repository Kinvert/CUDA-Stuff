#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for thresholding
__global__ void thresholdKernel(unsigned char* input, unsigned char* output, int width, int height, int threshold) {
    // Calculate current thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if thread is within image bounds
    if (x < width && y < height) {
        // Calculate 1D index from 2D position
        int idx = y * width + x;
        
        // Apply threshold
        if (input[idx] >= threshold) {
            output[idx] = 255; // White
        } else {
            output[idx] = 0;   // Black
        }
    }
}

int main(int argc, char** argv) {
    // Check if threshold value is provided
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <threshold_value> <output_filename>" << std::endl;
        return -1;
    }

    // Parse threshold value
    int threshold = std::atoi(argv[1]);
    
    // Check if threshold is valid
    if (threshold < 0 || threshold > 255) {
        std::cout << "Threshold value must be between 0 and 255" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat image = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    
    // Check if image is loaded successfully
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Image dimensions
    int width = image.cols;
    int height = image.rows;
    int size = width * height * sizeof(unsigned char);

    // Host and device pointers
    unsigned char *h_input, *h_output, *d_input, *d_output;
    
    // Allocate host memory
    h_input = image.data;
    h_output = (unsigned char*)malloc(size);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    
    // Copy input image to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    thresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, threshold);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Create output image
    cv::Mat output_image(height, width, CV_8UC1, h_output);
    
    // Save the output image
    cv::imwrite(argv[2], output_image);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Free host memory (don't free h_input as it's managed by OpenCV)
    free(h_output);
    
    std::cout << "CUDA thresholding complete. Output saved to " << argv[2] << std::endl;
    
    return 0;
}
