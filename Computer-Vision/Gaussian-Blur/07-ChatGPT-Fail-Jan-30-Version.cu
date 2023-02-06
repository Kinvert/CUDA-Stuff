// Written by ChatGPT Jan 30 Version

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define filterSize 5

__global__ void gaussianBlur(uchar4* input, uchar4* output, int width, int height, float* filter) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int filterRadius = filterSize / 2;

  if (x < width && y < height) {
    float4 sum = make_float4(0, 0, 0, 0);
    int filterIndex = 0;

    for (int i = -filterRadius; i <= filterRadius; i++) {
      for (int j = -filterRadius; j <= filterRadius; j++) {
        int imageX = x + j;
        int imageY = y + i;

        if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
          uchar4 pixel = input[imageY * width + imageX];
          sum.x += pixel.x * filter[filterIndex];
          sum.y += pixel.y * filter[filterIndex];
          sum.z += pixel.z * filter[filterIndex];
          filterIndex++;
        }
      }
    }

    output[y * width + x] = make_uchar4(sum.x, sum.y, sum.z, 255);
  }
}

int main(int argc, char* argv[]) {
  cv::Mat image = cv::imread("image.jpg");

  if (!image.data) {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  int width = image.cols;
  int height = image.rows;

  // Allocate memory on the GPU for the input and output images
  uchar4* d_input;
  cudaMalloc((void**) &d_input, width * height * sizeof(uchar4));

  uchar4* d_output;
  cudaMalloc((void**) &d_output, width * height * sizeof(uchar4));

  // Copy the input image from host to GPU memory
  cudaMemcpy(d_input, h_input, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);

  // Invoke the CUDA kernel function to perform the Gaussian blur
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  gaussianBlur<<<blocksPerGrid, threadsPerBlock>>>(d_input, width, height, d_output, filter, filterRadius, filterSize);

  // Copy the output image from GPU memory to host memory
  cudaMemcpy(h_output, d_output, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

  // Deallocate GPU memory
  cudaFree(d_input);
  cudaFree(d_output);

  // Create an OpenCV Mat structure to hold the output image
  Mat outputImage(height, width, CV_8UC4, h_output);

  // Display the output image
  imshow("Gaussian Blur", outputImage);

  // Wait for a key press
  waitKey(0);

  // Save the output image to file
  imwrite("output.jpg", outputImage);

  // Deallocate host memory
  delete[] h_input;
  delete[] h_output;

  return 0;
}
