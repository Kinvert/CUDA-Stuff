// Written by ChatGPT Jan 30 Version

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

// CUDA kernel function to perform the Gaussian blur
__global__ void GaussianBlurKernel(float* input, float* output, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int filterSize = ceil(6 * sigma) | 1;
        int filterRadius = filterSize / 2;

        // Calculate the filter values
        for (int yf = 0; yf < filterSize; ++yf) {
            for (int xf = 0; xf < filterSize; ++xf) {
                int xDistance = xf - filterRadius;
                int yDistance = yf - filterRadius;
                float value = exp(-(xDistance * xDistance + yDistance * yDistance) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
                int filterIndex = yf * filterSize + xf;
                sum += value;
            }
        }

        // Normalize the filter values
        for (int i = 0; i < filterSize * filterSize; ++i) {
            filter[i] /= sum;
        }

        // Perform the convolution with the filter
        for (int yf = -filterRadius; yf <= filterRadius; ++yf) {
            for (int xf = -filterRadius; xf <= filterRadius; ++xf) {
                int x_ = x + xf;
                int y_ = y + yf;

                if (x_ >= 0 && x_ < width && y_ >= 0 && y_ < height) {
                    int filterIndex = (yf + filterRadius) * filterSize + (xf + filterRadius);
                    output[y * width + x] += input[y_ * width + x_] * filter[filterIndex];
                }
            }
        }
    }
}

int main() {
  // Load image using OpenCV
  cv::Mat image = cv::imread("image.jpg", cv::IMREAD_COLOR);
  if (!image.data) {
    std::cerr << "Error: Could not open or find the image" << std::endl;
    return -1;
  }

  int width = image.cols;
  int height = image.rows;
  int channel_count = image.channels();
  int image_size = width * height * channel_count;

  // Allocate memory on the GPU for input and output image data
  unsigned char *d_input, *d_output;
  cudaMalloc((void **)&d_input, image_size * sizeof(unsigned char));
  cudaMalloc((void **)&d_output, image_size * sizeof(unsigned char));

  // Copy input image data to GPU
  cudaMemcpy(d_input, image.data, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Launch CUDA kernel to perform Gaussian blur
  int block_size = 32;
  int grid_size = (width * height * channel_count + block_size - 1) / block_size;
  gaussianBlur<<<grid_size, block_size>>>(d_input, d_output, width, height, channel_count);

  // Copy output image data back to host
  cudaMemcpy(image.data, d_output, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_input);
  cudaFree(d_output);

  // Display and save the output image
  cv::imshow("Blurred Image", image);
  cv::imwrite("blurred_image.jpg", image);

  cv::waitKey(0);

  return 0;
}
