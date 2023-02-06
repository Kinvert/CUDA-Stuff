// Written by ChatGPT Jan 30 Version

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_math.h>

#define WIDTH input.cols
#define HEIGHT input.rows
#define BLOCK_SIZE 16

__global__ void sobel_kernel(uchar *input, uchar *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = x + y * width;
        int gx = input[index - width - 1] + 2 * input[index - 1] + input[index + width - 1] - input[index - width + 1] - 2 * input[index + 1] - input[index + width + 1];
        int gy = input[index - width - 1] + 2 * input[index - width] + input[index - width + 1] - input[index + width - 1] - 2 * input[index + width] - input[index + width + 1];
        output[index] = sqrt(gx * gx + gy * gy);
    }
}

int main() {
    cv::Mat input = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat output = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);

    uchar *input_d, *output_d;
    cudaMalloc((void**)&input_d, WIDTH * HEIGHT * sizeof(uchar));
    cudaMalloc((void**)&output_d, WIDTH * HEIGHT * sizeof(uchar));

    cudaMemcpy(input_d, input.ptr(), WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sobel_kernel<<<grid, block>>>(input_d, output_d, WIDTH, HEIGHT);

    cudaMemcpy(output.ptr(), output_d, WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);

//    cv::imshow("Input", input);
    cv::imshow("Output", output);
    cv::waitKey();
    cv::imwrite("edges.jpg", output);

    return 0;
}
