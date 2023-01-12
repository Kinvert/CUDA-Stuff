// Works for me: nvcc 09-Race-Condition.cu -o 9.out `pkg-config opencv4 --cflags --libs` && ./4.out
// Race Condition
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdio.h>

const int N = 100;
const int IMGSIZE = 280;

struct Point {
    double x = 0;
    double y = 0;
};

std::array<Point, N> h_p;
Point* d_p;
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<double> dist(0.0, 280.0);

__device__ double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__global__ void closestPairPoints(int* a, int* b, Point* const p, double* res) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x != y && x < N && y < N) {
        double thisDist = distance(p[x], p[y]);
        if (thisDist < *res) {
            *res = thisDist;
            *a = x;
            *b = y;
        }
    }
    __syncthreads();
}

int main() {
    srand(time(0));
    
    cudaMalloc((void**)&d_p, N * sizeof(Point));
    
    int* h_a = new int[1];
    int* h_b = new int[1];
    int* d_a;
    int* d_b;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    double* h_res = new double[1];
    double* d_res;
    cudaMalloc(&d_res, sizeof(double));
    
    *h_res = 65353.0;
    *h_a = 999;
    *h_b = 999;
    
    for (int i = 0; i < N; i++) {
        Point& p = h_p[i];
        p.x = dist(rng);
        p.y = dist(rng);
    }
    
    cudaMemcpy(d_p, h_p.data(), N * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 gridDim(2, 2, 1);
    dim3 blockDim(32, 32, 1);
    
    printf("BEFORE KERNEL\n");
    closestPairPoints<<<gridDim, blockDim>>>(d_a, d_b, d_p, d_res);
    cudaDeviceSynchronize();
    printf("Before Device to Host\n");
    cudaMemcpy(h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    
    cv::Mat image(IMGSIZE, IMGSIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < N; i++) {
        cv::Scalar color = cv::Scalar(0, 0, 0);
        if (i == *h_a || i == *h_b) {
            color = cv::Scalar(0, 255, 0);
        } else {
            color = cv::Scalar(255, 0, 0);
        }
        cv::circle(image, cv::Point(int(h_p[i].x), int(h_p[i].y)), 2, color, -1); // draw a red point
    }
    
    printf("Result = %.3f\n", *h_res);
    cv::namedWindow("Largest empty circle", cv::WINDOW_NORMAL);
    cv::imshow("Largest empty circle", image);
    cv::imwrite("09-Result.png", image);
    cv::waitKey(0); // wait for a key press to exit
    
    return 0;
}

