// Written by ChatGPT Jan 30 Version

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 128

__device__ float2 circumcenter(float2 a, float2 b, float2 c) {
    float D = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    float Ux = (a.x*a.x + a.y*a.y) * (b.y - c.y) + (b.x*b.x + b.y*b.y) * (c.y - a.y) + (c.x*c.x + c.y*c.y) * (a.y - b.y);
    float Uy = (a.x*a.x + a.y*a.y) * (c.x - b.x) + (b.x*b.x + b.y*b.y) * (a.x - c.x) + (c.x*c.x + c.y*c.y) * (b.x - a.x);
    return make_float2(Ux / D, Uy / D);
}

__global__ void smallest_bounding_circle(float2 *points, int n, float2 *center, float *radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    
    float2 o = circumcenter(points[i], points[j], points[(i + 1) % n]);
    float r = sqrt((o.x - points[i].x) * (o.x - points[i].x) + (o.y - points[i].y) * (o.y - points[i].y));
    if (r > *radius) {
        *center = o;
        *radius = r;
    }
}

int main(int argc, char *argv[]) {
    vector<Point2f> points = {
        Point2f(1, 1),
        Point2f(3, 2),
        Point2f(2, 5),
        Point2f(5, 4),
        Point2f(4, 1)
    };
    int n = points.size();

    float2 *d_points;
    cudaMalloc((void**)&d_points, n * sizeof(float2));
    cudaMemcpy(d_points, points.data(), n * sizeof(float2), cudaMemcpyHostToDevice);
    
    // Define the parameters of the CUDA kernel call
    dim3 threadsPerBlock(512);
    dim3 blocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Call the CUDA kernel to perform the calculation
    circumcenter_kernel<<<blocks, threadsPerBlock>>>(d_points, n, &o);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(&o, &o, sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_points);

    // Draw the circle on the image
    circle(img, Point2f(o.x, o.y), r, Scalar(0, 255, 0), 2);

    // Display the image
    namedWindow("Bounding Circle", WINDOW_NORMAL);
    imshow("Bounding Circle", img);
    waitKey(0);

    return 0;
}

