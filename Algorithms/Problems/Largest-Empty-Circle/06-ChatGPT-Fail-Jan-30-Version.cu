// Written by ChatGPT Jan 30 Version

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <random>

using namespace std;
using namespace cv;

// define the number of points
#define N 1000

// define the size of the image
#define size 280

// define the maximum radius
#define R 200

// define the minimum radius
#define r_min 10

// define a struct to store a point
struct Point {
    int x, y;
};

// device function to calculate the distance between two points
__device__ double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// kernel function to calculate the largest empty circle
__global__ void largest_empty_circle(Point *points, int N, double *r, Point *c) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (i == j) return; // skip same points

    // calculate the midpoint of the two points
    Point mid;
    mid.x = (points[i].x + points[j].x) / 2;
    mid.y = (points[i].y + points[j].y) / 2;

    // calculate the distance between the two points
    double d = distance(points[i], points[j]);
    double r_tmp = d / 2;

    // check if the radius is too big
    if (r_tmp > R) return;

    // check if all other points are outside the circle
    bool valid = true;
    for (int k = 0; k < N; k++) {
        if (k == i || k == j) continue; // skip the two points used to calculate the circle
        if (distance(points[k], mid) < r_tmp) {
            valid = false;
            break;
        }
    }

    // update the result if a larger empty circle is found
    if (valid && r_tmp > *r) {
        *r = r_tmp;
        *c = mid;
    }
}

int main() {
    // generate random points
    Point points[N];
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, size);
    for (int i = 0; i < N; i++) {
        points[i].x = distribution(generator);
        points[i].y = distribution(generator);
    }

    // allocate memory on the device
    Point *d_points;
    cudaMalloc((void**)&d_points, N * sizeof(Point));
    double *d_r;
    cudaMalloc((void**)&d_r, sizeof(double));
    Point *d_c;
    cudaMalloc((void**)&d_c, sizeof(Point));

    // copy the data to the device
    cudaMemcpy(d_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
    double r = r_min;
    cudaMemcpy(d_r, &r, sizeof(double), cudaMemcpyHostToDevice);
    Point c = {0, 0};
    cudaMemcpy(d_c, &c, sizeof(Point), cudaMemcpyHostToDevice);

    // launch the kernel
    int blocks = N;
    int threads = N;
    largest_empty_circle<<<blocks, threads>>>(d_points, N, d_r, d_c);

    // copy the result back to the host
    cudaMemcpy(&r, d_r, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, d_c, sizeof(Point), cudaMemcpyDeviceToHost);

    // free the device memory
    cudaFree(d_points);
    cudaFree(d_r);
    cudaFree(d_c);

    // create an image
    Mat image = Mat::zeros(size, size, CV_8UC3);

    // draw the largest empty circle
    circle(image, Point(c.x, c.y), r, Scalar(0, 255, 0), 2);

    // show the image
    imshow("largest empty circle", image);
    waitKey(0);

    return 0;
}
