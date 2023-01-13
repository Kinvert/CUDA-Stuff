// Written by ChatGPT Jan 9 Version
// Note - It really didn't want to finish the code and was in a rather annoying mood today.

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// A structure to represent a point in 2D
struct Point {
    int x, y;
};

__global__ void nextToTop(Point *S, Point *result) {
    Point p = S[blockDim.x*blockIdx.x + threadIdx.x];
    result[blockDim.x*blockIdx.x + threadIdx.x] = S[blockDim.x*blockIdx.x + threadIdx.x-1];
    S[blockDim.x*blockIdx.x + threadIdx.x-1] = p;
}

__global__ void swap(Point *p1, Point *p2) {
    Point temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

__global__ void distSq(Point p1, Point p2, int *result) {
    result[blockDim.x*blockIdx.x + threadIdx.x] = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

__global__ void orientation(Point p, Point q, Point r, int *result) {
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) {
        result[blockDim.x*blockIdx.x + threadIdx.x] = 0;
    } else {
        result[blockDim.x*blockIdx.x + threadIdx.x] = (val > 0) ? 1 : 2;
    }
}

__global__ void compare(Point p0, Point *p1, Point *p2, int *result) {
    int o = orientation(p0, *p1, *p2);
    if (o == 0) {
        int dist1 = distSq(p0, *p1);
        int dist2 = distSq(p0, *p2);
        result[blockDim.x*blockIdx.x + threadIdx.x] = (dist2 >= dist1) ? -1 : 1;
    } else {
        result[blockDim.x*blockIdx.x + threadIdx.x] = (o == 2) ? -1 : 1;
    }
}

vector<Point> convexHull(vector<Point> points) {
    int n = points.size();

    // Find the bottommost point
    int ymin = points[0].y, min = 0;
    for (int i = 1; i < n; i++) {
        int y = points[i].y;

