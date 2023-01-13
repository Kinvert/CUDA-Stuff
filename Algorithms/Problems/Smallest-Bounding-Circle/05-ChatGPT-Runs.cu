// Written by ChatGPT Jan 9 Version

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__global__ void smallest_bounding_circle(Point2f *points, int n, Point2f *center, float *radius) {
    float sumX = 0;
    float sumY = 0;
    for (int i = 0; i < n; i++) {
        sumX += points[i].x;
        sumY += points[i].y;
    }
    sumX /= n;
    sumY /= n;
    center->x = sumX;
    center->y = sumY;
    *radius = 0;
    for (int i = 0; i < n; i++) {
        float dist = sqrtf((points[i].x - sumX) * (points[i].x - sumX) + (points[i].y - sumY) * (points[i].y - sumY));
        *radius = max(*radius, dist);
    }
}

int main() {
    Mat img(280, 280, CV_8UC3, Scalar(0, 0, 0));
    vector<Point2f> points;
    RNG rng(12345);
    for (int i = 0; i < 50; i++) {
        Point2f point(rng.uniform(40, 240), rng.uniform(40, 240));
        points.push_back(point);
        circle(img, point, 2, Scalar(255, 0, 0), -1);
    }

    Point2f *d_points;
    Point2f *d_center;
    float *d_radius;
    cudaMalloc((void **)&d_points, points.size() * sizeof(Point2f));
    cudaMalloc((void **)&d_center, sizeof(Point2f));
    cudaMalloc((void **)&d_radius, sizeof(float));
    cudaMemcpy(d_points, &points[0], points.size() * sizeof(Point2f), cudaMemcpyHostToDevice);

    smallest_bounding_circle<<<1, 1>>>(d_points, points.size(), d_center, d_radius);

    Point2f center;
    float radius;
    cudaMemcpy(&center, d_center, sizeof(Point2f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&radius, d_radius, sizeof(float), cudaMemcpyDeviceToHost);

    circle(img, center, radius, Scalar(0, 255, 0), 2);
    imshow("Bounding Circle", img);
    waitKey(0);
    imwrite("05-Result.png", img);

    cudaFree(d_points);
    cudaFree(d_center);
    cudaFree(d_radius);
    return 0;
}
