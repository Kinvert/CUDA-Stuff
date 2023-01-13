// Written by ChatGPT Jan 9 Version

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__device__ void circumcenter(Point2f a, Point2f b, Point2f c, Point2f *o) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float ex = c.x - a.x;
    float ey = c.y - a.y;
    float bl = dx * dx + dy * dy;
    float cl = ex * ex + ey * ey;
    float d = 0.5f / (dx * ey - dy * ex);
    o->x = a.x + (ey * bl - dy * cl) * d;
    o->y = a.y + (dx * cl - ex * bl) * d;
}

__global__ void smallest_bounding_circle(Point2f *points, int n, Point2f *center, float *radius) {
    Point2f o;
    *radius = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            for (int k = j + 1; k < n; k++) {
                circumcenter(points[i], points[j], points[k], &o);
                cudaDeviceSynchronize();

                float r = sqrtf((o.x - points[i].x) * (o.x - points[i].x) + (o.y - points[i].y) * (o.y - points[i].y));
                bool contains_all = true;
                for (int l = 0; l < n; l++) {
                    float dist = sqrtf((o.x - points[l].x) * (o.x - points[l].x) + (o.y - points[l].y) * (o.y - points[l].y));
                    if (dist > r) {
                        contains_all = false;
                        break;
                    }
                }
                if (contains_all && (r > *radius || *radius == 0)) {
                    *center = o;
                    *radius = r;
                }
            }
        }
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
    cudaDeviceSynchronize();

    Point2f center;
    float radius;
    cudaMemcpy(&center, d_center, sizeof(Point2f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&radius, d_radius, sizeof(float), cudaMemcpyDeviceToHost);

    circle(img, center, radius, Scalar(0, 255, 0), 1);
    imshow("Bounding Circle", img);
    waitKey(0);
    imwrite("06-Result.png", img);

    cudaFree(d_points);
    cudaFree(d_center);
    cudaFree(d_radius);
    return 0;
}
