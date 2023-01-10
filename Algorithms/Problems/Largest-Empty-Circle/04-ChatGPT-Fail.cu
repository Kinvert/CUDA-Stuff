// Written by ChatGPT Dec 15 Version

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int N = 100; // number of points
const int size = 280; // size of the image

struct Point {
    double x, y;
};

// function to check if a point lies inside a circle
__device__ bool isInside(Point p, Point c, double r)
{
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    double d = sqrt(dx * dx + dy * dy);
    return d <= r;
}

int main()
{
    Point points[N]; // points on the plane

    // generate random points
    for (int i = 0; i < N; i++) {
        points[i].x = rand() % size;
        points[i].y = rand() % size;
    }

    double r = -1; // radius of the largest empty circle
    Point c; // center of the largest empty circle

    // allocate memory on the GPU
    Point *dev_points;
    cudaMalloc((void **)&dev_points, N * sizeof(Point));
    cudaMemcpy(dev_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);

    // launch a kernel to find the largest empty circle
    dim3 blocks(N, N);
    dim3 threads(1, 1);
    cudaLaunchKernel(findLargestEmptyCircle, blocks, threads, 0, 0, dev_points, r, c);

    // copy the result back to the host
    cudaMemcpy(&r, &r, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, &c, sizeof(Point), cudaMemcpyDeviceToHost);

    // create an image and draw the points and circle
    Mat image(size, size, CV_8UC3, Scalar(0, 0, 0)); // create a black image
    for (int i = 0; i < N; i++) {
        circle(image, Point(points[i].x, points[i].y), 3, Scalar(0, 0, 255), -1); // draw a red point
    }
    circle(image, Point(c.x, c.y), r, Scalar(0, 255, 0), 2); // draw a green circle

    // show the image
    namedWindow("Largest empty circle", WINDOW_NORMAL);
    imshow("Largest empty circle", image);

    waitKey(0); // wait for a key press to exit

    return 0;
}

__global__ void findLargestEmptyCircle(Point *points, double r, Point c)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i == j) return; // skip same points

    double dx = points[i].x - points[j].x;
    double dy = points[i].y - points[j].y;
    double rr = sqrt(dx * dx + dy * dy) / 2; // try radius as half the distance between i and j
    Point cc = {(points[i].x + points[j].x) / 2, (points[i].y + points[j].y) / 2}; // try center as midpoint of i and j

    // check if all other points are outside the circle
    bool valid = true;
    for (int k = 0; k < N; k++) {
        if (k == i || k == j) continue; // skip points i and j
        if (isInside(points[k], cc, rr)) { // check if point k is inside the circle
            valid = false;
            break;
        }
    }

    // update radius and center if valid and larger than current maximum
    if (valid && rr > r) {
        r = rr;
        c = cc;
    }
}

