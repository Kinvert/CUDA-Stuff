// Written by ChatGPT Jan 9 Version

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

#define MAX_NUM_POINTS 20

__global__ void delaunay_kernel(int width, int height, Point2f* points, int numPoints, int* triangles)
{
    // Perform Delaunay triangulation on the GPU
    // This is just an example, you will need to implement the actual triangulation algorithm on the GPU
    __shared__ Point2f sharedPoints[MAX_NUM_POINTS];
    __shared__ int sharedTriangles[MAX_NUM_POINTS * 3];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numPoints) {
        sharedPoints[tid] = points[tid];
    }
    __syncthreads();
    // ... perform triangulation algorithm here ...
    __syncthreads();

    // Store the triangles in the output array
    // This is just an example, you will need to implement the correct indexing for the triangles array
    if (tid < numPoints * 3) {
        triangles[tid] = sharedTriangles[tid];
    }
}

int main()
{
    // Create an empty image with a black background
    Mat img(280, 280, CV_8UC3, Scalar(0, 0, 0));

    // Generate random points
    vector<Point2f> points;
    for (int i = 0; i < 20; i++)
    {
        Point2f p(rand() % 200 + 40, rand() % 200 + 40); // points only inside 200X200 central area
        points.push_back(p);
    }

    // Copy the points to the GPU
    Point2f* d_points;
    cudaMalloc(&d_points, points.size() * sizeof(Point2f));
    cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point2f), cudaMemcpyHostToDevice);

    // Allocate memory for the triangles on the GPU
    int* d_triangles;
    cudaMalloc(&d_triangles, points.size() * 3 * sizeof(int));

    // Launch the CUDA kernel
    delaunay_kernel<<<1, 1>>>(280, 280, d_points, points.size(), d_triangles);

    // Copy the triangles from the GPU to the host
    int* h_triangles = new int[points.size() * 3];
    cudaMemcpy(h_triangles, d_triangles, points.size() * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Draw the points on the image
    for (int i = 0; i < points.size(); i++)
    {
        circle(img, points[i], 2, Scalar(255, 0, 0), -1);
    }

    // Draw the Delaunay triangles
    for (int i = 0; i < points.size()*3; i+=3)
    {
        line(img, points[h_triangles[i]], points[h_triangles[i+1]], Scalar(255, 255, 255), 1);
        line(img, points[h_triangles[i+1]], points[h_triangles[i+2]], Scalar(255, 255, 255), 1);
        line(img, points[h_triangles[i+2]], points[h_triangles[i]], Scalar(255, 255, 255), 1);
    }

    // Show the image
    imshow("Delaunay", img);
    waitKey(0);

    // Free memory
    cudaFree(d_points);
    cudaFree(d_triangles);

    return 0;
}
