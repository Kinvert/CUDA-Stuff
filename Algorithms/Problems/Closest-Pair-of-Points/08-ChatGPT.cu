// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <cfloat>

using namespace std;

const int N = 100;

struct Point {
    double x, y;
} p[N];

__device__ double dist(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__global__ void closest_pair_kernel(Point *p, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx >= N || idy >= N) return;
    double d = dist(p[idx], p[idy]);
    if (d < *res) *res = d;
}

double closest_pair(Point *p) {
    double res = DBL_MAX;
    cudaMemcpy(p, p, sizeof(Point) * N, cudaMemcpyHostToDevice);
    closest_pair_kernel<<<(N + 31) / 32, 32, 32>>>(p, &res);
    cudaDeviceSynchronize();
    cudaMemcpy(&res, &res, sizeof(double), cudaMemcpyDeviceToHost);
    return res;
}

int main() {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        p[i].x = (rand() / double(RAND_MAX)) * 280;
        p[i].y = (rand() / double(RAND_MAX)) * 280;
    }
    double res = closest_pair(p);
    cout << res << endl;
    return 0;
}

