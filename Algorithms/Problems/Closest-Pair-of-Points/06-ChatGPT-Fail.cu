// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

const int N = 1e5 + 10;

struct Point {
    double x, y;

    bool operator < (const Point &p) const {
        return x < p.x;
    }
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
    int n;
    cin >> n;
    for (int i = 0; i < n; i++)
        cin >> p[i].x >> p[i].y;
    sort(p, p + n);
    double res = closest_pair(p);
    cout << res << endl;
    return 0;
}

