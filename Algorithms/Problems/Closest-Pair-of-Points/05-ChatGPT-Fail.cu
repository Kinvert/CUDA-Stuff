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
    double d = *res;
    for (int i = 0; i < N; i++) {
        if (idx == i) continue;
        d = min(d, dist(p[idx], p[i]));
    }
    atomicMin(res, d);
}

double closest_pair(Point *p) {
    double res;
    cudaMemcpyToSymbol(p, p, sizeof(Point) * N);
    cudaMemcpyToSymbol(&res, &res, sizeof(double));
    closest_pair_kernel<<<(N + 255) / 256, 256>>>(p, &res);
    cudaDeviceSynchronize();
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

