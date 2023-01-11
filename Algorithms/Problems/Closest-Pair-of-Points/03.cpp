#include <cmath>
#include <ctime>
#include <stdio.h>

const int N = 100;

struct Point {
    double x, y;
} p[N];

double dist(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double closest_pair(Point *p, int n) {
    double res = 65535;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ( i != j ) {
                res = std::fmin(res, dist(p[i], p[j]));
            }
        }
    }
    return res;
}

int main() {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        p[i].x = (rand() / double(RAND_MAX)) * 280;
        p[i].y = (rand() / double(RAND_MAX)) * 280;
    }
    double res = closest_pair(p, N);
    printf("Result = %.3f\n", res);
    return 0;
}

