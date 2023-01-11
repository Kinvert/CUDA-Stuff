// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

const int N = 1e2 + 10;

struct Point {
    double x, y;

    bool operator < (const Point &p) const {
        return x < p.x;
    }
} p[N];

double dist(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double closest_pair(int l, int r) {
    if (r - l <= 1) return 1e20;
    int mid = (l + r) / 2;
    double d = min(closest_pair(l, mid), closest_pair(mid, r));
    inplace_merge(p + l, p + mid, p + r);
    static Point t[N];
    int cnt = 0;
    for (int i = l; i < r; i++)
        if (abs(p[i].x - p[mid].x) < d)
            t[cnt++] = p[i];
    double res = d;
    for (int i = 0; i < cnt; i++)
        for (int j = i + 1; j < cnt && t[j].y - t[i].y < d; j++)
            res = min(res, dist(t[i], t[j]));
    return res;
}

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++)
        cin >> p[i].x >> p[i].y;
    sort(p, p + n);
    cout << closest_pair(0, n) << endl;
    return 0;
}

