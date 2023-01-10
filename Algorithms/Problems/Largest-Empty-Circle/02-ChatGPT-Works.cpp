// Written by ChatGPT Dec 15 Version

#include <iostream>
#include <cmath>

using namespace std;

const int N = 100; // number of points

struct Point {
    double x, y;
};

// function to check if a point lies inside a circle
bool isInside(Point p, Point c, double r)
{
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    double d = sqrt(dx * dx + dy * dy);
    return d <= r;
}

int main()
{
    srand(time(0));
    Point points[N]; // points on the plane

    // generate random points
    for (int i = 0; i < N; i++) {
        points[i].x = rand() % 100;
        points[i].y = rand() % 100;
    }

    double r = -1; // radius of the largest empty circle
    Point c; // center of the largest empty circle

    // brute force algorithm:
    // try all possible centers c and all possible radii r
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue; // skip same points

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

            if (valid && rr > r) { // update radius and center if valid and larger than current maximum
                r = rr;
                c = cc;
            }
        }
    }

    cout << "Largest empty circle with radius " << r << " and center (" << c.x << ", " << c.y << ")" << endl;

    return 0;
}
