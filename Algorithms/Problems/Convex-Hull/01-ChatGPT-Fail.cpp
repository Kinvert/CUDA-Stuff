// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// A structure to represent a point in 2D
struct Point {
    int x, y;
};

// Utility function to find next to top in a stack
Point nextToTop(vector<Point> &S) {
    Point p = S.back();
    S.pop_back();
    Point res = S.back();
    S.push_back(p);
    return res;
}

// A utility function to swap two points
void swap(Point &p1, Point &p2) {
    Point temp = p1;
    p1 = p2;
    p2 = temp;
}

// A utility function to return square of distance
// between p1 and p2
int distSq(Point p1, Point p2) {
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 : Collinear points
// 1 : Clockwise points
// 2 : Counterclockwise
int orientation(Point p, Point q, Point r) {
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // collinear
    return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// A function used by library function qsort() to sort an array of
// points with respect to the first point
int compare(const void *vp1, const void *vp2) {
    Point *p1 = (Point *) vp1;
    Point *p2 = (Point *) vp2;

    // Find orientation
    int o = orientation(p0, *p1, *p2);
    if (o == 0)
        return (distSq(p0, *p2) >= distSq(p0, *p1)) ? -1 : 1;

    return (o == 2) ? -1 : 1;
}

// Prints convex hull of a set of n points.
vector<Point> convexHull(vector<Point> points) {
    int n = points.size();

    // Find the bottommost point
    int ymin = points[0].y, min = 0;
    for (int i = 1; i < n; i++) {
        int y = points[i].y;

        // Pick the bottom-most or chose the left
        // most point in case of tie
        if ((y < ymin) || (ymin == y && points[i].x < points[min].x))
            ymin = points[i].y, min = i;
    // Place the bottom-most point at first position
    swap(points[0], points[min]);

    // Sort n-1 points with respect to the first point.
    // A point p1 comes before p2 in sorted ouput if p2
    // has larger polar angle (in counterclockwise
    // direction) than p1
    p0 = points[0];
    sort(&points[1], &points[n], compare);

    // Create an empty stack and push first three points
    // to it.
    vector<Point> S;
    S.push_back(points[0]);
    S.push_back(points[1]);
    S.push_back(points[2]);

    // Process remaining n-3 points
    for (int i = 3; i < n; i++) {
        // Keep removing top while the angle formed by
        // points next-to-top, top, and points[i] makes
        // a non-left turn
        while (S.size() > 1 && orientation(nextToTop(S), S.back(), points[i]) != 2)
            S.pop_back();
        S.push_back(points[i]);
    }

    return S;
}

int main() {
    int width = 280;
    int height = 280;
    int num_points = 50;

    // Create a black image
    Mat image(height, width, CV_8UC3, Scalar(0, 0, 0));

    // Generate random points
    vector<Point> points;
    default_random_engine generator;
    uniform_int_distribution<int> x_distribution(40, 240);
    uniform_int_distribution<int> y_distribution(40, 240);
    for (int i = 0; i < num_points; i++) {
        Point p;
        p.x = x_distribution(generator);
        p.y = y_distribution(generator);
        points.push_back(p);
    }

    // Draw the points on the image
    for (int i = 0; i < num_points; i++) {
        Point p = points[i];
        circle(image, Point(p.x, p.y), 2, Scalar(255, 0, 0), 2);
    }

    // Compute the convex hull
    vector<Point> hull = convexHull(points);

    // Draw the convex hull on the image
    for (int i = 0; i < hull.size(); i++) {
        Point p1 = hull[i];
        Point p2 = hull[(i + 1) % hull.size()];
        line(image, Point(p1.x, p1.y), Point(p2.x, p2.y), Scalar(0, 255, 0), 2);
    }

    // Show the image
    imshow("Convex Hull", image);
    waitKey(0);

    return 0;
}

