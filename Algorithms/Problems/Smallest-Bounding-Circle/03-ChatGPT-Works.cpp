// Written by ChatGPT Jan 9 Version

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace cv;

const float eps = 1e-5;

struct Circle {
    Point2f center;
    float radius;

    Circle(Point2f c = Point2f(0, 0), float r = 0) : center(c), radius(r) {}
    Circle(float x, float y, float r) : center(x, y), radius(r) {}

    bool contains(Point2f p) const {
        return norm(p - center) <= radius + eps;
    }
};

Point2f circumcenter(Point2f a, Point2f b, Point2f c) {
    float Bx = b.x, By = b.y, Cx = c.x, Cy = c.y, Ax = a.x, Ay = a.y;
    float D = 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By));
    float x = (Ax * Ax + Ay * Ay) * (By - Cy) + (Bx * Bx + By * By) * (Cy - Ay) + (Cx * Cx + Cy * Cy) * (Ay - By);
    float y = (Ax * Ax + Ay * Ay) * (Cx - Bx) + (Bx * Bx + By * By) * (Ax - Cx) + (Cx * Cx + Cy * Cy) * (Bx - Ax);
    return Point2f(x / D, y / D);
}

Circle min_bounding_circle(vector<Point2f> &points) {
    random_shuffle(points.begin(), points.end());
    Circle c(Point2f(0, 0), 0);
    for (int i = 0; i < points.size(); i++) {
        if (!c.contains(points[i])) {
            c = Circle(points[i], 0);
            for (int j = 0; j < i; j++) {
                if (!c.contains(points[j])) {
                    c = Circle((points[i] + points[j]) / 2, norm(points[i] - points[j]) / 2);
                    for (int k = 0; k < j; k++) {
                        if (!c.contains(points[k])) {
                            Point2f o = circumcenter(points[i], points[j], points[k]);
                            c = Circle(o, norm(o - points[k]));
                        }
                    }
                }
            }
        }
    }
    return c;
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

    Circle c = min_bounding_circle(points);
    circle(img, c.center, c.radius, Scalar(0, 255, 0), 1);
    imshow("Bounding Circle", img);
    imwrite("03-Result.png", img);
    waitKey(0);
    return 0;
}
