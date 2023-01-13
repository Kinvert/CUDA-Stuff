// Written by ChatGPT Jan 9 Version

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace cv;

void smallest_bounding_circle(const vector<Point2f> &points, Point2f &center, float &radius) {
    center = Point2f(0, 0);
    radius = 0;
    for (int i = 0; i < points.size(); i++) {
        center += points[i];
    }
    center *= (1. / points.size());

    for (int i = 0; i < points.size(); i++) {
        float dist = norm(points[i] - center);
        radius = max(radius, dist);
    }
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

    Point2f center;
    float radius;
    smallest_bounding_circle(points, center, radius);

    circle(img, center, radius, Scalar(0, 255, 0), 2);
    imshow("Bounding Circle", img);
    imwrite("02-Result.png", img);
    waitKey(0);
    return 0;
}

