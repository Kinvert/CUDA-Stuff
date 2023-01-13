// Written by ChatGPT Jan 9 Version
// It really didn't want to do CUDA
// After many attempts it just complained more and send OpenCV code instead.

#include <opencv2/opencv.hpp>

using namespace cv;

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
    vector<Point> hull;
    convexHull(points, hull, true);

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
// ... and it still failed
