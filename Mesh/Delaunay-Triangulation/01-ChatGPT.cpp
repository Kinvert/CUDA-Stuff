// Written by ChatGPT Jan 9 Version

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    // Create an empty image with a black background
    Mat img(280, 280, CV_8UC3, Scalar(0, 0, 0));

    // Generate random points
    vector<Point2f> points;
    for (int i = 0; i < 20; i++)
    {
        Point2f p(rand() % 200 + 40, rand() % 200 + 40); // points only inside 200X200 central area
        points.push_back(p);
    }

    // Draw the points on the image
    for (int i = 0; i < points.size(); i++)
    {
        circle(img, points[i], 2, Scalar(255, 0, 0), -1);
    }

    // Perform Delaunay triangulation
    Subdiv2D subdiv(Rect(0, 0, 280, 280));
    for (int i = 0; i < points.size(); i++)
    {
        subdiv.insert(points[i]);
    }

    // Draw the Delaunay triangles
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    for (int i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        line(img, Point(t[0], t[1]), Point(t[2], t[3]), Scalar(255, 255, 255), 1);
        line(img, Point(t[2], t[3]), Point(t[4], t[5]), Scalar(255, 255, 255), 1);
        line(img, Point(t[4], t[5]), Point(t[0], t[1]), Scalar(255, 255, 255), 1);
    }

    // Show the image
    imshow("Delaunay", img);
    imwrite("01-Result.png", img);
    waitKey(0);

    return 0;
}

