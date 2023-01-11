// Works for me: g++ 04.cpp -o 4.out `pkg-config opencv4 --cflags --libs` && ./4.out
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <stdio.h>

const int N = 100;
const int IMGSIZE = 280;

struct Point {
    double x, y;
} p[N];

double dist(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

int main() {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        p[i].x = (rand() / double(RAND_MAX)) * IMGSIZE;
        p[i].y = (rand() / double(RAND_MAX)) * IMGSIZE;
    }
    
    double res = 65535;
    int idxA = 999;
    int idxB = 999;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if ( i != j ) {
                double thisDist = dist(p[i], p[j]);
                if (thisDist < res) {
                    res = thisDist;
                    idxA = i;
                    idxB = j;
                }
            }
        }
    }
    
    cv::Mat image(IMGSIZE, IMGSIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < N; i++) {
        cv::Scalar color = cv::Scalar(0, 0, 0);
        if (i == idxA || i == idxB) {
            color = cv::Scalar(0, 255, 0);
        } else {
            color = cv::Scalar(255, 0, 0);
        }
        cv::circle(image, cv::Point(p[i].x, p[i].y), 2, color, -1); // draw a red point
    }
    
    printf("Result = %.3f\n", res);
    cv::namedWindow("Largest empty circle", cv::WINDOW_NORMAL);
    cv::imshow("Largest empty circle", image);
    cv::waitKey(0); // wait for a key press to exit
    return 0;
}

