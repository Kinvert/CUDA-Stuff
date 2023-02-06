// Written by ChatGPT Jan 30 Version

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

const int sobel_x[3][3] = {{-1, 0, 1}, 
                           {-2, 0, 2}, 
                           {-1, 0, 1}};

const int sobel_y[3][3] = {{-1, -2, -1}, 
                           {0, 0, 0}, 
                           {1, 2, 1}};

void sobel(const cv::Mat &input, cv::Mat &output) {
    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int gradient_x = 0, gradient_y = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = input.at<uchar>(y + i, x + j);
                    gradient_x += sobel_x[i + 1][j + 1] * pixel;
                    gradient_y += sobel_y[i + 1][j + 1] * pixel;
                }
            }
            int gradient = (int) round(sqrt(gradient_x * gradient_x + gradient_y * gradient_y));
            output.at<uchar>(y, x) = gradient;
        }
    }
}

int main(int argc, char** argv) {
    // Load image.jpg using OpenCV
    cv::Mat input = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cout << "Failed to open image.jpg" << std::endl;
        return 1;
    }

    // Apply Sobel operator
    cv::Mat output = cv::Mat(input.size(), CV_8UC1);
    sobel(input, output);

    // Display the result using OpenCV imshow
    cv::imshow("Edges", output);
    cv::waitKey(0);

    // Save the result as edges.jpg using OpenCV imwrite
    cv::imwrite("01-result.jpg", output);

    return 0;
}
