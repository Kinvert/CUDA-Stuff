// Written by ChatGPT Jan 30 Version

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

using namespace cv;
using namespace std;

Mat image1; // original image
Mat result; // result image

// Function to perform Wave Function Collapse
void waveFunctionCollapse(Mat& src, Mat& dst, int size) {
  int width = src.cols;
  int height = src.rows;

  // Create a vector to store the possible values for each cell
  vector<vector<int>> possibilities(width * height);

  // Initialize the possibilities for each cell
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int currIndex = i * width + j;
      // Check the surrounding cells to determine the possible values
      for (int ii = max(0, i - size); ii < min(height, i + size + 1); ii++) {
        for (int jj = max(0, j - size); jj < min(width, j + size + 1); jj++) {
          int index = ii * width + jj;
          if (index == currIndex) continue;
          possibilities[currIndex].push_back(src.at<uchar>(ii, jj));
        }
      }
    }
  }

  // Randomly select values for each cell
  default_random_engine generator;
  dst = Mat::zeros(height, width, CV_8UC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int currIndex = i * width + j;
      uniform_int_distribution<int> distribution(0, possibilities[currIndex].size() - 1);
      int index = distribution(generator);
      dst.at<uchar>(i, j) = possibilities[currIndex][index];
    }
  }
}

int main() {
  // Load the original image
  image1 = imread("imageB.jpg", IMREAD_GRAYSCALE);

  // Perform Wave Function Collapse
  waveFunctionCollapse(image1, result, 2);

  // Display the result image
  imshow("Result", result);
  imwrite("01-ResultB.jpg", result);
  waitKey(0);

  return 0;
}
