// g++ -o 1.out 01-Claude3.7.cpp `pkg-config --cflags --libs opencv4`
// ./1.out 128 01-result.jpg
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Check if threshold value is provided
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <threshold_value> <output_filename>" << std::endl;
        return -1;
    }

    // Parse threshold value
    int threshold = std::atoi(argv[1]);
    
    // Check if threshold is valid
    if (threshold < 0 || threshold > 255) {
        std::cout << "Threshold value must be between 0 and 255" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat image = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    
    // Check if image is loaded successfully
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Create output image of the same size
    cv::Mat output_image(image.rows, image.cols, CV_8UC1);
    
    // Simple threshold operation using for loops
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // Get pixel value
            uchar pixel = image.at<uchar>(y, x);
            
            // Apply threshold
            if (pixel >= threshold) {
                output_image.at<uchar>(y, x) = 255; // White
            } else {
                output_image.at<uchar>(y, x) = 0; // Black
            }
        }
    }
    
    // Save the output image
    cv::imwrite(argv[2], output_image);
    
    std::cout << "Thresholding complete. Output saved to " << argv[2] << std::endl;
    
    return 0;
}
