// Written by ChatGPT Dec 15 Version
// I changed some variable names and constants
// for example they named yBot as floor which caused errors
// they also named yTop as ceiling instead
// Compiling worked for me: nvcc 02-ChatGPT.cu -o 2.out `pkg-config opencv4 --cflags --libs` && ./2.out
#include <iostream>
#include <cmath>
#include <array>
#include <random>
#include <opencv2/opencv.hpp>

constexpr int numBalls = 50;
constexpr double g = -9.81; // acceleration due to gravity
constexpr double dt = 0.2; // time step
constexpr int numTimeSteps = 30;
constexpr double yBot = 0; // y-coordinate of the yBot
constexpr double yTop = 360; // y-coordinate of the yTop
constexpr double xLeft = 0; // x-coordinate of the left wall
constexpr double xRight = 480; // x-coordinate of the right wall

struct Ball
{
    double x = 0; // x-coordinate of the center
    double y = 0; // y-coordinate of the center
    double vx = 0; // velocity in the x direction
    double vy = 0; // velocity in the y direction
    double m = 1; // mass
    double r = 8; // radius
};

std::array<Ball, numBalls> hostBalls; // balls on the host (CPU)
Ball* deviceBalls; // balls on the device (GPU)
std::mt19937 rng(std::random_device{}()); // random number generator
std::uniform_real_distribution<double> xDist(0.0, 480.0); // random x-coordinate generator
std::uniform_real_distribution<double> yDist(0.0, 360.0); // random y-coordinate generator
std::uniform_real_distribution<double> xVelo(-20.0, 20.0); // random x-coordinate generator
std::uniform_real_distribution<double> yVelo(-20.0, 20.0); // random y-coordinate generator

__global__ void updatePositions(Ball* balls, int numBalls, double dt, double g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBalls) return;

    Ball& ball = balls[i];
    ball.x += ball.vx * dt;
    ball.y += ball.vy * dt;
    ball.vy += -g * dt;
}

__global__ void bounceOffWalls(Ball* balls, int numBalls, double yBot, double yTop, double xLeft, double xRight)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBalls) return;

    Ball& ball = balls[i];
    if (ball.y < yBot + ball.r)
    {
        ball.y = yBot + ball.r;
        ball.vy = std::abs(ball.vy);
    }
    else if (ball.y > yTop - ball.r)
    {
        ball.y = yTop - ball.r;
        ball.vy = -std::abs(ball.vy);
    }
    if (ball.x < xLeft + ball.r)
    {
        ball.x = xLeft + ball.r;
        ball.vx = std::abs(ball.vx);
    }
    else if (ball.x > xRight - ball.r)
    {
        ball.x = xRight - ball.r;
    ball.vx = -std::abs(ball.vx);
    }
}

int main()
{
    // allocate memory on the device
    cudaMalloc((void**)&deviceBalls, numBalls * sizeof(Ball));

    // initialize the balls with random positions on the device
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.x = xDist(rng);
        ball.y = yDist(rng);
        ball.vx = xVelo(rng);
        ball.vy = yVelo(rng);
        cudaMemcpy(deviceBalls + i, &ball, sizeof(Ball), cudaMemcpyHostToDevice);
    }

    // update positions and velocities
    for (int t = 0; t < numTimeSteps; t++)
    {
        // launch the updatePositions kernel
        int blockSize = 32;
        int numBlocks = (numBalls + blockSize - 1) / blockSize;
        updatePositions<<<numBlocks, blockSize>>>(deviceBalls, numBalls, dt, g);

        // launch the bounceOffWalls kernel
        bounceOffWalls<<<numBlocks, blockSize>>>(deviceBalls, numBalls, yBot, yTop, xLeft, xRight);

        // copy the device balls back to the host
        cudaMemcpy(hostBalls.data(), deviceBalls, numBalls * sizeof(Ball), cudaMemcpyDeviceToHost);

        // create an image with a white background
        cv::Mat image = cv::Mat::ones(360, 480, CV_8UC3) * 255;

        // draw the balls on the image
        for (int i = 0; i < numBalls; i++)
        {
            const Ball& ball = hostBalls[i];
            cv::circle(image, cv::Point(ball.x, ball.y), ball.r, cv::Scalar(0, 0, 0), -1);
        }

        // save the image
        cv::imwrite(std::to_string(t) + ".jpg", image);
    }

    // free memory on the device
    cudaFree(deviceBalls);

    return 0;
}
