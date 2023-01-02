// Written by ChatGPT Dec 15 Version
// I did have to make some small fixes such as color being Scalar
// and how the frames were saved
// Compiling worked for me: nvcc 03-ChatGPT.cu -o 3.out `pkg-config opencv4 --cflags --libs` && ./3.out
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
//    cv::Vec3b color; // color of the ball
    cv::Scalar color;
};

std::array<Ball, numBalls> hostBalls; // balls on the host (CPU)
Ball* deviceBalls; // balls on the device (GPU)
std::mt19937 rng(std::random_device{}()); // random number generator
std::uniform_real_distribution<double> xDist(0.0, 480.0); // random x-coordinate generator
std::uniform_real_distribution<double> yDist(0.0, 360.0); // random y-coordinate generator
std::uniform_real_distribution<double> xVelo(-20.0, 20.0); // random x-velocity generator
std::uniform_real_distribution<double> yVelo(-20.0, 20.0); // random y-velocity generator
std::uniform_int_distribution<int> colorDist(0, 255); // random color generator

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

__global__ void bounceOffBalls(Ball* balls, int numBalls)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBalls) return;

    Ball& ball = balls[i];
    for (int j = 0; j < numBalls; j++)
    {
        if (i == j) continue;

        Ball& other = balls[j];
        double dx = ball.x - other.x;
        double dy = ball.y - other.y;
        double distance = std::sqrt(dx * dx + dy * dy);
        if (distance < ball.r + other.r)
        {
            double angle = std::atan2(dy, dx);
            double sine = std::sin(angle);
            double cosine = std::cos(angle);

            // rotate the ball positions
            double x = 0;
            double y = 0;
            x = dx * cosine + dy * sine;
            y = dy * cosine - dx * sine;

            // rotate the velocities
            double vx = 0;
            double vy = 0;
            vx = ball.vx * cosine + ball.vy * sine;
            vy = ball.vy * cosine - ball.vx * sine;

            // collide
            double vTotal = vx - other.vx;
            double xTotal = x - other.x;
            double yTotal = y - other.y;
            vx = ((ball.m - other.m) * vx + 2 * other.m * other.vx) / (ball.m + other.m);
            vy *= -1;
            other.vx = ((other.m - ball.m) * other.vx + 2 * ball.m * ball.vx) / (ball.m + other.m);
            other.vy *= -1;

            // rotate ball positions back
            x = x * cosine - y * sine;
            y = y * cosine + x * sine;

            // rotate velocities back
            ball.vx = vx * cosine - vy * sine;
            ball.vy = vy * cosine + vx * sine;

            // adjust positions to actual screen position
            ball.x = ball.x + (x - xTotal) / 2;
            ball.y = ball.y + (y - yTotal) / 2;
            other.x = other.x - (x - xTotal) / 2;
            other.y = other.y - (y - yTotal) / 2;
        }
    }
}

int main()
{
    // allocate memory on the device
    cudaMalloc((void**)&deviceBalls, numBalls * sizeof(Ball));

    // initialize the balls with random positions and colors on the device
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.x = xDist(rng);
        ball.y = yDist(rng);
        ball.vx = xVelo(rng);
        ball.vy = yVelo(rng);
        ball.color = cv::Scalar(colorDist(rng), colorDist(rng), colorDist(rng));
        cudaMemcpy(deviceBalls + i, &ball, sizeof(Ball), cudaMemcpyHostToDevice);
    }

    cv::Mat frame(480, 360, CV_8UC3, cv::Scalar::all(0)); // create a black image
    std::vector<cv::Scalar> colors(numBalls);

    // update positions and velocities, and draw the balls
    for (int t = 0; t < numTimeSteps; t++)
    {
        // launch the updatePositions kernel
        int blockSize = 32;
        int numBlocks = (numBalls + blockSize - 1) / blockSize;
        updatePositions<<<numBlocks, blockSize>>>(deviceBalls, numBalls, dt, g);

        // launch the bounceOffWalls kernel
        bounceOffWalls<<<numBlocks, blockSize>>>(deviceBalls, numBalls, yBot, yTop, xLeft, xRight);

        // launch the bounceOffBalls kernel
        bounceOffBalls<<<numBlocks, blockSize>>>(deviceBalls, numBalls);

        // copy the device balls back to the host
        cudaMemcpy(hostBalls.data(), deviceBalls, numBalls * sizeof(Ball), cudaMemcpyDeviceToHost);

        // draw the balls on the image
        frame = cv::Scalar::all(0); // set the image to black
        for (int i = 0; i < numBalls; i++)
        {
            const Ball& ball = hostBalls[i];
            cv::ellipse(frame, cv::Point(ball.x, ball.y), cv::Size(ball.r, ball.r), 0, 0, 360, ball.color, -1);
        }

        // show the image and save it to a file
        cv::imshow("Bouncing Balls", frame);
        cv::waitKey(10);
        
        // save the image
        cv::imwrite(std::to_string(t) + ".jpg", frame);
    }

    // free memory on the device
    cudaFree(deviceBalls);

    return 0;
}
