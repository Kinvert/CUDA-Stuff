// Compiling worked for me: nvcc 04-Works.cu -o 4.out `pkg-config opencv4 --cflags --libs` && ./4.out
// To keep image size down for GitHub, I don't output every single frame to the gif
// Reference: Statics and Dynamics 11th - R.C. Hibbeler, Page 245 in the Dynamics half of the book
//     similar to homework on page 254 of Dynamics half of the book
//     Amazon Link (I'll get a small percentage of Amazon's profit): https://amzn.to/3GfJINv
#include <array>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

constexpr int numBalls = 50; // Number of Balls
constexpr double g = -9.81; // Gravity
constexpr double dt = 0.01; // Timestep
constexpr double COR = 0.99; // Coefficient of Restitution
constexpr int numTimeSteps = 600; // Total number of time steps (not all turn in to image frames)
constexpr int framesEvery = 10; // Skip this many time steps before making next frame (for animation purposes)
constexpr double yBot = 0; // y-coordinate of the yBot
constexpr double yTop = 360; // y-coordinate of the yTop (this is actually the floor in image space)
constexpr double xLeft = 0; // x-coordinate of the left wall
constexpr double xRight = 480; // x-coordinate of the right wall

struct Ball
{
    double x = 0.0; // x-coordinate of the center
    double y = 0.0; // y-coordinate of the center
    double vx = 0.0; // velocity in the x direction
    double vy = 0.0; // velocity in the y direction
    double m = 1.0; // Mass
    int r = 8; // Radius
    int stroke = 1; // Drawing line thickness
    int contact = 0; // Keep track of contact for Fill
    cv::Scalar color; // Color of the ball
};

std::array<Ball, numBalls> hostBalls; // Balls on the host (CPU)
Ball* deviceBalls; // Balls on the device (GPU)
std::mt19937 rng(std::random_device{}()); // random number generator
std::uniform_real_distribution<double> xDist(20.0, 460.0); // random x-coordinate generator
std::uniform_real_distribution<double> yDist(20.0, 340.0); // random y-coordinate generator
std::uniform_real_distribution<double> xVelo(-40.0, 40.0); // random x-velocity generator
std::uniform_real_distribution<double> yVelo(-40.0, 40.0); // random y-velocity generator
std::uniform_int_distribution<int> colorDist(0, 255); // random color generator
std::uniform_int_distribution<int> radius(6, 16); // random radius generator

__global__ void updatePositions(Ball* balls, int numBalls, double dt, double g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBalls) return;

    Ball& ball = balls[i];
    ball.x += ball.vx * dt;
    ball.y += ball.vy * dt;
    ball.vy += -g * dt; // Gravity
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
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= numBalls) return;
    if (j >= numBalls) return;
    
    Ball& ball1 = balls[i];
    Ball& ball2 = balls[j];
    ball1.contact = 0;
    ball2.contact = 0;
    ball1.stroke = 1;
    ball2.stroke = 1;
    
    __syncthreads();

    if (i != j) {
        double x1 = ball1.x;
        double y1 = ball1.y;
        double x2 = ball2.x;
        double y2 = ball2.y;
        double dx = x2 - x1;
        double dy = y2 - y1;
        double distance = std::sqrt(dx * dx + dy * dy); // Distance between centers of the two balls
        
        // Collision?
        if (distance < ball1.r + ball2.r)
        {
            // Collision detected, so execute the following
            
            // Initial Conditions
            double m1 = ball1.m;
            double m2 = ball2.m;
            double vx1 = ball1.vx; // Vel X Component for ball 1
            double vy1 = ball1.vy;
            double vx2 = ball2.vx;
            double vy2 = ball2.vy;
            double v1 = std::sqrt(vx1 * vx1 + vy1 * vy1); // Vel Magnitude
            double v2 = std::sqrt(vx2 * vx2 + vy2 * vy2);
            double r1 = ball1.r;
            double r2 = ball2.r;
            double theta1 = std::atan2(vy1, vx1); // Angle of vel vector ball1
            double theta2 = std::atan2(vy2, vx2);
            double dvx = vx2 - vx1; // Delta V x component
            double dvy = vy2 - vy1;
            
            // Initial Normal-Tangential
            double angleN = std::atan2(dy, dx); // Normal angle for Normal-Tangential reference frame
            double angleT = angleN - 3.14159 / 2.0; // Tangential angle for Normal-Tangential reference frame
            double theta1t = theta1 - angleT; // This is effectively the new x axis in this reference frame
            double theta2t = theta2 - angleT;
            double vt1 = v1 * cos(theta1t);
            double vn1 = v1 * sin(theta1t);
            //double vt2 = v2 * cos(theta2t);
            double vn2 = v2 * sin(theta2t);
            
            // Final
            double vt1P = vt1;
            //double vt2P = vt2;
            double vn1P = (COR * m2 * (vn2-vn1) + m1 * vn1 + m2 * vn2) / (m1 + m2);
            //double vn2P = (COR * m1 * (vn1-vn2) + m1 * vn1 + m2 * vn2) / (m1 + m2);
            double v1P = std::sqrt(vt1P * vt1P + vn1P * vn1P);
            //double v2P = std::sqrt(vt2P * vt2P + vn2P * vn2P);
            double thetaNT1 = std::atan2(vn1P, vt1P);
            //double thetaNT2 = std::atan2(vn2P, vt2P);
            double thetaXY1 = thetaNT1 + angleT;
            //double thetaXY2 = thetaNT2 + angleT;
            
            // Values to Assign
            double vx1P = v1P * cos(thetaXY1);
            double vy1P = v1P * sin(thetaXY1);
            //double vx2P = v2P * cos(thetaXY2);
            //double vy2P = v2P * sin(thetaXY2);
            ball1.vx = vx1P;
            ball1.vy = vy1P;
            
            // Update Position to remove overlap
            double overlap = r1 + r2 - distance;
            double hOverlap = overlap / 2.0;
            double x1Pcorr = cos(angleN) * hOverlap; // Correction X
            double y1Pcorr = sin(angleN) * hOverlap; // Correction Y
            ball1.x -= x1Pcorr;
            ball1.y -= y1Pcorr;
            
            // Contact for color Fill
            ball1.contact = 1;
            ball2.contact = 1;
            ball1.stroke = -1;
            ball2.stroke = -1;
        }
    }
    __syncthreads();
}

int main()
{
    // Allocate Memory on GPU
    cudaMalloc((void**)&deviceBalls, numBalls * sizeof(Ball));

    // Initialize Balls
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.x = xDist(rng);
        ball.y = yDist(rng);
        ball.vx = xVelo(rng);
        ball.vy = yVelo(rng);
        ball.r= radius(rng);
        ball.m = ball.r * ball.r;
        ball.color = cv::Scalar(colorDist(rng), colorDist(rng), colorDist(rng));
        cudaMemcpy(deviceBalls + i, &ball, sizeof(Ball), cudaMemcpyHostToDevice);
    }

    cv::Mat frame(yTop, xRight, CV_8UC3, cv::Scalar::all(0)); // create a black image

    // update positions and velocities, and draw the balls
    for (int t = 0; t < numTimeSteps; t++)
    {
        // launch the updatePositions kernel
        int blockSize = 32;
        int numBlocks = (numBalls + blockSize - 1) / blockSize;
        dim3 gridSize(numBlocks, numBlocks);
        dim3 blockSizeDim3(blockSize, blockSize);
        updatePositions<<<numBlocks, blockSize>>>(deviceBalls, numBalls, dt, g);

        // launch the bounceOffWalls kernel
        bounceOffWalls<<<numBlocks, blockSize>>>(deviceBalls, numBalls, yBot, yTop, xLeft, xRight);

        // launch the bounceOffBalls kernel
        bounceOffBalls<<<gridSize, blockSizeDim3>>>(deviceBalls, numBalls);

        // copy the device balls back to the host
        cudaMemcpy(hostBalls.data(), deviceBalls, numBalls * sizeof(Ball), cudaMemcpyDeviceToHost);

        // Draw Frame
        if (t % framesEvery == 0) {
            frame = cv::Scalar::all(0); // set the image to black
            
            // Draw Walls
            cv::line(frame, cv::Point(0, 0), cv::Point(xRight, 0), cv::Scalar::all(50), 2);
            cv::line(frame, cv::Point(0, yTop), cv::Point(xRight, yTop), cv::Scalar::all(255), 2);
            cv::line(frame, cv::Point(0, 0), cv::Point(0, yTop), cv::Scalar::all(50), 2);
            cv::line(frame, cv::Point(xRight, 0), cv::Point(xRight, yTop), cv::Scalar::all(50), 2);
            
            // Draw Balls
            for (int i = 0; i < numBalls; i++)
            {
                const Ball& ball = hostBalls[i];
                cv::ellipse(frame, cv::Point(ball.x, ball.y), cv::Size(ball.r, ball.r), 0, 0, 360, ball.color, ball.stroke);
            }

            // show the image and save it to a file
            cv::imshow("Bouncing Balls", frame);
            cv::waitKey(10);
        
            // save the image
            cv::imwrite(std::to_string(t) + ".jpg", frame);
        }
    }

    // free memory on the device
    cudaFree(deviceBalls);

    return 0;
}

// I took the images created here and used GIMP to make the gif
