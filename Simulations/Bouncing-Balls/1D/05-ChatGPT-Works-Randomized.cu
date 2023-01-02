// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <cmath>
#include <array>
#include <random>

constexpr int numBalls = 1000;
constexpr double g = 9.81; // acceleration due to gravity
constexpr double dt = 0.01; // time step
constexpr int numTimeSteps = 1000;

struct Ball
{
    double x = 0; // position
    double v = 0; // velocity
    double m = 1; // mass
    double r = 1; // radius
};

std::array<Ball, numBalls> hostBalls; // balls on the host (CPU)
Ball* deviceBalls; // balls on the device (GPU)
std::mt19937 rng(std::random_device{}()); // random number generator
std::uniform_real_distribution<double> dist(0.0, 100.0); // random position generator

__global__ void updatePositions(Ball* balls, int numBalls, double dt, double g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBalls) return;

    Ball& ball = balls[i];
    ball.x += ball.v * dt;
    ball.v += -g * dt;
}

__global__ void bounceOffGround(Ball* balls, int numBalls)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBalls) return;

    Ball& ball = balls[i];
    if (ball.x < ball.r)
    {
        ball.x = ball.r;
        ball.v = std::abs(ball.v);
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
        ball.x = dist(rng);
        cudaMemcpy(deviceBalls + i, &ball, sizeof(Ball), cudaMemcpyHostToDevice);
    }

    // update positions and velocities
    for (int t = 0; t < numTimeSteps; t++)
    {
        // launch the updatePositions kernel
        int blockSize = 32;
        int numBlocks = (numBalls + blockSize - 1) / blockSize;
        updatePositions<<<numBlocks, blockSize>>>(deviceBalls, numBalls, dt, g);

        // launch the bounceOffGround kernel
        bounceOffGround<<<numBlocks, blockSize>>>(deviceBalls, numBalls);
    }

    // copy the device balls back to the host
    cudaMemcpy(hostBalls.data(), deviceBalls, numBalls * sizeof(Ball), cudaMemcpyDeviceToHost);

    // print the final positions of the balls
    for (int i = 0; i < numBalls; i++)
    {
        const Ball& ball = hostBalls[i];
        std::cout << "Ball " << i << ": x = " << ball.x << ", v = " << ball.v << std::endl;
    }

    // free memory on the device
    cudaFree(deviceBalls);

    return 0;
}
