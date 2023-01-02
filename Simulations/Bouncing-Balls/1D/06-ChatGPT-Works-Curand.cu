// Written by ChatGPT Dec 15 Version
// In my case: nvcc 06-ChatGPT-Works-Curand.cu -o 6.out -I /usr/local/cuda/lib64 -lcurand && ./6.out
#include <iostream>
#include <cmath>
#include <array>
#include <random>
#include <curand_kernel.h>

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
curandGenerator_t rng; // random number generator on the device

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
    // create a random number generator on the device
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, std::random_device{}());

    // allocate memory on the device
    cudaMalloc((void**)&deviceBalls, numBalls * sizeof(Ball));

    // initialize the balls with random positions on the device
    curandGenerateUniform(rng, (float*)deviceBalls, numBalls);

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

    // destroy the random number generator and free memory on the device
    curandDestroyGenerator(rng);
    cudaFree(deviceBalls);

    return 0;
}
