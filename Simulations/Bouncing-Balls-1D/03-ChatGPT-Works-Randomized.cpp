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

std::array<Ball, numBalls> balls;
std::mt19937 rng(std::random_device{}()); // random number generator
std::uniform_real_distribution<double> dist(0.0, 100.0); // random position generator

int main()
{
    // initialize the balls with random positions
    for (int i = 0; i < numBalls; i++)
    {
        Ball& ball = balls[i];
        ball.x = dist(rng);
    }

    // update positions and velocities
    for (int t = 0; t < numTimeSteps; t++)
    {
        for (int i = 0; i < numBalls; i++)
        {
            Ball& ball = balls[i];
            ball.x += ball.v * dt;
            ball.v += -g * dt;

            // bounce off the ground
            if (ball.x < ball.r)
            {
                ball.x = ball.r;
                ball.v = std::abs(ball.v);
            }
        }
    }

    // print the final positions of the balls
    for (int i = 0; i < numBalls; i++)
    {
        const Ball& ball = balls[i];
        std::cout << "Ball " << i << ": x = " << ball.x << ", v = " << ball.v << std::endl;
    }

    return 0;
}
