// Written by ChatGPT Dev 15 Version
#include <iostream>
#include <cmath>
#include <array>
#include <chrono>
#include <thread>

constexpr int numBalls = 1000;
constexpr double g = 9.81; // acceleration due to gravity
constexpr double dt = 0.01; // time step

struct Ball
{
    double x = 0; // position
    double v = 0; // velocity
    double m = 1; // mass
    double r = 1; // radius
};

std::array<Ball, numBalls> balls;

int main()
{
    while (true)
    {
        // update positions and velocities
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

        // render the balls
        for (int i = 0; i < numBalls; i++)
        {
            const Ball& ball = balls[i];
            std::cout << "Ball " << i << ": x = " << ball.x << ", v = " << ball.v << std::endl;
        }

        // sleep for a little bit to slow down the simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
