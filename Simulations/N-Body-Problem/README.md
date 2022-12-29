# N Body Problem

**Written by ChatGPT**

**Note: It added that extra sun in there for fun on it's own haha**

## Introduction

Certainly! The N-body problem refers to the problem of predicting the individual motions of a group of celestial objects interacting with each other gravitationally. It can be a computationally intensive problem, particularly when simulating a large number of bodies over a long period of time.

## Naive C++ Implementation

Here is a simple C++ program that solves the N-body problem using a brute-force approach, where the force between each pair of bodies is calculated directly. This program uses the Euler method to integrate the equations of motion, which is a simple and widely-used method, but not particularly accurate or efficient. However, it is easy to understand and implement, and sufficient for demonstrating the basic concepts of the N-body problem.

The program defines a Body class to represent each celestial body, with member variables for its position, velocity, mass, and force. It also defines a Universe class to represent the collection of bodies and simulate their interactions. The Universe class contains a vector of Body objects and a method step to advance the simulation by a small time step.

'''cpp
#include <iostream>
#include <cmath>
#include <vector>

constexpr double G = 6.67430e-11;  // gravitational constant
constexpr double DT = 1.0;  // time step (s)

class Body {
 public:
  double x;  // x position (m)
  double y;  // y position (m)
  double vx;  // x velocity (m/s)
  double vy;  // y velocity (m/s)
  double mass;  // mass (kg)
  double fx;  // x force (N)
  double fy;  // y force (N)

  Body(double x, double y, double vx, double vy, double mass)
      : x(x), y(y), vx(vx), vy(vy), mass(mass), fx(0), fy(0) {}
};

class Universe {
 public:
  std::vector<Body> bodies;

  void step() {
    for (auto& body : bodies) {
      body.fx = body.fy = 0;  // reset forces
      for (const auto& other : bodies) {
        if (&body == &other) continue;  // skip self
        double dx = other.x - body.x;
        double dy = other.y - body.y;
        double r2 = dx * dx + dy * dy;
        double r = std::sqrt(r2);
        double F = (G * body.mass * other.mass) / r2;
        body.fx += F * dx / r;
        body.fy += F * dy / r;
      }
    }
    for (auto& body : bodies) {
      body.x += DT * body.vx;
      body.y += DT * body.vy;
      body.vx += DT * body.fx / body.mass;
      body.vy += DT * body.fy / body.mass;
    }
  }
};

int main() {
  Universe universe;
  // add some bodies to the universe
  //                           x,        y, vx, vy,     m
  universe.bodies.emplace_back(0,        0, 0, 0,      1.989e30);  // Sun
  universe.bodies.emplace_back(149.6e9,  0, 0, 29800,  5.972e24);  // Earth
  universe.bodies.emplace_back(-149e9,   0, 0, 0,      1.989e30);  // Another sun for fun
  universe.bodies.emplace_back(-149.6e9, 0, 0, -29800, 5.972e24);  // Moon orbiting the other sun

  // simulate the universe for 1000 time steps
  for (int i = 0; i < 1000; ++i) {
    universe.step();
  }

  // print final positions of the bodies
  for (const auto& body : universe.bodies) {
    std::cout << body.x << ' ' << body.y << '\n';
  }
  return 0;
}
```

This program initializes a few bodies in the universe, advances the simulation by 1000 time steps, and then prints the final positions of the bodies.

This is a very basic implementation of the N-body problem and there are many ways it could be improved and optimized. For example, the Euler method is relatively inaccurate and inefficient compared to more advanced methods like the leapfrog method or the Runge-Kutta method. Additionally, the brute-force approach of calculating the force between every pair of bodies is O(N^2) in complexity, which can be slow for large N. There are also many optimization techniques that can be used to improve the performance of the simulation, such as hierarchical tree methods or approximations like the Barnes-Hut algorithm. However, this simple program should give you a basic understanding of how to solve the N-body problem in C++.
