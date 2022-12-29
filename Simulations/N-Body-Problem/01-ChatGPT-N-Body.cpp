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
  universe.bodies.emplace_back(108.9e9,  0, 0, 35074,  4.867e24);  // Venus
  universe.bodies.emplace_back(227.9e9,  0, 0, 24077,  6.39e23);   // Mars
  universe.bodies.emplace_back(778.3e9,  0, 0, 13070,  1.898e27);  // Jupiter


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
