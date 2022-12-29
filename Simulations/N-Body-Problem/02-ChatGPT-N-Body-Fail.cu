#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

constexpr double G = 6.67430e-11;  // gravitational constant
constexpr double DT = 1.0;  // time step (s)

// host (CPU) version of Body
struct Body {
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

// device (GPU) version of Body
struct BodyDevice {
  double x;  // x position (m)
  double y;  // y position (m)
  double vx;  // x velocity (m/s)
  double vy;  // y velocity (m/s)
  double mass;  // mass (kg)
  double fx;  // x force (N)
  double fy;  // y force (N)
};

// host (CPU) version of Universe
class Universe {
 public:
  std::vector<Body> bodies;

  void step() {
    // compute forces on bodies
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
    // update positions and velocities of bodies
    for (auto& body : bodies) {
      body.x += DT * body.vx;
      body.y += DT * body.vy;
      body.vx += DT * body.fx / body.mass;
      body.vy += DT * body.fy / body.mass;
    }
  }
};

  // device (GPU) version of Universe
  class UniverseDevice {
   public:
    BodyDevice* bodies;  // pointer to device memory
    int num_bodies;  // number of bodies

    __device__ void compute_forces(int i) {
      // compute forces on body i
      BodyDevice& body = bodies[i];
      body.fx = body.fy = 0;  // reset forces
      for (int j = 0; j < num_bodies; ++j) {
        if (i == j) continue;  // skip self
        BodyDevice& other = bodies[j];
        double dx = other.x - body.x;
        double dy = other.y - body.y;
        double r2 = dx * dx + dy * dy;
        double r = sqrt(r2);
        double F = (G * body.mass * other.mass) / r2;
        body.fx += F * dx / r;
        body.fy += F * dy / r;
      }
    }

    __device__ void update_positions_and_velocities(int i) {
      // update positions and velocities of body i
      BodyDevice& body = bodies[i];
      body.x += DT * body.vx;
      body.y += DT * body.vy;
      body.vx += DT * body.fx / body.mass;
      body.vy += DT * body.fy / body.mass;
    }
/*
    void step() {
      // compute forces on all bodies
      for (int i = 0; i < num_bodies; ++i) {
        compute_forces(i);
      }
      // update positions and velocities of all bodies
      for (int i = 0; i < num_bodies; ++i) {
        update_positions_and_velocities(i);
      }
    }
*/
  void step() {
    // launch compute_forces kernel
    int num_threads = num_bodies;
    int num_blocks = 1;
    compute_forces<<<num_blocks, num_threads>>>(bodies, num_bodies);

    // launch update_positions_and_velocities kernel
    //update_positions_and_velocities<<<num_blocks, num_threads>>>(bodies, num_bodies);
    update_positions_and_velocities<<<num_blocks, num_threads>>>(bodies, num_bodies);
  }

  };

int main() {
  // create host (CPU) universe
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

  // copy bodies to device (GPU) memory
  int num_bodies = universe.bodies.size();
  BodyDevice* bodies_device;
  cudaMalloc(&bodies_device, num_bodies * sizeof(BodyDevice));
  cudaMemcpy(bodies_device, universe.bodies.data(),
             num_bodies * sizeof(BodyDevice), cudaMemcpyHostToDevice);

  // create device (GPU) universe
  UniverseDevice universe_device;
  universe_device.bodies = bodies_device;
  universe_device.num_bodies = num_bodies;

  // simulate the universe for 1000 time steps
  for (int i = 0; i < 1000; ++i) {
    universe_device.step();
  }

  // copy bodies back to host (CPU) memory
  cudaMemcpy(universe.bodies.data(), bodies_device,
             num_bodies * sizeof(BodyDevice), cudaMemcpyDeviceToHost);

  // print final positions of the bodies
  for (const auto& body : universe.bodies) {
    std::cout << body.x << ' ' << body.y << '\n';
  }

  // free device memory
  cudaFree(bodies_device);

  return 0;
}
