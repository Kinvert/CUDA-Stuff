// ChatGPT also wrote this
// This one also does not work
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

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

  void step() {
    // define the kernel launch configuration
    dim3 block_size(16, 16);
    dim3 grid_size((num_bodies + block_size.x - 1) / block_size.x,
                   (num_bodies + block_size.y - 1) / block_size.y);

    // launch the kernel
    //compute_forces<<<grid_size, block_size>>>(bodies, num_bodies); // Line 110
    compute_forces<<<grid_size, block_size>>>(num_bodies); // Line 110

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cerr << "Error launching kernel: " << cudaGetErrorString(error) << std::endl;
      exit(1);
    }
  }
};

int main() {
  // create host (CPU) universe
  Universe universe;
  universe.bodies.emplace_back(0, 0, 0, 0, 1.989e30);  // Sun
  universe.bodies.emplace_back(1.496e11, 0, 0, 29800, 5.972e24);  // Earth
  universe.bodies.emplace_back(-149.6e9, 0, 0, -29800, 5.972e24);  // Moon orbiting the other sun
  universe.bodies.emplace_back(108.9e9,  0, 0, 35074,  4.867e24);  // Venus

  // create device (GPU) universe
  int num_bodies = universe.bodies.size();
  BodyDevice* bodies_device;
  cudaMalloc((void**)&bodies_device, num_bodies * sizeof(BodyDevice));
  cudaMemcpy(bodies_device, universe.bodies.data(), num_bodies * sizeof(BodyDevice), cudaMemcpyHostToDevice);
  UniverseDevice universe_device;
  universe_device.bodies = bodies_device;
  universe_device.num_bodies = num_bodies;

  // simulate for 1000 steps
  for (int i = 0; i < 1000; ++i) {
    universe_device.step();
  }

  // copy device memory back to host
  cudaMemcpy(universe.bodies.data(), universe_device.bodies,
             num_bodies * sizeof(Body), cudaMemcpyDeviceToHost);

  // print final positions of bodies
  for (const auto& body : universe.bodies) {
    std::cout << body.x << " " << body.y << " " << body.vx << " " << body.vy
              << " " << body.mass << std::endl;
  }

  // clean up device memory
  cudaFree(universe_device.bodies);

  return 0;
}



