// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <cmath>

const int N = 100;
const float dt = 0.01;
const float dx = 1.0;
const float viscosity = 0.1;

void update_velocity(float *u, float *v, float *u_prev, float *v_prev) {
  for (int i = 1; i < N-1; i++) {
    for (int j = 1; j < N-1; j++) {
      u[i*N + j] = u_prev[i*N + j] + viscosity * (u_prev[(i+1)*N + j] - 2*u_prev[i*N + j] + u_prev[(i-1)*N + j]) / (dx*dx) * dt;
      v[i*N + j] = v_prev[i*N + j] + viscosity * (v_prev[i*N + j+1] - 2*v_prev[i*N + j] + v_prev[i*N + j-1]) / (dx*dx) * dt;
    }
  }
}

void update_density(float *density, float *density_prev, float *u, float *v) {
  for (int i = 1; i < N-1; i++) {
    for (int j = 1; j < N-1; j++) {
      density[i*N + j] = density_prev[i*N + j] + -density_prev[i*N + j] * (u[(i+1)*N + j] - u[(i-1)*N + j]) / (2*dx) * dt 
                        - density_prev[i*N + j] * (v[i*N + j+1] - v[i*N + j-1]) / (2*dx) * dt;
    }
  }
}

int main() {
  float u[N*N] = {0};
  float v[N*N] = {0};
  float u_prev[N*N] = {0};
  float v_prev[N*N] = {0};
  float density[N*N] = {0};
  float density_prev[N*N] = {0};

  // Initialize velocity and density
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (j >= N/4 && j <= 3*N/4) {
        u_prev[i*N + j] = 1.0;
      }
      density_prev[i*N + j] = 1.0;
    }
  }

  // Simulate for 1000 time steps
  for (int t = 0; t < 1000; t++) {
    update_velocity(u, v, u_prev, v_prev);
    update_density(density, density_prev, u, v);

    // Copy new velocity and density to previous arrays
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        u_prev[i*N + j] = u[i*N + j];
        v_prev[i*N + j] = v[i*N + j];
        density_prev[i*N + j] = density[i*N + j];
      }
    }
  }

  // Output final density
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << density[i*N + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
