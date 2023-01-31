// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <cmath>

const int N = 100;
const float dt = 0.01;
const float dx = 1.0;
const float viscosity = 0.1;
const float inlet_velocity = 1.0;

void update_velocity(float* u, float* v, float* u_prev, float* v_prev) {
  for (int i = 1; i < N-1; i++) {
    for (int j = 1; j < N-1; j++) {
      float u_x = (u_prev[(i+1)*N + j] - u_prev[(i-1)*N + j]) / (2*dx);
      float u_y = (u_prev[i*N + j+1] - u_prev[i*N + j-1]) / (2*dx);
      float v_x = (v_prev[(i+1)*N + j] - v_prev[(i-1)*N + j]) / (2*dx);
      float v_y = (v_prev[i*N + j+1] - v_prev[i*N + j-1]) / (2*dx);

      float du_dt = -u_prev[i*N + j] * u_x - v_prev[i*N + j] * u_y + viscosity * (u_x + v_y);
      float dv_dt = -u_prev[i*N + j] * v_x - v_prev[i*N + j] * v_y + viscosity * (u_x + v_y);

      u[i*N + j] = u_prev[i*N + j] + du_dt * dt;
      v[i*N + j] = v_prev[i*N + j] + dv_dt * dt;
    }
  }

  // Set velocity on left boundary to inlet velocity
  for (int j = 1; j < N-1; j++) {
    u[j] = inlet_velocity;
    v[j] = 0.0;
  }
}

int main() {
  float* u = new float[N*N];
  float* v = new float[N*N];
  float* u_prev = new float[N*N];
  float* v_prev = new float[N*N];

  // Initialize velocity
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      u_prev[i*N + j] = 0.0;
      v_prev[i*N + j] = 0.0;
    }
  }

  for (int t = 0; t < 1000; t++) {
    update_velocity(u, v, u_prev, v_prev);

    // Copy current velocity to previous velocity
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        u_prev[i*N + j] = u[i*N + j];
        v_prev[i*N + j] = v[i*N + j];
      }
    }
  }

  // Output velocity at (98, 99)
  std::cout << "Velocity at (98, 99): " << u[98*N + 99] << ", " << v[98*N + 99] << std::endl;

  delete[] u;
  delete[] v;
  delete[] u_prev;
  delete[] v_prev;

  return 0;
}
