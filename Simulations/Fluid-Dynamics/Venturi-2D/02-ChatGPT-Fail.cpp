// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <cmath>

const int N = 100;
const float dt = 0.1;
const float visc = 0.001;
const float force = 5.0;

void update_velocity(float *u, float *v, float *u_prev, float *v_prev) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      float u_x = (u_prev[i*N + j + 1] - u_prev[i*N + j - 1]) / 2.0;
      float u_y = (u_prev[(i + 1)*N + j] - u_prev[(i - 1)*N + j]) / 2.0;
      float v_x = (v_prev[i*N + j + 1] - v_prev[i*N + j - 1]) / 2.0;
      float v_y = (v_prev[(i + 1)*N + j] - v_prev[(i - 1)*N + j]) / 2.0;

      float du_dt = -u_prev[i*N + j] * u_x - v_prev[i*N + j] * u_y + visc * (u_x + u_y) + force;
      float dv_dt = -u_prev[i*N + j] * v_x - v_prev[i*N + j] * v_y + visc * (v_x + v_y);

      u[i*N + j] = u_prev[i*N + j] + du_dt * dt;
      v[i*N + j] = v_prev[i*N + j] + dv_dt * dt;
    }
  }

  // Apply boundary conditions
  for (int i = 0; i < N; i++) {
    u[i*N] = 1.0;
    u[i*N + N - 1] = 1.0;
    v[i*N] = 0.0;
    v[i*N + N - 1] = 0.0;
  }
  for (int j = 0; j < N; j++) {
    u[j] = 0.0;
    u[(N - 1)*N + j] = 0.0;
    v[j] = 0.0;
    v[(N - 1)*N + j] = 0.0;
  }
}

int main() {
  float *u = new float[N*N];
  float *v = new float[N*N];
  float *u_prev = new float[N*N];
  float *v_prev = new float[N*N];

  // Initialize velocity
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      u[i*N + j] = 0.0;
      v[i*N + j] = 0.0;
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

  // Output final velocity
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << "Velocity at (" << i << ", " << j << "): " << u[i*N + j] << ", " << v[i*N + j] << std::endl;
    }
  }

  delete[] u;
  delete[] v;
  delete[] u_prev;
  delete[] v_prev;

  return 0;
}

