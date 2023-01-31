// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <cmath>

const int N = 100;
const int M = 100;
const float dt = 0.01;
const float h0 = 0.1;
const float g = 9.8;
const float u0 = 1.0;

void update_depth(float h[N][M], float u[N][M], float v[N][M]) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < M - 1; j++) {
      h[i][j] = h[i][j] - (h0 / u0) * (u[i][j] - u[i][j - 1]) * dt - (h0 / u0) * (v[i][j] - v[i - 1][j]) * dt;
    }
  }
}

void update_velocity(float u[N][M], float v[N][M], float h[N][M]) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < M - 1; j++) {
      u[i][j] = u[i][j] - g * (h[i][j + 1] - h[i][j - 1]) * dt / (2 * h0);
      v[i][j] = v[i][j] - g * (h[i + 1][j] - h[i - 1][j]) * dt / (2 * h0);
    }
  }
}

int main() {
  float h[N][M];
  float u[N][M];
  float v[N][M];

  // Set initial conditions for depth
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (i >= N / 4 && i <= N / 4 * 3 && j >= M / 4 && j <= M / 4 * 3) {
        h[i][j] = 1;
      } else {
        h[i][j] = h0;
      }
      u[i][j] = u0;
      v[i][j] = 0.0;
    }
  }

  for (int t = 0; t < 1000; t++) {
    update_depth(h, u, v);
    update_velocity(u, v, h);

    // Output the result
    if (t % 100 == 0) {
      std::cout << "Depth at time step " << t << ":" << std::endl;
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
          std::cout << h[i][j] << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  return 0;
}
