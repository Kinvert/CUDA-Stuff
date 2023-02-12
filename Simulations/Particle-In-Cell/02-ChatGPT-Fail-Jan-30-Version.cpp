// Written by ChatGPT Jan 30 Version

#include <iostream>
#include <cmath>
#include <cstdlib>

// Constants
const int num_particles = 1000;
const int num_cells = 100;

const double dt = 0.01;
const double L = 1.0;
const double h = L / num_cells;
const double q = 1.0;
const double m = 1.0;

// Particle struct
struct Particle {
    double x, v;
};

// Grid struct
struct Grid {
    double rho, E;
};

int main() {
    // Initialize particles
    Particle particles[num_particles];
    for (int i = 0; i < num_particles; i++) {
        particles[i].x = L * rand() / RAND_MAX;
        particles[i].v = 0.0;
    }

    // Initialize grid
    Grid grid[num_cells];
    for (int i = 0; i < num_cells; i++) {
        grid[i].rho = 0.0;
        grid[i].E = 0.0;
    }

    // Main loop
    for (int t = 0; t < 1000; t++) {
        // Update particle velocities
        for (int i = 0; i < num_particles; i++) {
            int cell = particles[i].x / h;
            if (cell < 0 || cell >= num_cells) {
                std::cerr << "Particle out of bounds!" << std::endl;
                return 1;
            }
            particles[i].v += q / m * grid[cell].E * dt;
        }

        // Update particle positions
        for (int i = 0; i < num_particles; i++) {
            particles[i].x += particles[i].v * dt;
            if (particles[i].x < 0) {
                particles[i].x += L;
            } else if (particles[i].x > L) {
                particles[i].x -= L;
            }
        }

        // Calculate charge density
        for (int i = 0; i < num_particles; i++) {
            int cell = particles[i].x / h;
            if (cell < 0 || cell >= num_cells) {
                std::cerr << "Particle out of bounds!" << std::endl;
                return 1;
            }
            grid[cell].rho += q / h;
        }

        // Calculate electric field
        for (int i = 0; i < num_cells; i++) {
            grid[i].E = (grid[(i+1)%num_cells].rho - grid[(i-1+num_cells)%num_cells].rho) / (2*h);
        }
    }

    // Print final particle positions
    for (int i = 0; i < num_particles; i++) {
        std::cout << particles[i].x << std::endl;
    }

    return 0;
}

