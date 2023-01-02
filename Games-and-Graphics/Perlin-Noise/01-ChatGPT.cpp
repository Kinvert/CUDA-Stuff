// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <cstdlib>
#include <cmath>

// Generates a random float between 0 and 1
float randomFloat()
{
    return (float)rand() / RAND_MAX;
}

// Interpolates between a and b by t
float interpolate(float a, float b, float t)
{
    return a * (1 - t) + b * t;
}

// Returns the dot product of the distance and gradient vectors
float dotGridGradient(int ix, int iy, float x, float y)
{
    // Precomputed (or otherwise) gradient vectors at each grid node
    float gradient[2][2] = {{randomFloat(), randomFloat()}, {randomFloat(), randomFloat()}};

    // Distance vectors
    float dx = x - (float)ix;
    float dy = y - (float)iy;

    // Compute the dot-product
    return (dx * gradient[iy][ix] + dy * gradient[iy][ix+1]);
}

// Computes Perlin noise at coordinates x, y
float perlin(float x, float y)
{
    // Determine grid cell coordinates
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    // Determine interpolation weights
    // Could also use higher order polynomial/s-curve here
    float sx = x - (float)x0;
    float sy = y - (float)y0;

    // Interpolate between grid point gradients
    float n0, n1, ix0, ix1, value;
    n0 = dotGridGradient(x0, y0, x, y);
    n1 = dotGridGradient(x1, y0, x, y);
    ix0 = interpolate(n0, n1, sx);
    n0 = dotGridGradient(x0, y1, x, y);
    n1 = dotGridGradient(x1, y1, x, y);
    ix1 = interpolate(n0, n1, sx);
    value = interpolate(ix0, ix1, sy);

    return value;
}

int main()
{
    // Generate Perlin noise at coordinate (x, y)
    float x = 0.5;
    float y = 0.5;
    std::cout << perlin(x, y) << std::endl;

    return 0;
}
