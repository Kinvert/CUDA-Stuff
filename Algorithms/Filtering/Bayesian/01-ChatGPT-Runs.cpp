// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <random>

const int N = 100;

void bayesianFilter(float state[], float measurement[], float control[]) {
    for (int i = 0; i < N; i++) {
        // prediction step
        state[i] = state[i] + control[i];
        // correction step
        state[i] = state[i] + (measurement[i] - state[i]) / 2;
    }
}

int main() {
    float state[N];
    float measurement[N];
    float control[N];

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0,1.0);
    
    // initialize state, measurement, and control with random values between -1 and 1
    for (int i = 0; i < N; i++) {
        state[i] = distribution(generator);
        measurement[i] = distribution(generator);
        control[i] = distribution(generator);
    }
    
    bayesianFilter(state, measurement, control);
    // use state as the updated estimate
    for(int i =0; i<N; i++){
        std::cout<< " State: " << state[i] << std::endl;
    }
    return 0;
}

