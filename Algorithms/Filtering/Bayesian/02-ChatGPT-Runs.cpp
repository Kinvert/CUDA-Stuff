// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <random>

const int N = 40; // 40 samples, with step size of 0.05s, this will give 2 seconds of simulation

void bayesianFilter(float state[], float measurement[], float control[], float dt, float target_rate) {
    float error;
    float kp = 1;
    float throttle;
    for (int i = 0; i < N; i++) {
        error = target_rate - state[i];
        throttle = kp * error + 0.6;
        if(throttle < 0.6) throttle = 0.6;
        else if(throttle > 1.0) throttle = 1.0;
        // prediction step
        state[i+1] = state[i] + (control[i]*throttle/15000)*dt; // update the velocity
        // correction step
        state[i+1] = state[i+1] + (measurement[i] - state[i+1]) / 2;
    }
}

int main() {
    float state[N+1];
    float measurement[N];
    float control[N];
    float dt = 0.05;
    float target_rate = 2;
    std::default_random_engine generator;
    std::normal_distribution<float> control_noise(0,100); 
    std::normal_distribution<float> measurement_noise(0,0.1); 
    // initialize state, measurement, and control with random noise
    state[0] = 3;
    for (int i = 0; i < N; i++) {
        control[i] = 16000 + control_noise(generator);
        measurement[i] = state[i] + measurement_noise(generator);
    }
    bayesianFilter(state, measurement, control, dt, target_rate);
    for(int i = 0; i <= N; i++){
        std::cout << "Time: " << i*dt << "s | Descent Rate: " << state[i] << "m/s" << std::endl;
    }
    return 0;
}
