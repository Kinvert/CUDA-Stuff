// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

const int N = 40; // 40 samples, with step size of 0.05s, this will give 2 seconds of simulation
const int WIDTH = 280;
const int HEIGHT = 280;
const int MOON_HEIGHT = 10;
const int LEM_SIZE = 20;
const int PIXEL_TO_METER = 50;
const int MAX_THRUST_LENGTH = 20;
const float GRAVITY = 1.625;
const int INITIAL_ALTITUDE = 4;

__global__ void bayesianFilter(float rate[], float altitude[], float measurement[], float control[], float dt, float target_rate) {
    int i = threadIdx.x;
    float error;
    float kp = 2.0;
    float throttle;
    measurement[i] += rate[i];
    error = target_rate - measurement[i];
    throttle = kp * error;
    if(throttle > 0.3 && throttle < 0.6) throttle = 0.6;
    else if(throttle > 1.0) throttle = 1.0;
    else throttle = 0.0;

    // prediction step
    rate[i+1] = rate[i] + (control[i] * throttle / 1500.0) * dt - GRAVITY * dt; // update the velocity with gravity
    altitude[i+1] = altitude[i] + rate[i+1] * dt;
    // correction step
    rate[i+1] = rate[i+1] + (measurement[i] - rate[i+1]) / 2.0;
    altitude[i+1] = altitude[i+1] + (altitude[i] - altitude[i+1]) / 2.0;
}

int main() {
    float rate[N+1];
    float altitude[N+1];
    float measurement[N];
    float control[N];
    float dt = 0.05;
    float target_rate = -2;
    std::default_random_engine generator;
    std::normal_distribution<float> control_noise(0,100); 
    std::normal_distribution<float> measurement_noise(0,0.1); 
    // initialize state, measurement, and control with random noise
    rate[0] = 0;
    altitude[0] = INITIAL_ALTITUDE;
    for (int i = 0; i < N; i++) {
        control[i] = 16000 + control_noise(generator);
        measurement[i] = rate[i] + measurement_noise(generator);
    }

    float *d_rate, *d_altitude, *d_measurement, *d_control;
    cudaMalloc(&d_rate, (N+1) * sizeof(float));
    cudaMalloc(&d_altitude, (N+1) * sizeof(float));
    cudaMalloc(&d_measurement, N * sizeof(float));
    cudaMalloc(&d_control, N * sizeof(float));
    cudaMemcpy(d_rate, rate, (N+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_altitude, altitude, (N+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurement, measurement, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_control, control, N * sizeof(float), cudaMemcpyHostToDevice);
    bayesianFilter<<<1, N>>>(d_rate, d_altitude, d_measurement, d_control, dt, target_rate);
    cudaMemcpy(rate, d_rate, (N+1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(altitude, d_altitude, (N+1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_rate);
    cudaFree(d_altitude);
    cudaFree(d_measurement);
    cudaFree(d_control);
    for (int i = 0; i < N; i++) {
        cv::Mat image(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::rectangle(image, cv::Rect(0, HEIGHT - MOON_HEIGHT, WIDTH, MOON_HEIGHT), cv::Scalar(128, 128, 128), -1); // Moon
        // draw the LEM
        int lem_y = HEIGHT - altitude[i] * PIXEL_TO_METER - LEM_SIZE;
        cv::Point lem_points[1][3];
        lem_points[0][0] = cv::Point(WIDTH/2, lem_y);
        lem_points[0][1] = cv::Point(WIDTH/2 - LEM_SIZE/2, lem_y + LEM_SIZE);
        lem_points[0][2] = cv::Point(WIDTH/2 + LEM_SIZE/2, lem_y + LEM_SIZE);
        const cv::Point* ppt[1] = {lem_points[0]};
        int npt[] = {3};
        cv::fillPoly(image, ppt, npt, 1, cv::Scalar(255, 255, 255));
        if(throttle > 0.6) {
            int thrust_length = MIN(MAX_THRUST_LENGTH, (int)(throttle * MAX_THRUST_LENGTH));
            cv::line(image, cv::Point(WIDTH/2, lem_y + LEM_SIZE), cv::Point(WIDTH/2, lem_y + LEM_SIZE + thrust_length), cv::Scalar(0, 0, 255), 5);
        }
        char str1[100];
        char str2[100];
        sprintf(str1,"current = %.3f", rate[i]); 
        sprintf(str2,"target   = %.3f", target_rate);
        cv::putText(image, str1, cv::Point(5,20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255));
        cv::putText(image, str2, cv::Point(5,40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255));
        cv::imshow("Lunar Lander", image);
        cv::waitKey(50);
        std::string img_path = std::to_string(i) + ".jpg";
        cv::imwrite(img_path, image);
    }
    return 0;
}
