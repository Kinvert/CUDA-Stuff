#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

struct DataPoint {
    std::vector<float> features;
    float target;
};

struct DistTarget {
    float distance;
    float target;
    
    __host__ __device__ bool operator<(const DistTarget& other) const {
        return distance < other.distance;
    }
};

// CUDA kernel to calculate distances and create distance-target pairs
__global__ void calculate_distances_and_targets(float* query, float* train_data, float* train_targets,
                                               DistTarget* dist_targets, int num_features, int num_train_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_train_points) {
        float sum = 0.0f;
        for (int f = 0; f < num_features; f++) {
            float diff = query[f] - train_data[idx * num_features + f];
            sum += diff * diff;
        }
        dist_targets[idx].distance = sqrtf(sum);
        dist_targets[idx].target = train_targets[idx];
    }
}

// CUDA kernel to sum the first k targets for averaging
__global__ void sum_k_targets(DistTarget* sorted_dist_targets, float* result, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {  // Only one thread does this
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += sorted_dist_targets[i].target;
        }
        *result = sum / k;
    }
}

std::vector<std::vector<float>> load_csv(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        data.push_back(row);
    }
    
    return data;
}

class ParallelCudaKNN {
private:
    std::vector<DataPoint> training_data;
    int k;
    float* d_train_data;
    float* d_train_targets;
    float* d_query;
    DistTarget* d_dist_targets;
    float* d_result;
    int num_features;
    
public:
    ParallelCudaKNN(int k_value) : k(k_value) {}
    
    ~ParallelCudaKNN() {
        if (d_train_data) cudaFree(d_train_data);
        if (d_train_targets) cudaFree(d_train_targets);
        if (d_query) cudaFree(d_query);
        if (d_dist_targets) cudaFree(d_dist_targets);
        if (d_result) cudaFree(d_result);
    }
    
    void fit(const std::vector<DataPoint>& data) {
        training_data = data;
        num_features = data[0].features.size();
        int num_train_points = data.size();
        
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_train_data, num_train_points * num_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_train_targets, num_train_points * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_query, num_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dist_targets, num_train_points * sizeof(DistTarget)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        
        // Flatten training data and copy to GPU
        std::vector<float> flat_data;
        std::vector<float> targets;
        
        for (const auto& point : data) {
            for (float f : point.features) {
                flat_data.push_back(f);
            }
            targets.push_back(point.target);
        }
        
        CUDA_CHECK(cudaMemcpy(d_train_data, flat_data.data(), 
                             flat_data.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_train_targets, targets.data(), 
                             targets.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    float predict(const std::vector<float>& query) {
        int num_train_points = training_data.size();
        
        // Copy query to GPU
        CUDA_CHECK(cudaMemcpy(d_query, query.data(), num_features * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel to calculate distances and prepare distance-target pairs
        int block_size = 256;
        int grid_size = (num_train_points + block_size - 1) / block_size;
        
        calculate_distances_and_targets<<<grid_size, block_size>>>(d_query, d_train_data, d_train_targets,
                                                                  d_dist_targets, num_features, num_train_points);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Sort using Thrust (parallel GPU sort)
        thrust::device_ptr<DistTarget> thrust_ptr(d_dist_targets);
        thrust::sort(thrust_ptr, thrust_ptr + num_train_points);
        
        // Sum and average the k nearest targets on GPU
        sum_k_targets<<<1, 1>>>(d_dist_targets, d_result, k);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back to CPU
        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        
        return result;
    }
};

float calculate_mse(const std::vector<float>& predictions, const std::vector<float>& actual) {
    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); i++) {
        float diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    return sum / predictions.size();
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Load concrete strength data from CSV
    std::vector<std::vector<float>> raw_data = load_csv("../../../../zData/Concrete/concrete.csv");
    
    // Convert to DataPoint format
    std::vector<DataPoint> dataset;
    for (const auto& row : raw_data) {
        DataPoint point;
        point.features = std::vector<float>(row.begin(), row.end() - 1);
        point.target = row.back();
        dataset.push_back(point);
    }
    
    // Shuffle data with fixed seed for deterministic results
    std::mt19937 g(42);
    std::shuffle(dataset.begin(), dataset.end(), g);
    
    // Split 80/20
    size_t split_idx = (size_t)(dataset.size() * 0.8);
    std::vector<DataPoint> train_data(dataset.begin(), dataset.begin() + split_idx);
    std::vector<DataPoint> test_data(dataset.begin() + split_idx, dataset.end());
    
    std::cout << "Training samples: " << train_data.size() << std::endl;
    std::cout << "Test samples: " << test_data.size() << std::endl;
    
    // Create and train parallel CUDA KNN
    ParallelCudaKNN knn(3); // k=3
    knn.fit(train_data);
    
    // Make predictions on test set
    std::vector<float> predictions;
    std::vector<float> actual;
    
    for (const auto& test_point : test_data) {
        float pred = knn.predict(test_point.features);
        predictions.push_back(pred);
        actual.push_back(test_point.target);
    }
    
    // Calculate MSE
    float mse = calculate_mse(predictions, actual);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Parallel CUDA Implementation" << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    std::cout << "Test MSE: " << mse << std::endl;
    
    // Show some predictions vs actual
    std::cout << "\nSample predictions:" << std::endl;
    std::cout << "Predicted\tActual" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, predictions.size()); i++) {
        std::cout << predictions[i] << "\t\t" << actual[i] << std::endl;
    }
    
    return 0;
}

