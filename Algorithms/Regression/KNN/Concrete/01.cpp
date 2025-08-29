#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <fstream>
#include <chrono>

struct DataPoint {
    std::vector<float> features;
    float target;
};

struct Neighbor {
    float distance;
    float target;
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

class SimpleKNN {
private:
    std::vector<DataPoint> training_data;
    int k;
    
public:
    SimpleKNN(int k_value) : k(k_value) {}
    
    void fit(const std::vector<DataPoint>& data) {
        training_data = data;
    }
    
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    
    float predict(const std::vector<float>& query) {
        std::vector<Neighbor> neighbors;
        
        // Calculate distances to all training points
        for (const auto& point : training_data) {
            float dist = euclidean_distance(query, point.features);
            neighbors.push_back({dist, point.target});
        }
        
        // Sort by distance and take k nearest
        std::sort(neighbors.begin(), neighbors.end());
        
        // Average the k nearest targets
        float sum = 0.0f;
        for (int i = 0; i < k && i < neighbors.size(); i++) {
            sum += neighbors[i].target;
        }
        
        return sum / k;
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

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Load concrete strength data from CSV
    std::vector<std::vector<float>> raw_data = load_csv("../../../../zData/Concrete/concrete.csv");
    
    // Convert to DataPoint format
    std::vector<DataPoint> dataset;
    for (const auto& row : raw_data) {
        DataPoint point;
        point.features = std::vector<float>(row.begin(), row.end() - 1); // All but last column
        point.target = row.back(); // Last column is target
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
    
    // Create and train KNN
    SimpleKNN knn(3); // k=3
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
    
    std::cout << "C++ Implementation" << std::endl;
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

