#include <chrono>
#include <iostream>

void insertionSort(int *data, int size) {
    for (int i = 1; i < size; i++) {
        int key = data[i];
        int j = i - 1;
        while (j >= 0 && data[j] > key) {
            data[j + 1] = data[j];
            j--;
        }
        data[j + 1] = key;
    }
}

int main() {
    // Initialize data
    const int size = 100;
    int data[size];
    for (int i = 0; i < size; i++) {
        data[i] = rand();
    }

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();

    // Sort data
    insertionSort(data, size);

    // Print sorted data
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    // Calculate elapsed time
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print elapsed time
    std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;

    return 0;
}

