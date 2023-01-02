// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <algorithm>

// define the input data size
const int DATA_SIZE = 100;

// define the threshold for filtering
const int THRESHOLD = 50;

// function to generate input data
void generateData(int* data) {
  for (int i = 0; i < DATA_SIZE; i++) {
    data[i] = rand() % 100; // generate a random number between 0 and 99
  }
}

int main() {
  // allocate memory for the input data and the filtered data
  int* data = new int[DATA_SIZE];
  int* filtered = new int[DATA_SIZE];

  // generate input data
  generateData(data);

  // filter the data
  int numFiltered = 0;
  for (int i = 0; i < DATA_SIZE; i++) {
    if (data[i] > THRESHOLD) {
      filtered[numFiltered++] = data[i];
    }
  }

  // print the filtered data
  for (int i = 0; i < numFiltered; i++) {
    std::cout << filtered[i] << " ";
  }

  // free memory
  delete[] data;
  delete[] filtered;

  return 0;
}

