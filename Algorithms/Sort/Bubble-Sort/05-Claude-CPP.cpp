#include <iostream>
#include <ctime>
#include <cstdlib>

void bubbleSort(int arr[], int n) {
  bool swapped = true;
  int i = 0;
  int j = 0;
  int temp;
  
  while (swapped) {
    swapped = false;
    j++;
    for (i = 0; i < n - j; i++) {
      if (arr[i] > arr[i + 1]) {
        temp = arr[i];
        arr[i] = arr[i + 1];  
        arr[i + 1] = temp;
        swapped = true;
      }
    }
  }
}

int main() {

  int n;
  std::cout << "Enter number of elements: ";
  std::cin >> n;
  
  int arr[n];
  
  // Generate random array
  srand(time(NULL));
  for(int i = 0; i < n; i++) {
    arr[i] = rand(); 
  }

  clock_t start = clock();
  
  bubbleSort(arr, n);
  
  clock_t end = clock();
  double timeTaken = double(end - start) / double(CLOCKS_PER_SEC);

  std::cout << "Sorted array: ";
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << " ";
  }
  
  std::cout << "\nTime taken: " << timeTaken << " seconds" << std::endl;

  return 0;
}
