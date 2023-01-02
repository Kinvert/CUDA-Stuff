// ChatGPT Dec 15 Version wrote this
#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to perform bubble sort
void bubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to print the array
void printArray(int arr[], int n)
{
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main()
{
    // Generate a random array of size 1000
    const int n = 1000;
    int arr[n];
    srand(time(0));
    for (int i = 0; i < n; i++)
        arr[i] = rand();

    // Measure the time taken by bubble sort
    clock_t start = clock();
    bubbleSort(arr, n);
    clock_t end = clock();

    // Print the sorted array and the time taken
    printArray(arr, n);
    std::cout << "Time taken: " << (end - start) / (double)CLOCKS_PER_SEC << " seconds" << std::endl;

    return 0;
}

