# Insertion Sort

**Note: This was written by ChatGPT**

## Introduction

Insertion sort is a simple sorting algorithm that works by iterating through the elements of an array and inserting each element in its correct position in the sorted portion of the array. It is a comparison-based algorithm, meaning it compares elements to determine their order.

Insertion sort has a time complexity of O(n^2) in the worst case, making it less efficient than other algorithms such as quicksort or merge sort. However, it has a number of advantages, including simplicity and a low overhead, which make it a good choice for sorting small arrays or partially sorted arrays.

Here is the pseudocode for the insertion sort algorithm:

```cpp
INSERTION-SORT(A)
1  for i = 2 to A.length
2      key = A[i]
3      j = i - 1
4      while j > 0 and A[j] > key
5          A[j + 1] = A[j]
6          j = j - 1
7      A[j + 1] = key
```

The algorithm works by iterating through the elements of the array from left to right (line 1). For each element, it compares it with the elements to its left and inserts it in its correct position (lines 3-7).

Here is the same algorithm in LaTeX format:

$$
\begin{algorithm}
\caption{Insertion Sort}
\label{alg:insertion-sort}
\begin{algorithmic}[1]
\Procedure{INSERTION-SORT}{$A$}
\For{$i \gets 2$ \textbf{to} $A.length$}
    \State $key \gets A[i]$
    \State $j \gets i - 1$
    \While{$j > 0$ and $A[j] > key$}
        \State $A[j + 1] \gets A[j]$
        \State $j \gets j - 1$
    \EndWhile
    \State $A[j + 1] \gets key$
\EndFor
\EndProcedure
\end{algorithmic}
\end{algorithm}
$$

## C++ Implementation

The C++ implementation of insertion sort is relatively straightforward. It consists of a single function, insertionSort, which takes an array data and its size size as input and sorts the array in ascending order.

The function works by iterating through the elements of the array from left to right (line 5). For each element, it compares it with the elements to its left and inserts it in its correct position (lines 7-11).

Here is a description of the C++ implementation of the insertion sort algorithm in the form of code snippets with explanations:

Copy code
// Function to perform insertion sort
void insertionSort(int *data, int size) {
    // Iterate through the elements of the array
    for (int i = 1; i < size; i++) {
        // Store the current element
        int key = data[i];

        // Compare the current element with the elements to its left
        int j = i - 1;
        while (j >= 0 && data[j] > key) {
            // Shift elements to the right to make room for the current element
            data[j + 1] = data[j];
            j--;
        }

        // Insert the current element in its correct position
        data[j + 1] = key;
    }
}
This code defines the insertionSort function, which takes an array data and its size size as input and sorts the array in ascending order.

The function works by iterating through the elements of the array from left to right (line 4). For each element, it compares it with the elements to its left and inserts it in its correct position (lines 6-10).

```cpp
int main() {
    // Initialize data
    const int size = 100;
    int data[size];
    for (int i = 0; i < size; i++) {
        data[i] = rand();
    }

    // Sort data
    insertionSort(data, size);

    // Print sorted data
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

This code initializes an array data of size 100 with random values (line 4). It then sorts the array using the insertionSort function (line 7) and prints the sorted array to the console (lines 10-12).

## CUDA

```cpp
__global__ void insertionSortKernel(int *data, int size) {
    // Each thread sorts one element
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the current thread is within the bounds of the array
    if (i < size) {
        // Store the current element
        int key = data[i];

        // Compare the current element with the elements to its left
        int j = i - 1;
        while (j >= 0 && data[j] > key) {
            // Shift elements to the right to make room for the current element
            data[j + 1] = data[j];
            j--;
        }

        // Insert the current element in its correct position
        data[j + 1] = key;
    }
}
```

This code defines the insertionSortKernel function, which is a CUDA kernel that takes an array data and its size size as input and sorts the array in ascending order.

The kernel works by launching one thread for each element of the array (line 3). Each thread compares the element it is responsible for with the elements to its left and inserts it in its correct position (lines 6-10).

```cpp
int main() {
    // Initialize data
    const int size = 1000;
    int data[size];
    for (int i = 0; i < size; i++) {
        data[i] = rand();
    }

    // Allocate memory on the device for the array
    int *deviceData;
    cudaMalloc((void **)&deviceData, size * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(deviceData, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Set the number of threads and blocks for the kernel
    const int numThreads = 256;
    const int numBlocks = (size + numThreads - 1) / numThreads;

    // Launch the kernel
    insertionSortKernel<<<numBlocks, numThreads>>>(deviceData, size);

    // Copy the sorted array from device to host
    cudaMemcpy(data, deviceData, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(deviceData);

    // Print sorted data
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

This code copies the sorted array from device to host memory (line 17) and frees the GPU memory that was allocated for the array (line 20). It then prints the sorted array to the console (lines 23-25).

Finally, the main function returns 0 to indicate that it has completed successfully (line 27).

## CUDA Optimized

```cpp
// Function to perform insertion sort
__global__ void insertionSortKernel(int *data, int size) {
    // Each thread block sorts a block of elements
    __shared__ int block[256];
```

This code defines the insertionSortKernel function, which is a CUDA kernel that takes an array data and its size size as input and sorts the array in ascending order.

The kernel works by launching one thread block for each block of elements to be sorted. Each thread block consists of blockDim.x threads, which sort a block of blockDim.x elements in parallel using a modified version of the insertion sort algorithm.

The kernel declares a shared memory array block that will be used to store the block of elements being sorted by each thread block (line 3).

```cpp
    // Load block of elements into shared memory
    int i = threadIdx.x;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < size) {
        block[i] = data[j];
    }
    __syncthreads();
```

This code loads a block of blockDim.x elements from the input array data into shared memory.

The threadIdx.x variable is a built-in variable that holds the unique 1D index of the current thread within the thread block. It ranges from 0 to blockDim.x - 1.

The blockIdx.x variable is another built-in variable that holds the unique 1D index of the thread block within the grid. It ranges from 0 to gridDim.x - 1.

The code uses these variables to determine the index j of the element in the input array that should be loaded by the current thread (line 2). It then uses the threadIdx.x variable to determine the index i of the element in the shared memory array block where the element should be stored (line 3).

Finally, the code uses an if statement to ensure that the index j is within the bounds of the input array (line 4). If the index is valid, the element at index j in the input array is copied to index i in the shared memory array (line 5).

The __syncthreads() function is a built-in function that synchronizes all threads in the thread block. It ensures that all threads have completed the code block before any thread can proceed to the next code block. In this case, it is used to ensure that all elements have been loaded into shared memory before the threads start sorting them.

```cpp
    // Sort block of elements in shared memory
    for (int k = 1; k < blockDim.x; k *= 2) {
        i = (threadIdx.x + k) % blockDim.x;
        if (i > 0 && block[i - 1] > block[i]) {
            int temp = block[i - 1];
            block[i - 1] = block[i];
            block[i] = temp;
        }
        __syncthreads();
    }
```

This code performs an iterative sorting pass on the elements in shared memory using a binary tree sorting algorithm (lines 2-6).

The algorithm works by having each thread compare the element at its current index with the element at an offset index k (line 3). If the element at the offset index is smaller, the two elements are swapped (lines 4-6). The threads synchronize after each iteration to ensure that all elements have been compared (line 7).

```cpp
    // Store sorted block of elements in global memory
    i = threadIdx.x;
    j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < size) {
        data[j] = block[i];
    }
}
```

This code stores the sorted block of elements from shared memory back into the input array data (lines 2-5). It uses the threadIdx.x and blockIdx.x variables to determine the index of the current thread and block and the corresponding element in the input array.
