# Bubble Sort

**Note: This was written by ChatGPT**

Bubble sort is a simple sorting algorithm that repeatedly iterates through the list of elements, compares adjacent elements, and swaps them if they are in the wrong order. It is called bubble sort because the smaller elements "bubble" to the top of the list. The algorithm is inefficient for large lists, but it can be useful for small lists or as a learning exercise.

## Algorithm

The bubble sort algorithm can be described using the following pseudocode:

```cpp
procedure bubbleSort(A: list of sortable items)
    n = length(A)
    repeat
        swapped = false
        for i = 1 to n-1 inclusive do
            if A[i-1] > A[i] then
                swap(A[i-1], A[i])
                swapped = true
            end if
        end for
    until not swapped
end procedure
```

This algorithm compares each pair of adjacent elements and swaps them if they are in the wrong order. It repeats this process until no more swaps are needed, indicating that the list is sorted. The time complexity of bubble sort is $O(n^2)$ in the worst case, making it inefficient for large lists.

**Note: ChatGPT also tried LaTeX**

```math
\begin{algorithmic}[1]
\Procedure{bubbleSort}{$A$}
    \State $n \gets \text{length}(A)$
    \Repeat
        \State swapped $\gets$ \textbf{false}
        \For{$i \gets 1$ to $n-1$}
            \If{$A[i-1] > A[i]$}
                \State $\text{swap}(A[i-1], A[i])$
                \State swapped $\gets$ \textbf{true}
            \EndIf
        \EndFor
    \Until{not swapped}
\EndProcedure
\end{algorithmic}
```

## C++ Implementation

Here is a C++ implementation of the bubble sort algorithm:

```cpp
void bubbleSortCpp(int arr[], int n)
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
```

This function iterates through the list of elements and compares adjacent elements, swapping them if they are in the wrong order. It repeats this process until the list is sorted.

## CUDA Implementation

Bubble sort can also be implemented on the GPU using CUDA. Here is a naive implementation of bubble sort in CUDA:

```cpp
__global__ void bubbleSortKernel(int* arr, int n)
{
    // Determine the position of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread is within the bounds of the array
    if (i < n)
    {
        // Perform bubble sort on the current element
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
```

This kernel function is similar to the C++ implementation, but it launches multiple threads to perform bubble sort in parallel on the GPU. However, this implementation may not be efficient due to the high number of global memory accesses.

## Optimized CUDA Implementation

To optimize the CUDA implementation of bubble sort, we can use shared memory to reduce the number of global memory accesses. Shared memory is a fast memory location that is shared among threads in a block and can be used to store data that will be accessed frequently by multiple threads. By using shared memory to store the data being sorted, we can reduce the number of global memory accesses, which can significantly improve the performance of the algorithm.

Here is an optimized implementation of bubble sort in CUDA using shared memory:

```cpp
__global__ void bubbleSortKernel(int* arr, int n)
{
    // Allocate shared memory for the current block
    extern __shared__ int shared[];

    // Determine the position of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread is within the bounds of the array
    if (i < n)
    {
        // Load the current element into shared memory
        shared[threadIdx.x] = arr[i];
        __syncthreads();

        // Perform bubble sort on the current element using shared memory
        for (int j = 0; j < n - i - 1; j++)
        {
            if (shared[j] > shared[j + 1])
            {
                int temp = shared[j];
                shared[j] = shared[j + 1];
                shared[j + 1] = temp;
            }
        }

        // Store the sorted element back in global memory
        arr[i] = shared[threadIdx.x];
    }
}

```

This kernel function is similar to the C++ implementation, but it uses shared memory to reduce the number of global memory accesses. It loads the current element into shared memory, performs bubble sort on the element using shared memory, and then stores the sorted element back in global memory.
