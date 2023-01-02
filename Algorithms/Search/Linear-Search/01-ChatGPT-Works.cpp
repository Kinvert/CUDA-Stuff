// ChatGPT Dec 15 Version wrote this First Try
#include <iostream>
#include <vector>

// Function to perform linear search
int linearSearch(std::vector<int>& arr, int target)
{
    // search for target in the array
    for (int i = 0; i < arr.size(); i++)
    {
        // if target is found, return its index
        if (arr[i] == target)
            return i;
    }

    // if target is not found, return -1
    return -1;
}

int main()
{
    std::vector<int> arr = { 4, 2, 6, 1, 3, 7, 8, 5 };
    int target = 5;

    int index = linearSearch(arr, target);

    if (index != -1)
        std::cout << "Target found at index: " << index << std::endl;
    else
        std::cout << "Target not found" << std::endl;

    return 0;
}
