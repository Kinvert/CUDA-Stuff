// Written by Claude 3.7 Sonnet Apr 16 2025
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

// Define a structure for grid points/nodes
struct Point {
    int x, y;
    
    // Constructor for host code
    Point(int x = 0, int y = 0) : x(x), y(y) {}
    
    // Equality operator
    __host__ __device__ bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    
    // Less than operator - needed for priority queue comparisons
    __host__ __device__ bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
};

// Custom hash function for using Point as a key in unordered_map
struct PointHash {
    size_t operator()(const Point& p) const {
        return std::hash<int>()(p.x) ^ std::hash<int>()(p.y);
    }
};

// Direction vectors for neighbors (up, right, down, left, and diagonals)
const int NUM_DIRECTIONS = 8;
__constant__ int directions[NUM_DIRECTIONS][2];

// Initialize directions on device
void initializeDirections() {
    int hostDirections[NUM_DIRECTIONS][2] = {
        {0, -1}, {1, 0}, {0, 1}, {-1, 0},
        {1, -1}, {1, 1}, {-1, 1}, {-1, -1}
    };
    cudaMemcpyToSymbol(directions, hostDirections, NUM_DIRECTIONS * 2 * sizeof(int));
}

// CUDA kernel to calculate heuristic values for all points
__global__ void calculateHeuristic(float* heuristicMap, Point goal, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        heuristicMap[idx] = sqrtf(powf(goal.x - col, 2) + powf(goal.y - row, 2));
    }
}

// A* algorithm implementation (host side)
std::vector<Point> findPath(const std::vector<std::vector<char>>& grid, Point start, Point goal) {
    int height = grid.size();
    int width = grid[0].size();
    
    // Initialize CUDA directions
    initializeDirections();
    
    // Create a flat representation of the grid for CUDA
    char* hostGrid = new char[width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            hostGrid[y * width + x] = grid[y][x];
        }
    }
    
    // Allocate device memory for grid
    char* deviceGrid;
    cudaMalloc(&deviceGrid, width * height * sizeof(char));
    cudaMemcpy(deviceGrid, hostGrid, width * height * sizeof(char), cudaMemcpyHostToDevice);
    
    // Allocate device memory for heuristic values
    float* deviceHeuristic;
    cudaMalloc(&deviceHeuristic, width * height * sizeof(float));
    
    // Calculate heuristic values using CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    calculateHeuristic<<<gridSize, blockSize>>>(deviceHeuristic, goal, width, height);
    cudaDeviceSynchronize();
    
    // Copy heuristic values back to host
    float* hostHeuristic = new float[width * height];
    cudaMemcpy(hostHeuristic, deviceHeuristic, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Rest of A* algorithm on CPU (similar to original code)
    std::priority_queue<std::pair<float, Point>, 
                       std::vector<std::pair<float, Point>>,
                       std::greater<std::pair<float, Point>>> openSet;
    
    std::unordered_map<Point, float, PointHash> gScore;
    std::unordered_map<Point, float, PointHash> fScore;
    std::unordered_map<Point, Point, PointHash> cameFrom;
    std::unordered_map<Point, bool, PointHash> closedSet;
    
    // Initialize the start node
    gScore[start] = 0;
    fScore[start] = hostHeuristic[start.y * width + start.x];
    openSet.push({fScore[start], start});
    
    // Create host copy of directions for CPU part
    Point hostDirections[NUM_DIRECTIONS];
    cudaMemcpyFromSymbol(hostDirections, directions, NUM_DIRECTIONS * sizeof(Point));
    
    while (!openSet.empty()) {
        Point current = openSet.top().second;
        openSet.pop();
        
        if (current == goal) {
            std::vector<Point> path;
            while (!(current == start)) {
                path.push_back(current);
                current = cameFrom[current];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            
            // Clean up
            delete[] hostGrid;
            delete[] hostHeuristic;
            cudaFree(deviceGrid);
            cudaFree(deviceHeuristic);
            
            return path;
        }
        
        if (closedSet[current]) continue;
        closedSet[current] = true;
        
        for (int i = 0; i < NUM_DIRECTIONS; i++) {
            const Point& dir = hostDirections[i];
            Point neighbor{current.x + dir.x, current.y + dir.y};
            
            if (neighbor.x < 0 || neighbor.x >= width || 
                neighbor.y < 0 || neighbor.y >= height || 
                hostGrid[neighbor.y * width + neighbor.x] == '#' ||
                closedSet[neighbor]) {
                continue;
            }
            
            float tentativeGScore = gScore[current] + 
                (dir.x != 0 && dir.y != 0 ? 1.414f : 1.0f);
            
            if (!gScore.count(neighbor) || tentativeGScore < gScore[neighbor]) {
                cameFrom[neighbor] = current;
                gScore[neighbor] = tentativeGScore;
                fScore[neighbor] = gScore[neighbor] + hostHeuristic[neighbor.y * width + neighbor.x];
                openSet.push({fScore[neighbor], neighbor});
            }
        }
    }
    
    // Clean up
    delete[] hostGrid;
    delete[] hostHeuristic;
    cudaFree(deviceGrid);
    cudaFree(deviceHeuristic);
    
    // No path found
    return {};
}

// Function to print the grid with the path
void printGridWithPath(std::vector<std::vector<char>> grid, const std::vector<Point>& path) {
    // Mark the path on the grid
    for (const auto& point : path) {
        if (grid[point.y][point.x] != 'S' && grid[point.y][point.x] != 'G') {
            grid[point.y][point.x] = '*';
        }
    }
    
    // Print the grid
    std::cout << "Grid with Path:\n";
    for (const auto& row : grid) {
        for (char cell : row) {
            if (cell == '.') { // I added this so it prints more clearly
                cell = ' ';
            }
            std::cout << cell << ' ';
        }
        std::cout << '\n';
    }
}

// CUDA error checking helper
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
inline void __checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Define the grid map ('.' is open space, '#' is wall, 'S' is start, 'G' is goal)
    std::vector<std::vector<char>> grid = {
        {'.', '.', '.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '#', '#', '#', '#', '#', '.', '#', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '#', '.', '.'},
        {'.', '#', '#', '#', '#', '.', '#', '#', '.', '.'},
        {'.', '.', '.', '.', '#', '.', '.', '.', '.', '.'},
        {'#', '#', '#', '.', '#', '#', '#', '#', '#', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '#', '#', '#', '#', '#', '#', '#', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.', '.', '.'}
    };
    
    // Define start and goal points
    Point start{0, 0};
    Point goal{9, 9};
    
    // Mark start and goal on the grid
    grid[start.y][start.x] = 'S';
    grid[goal.y][goal.x] = 'G';
    
    // Print the initial grid
    std::cout << "Initial Grid:\n";
    for (const auto& row : grid) {
        for (char cell : row) {
            std::cout << cell << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    
    // Find and print the path using CUDA-accelerated A* algorithm
    std::vector<Point> path = findPath(grid, start, goal);
    
    if (path.empty()) {
        std::cout << "No path found!\n";
    } else {
        std::cout << "Path found! Path length: " << path.size() << '\n';
        std::cout << "Path coordinates: ";
        for (const auto& point : path) {
            std::cout << "(" << point.x << "," << point.y << ") ";
        }
        std::cout << "\n\n";
        
        printGridWithPath(grid, path);
    }
    
    return 0;
}
