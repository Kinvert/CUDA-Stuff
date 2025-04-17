// Written by Claude 3.7 Sonnet Apr 16 2025
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>

// Define a structure for grid points/nodes
struct Point {
    int x, y;
    
    Point(int x = 0, int y = 0) : x(x), y(y) {}
    
    // Equality operator
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    
    // Less than operator - needed for priority queue comparisons
    bool operator<(const Point& other) const {
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

// A* algorithm implementation
std::vector<Point> findPath(const std::vector<std::vector<char>>& grid, Point start, Point goal) {
    int height = grid.size();
    int width = grid[0].size();
    
    // Define directions: up, right, down, left, and diagonals
    std::vector<Point> directions = {
        {0, -1}, {1, 0}, {0, 1}, {-1, 0}, 
        {1, -1}, {1, 1}, {-1, 1}, {-1, -1}
    };
    
    // Priority queue for open set
    // We use pair<float, Point> where float is the f-score (lower is better)
    std::priority_queue<std::pair<float, Point>, 
                       std::vector<std::pair<float, Point>>,
                       std::greater<std::pair<float, Point>>> openSet;
    
    // Maps for tracking various scores
    std::unordered_map<Point, float, PointHash> gScore; // Cost from start to node
    std::unordered_map<Point, float, PointHash> fScore; // Estimated total cost from start to goal through node
    std::unordered_map<Point, Point, PointHash> cameFrom; // For reconstructing path
    std::unordered_map<Point, bool, PointHash> closedSet; // Set of evaluated nodes
    
    // Initialize the start node
    gScore[start] = 0;
    // Heuristic: Euclidean distance
    fScore[start] = std::sqrt(std::pow(goal.x - start.x, 2) + std::pow(goal.y - start.y, 2));
    openSet.push({fScore[start], start});
    
    while (!openSet.empty()) {
        // Get the node with lowest f-score
        Point current = openSet.top().second;
        openSet.pop();
        
        // If we've reached the goal, reconstruct and return the path
        if (current == goal) {
            std::vector<Point> path;
            while (!(current == start)) {
                path.push_back(current);
                current = cameFrom[current];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        // Skip if we've already processed this node
        if (closedSet[current]) continue;
        closedSet[current] = true;
        
        // Check all neighboring cells
        for (const auto& dir : directions) {
            Point neighbor{current.x + dir.x, current.y + dir.y};
            
            // Skip if out of bounds or is a wall
            if (neighbor.x < 0 || neighbor.x >= width || 
                neighbor.y < 0 || neighbor.y >= height || 
                grid[neighbor.y][neighbor.x] == '#' ||
                closedSet[neighbor]) {
                continue;
            }
            
            // Calculate tentative g-score
            float tentativeGScore = gScore[current] + 
                (dir.x != 0 && dir.y != 0 ? 1.414f : 1.0f); // Diagonal movement costs more
            
            // If we've found a better path to this neighbor
            if (!gScore.count(neighbor) || tentativeGScore < gScore[neighbor]) {
                // Update the path
                cameFrom[neighbor] = current;
                gScore[neighbor] = tentativeGScore;
                fScore[neighbor] = gScore[neighbor] + 
                    std::sqrt(std::pow(goal.x - neighbor.x, 2) + std::pow(goal.y - neighbor.y, 2));
                openSet.push({fScore[neighbor], neighbor});
            }
        }
    }
    
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
    
    // Find and print the path
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
