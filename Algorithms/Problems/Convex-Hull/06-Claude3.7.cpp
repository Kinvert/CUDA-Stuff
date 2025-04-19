#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

// A simple Point structure to represent 2D points
struct Point {
    int x, y;
    
    // Constructor
    Point(int _x = 0, int _y = 0) : x(_x), y(_y) {}
    
    // Operator overloading for comparing points
    bool operator<(const Point& p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
    
    bool operator==(const Point& p) const {
        return x == p.x && y == p.y;
    }
};

// Global point needed for sorting points based on polar angle
Point pivot;

// Calculate the square of distance between two points
int distSq(const Point& p1, const Point& p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

// Determine the orientation of triplet (p, q, r)
// Returns:
// 0 --> Collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(const Point& p, const Point& q, const Point& r) {
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    
    if (val == 0) return 0;         // Collinear
    return (val > 0) ? 1 : 2;       // Clockwise or Counterclockwise
}

// Compare function used for sorting points based on polar angle
// with respect to the pivot point
bool polarAngleCompare(const Point& p1, const Point& p2) {
    // Find orientation
    int orient = orientation(pivot, p1, p2);
    
    if (orient == 0) {
        // If collinear, choose the point that is farther from pivot
        return distSq(pivot, p1) < distSq(pivot, p2);
    }
    
    // If not collinear, return true if counterclockwise orientation
    return orient == 2;
}

// Print the convex hull to console
void printConvexHull(const std::vector<Point>& hull) {
    std::cout << "Convex Hull Points:" << std::endl;
    for (size_t i = 0; i < hull.size(); i++) {
        std::cout << "(" << hull[i].x << ", " << hull[i].y << ")" << std::endl;
    }
}

// Function to check if a point is in the hull
bool isInHull(const Point& p, const std::vector<Point>& hull) {
    for (const auto& hp : hull) {
        if (p.x == hp.x && p.y == hp.y) {
            return true;
        }
    }
    return false;
}

// Function to check if a point lies on a line segment
bool isOnLine(const Point& p, const Point& start, const Point& end) {
    // Check if point p is on the line segment from start to end
    if (orientation(start, p, end) != 0) return false;
    
    // Check if p is within the bounding box of the line segment
    return p.x >= std::min(start.x, end.x) && p.x <= std::max(start.x, end.x) &&
           p.y >= std::min(start.y, end.y) && p.y <= std::max(start.y, end.y);
}

// Function to create and display an ASCII visualization of points and hull
void displayASCII(const std::vector<Point>& points, const std::vector<Point>& hull, int size = 15) {
    if (points.empty()) return;
    
    // Find the bounds of the points
    int min_x = points[0].x, max_x = points[0].x;
    int min_y = points[0].y, max_y = points[0].y;
    
    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    
    // Add some padding
    min_x -= 1; max_x += 1;
    min_y -= 1; max_y += 1;
    
    // Make sure we don't exceed the size limit
    int width = std::min(size, max_x - min_x + 1);
    int height = std::min(size, max_y - min_y + 1);
    
    // Create the grid
    std::vector<std::string> grid(height, std::string(width, ' '));
    
    // Scale factors to map coordinates to grid indices
    double scale_x = static_cast<double>(width - 1) / (max_x - min_x);
    double scale_y = static_cast<double>(height - 1) / (max_y - min_y);
    
    // Function to map real coordinates to grid indices
    auto mapToGrid = [&](const Point& p) -> std::pair<int, int> {
        int grid_x = static_cast<int>((p.x - min_x) * scale_x);
        int grid_y = height - 1 - static_cast<int>((p.y - min_y) * scale_y); // Invert y for display
        return {grid_x, grid_y};
    };
    
    // Draw hull lines
    if (hull.size() >= 3) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Map grid coordinates back to real coordinates
                Point p(
                    static_cast<int>(min_x + x / scale_x),
                    static_cast<int>(min_y + (height - 1 - y) / scale_y)
                );
                
                // Check if this point is on any hull edge
                bool onHullEdge = false;
                for (size_t i = 0; i < hull.size(); i++) {
                    const Point& start = hull[i];
                    const Point& end = hull[(i + 1) % hull.size()];
                    
                    if (isOnLine(p, start, end)) {
                        onHullEdge = true;
                        break;
                    }
                }
                
                if (onHullEdge) {
                    grid[y][x] = '#';
                }
            }
        }
    }
    
    // Draw points
    for (const auto& p : points) {
        std::pair<int, int> gridPos = mapToGrid(p);
        int grid_x = gridPos.first;
        int grid_y = gridPos.second;
        
        // Make sure we're in bounds
        if (grid_x >= 0 && grid_x < width && grid_y >= 0 && grid_y < height) {
            // Use different symbols for hull vs non-hull points
            if (isInHull(p, hull)) {
                grid[grid_y][grid_x] = 'H'; // Hull point
            } else {
                grid[grid_y][grid_x] = 'o'; // Regular point
            }
        }
    }
    
    // Display the grid
    std::cout << "\nASCII Visualization (H = Hull Point, o = Regular Point, # = Hull Edge):\n";
    std::cout << std::string(width + 2, '-') << std::endl;
    for (const auto& row : grid) {
        std::cout << "|" << row << "|" << std::endl;
    }
    std::cout << std::string(width + 2, '-') << std::endl;
}

// The main function to find the convex hull of a set of points
std::vector<Point> convexHull(std::vector<Point> points) {
    int n = points.size();
    
    // If less than 3 points, cannot form a convex hull
    if (n < 3) {
        std::cout << "Convex hull not possible with less than 3 points!" << std::endl;
        return points;
    }
    
    // Find the bottommost point (or leftmost if tie)
    int min_y_idx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y < points[min_y_idx].y || 
            (points[i].y == points[min_y_idx].y && points[i].x < points[min_y_idx].x)) {
            min_y_idx = i;
        }
    }
    
    // Place the bottommost point at first position
    std::swap(points[0], points[min_y_idx]);
    
    // Set the pivot for sorting the remaining points
    pivot = points[0];
    
    // Sort the remaining points based on polar angle with respect to pivot
    std::sort(points.begin() + 1, points.end(), polarAngleCompare);
    
    // Remove collinear points (keep only the farthest)
    int m = 1;  // Initialize size of modified array
    for (int i = 1; i < n; i++) {
        // Keep removing while the angle formed by points pivot, m-1, and i makes a non-left turn
        while (i < n - 1 && orientation(pivot, points[i], points[i + 1]) == 0) {
            i++;
        }
        points[m] = points[i];
        m++;
    }
    
    // If modified array has less than 3 points, cannot form a convex hull
    if (m < 3) {
        std::cout << "Convex hull not possible!" << std::endl;
        return {};
    }
    
    // Create a stack and push the first three points
    std::vector<Point> hull;
    hull.push_back(points[0]);
    hull.push_back(points[1]);
    hull.push_back(points[2]);
    
    // Process remaining points
    for (int i = 3; i < m; i++) {
        // Keep removing top while the angle formed by points next-to-top, top, and points[i] makes a non-left turn
        while (hull.size() > 1 && orientation(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) != 2) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }
    
    return hull;
}

int main() {
    // Example usage with more points for better visualization
    std::vector<Point> points = {
        {0, 0}, {1, 1}, {2, 2}, {4, 4}, {0, 3}, 
        {1, 2}, {3, 1}, {3, 3}, {2, 1}, {1, 0},
        {2, 3}, {4, 2}, {0, 1}, {3, 0}, {2, 4}
    };
    
    std::cout << "Original Points:" << std::endl;
    for (const auto& p : points) {
        std::cout << "(" << p.x << ", " << p.y << ")" << std::endl;
    }
    
    // Calculate the convex hull
    std::vector<Point> hull = convexHull(points);
    
    // Print the result
    printConvexHull(hull);
    
    // Display ASCII visualization
    displayASCII(points, hull);
    
    return 0;
}
