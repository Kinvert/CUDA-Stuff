// Written by ChatGPT Dec 15 Version
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// A pair used to store the coordinates of a cell in the grid
using Coord = std::pair<int, int>;

// A struct used to store the current state of a cell in the grid
struct Cell {
  // The coordinates of the cell
  Coord coord;
  // The distance from the start cell
  int g;
  // The estimated distance to the goal cell
  int h;
  // The total cost of the cell (g + h)
  int f;
  // The parent cell of the current cell (used to reconstruct the path)
  Cell *parent;
};

// A function to check if the given coordinates are within the bounds of the grid
bool inBounds(int x, int y, int width, int height) {
  return x >= 0 && x < width && y >= 0 && y < height;
}

// A function to check if the given cell is the goal cell
bool isGoal(const Cell &cell, const Coord &goal) {
  return cell.coord == goal;
}

// A function to calculate the Euclidean distance between two cells
int euclidean(const Cell &a, const Cell &b) {
  int dx = b.coord.first - a.coord.first;
  int dy = b.coord.second - a.coord.second;
  return sqrt(dx * dx + dy * dy);
}

// A function to calculate the Manhattan distance between two cells
//int manhattan(const Cell &a, const Cell &b) {
//  return abs(b.coord.first - a.coord.first) + abs(b.coord.second - a.coord.second);
//}
int manhattan(const Coord &a, const Coord &b) {
  return abs(b.first - a.first) + abs(b.second - a.second);
}

// A function to retrieve the neighbors of a given cell
std::vector<Cell> getNeighbors(const Cell &cell, int width, int height) {
  std::vector<Cell> neighbors;
  int x = cell.coord.first;
  int y = cell.coord.second;
  // Check the cell to the right
  if (inBounds(x + 1, y, width, height)) {
    neighbors.push_back({{x + 1, y}, cell.g + 1, 0, 0, nullptr});
  }
  // Check the cell to the left
  if (inBounds(x - 1, y, width, height)) {
    neighbors.push_back({{x - 1, y}, cell.g + 1, 0, 0, nullptr});
  }
  // Check the cell above
  if (inBounds(x, y + 1, width, height)) {
    neighbors.push_back({{x, y + 1}, cell.g + 1, 0, 0, nullptr});
  }
  // Check the cell below
  if (inBounds(x, y - 1, width, height)) {
    neighbors.push_back({{x, y - 1}, cell.g + 1, 0, 0, nullptr});
  }
  return neighbors;
}

// A function to retrieve the path from the start cell to the goal
std::vector<Coord> getPath(const Cell &goal) {
  std::vector<Coord> path;
  const Cell *current = &goal;
  while (current) {
    path.push_back(current->coord);
    current = current->parent;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

// The A* algorithm
std::vector<Coord> aStar(const Coord &start, const Coord &goal, int width, int height) {
  // A vector of open cells
  std::vector<Cell> open;
  // A vector of closed cells
  std::vector<Cell> closed;
  // Add the start cell to the open list
  //open.push_back({start, 0, manhattan({0, 0}, goal), manhattan({0, 0}, goal), nullptr});
  open.push_back({start, 0, manhattan(start, goal), manhattan(start, goal), nullptr});
  //Cell startCell {start, 0, 0, 0, nullptr};
  //Cell goalCell {goal, 0, 0, 0, nullptr};
  //open.push_back({start, 0, manhattan(startCell, goalCell), manhattan(startCell, goalCell), nullptr});
  // While there are cells in the open list
  while (!open.empty()) {
    // Find the cell with the lowest total cost (f)
    auto it = std::min_element(open.begin(), open.end(), [](const Cell &a, const Cell &b) {
      return a.f < b.f;
    });
    // Remove the cell from the open list
    Cell current = *it;
    open.erase(it);
    // Add the cell to the closed list
    closed.push_back(current);
    // If the current cell is the goal cell, return the path
    if (isGoal(current, goal)) {
      return getPath(current);
    }
    // Get the neighbors of the current cell
    std::vector<Cell> neighbors = getNeighbors(current, width, height);
    // For each neighbor of the current cell
    for (auto &neighbor : neighbors) {
      // If the neighbor is already in the closed list, skip it
      if (std::find(closed.begin(), closed.end(), neighbor) != closed.end()) {
        continue;
      }
      // Calculate the distance from the start cell to the neighbor
      neighbor.g = current.g + 1;
      // Calculate the estimated distance from the neighbor to the goal cell
      neighbor.h = manhattan(neighbor, goal);
      // Calculate the total cost of the neighbor
      neighbor.f = neighbor.g + neighbor.h;
      // Set the parent of the neighbor to the current cell
      neighbor.parent = &closed.back();
      // If the neighbor is not in the open list, add it
      if (std::find(open.begin(), open.end(), neighbor) == open.end()) {
        open.push_back(neighbor);
      }
    }
  }
  // If the open list is empty and the goal has not been reached, return an empty path
  return {};
}

int main() {
  int width = 5;
  int height = 5;
  // Find the path using A*
  std::vector<Coord> path = aStar({0, 0}, {4, 4}, width, height);
  // Print the path
  for (const auto &coord : path) {
    std::cout << coord.first << ", " << coord.second << std::endl;
  }
  return 0;
}

