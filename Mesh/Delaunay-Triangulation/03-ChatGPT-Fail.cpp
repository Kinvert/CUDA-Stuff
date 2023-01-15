// Written by ChatGPT Jan 9 Version

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <utility>

// A simple point structure
struct Point
{
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

// Comparison function for sorting points by their x-coordinate
bool x_cmp(Point a, Point b) { return a.x < b.x; }

// Comparison function for sorting points by their y-coordinate
bool y_cmp(Point a, Point b) { return a.y < b.y; }

// Custom implementation of the cross product
double cross(Point O, Point A, Point B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

// Function to compute the Delaunay triangulation of a set of points
vector<pair<Point, Point>> delaunay(vector<Point> P)
{
    int n = P.size();

    // Sort the points by their x-coordinate
    sort(P.begin(), P.end(), x_cmp);

    // Build the lower convex hull
    vector<Point> L;
    for (int i = 0; i < n; i++)
    {
        while (L.size() >= 2 && cross(L[L.size() - 2], L[L.size() - 1], P[i]) <= 0) L.pop_back();
        L.push_back(P[i]);
    }

    // Build the upper convex hull
    vector<Point> U;
    for (int i = n - 1; i >= 0; i--)
    {
        while (U.size() >= 2 && cross(U[U.size() - 2], U[U.size() - 1], P[i]) <= 0) U.pop_back();
        U.push_back(P[i]);
    }

    // Remove the duplicate point at the end of each hull
    L.pop_back();
    U.pop_back();

    // Concatenate the lower and upper hulls to form the convex hull
    L.insert(L.end(), U.begin(), U.end());

    // Sort the points by their y-coordinate
    sort(P.begin(), P.end(), y_cmp);

    // Initialize the edge buffer
    vector<pair<Point, Point>> edges;

    // Iterate over the points and form the Delaunay triangles
    for (auto p : P)
    {
        // Find the next and previous point in the convex hull
        auto prev = lower_bound(L.begin(), L.end(), p, y_cmp) - 1;
        auto next = upper_bound(L.begin(), L.end(), p, y_cmp);

        // Form the Delaunay triangles
        if (next != L.end() && prev != L.end()) edges.push_back({*prev, *next});
        if (next != L.end() && prev != L.begin()) edges.push_back({*prev, *next});
    }

    return edges;
}

int main()
{
    // Generate some random points
    int numPoints = 20;
    vector<Point> points;
    for (int i = 0; i < numPoints; i++)
    {
        Point p(rand() % 280, rand() % 280);
        points.push_back(p);
    }

    // Compute the Delaunay triangulation
    vector<pair<Point, Point>> edges = delaunay(points);

    // Print the edges of the Delaunay triangles
    for (auto edge : edges)
        std::cout<<"("<<edge.first.x<<","<<edge.first.y<<"), ("<<edge.second.x<<","<<edge.second.y<<")"<<std::endl;

    return 0;
}
