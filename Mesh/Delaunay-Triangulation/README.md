# Delaunay Triangulation

## 01

![01 Results](01-Result.png?raw=true "01 Result")

## 04

This one exports a CSV so we can do other mesh algorithms after loading this as the initial mesh.

delaynay_edges.csv is where the edges are stored.

## ChatGPT Other Ideas

ChatGPT:
There are several algorithms that can be used to optimize the triangulation of a set of points, some of them are:

- Lloyd's relaxation algorithm: This algorithm iteratively moves each point to the centroid of its surrounding triangles, creating a more uniform triangulation.

- Chew's second algorithm: This algorithm uses a global optimization approach to improve the triangulation by minimizing the circumcircle criterion.

- Incremental insertion algorithm: This algorithm starts with a small number of points and successively inserts new points into the triangulation, keeping it Delaunay-compliant.

- Divide and conquer algorithm: This algorithm recursively subdivides the point set and triangulates each sub-problem.

- Bowyer-Watson algorithm: This is a incremental algorithm that insert point in the triangulation and then repair it.

- Advancing front technique: This algorithm uses a moving front to maintain the Delaunay triangulation as new points are inserted.

- Alpha shapes : This algorithm generalizes the Delaunay triangulation to a family of shapes called alpha shapes, which can be used to represent point clouds with holes and boundaries.

- Voronoi diagram: This algorithm is closely related to Delaunay triangulation and can be used to optimize it. It partitions a plane into regions based on the distance to a set of points, and the Delaunay triangulation can be obtained as the dual graph of the Voronoi diagram.

Keep in mind that Delaunay triangulation is an NP-hard problem, so it is difficult to find a perfect solution. In practice, a good trade-off between quality and computational complexity must be found.
