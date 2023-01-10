// Written by ChatGPT Dec 15 Version

#include <iostream>
#include <vector>

// Evaluates a B-spline curve at a specific parameter value using DeBoor's algorithm
std::vector<double> evaluateBSpline(const std::vector<double>& knots, const std::vector<std::vector<double>>& controlPoints, int degree, double t)
{
    // Number of control points
    int numControlPoints = controlPoints.size();

    // Find the span of the curve at the parameter value
    int span = findSpan(numControlPoints - 1, degree, t, knots);

    // Create a temporary array to store the control points that will be used in the DeBoor algorithm
    std::vector<std::vector<double>> d(numControlPoints);
    for (int i = 0; i < numControlPoints; i++)
    {
        d[i].resize(degree + 1);
    }

    // Initialize the temporary array with the control points that are in the span
    for (int i = 0; i <= degree; i++)
    {
        d[i][0] = controlPoints[span - degree + i];
    }

    // Apply DeBoor's algorithm to compute the point on the curve
    for (int k = 1; k <= degree; k++)
    {
        for (int i = 0; i <= degree - k; i++)
        {
            double alpha = (t - knots[span - degree + i]) / (knots[i + span + 1] - knots[span - degree + i]);
            d[i][k] = (1.0 - alpha) * d[i][k - 1] + alpha * d[i + 1][k - 1];
        }
    }

    // Return the final point on the curve, which is the last element in the temporary array
    return d[0][degree];
}

// Finds the span of the curve at a specific parameter value
int findSpan(int numControlPoints, int degree, double t, const std::vector<double>& knots)
{
    if (t == knots[numControlPoints + 1])
    {
        return numControlPoints;
    }

    int low = degree;
    int high = numControlPoints;
    int mid = (low + high) / 2;

    while (t < knots[mid] || t >= knots[mid + 1])
    {
        if (t < knots[mid])
        {
            high = mid;
        }
        else
        {
            low = mid;
        }
        mid = (low + high) / 2;
    }

    return mid;
}

int main()
{
    // Test the B-spline curve evaluation
    std::vector<double> knots = { 0, 0, 0, 1, 2, 3, 3, 3 };
    std::vector<std::vector<double>> controlPoints = { { 0, 0 }, { 1, 1 }, { 2, 0 }, { 3, 1 } };
    int degree = 2;
    double t = 1.5;

    std::vector<double> point = evaluateBSpline(knots, controlPoints, degree, t);

    std::cout << "Point on B-spline curve: (" << point[0] << ", " << point[1] << ")" << std::endl;

    return 0;
}
