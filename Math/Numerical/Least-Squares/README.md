# Least Squares

**Note: This was written by ChatGPT**

## Introduction

Least squares is a method used to fit a linear model to a set of data points by minimizing the residual sum of squares (RSS) between the predicted values and the actual values. The linear model is of the form y = w[0] + w[1] * x, where x is the input variable and y is the target variable. Mathematically, the RSS can be expressed as:

$$
RSS = \sum_{i=1}^{n} (y_i - y_i')^2
$$

Where $y_i' = w[0] + w[1] * x_i$ and $x_i$ and $y_i$ are the input and target values, respectively, for the $i^{th}$ data point. The goal is to find the values of the coefficients $w[0]$ and $w[1]$ that minimize the RSS.

## Algorithm

There are several methods that can be used to compute the least squares solution, including the QR decomposition, the SVD decomposition, and the normal equations method. Here, we will discuss the QR decomposition method, which involves decomposing the matrix X into an orthogonal matrix Q and an upper triangular matrix R such that X = Q * R.

To compute the QR decomposition, we can use the Gram-Schmidt process, which involves iterating over the columns of X and orthogonalizing them one by one. Specifically, for each column x in X, we compute the projection of x onto the subspace spanned by the previous columns, and subtract this projection from x to obtain the orthogonalized column q. We then normalize q and store it in the corresponding column of Q. The matrix R is then obtained by taking the inner products of X and Q.

The least squares solution can then be computed as follows:

$$
w = (X^T * X)^{-1} * X^T * y
$$

Where y is the vector of target values and X^T is the transpose of X.

## C++

```cpp
// Number of data points
int n = N;

// Data points
double X[N][2] = {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5},
                  {1, 6}, {1, 7}, {1, 8}, {1, 9}, {1, 10}};

// Target values
double y[N] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// Coefficients
double w[2] = {0, 0};
```

This initializes the number of data points n, the input data matrix X, the target values y, and the coefficients w. The input data matrix X consists of 10 data points, each with 2 features. The target values y are the corresponding values for each data point. The coefficients w are initialized to 0.

```cpp
// Solve least squares problem
double XtX[2][2] = {{0, 0}, {0, 0}};
double Xty[2] = {0, 0};

for (int i = 0; i < n; i++) {
  for (int j = 0; j < 2; j++) {
    XtX[j][0] += X[i][j] * X[i][0];
    XtX[j][1] += X[i][j] * X[i][1];
    Xty[j] += X[i][j] * y[i];
  }
}
```

This block of code computes the matrices XtX and Xty, which are used to solve the least squares problem. The matrix XtX is the outer product of the transpose of X with itself, and Xty is the outer product of the transpose of X with y. These matrices are computed using a double nested loop over the data points in X and the features of each data point.

```cpp
double det = XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0];
w[0] = (XtX[1][1] * Xty[0] - XtX[0][1] * Xty[1]) / det;
w[1] = (XtX[0][0] * Xty[1] - XtX[1][0] * Xty[0]) / det;
```

This block of code computes the least squares solution using the matrices XtX and Xty. It does this by first computing the determinant of XtX, and then using it to solve the linear system of equations XtX * w = Xty. The resulting values of w are the least squares solution.

```cpp
std::cout << "Coefficients: " << w[0] << " " << w[1] << std::endl;
```

This prints the least squares solution, which are the values of the coefficients w[0] and w[1].

And that's it! We have now implemented the least squares method in C++.

## Conclusion
In this lesson, we learned about the least squares method and how to implement it in C++. We discussed the QR decomposition method, which involves decomposing the input matrix X into an orthogonal matrix Q and an upper triangular matrix R, and then using these matrices to solve the linear system of equations XtX * w = Xty. We also looked at an example implementation of the least squares method in C++.

I hope you found this lesson helpful! Let me know if you have any questions or feedback.
