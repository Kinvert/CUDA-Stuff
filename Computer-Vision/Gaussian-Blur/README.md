# Gaussian Blur

**Written by ChatGPT**

Gaussian blur is a popular image processing technique used to smooth images and reduce noise. It works by applying a convolution operation to the image, using a kernel (also known as a filter) that is designed to blur the image.

The kernel is typically a small matrix of weights, often with an odd number of rows and columns (e.g. 3x3, 5x5, etc.). The weights are chosen such that the kernel has a Gaussian distribution, which means that values near the center of the kernel have a higher weight than values farther away from the center.

To apply the Gaussian blur, the kernel is convolved with the image by sliding it over the image and computing a weighted sum of the overlapping pixels. The resulting value for each pixel in the output image is the sum of the weighted input pixels. For example, given a 3x3 kernel and an input image, the output pixel value is calculated as follows:

$O_{i,j} = \sum_{k=-1}^{1}\sum_{l=-1}^{1}I_{i+k,j+l}K_{k+1,l+1}$

where $O_{i,j}$ is the pixel value at position $(i,j)$ in the output image, $I_{i,j}$ is the pixel value at position $(i,j)$ in the input image, and $K_{k,l}$ is the weight at position $(k,l)$ in the kernel.

For example, given the following input image:

$I = {\left\lbrack \matrix{I_{1,1} & I_{1,2} & I_{1,3} \cr I_{2,1} & I_{2,2} & I_{2,3} \cr I_{3,1} & I_{3,2} & I_{3,3}} \right\rbrack} = {\left\lbrack \matrix{1 & 2 & 3 \cr 4 & 5 & 6 \cr 7 & 8 & 9} \right\rbrack}$

and a 3x3 kernel with weights 1/16, 2/16, 1/16 for the top row, 2/16, 4/16, 2/16 for the middle row, and 1/16, 2/16, 1/16 for the bottom row:

$K = {\left\lbrack \matrix{K_{1,1} & K_{1,2} & K_{1,3} \cr K_{2,1} & K_{2,2} & K_{2,3} \cr K_{3,1} & K_{3,2} & K_{3,3}} \right\rbrack} = {\left\lbrack \matrix{\frac{1}{16} & \frac{2}{16} & \frac{1}{16} \cr \frac{2}{16} & \frac{4}{16} & \frac{2}{16} \cr \frac{1}{16} & \frac{2}{16} & \frac{1}{16}} \right\rbrack}$

The output pixel values are calculated as follows:

For each pixel in the input image, a corresponding pixel is calculated in the output image. To do this, the value of the pixel is replaced with a weighted average of the values of the surrounding pixels. The weight of each surrounding pixel is determined by a Gaussian function, which gives higher weight to pixels that are closer to the center pixel and lower weight to pixels that are farther away.

The formula for calculating the output pixel value at position $(x,y)$ in the output image is:

$O(x,y) = \frac{1}{K}\sum_{i=-k}^{k}\sum_{j=-k}^{k}G(i,j)I(x+i,y+j)$

where $O(x,y)$ is the output pixel value at position $(x,y)$, $I(x+i,y+j)$ is the input pixel value at position $(x+i,y+j)$, $G(i,j)$ is the weight for the pixel at position $(i,j)$ in the kernel, and $K$ is a normalization constant equal to the sum of all the weights in the kernel.

The Gaussian function used to calculate the weights is:

$G(i,j) = \frac{1}{2\pi\sigma^2}e^{-\frac{i^2+j^2}{2\sigma^2}}$

where $\sigma$ is a parameter that controls the spread of the weights and is typically set to a small value such as 1 or 2.

In summary, Gaussian blur works by replacing each pixel in the input image with a weighted average of the surrounding pixels, where the weights are determined by a Gaussian function. This helps to smooth out noise and reduce high-frequency details in the image, resulting in a blur effect.
