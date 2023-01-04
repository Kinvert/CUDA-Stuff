# CUDA-Stuff

CUDA Stuff Here

## Goal

To make a great reference for CUDA code that can take advantage of parallel computing on GPUs.

Hopefully this is a place people can grab reference code, learn, and practice.

For example, I leave in any failed code (and mark it as such) so people can practice troubleshooting CUDA code.

[ChatGPT Code to Practice Fixing](https://github.com/Kinvert/CUDA-Stuff/search?q=filename%3A*ChatGPT-Fail)

Generally if ChatGPT wrote the code I'll give it the naming convention:
- ChatGPT-Works
- ChatGPT-Runs
- ChatGPT-Fail

Feel free to take ChatGPT code that doesn't work and try to fix it. I'm leaving in failed attempts since it shows how ChatGPT is doing, and I find it fun sometimes fixing their code.

## Outline

Here is the initial outline:

- [ChatGPT CUDA Lessons / Explanations](https://github.com/Kinvert/CUDA-Stuff/tree/master/ChatGPT-CUDA_Lessons)
  - The goal is Installation -> Hello World -> MNIST -> More
  - [Installation](https://github.com/Kinvert/CUDA-Stuff/tree/master/ChatGPT-CUDA_Lessons/000-Installation)
  - [Hello World](https://github.com/Kinvert/CUDA-Stuff/tree/master/ChatGPT-CUDA_Lessons/010-Hello-World)
  - [MNIST](https://github.com/Kinvert/CUDA-Stuff/tree/master/ChatGPT-CUDA_Lessons/500-MNIST) (not working yet)
- [Algorithms](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms)
  - [Search](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Search)
    - [Best First Search](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Search/Best-First-Search)
      - [A* Algorithm](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Search/Best-First-Search/A-Star-Algorithm)
    - [Linear-Search](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Search/Linear-Search)
  - [Sorting](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Sort)
    - [Bubble Sort](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Sort/Bubble-Sort)
    - [Insertion Sort](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Sort/Insertion-Sort)
  - [Monte-Carlo](https://github.com/Kinvert/CUDA-Stuff/tree/master/Algorithms/Monte-Carlo)
- [Computer Vision](https://github.com/Kinvert/CUDA-Stuff/tree/master/Computer-Vision)
  - Debayering
  - [Gaussian Blur](https://github.com/Kinvert/CUDA-Stuff/tree/master/Computer-Vision/Gaussian-Blur)
  - Image Denoising
  - Image Enhancement
  - Image Restoration
  - Image Segmentation
- [Data Analytics](https://github.com/Kinvert/CUDA-Stuff/tree/master/Data-Analytics)
  - Big Data Processing
  - [Data Filtering](https://github.com/Kinvert/CUDA-Stuff/tree/master/Data-Analytics/Data-Filtering)
  - Data Mining
  - Data Visualization
  - Hash Tables
- [Games and Graphics](https://github.com/Kinvert/CUDA-Stuff/tree/master/Games-and-Graphics)
  - 3D Rendering
  - [Perlin Noise](https://github.com/Kinvert/CUDA-Stuff/tree/master/Games-and-Graphics/Perlin-Noise)
  - Video Game Physics
  - Video Game AI
- Machine Learning and Deep Learning
  - Neural Network Training and Inference
  - Data Augmentation
  - Feature Extraction
- [Math](https://github.com/Kinvert/CUDA-Stuff/tree/master/Math)
  - [Linear Algebra](https://github.com/Kinvert/CUDA-Stuff/tree/master/Math/Linear-Algebra)
    - [Matrix Operations](https://github.com/Kinvert/CUDA-Stuff/tree/master/ChatGPT-CUDA_Lessons/060-Matrix-Operations)
      - [Matrix Addition](https://github.com/Kinvert/CUDA-Stuff/tree/master/ChatGPT-CUDA_Lessons/060-Matrix-Operations/061-Matrix-Addition)
      - [Matrix Multiplications](https://github.com/Kinvert/CUDA-Stuff/tree/master/ChatGPT-CUDA_Lessons/060-Matrix-Operations/066-Matrix-Multiplication)
    - [Row Reduction](https://github.com/Kinvert/CUDA-Stuff/tree/master/Math/Linear-Algebra/Row-Reduction)
  - Calculus 1
  - [Calculus 2](https://github.com/Kinvert/CUDA-Stuff/tree/master/Math/Calculus-2)
  - Calculus 3
  - [Numerical](https://github.com/Kinvert/CUDA-Stuff/tree/master/Math/Numerical)
    - [Least Squares](https://github.com/Kinvert/CUDA-Stuff/tree/master/Math/Numerical/Least-Squares)
- Scientific Computing
  - Molecular Dynamics
  - Fluid Dynamics
  - Climate Modeling
  - Finite Element Analysis
- [Simulations](https://github.com/Kinvert/CUDA-Stuff/tree/master/Simulations)
  - [Bouncing Balls](https://github.com/Kinvert/CUDA-Stuff/tree/master/Simulations/Bouncing-Balls)
    - [1D](https://github.com/Kinvert/CUDA-Stuff/tree/master/Simulations/Bouncing-Balls/1D)
    - [2D](https://github.com/Kinvert/CUDA-Stuff/tree/master/Simulations/Bouncing-Balls/2D) **ANIMATION GIFS**
  - [N-Body Problem](https://github.com/Kinvert/CUDA-Stuff/tree/master/Simulations/N-Body-Problem)
  - Particle-in-cell Simulations
  - Monte Carlo Simulations
  - Molecular Dynamics Simulations
  - Fluid Dynamics
  
## Useful Links

Looking these up Jan 2 2023 to help people find more good CUDA examples.

- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples)
- [zchee/cuda-sample](https://github.com/zchee/cuda-sample)
- [Erkaman/Awesome-CUDA](https://github.com/Erkaman/Awesome-CUDA)
- [CodedK/CUDA-by-Example-source...](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-)
- [NVIDIA/thrust](https://github.com/NVIDIA/thrust)
- [NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)
