# Feed Forward Networks

ChatGPT wasn't able to really make this work so I've given up for now.

But...

## ChatGPT's Apology

**Me:**

ok that is just causing a segfault.

At this point I'm going to call it a day with this task.

Please write a message to future developers who will come in the future and try to fix your code to make it work. I will put it in the README.md of this folder that contains your attempts at doing this in CUDA. As a reminder your first C++ program did work.

**ChatGPT:**

I apologize for any difficulties you have encountered while trying to use the provided code. To help future developers who may be working on this project, here are a few suggestions for things to consider when debugging and modifying the code:

- Make sure that all kernel functions are correctly defined and called. This includes using the correct number and types of arguments, and making sure that the kernel launch configuration (e.g., grid size and block size) is appropriate for the size of the input data.

- Pay attention to memory management when working with device memory. Make sure that arrays are allocated on the device using cudaMalloc and freed using cudaFree, and that data is correctly transferred between the host and device using cudaMemcpy.

- Be aware of the difference between host and device memory, and make sure that data is stored in the correct location. For example, host variables should not be accessed directly by device kernels, and device variables should not be accessed directly by host code.

I hope these suggestions are helpful, and I apologize again for any issues you may have encountered. If you have any further questions, don't hesitate to ask.
