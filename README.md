# Example of Inference Using Tensorflow and C 

This is a crude example of running inference on a locked graph using tensorflow's C library. The point of this repository is to act as a reference for future projects.
The code in this repo is an updated version of the following resources:
- [Inference in C using Tensorflow](http://iamsurya.com/inference-in-c-using-tensorflow/) by Surya Sharma
- [Tensorflow CMake/C++ Collection](https://github.com/PatWie/tensorflow-cmake/tree/master) by Patrick Wieschollek


# Setup
The only requirement is [Tensorflow's C API](https://www.tensorflow.org/install/lang_c).
The Makefile includes the build command which can be ran with: ```make build```, and it creates an executable file called ```test```.
