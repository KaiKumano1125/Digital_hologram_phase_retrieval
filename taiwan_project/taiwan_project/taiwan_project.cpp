//------------------------------------------------------------
//Step 1 : Generate the Ground Truth Data (Ideal Hologram)
//Step 2 : Generate the Training Data(Simulated Experimental Hologram)
//Step 3 : Train the Model using Stochastic Gradient Descent(SGD)
//------------------------------------------------------------
//This code generates the ground truth data (Step1) based on gabor hologram.

#include <iostream>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
