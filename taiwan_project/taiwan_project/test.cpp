#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <fstream>
#include "fftw3.h"
#include <opencv2/opencv.hpp>

// Define PI
const double PI = 3.14159265358979323846;

// Function to read a simple 24-bit uncompressed BMP file
// Now uses OpenCV's imread for simplicity
std::vector<double> readImageToVector(const std::string& filename, int& width, int& height) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open image file " << filename << std::endl;
        return {};
    }

    width = image.cols;
    height = image.rows;

    std::vector<double> imageVector(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            imageVector[y * width + x] = static_cast<double>(image.at<uchar>(y, x)) / 255.0;
        }
    }
    return imageVector;
}

// Function to generate a spherical reference wave
std::vector<std::complex<double>> generateSphericalWave(int width, int height, double lambda, double z) {
    std::vector<std::complex<double>> wave(width * height);
    double k = 2 * PI / lambda;

    double cx = width / 2.0;
    double cy = height / 2.0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double r_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy) + z * z;
            double r = std::sqrt(r_sq);

            wave[y * width + x] = std::exp(std::complex<double>(0.0, k * r));
        }
    }
    return wave;
}

// Function to perform Angular Spectrum Propagation using FFTW
std::vector<std::complex<double>> angularSpectrumPropagation(const std::vector<std::complex<double>>& inputWave, int width, int height, double lambda, double z) {
    int N = width * height;
    std::vector<std::complex<double>> outputWave(N);

    fftw_complex* in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));

    for (int i = 0; i < N; ++i) {
        in[i][0] = inputWave[i].real();
        in[i][1] = inputWave[i].imag();
    }

    fftw_plan p_forward = fftw_plan_dft_2d(height, width, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p_forward);

    double k = 2 * PI / lambda;
    double dx = 1.0;
    double dy = 1.0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double fx = (x - width / 2.0) / (width * dx);
            double fy = (y - height / 2.0) / (height * dy);

            std::complex<double> H = std::exp(std::complex<double>(0.0, k * z) * std::sqrt(std::complex<double>(1.0 - lambda * lambda * (fx * fx + fy * fy))));

            int idx = y * width + x;
            outputWave[idx] = std::complex<double>(out[idx][0], out[idx][1]) * H / static_cast<double>(N);
        }
    }

    fftw_plan p_backward = fftw_plan_dft_2d(height, width, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int i = 0; i < N; ++i) {
        in[i][0] = outputWave[i].real();
        in[i][1] = outputWave[i].imag();
    }

    fftw_execute(p_backward);

    for (int i = 0; i < N; ++i) {
        outputWave[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);
    fftw_free(in);
    fftw_free(out);

    return outputWave;
}

// Function to save intensity to a BMP file using OpenCV
void saveIntensity(const std::vector<double>& intensity, int width, int height, const std::string& filename) {
    cv::Mat outputImage(height, width, CV_8UC1);

    // Normalize intensity values and convert to 8-bit unsigned char
    double max_val = 0.0;
    for (double val : intensity) {
        if (val > max_val) max_val = val;
    }
    if (max_val > 0) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                outputImage.at<uchar>(y, x) = static_cast<uchar>((intensity[y * width + x] / max_val) * 255.0);
            }
        }
    }

    cv::imwrite(filename, outputImage);
    std::cout << "Image saved to " << filename << std::endl;
}


// Function to reconstruct the hologram and display the result
void reconstructHologram(const std::vector<double>& hologram_intensity, int width, int height, double lambda, double z1, double z2, const std::string& output_filename) {
    std::vector<std::complex<double>> hologram_amplitude(width * height);
    for (int i = 0; i < width * height; ++i) {
        hologram_amplitude[i] = hologram_intensity[i];
    }

    std::vector<std::complex<double>> reference_wave = generateSphericalWave(width, height, lambda, z1);
    std::vector<std::complex<double>> propagated_reference_wave = angularSpectrumPropagation(reference_wave, width, height, lambda, z1 + z2);

    std::vector<std::complex<double>> conjugated_reference_wave(width * height);
    for (int i = 0; i < width * height; ++i) {
        conjugated_reference_wave[i] = std::conj(propagated_reference_wave[i]);
    }

    std::vector<std::complex<double>> reconstructed_wave(width * height);
    for (int i = 0; i < width * height; ++i) {
        reconstructed_wave[i] = hologram_amplitude[i] * conjugated_reference_wave[i];
    }

    std::vector<std::complex<double>> final_wave = angularSpectrumPropagation(reconstructed_wave, width, height, lambda, -(z1 + z2));

    std::vector<double> reconstructed_intensity(width * height);
    for (int i = 0; i < width * height; ++i) {
        reconstructed_intensity[i] = std::norm(final_wave[i]);
    }

    saveIntensity(reconstructed_intensity, width, height, output_filename);
}

int main() {
    // Simulation parameters
    const double lambda = 532e-9; // Wavelength (e.g., green laser)
    const double Z1 = 0.5;        // Distance from light source to object
    const double Z2 = 0.2;        // Distance from object to hologram plane

    std::string object_filename = "input/Fishing Boat.bmp";
    int width, height;

    std::vector<double> object_data = readImageToVector(object_filename, width, height);
    if (object_data.empty()) {
        return 1;
    }

    std::vector<std::complex<double>> object_wave(width * height);
    for (int i = 0; i < width * height; ++i) {
        object_wave[i] = object_data[i];
    }

    std::vector<std::complex<double>> spherical_wave = generateSphericalWave(width, height, lambda, Z1);

    std::vector<std::complex<double>> reference_wave_at_hologram = angularSpectrumPropagation(spherical_wave, width, height, lambda, Z1 + Z2);

    std::vector<std::complex<double>> propagated_object_wave = angularSpectrumPropagation(object_wave, width, height, lambda, Z2);

    std::vector<double> hologram_intensity(width * height);
    for (int i = 0; i < width * height; ++i) {
        std::complex<double> total_wave = reference_wave_at_hologram[i] + propagated_object_wave[i];
        hologram_intensity[i] = std::norm(total_wave);
    }

    saveIntensity(hologram_intensity, width, height, "output/test.cpp/hologram_intensity.bmp");

    //reconstructHologram(hologram_intensity, width, height, lambda, Z1, Z2, "reconstructed_image.bmp");

    fftw_cleanup();

    return 0;
}
