//This code simulates gabor hologram generation and reconstruction using Angular Spectrum Propagation method.//
//This is main code (the old version is taiwan_project/taiwan_project.cpp).//

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include "fftw3.h"
#include <opencv2/opencv.hpp>


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

// FFTシフト:象限変換//Function to rearrange the quadrants of the FFT output for visualization
void fftshift(std::vector<double>& data, int width, int height) {
    int cx = width / 2;
    int cy = height / 2;

    cv::Mat image(height, width, CV_64F, data.data());
    cv::Mat q0(image, cv::Rect(0, 0, cx, cy));   // Top-Left - with size
    cv::Mat q1(image, cv::Rect(cx, 0, width - cx, cy));  // Top-Right
    cv::Mat q2(image, cv::Rect(0, cy, cx, height - cy));  // Bottom-Left
    cv::Mat q3(image, cv::Rect(cx, cy, width - cx, height - cy)); // Bottom-Right

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// Function to generate a spherical reference wave
std::vector<std::complex<double>> generateSphericalWave(int width, int height, double lambda, double z) {
    std::vector<std::complex<double>> wave(width * height);
    double k = 2 * M_PI / lambda;

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

    // Execute the FFT
    fftw_execute(p_forward);

    // Calculate the intensity spectrum (magnitude squared)
    std::vector<double> spectrum_intensity(N);
    for (int i = 0; i < N; ++i) {
        double real_part = out[i][0];
        double imag_part = out[i][1];
        spectrum_intensity[i] = real_part * real_part + imag_part * imag_part;
    }

    // Apply a logarithmic scale to the spectrum intensity to make it visible
    for (int i = 0; i < N; ++i) {
        spectrum_intensity[i] = std::log(1.0 + spectrum_intensity[i]);
    }

    // Find the new max value after log scaling
    double new_max_val = 0.0;
    for (int i = 0; i < N; ++i) {
        if (spectrum_intensity[i] > new_max_val) {
            new_max_val = spectrum_intensity[i];
        }
    }

    // Normalize the log-scaled intensity to a 0-1 range
    if (new_max_val > 0) {
        for (int i = 0; i < N; ++i) {
            spectrum_intensity[i] /= new_max_val;
        }
    }

    // Rearrange the quadrants of the FFT output for visualization
    fftshift(spectrum_intensity, width, height);

    // Save the intensity to a file
    saveIntensity(spectrum_intensity, width, height, "output/hologram_spectrum.bmp");


    double k = 2 * M_PI / lambda;
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

    // Apply logarithmic normalization to the reconstructed image
    double max_val_rec = 0.0;
    for (double val : reconstructed_intensity) {
        reconstructed_intensity[val] = std::log(1.0 + reconstructed_intensity[val]);
        if (reconstructed_intensity[val] > max_val_rec) {
            max_val_rec = reconstructed_intensity[val];
        }
    }
    if (max_val_rec > 0) {
        for (int i = 0; i < reconstructed_intensity.size(); ++i) {
            reconstructed_intensity[i] /= max_val_rec;
        }
    }

    saveIntensity(reconstructed_intensity, width, height, output_filename);
}

int main() {
    // Simulation parameters
    const double lambda = 500e-9; // Wavelength (green laser)
    const double Z1 = 0.04;        // Distance from light source to object
    const double Z2 = 0.2;         // Distance from object to hologram plane

    std::string object_filename = "input/Man.bmp";
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

    // --- NEW: Save the hologram's spectrum for visualization ---
    // Create FFTW arrays for the total wave
    int N = width * height;
    fftw_complex* in_total = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    fftw_complex* out_total = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));

    for (int i = 0; i < N; ++i) {
        std::complex<double> total_wave = reference_wave_at_hologram[i] + propagated_object_wave[i];
        in_total[i][0] = total_wave.real();
        in_total[i][1] = total_wave.imag();
    }

    // Perform the forward FFT on the total wave
    fftw_plan p_forward_total = fftw_plan_dft_2d(height, width, in_total, out_total, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p_forward_total);

    // Calculate intensity, apply log scaling, and normalize
    std::vector<double> hologram_spectrum(N);
    for (int i = 0; i < N; ++i) {
        double real_part = out_total[i][0];
        double imag_part = out_total[i][1];
        hologram_spectrum[i] = real_part * real_part + imag_part * imag_part;
    }

    double max_val_spec = 0.0;
    for (int i = 0; i < N; ++i) {
        hologram_spectrum[i] = std::log(1.0 + hologram_spectrum[i]);
        if (hologram_spectrum[i] > max_val_spec) {
            max_val_spec = hologram_spectrum[i];
        }
    }

    if (max_val_spec > 0) {
        for (int i = 0; i < N; ++i) {
            hologram_spectrum[i] /= max_val_spec;
        }
    }

    // Apply fftshift to center the spectrum
    fftshift(hologram_spectrum, width, height);

    // Save the hologram's spectrum image
    saveIntensity(hologram_spectrum, width, height, "output/hologram_spectrum.bmp");

    // Clean up FFTW resources for the spectrum
    fftw_destroy_plan(p_forward_total);
    fftw_free(in_total);
    fftw_free(out_total);
    // --- END OF NEW CODE ---

    saveIntensity(hologram_intensity, width, height, "output/hologram_intensity.bmp");

    // Add logarithmic normalization to the reconstructed image as well
    reconstructHologram(hologram_intensity, width, height, lambda, Z1, Z2, "output/reconstructed_image.bmp");

    fftw_cleanup();

    return 0;
}
