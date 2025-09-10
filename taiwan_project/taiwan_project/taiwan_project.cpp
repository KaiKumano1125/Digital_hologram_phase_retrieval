#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//FFT library//
#include <fftw3.h>
#pragma comment(lib, "libfftw3-3.lib")

// Define the constants values//
const double WL = 500e-9; //wavelength (green laser)
const double PIXEL_PITCH = 8.0e-6; //pixel pitch of SLM
const double Z_1 = 0.15e-3; //distance between light source and object
const double Z_2 = 0.75e-3; //distance between object and sensor /M=(z_1+z_2)/z_1 =6

//complex number type for our wave calculation//
typedef std::complex<double> Complex;

// Helper function to save a complex vector's magnitude to a normalized image
void save_complex_magnitude_as_image(const std::vector<Complex>& data, int nx, int ny, const std::string& filename) {
    cv::Mat image(ny, nx, CV_64FC1);
    double max_val = 0.0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            double magnitude_squared = std::norm(data[x + y * nx]);
            image.at<double>(y, x) = magnitude_squared;
            if (magnitude_squared > max_val) {
                max_val = magnitude_squared;
            }
        }
    }

    if (max_val > 0) {
        image.convertTo(image, CV_8UC1, 255.0 / max_val);
    }
    else {
        image.convertTo(image, CV_8UC1, 255.0);
    }
    cv::imwrite(filename, image);
    std::cout << "Image saved: " << filename << std::endl;
}

// Helper function to save a complex FFTW array's magnitude to a normalized image
void save_fftw_magnitude_as_image(fftw_complex* data, int nx, int ny, const std::string& filename, bool normalize_fft_output = true) {
    cv::Mat image(ny, nx, CV_64FC1);
    double max_val = 0.0;
    double norm_factor = normalize_fft_output ? (nx * ny) : 1.0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            double real_part = data[x + y * nx][0] / norm_factor;
            double imag_part = data[x + y * nx][1] / norm_factor;
            double magnitude_squared = real_part * real_part + imag_part * imag_part;
            image.at<double>(y, x) = magnitude_squared;
            if (magnitude_squared > max_val) {
                max_val = magnitude_squared;
            }
        }
    }

    if (max_val > 0) {
        image.convertTo(image, CV_8UC1, 255.0 / max_val);
    }
    else {
        image.convertTo(image, CV_8UC1, 255.0);
    }
    cv::imwrite(filename, image);
    std::cout << "Image saved: " << filename << std::endl;
}

//Function to perform FFT using FFTW library//
void fft2d(fftw_complex* data, int nx, int ny, int direction) {
    fftw_plan p;
    if (direction == FFTW_FORWARD) {
        p = fftw_plan_dft_2d(ny, nx, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    }
    else {
        p = fftw_plan_dft_2d(ny, nx, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    fftw_execute(p);
    fftw_destroy_plan(p);
}

//main simulation function//
int main() {
    // 1.define object transmission function from BMP
    const char* bmp_filename = "Clock.bmp"; //input object image file name
    cv::Mat object_img = cv::imread(bmp_filename, cv::IMREAD_GRAYSCALE);

    if (object_img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    int N_X = object_img.cols; //image width
    int N_Y = object_img.rows; //image height

    // The paper's method requires square matrices for FFT
    if (N_X != N_Y) {
        std::cerr << "Error: The input image must be square (N_X == N_Y)." << std::endl;
        return -1;
    }

    std::vector<Complex> t(N_X * N_Y);
    for (int y = 0; y < N_Y; ++y) {
        for (int x = 0; x < N_X; ++x) {
            // Normalize pixel value (0-255) to a transmission amplitude (0.0-1.0)
            double amplitude = static_cast<double>(object_img.at<uchar>(y, x)) / 255.0;
            // The phase is assumed to be 0 for a pure absorption object
            t[x + y * N_X] = Complex(amplitude, 0.0);
        }
    }

    // -----------------------------------------------------------
    // 2. Define Incident Spherical Wave U_incident(x, y)
    // -----------------------------------------------------------
    // Based on Eq. (37) from the paper: U_incident = exp(ikr)/r
    // where r = sqrt(x^2 + y^2 + z^2)
    std::vector<Complex> incident_wave(N_X * N_Y);
    const double k = 2.0 * M_PI / WL;
    for (int y = 0; y < N_Y; ++y) {
        for (int x = 0; x < N_X; ++x) {
            // Centered coordinates
            double real_x = (x - N_X / 2.0) * PIXEL_PITCH;
            double real_y = (y - N_Y / 2.0) * PIXEL_PITCH;
            double r = sqrt(real_x * real_x + real_y * real_y + Z_1 * Z_1);

            // U_incident = (1/r) * exp(ikr)
            incident_wave[x + y * N_X] = (1.0 / r) * Complex(cos(k * r), sin(k * r));
        }
    }

    // -----------------------------------------------------------
    // 3. Calculate the Exit Wave U_exit_wave
    // -----------------------------------------------------------
    // U_exit_wave = U_incident * t(x, y)
    std::vector<Complex> exit_wave(N_X * N_Y);
    for (int i = 0; i < N_X * N_Y; ++i) {
        exit_wave[i] = incident_wave[i] * t[i];
    }
    // Save intensity of initial object for visualization
    save_complex_magnitude_as_image(exit_wave, N_X, N_Y, "output/01_object_intensity.bmp");

    // -----------------------------------------------------------
    // 4. Propagate the Exit Wave to the Hologram Plane (ASM)
    // -----------------------------------------------------------
    // This is the core propagation step, based on Eq. (28) from the paper.

    fftw_complex* fft_exit_wave = (fftw_complex*)fftw_malloc(N_X * N_Y * sizeof(fftw_complex));
    for (int i = 0; i < N_X * N_Y; ++i) {
        fft_exit_wave[i][0] = exit_wave[i].real();
        fft_exit_wave[i][1] = exit_wave[i].imag();
    }

    // Perform forward FFT on the exit wave
    fft2d(fft_exit_wave, N_X, N_Y, FFTW_FORWARD);

    // Save magnitude of the angular spectrum
    save_fftw_magnitude_as_image(fft_exit_wave, N_X, N_Y, "output/02_asm_magnitude.bmp", false);

    // -----------------------------------------------------------
    // 4.1. Create the Angular Spectrum Propagation Kernel (Transfer Function)
    // -----------------------------------------------------------
    // This is the H(u,v) or exp[i * ...] term in Eq. (28).
    std::vector<Complex> prop_kernel(N_X * N_Y);
    double du = 1.0 / (N_X * PIXEL_PITCH);
    double dv = 1.0 / (N_Y * PIXEL_PITCH);

    for (int y = 0; y < N_Y; ++y) {
        for (int x = 0; x < N_X; ++x) {
            // Shifted frequency coordinates for centered FFT
            double u = (x - N_X / 2.0) * du;
            double v = (y - N_Y / 2.0) * dv;

            double k_z_squared = k * k - (2.0 * M_PI * u) * (2.0 * M_PI * u) - (2.0 * M_PI * v) * (2.0 * M_PI * v);

            // Check for evanescent waves (k_z becomes imaginary)
            if (k_z_squared >= 0) {
                // Propagating wave part
                double k_z = sqrt(k_z_squared);
                prop_kernel[x + y * N_X] = Complex(cos(k_z * Z_2), sin(k_z * Z_2));
            }
            else {
                // Evanescent wave part, which decays exponentially.
                prop_kernel[x + y * N_X] = Complex(0.0, 0.0);
            }
        }
    }

    // -----------------------------------------------------------
    // 4.2. Multiply by the Propagation Kernel in the Frequency Domain
    // -----------------------------------------------------------
    for (int i = 0; i < N_X * N_Y; ++i) {
        double real_part = fft_exit_wave[i][0] * prop_kernel[i].real() - fft_exit_wave[i][1] * prop_kernel[i].imag();
        double imag_part = fft_exit_wave[i][0] * prop_kernel[i].imag() + fft_exit_wave[i][1] * prop_kernel[i].real();
        fft_exit_wave[i][0] = real_part;
        fft_exit_wave[i][1] = imag_part;
    }

    // -----------------------------------------------------------
    // 4.3. Inverse FFT to get Propagated Wave in Hologram Plane
    // -----------------------------------------------------------
    fft2d(fft_exit_wave, N_X, N_Y, FFTW_BACKWARD);

    // -----------------------------------------------------------
    // 5. Generate Hologram (Intensity)
    // -----------------------------------------------------------
    // Hologram = |U_detector|^2 = U_detector * U_detector*
    cv::Mat hologram(N_Y, N_X, CV_64FC1);
    double max_intensity = 0.0;
    for (int y = 0; y < N_Y; ++y) {
        for (int x = 0; x < N_X; ++x) {
            int idx = x + y * N_X;
            // The backward FFT has a normalization factor of N*N that must be accounted for.
            double real_part = fft_exit_wave[idx][0] / (N_X * N_Y);
            double imag_part = fft_exit_wave[idx][1] / (N_X * N_Y);
            double intensity = real_part * real_part + imag_part * imag_part;

            if (intensity > max_intensity) {
                max_intensity = intensity;
            }
            hologram.at<double>(y, x) = intensity;
        }
    }

    // Save unnormalized hologram intensity
    cv::Mat unnormalized_hologram;
    hologram.convertTo(unnormalized_hologram, CV_8UC1, 255.0); // Save as is for visualization
    cv::imwrite("output/03_unnormalized_hologram_intensity.bmp", unnormalized_hologram);

    // Normalize to 8-bit image for viewing
    cv::Mat normalized_hologram;
    if (max_intensity > 0) {
        hologram.convertTo(normalized_hologram, CV_8UC1, 255.0 / max_intensity);
    }
    else {
        hologram.convertTo(normalized_hologram, CV_8UC1, 255.0);
    }

    // Save the hologram image
    cv::imwrite("output/intensity_simulation.bmp", normalized_hologram);
    std::cout << "Hologram saved as intensity_simulation.bmp" << std::endl;

    fftw_free(fft_exit_wave);

    return 0;
}
