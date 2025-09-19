"""
Computer Vision Lab 2: Image Representations, Processing, and Filtering
========================================================================

This lab covers the fundamental concepts of digital image processing including:
1. Sampling and aliasing effects
2. Color space conversions
3. Image enhancement techniques
4. Linear and non-linear filtering
5. Edge detection algorithms
6. Morphological operations
7. Performance analysis and optimization

Students will implement core algorithms from scratch and compare with library implementations.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.fft
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from skimage import morphology
import time
import os
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class ImageProcessingLab:
    """
    Main class containing all lab exercises for image processing fundamentals.
    Students will implement methods marked with TODO comments.
    """

    def __init__(self):
        self.results = {}

    def load_test_image(self, path: Optional[str] = None) -> np.ndarray:
        """Load a test image for the lab exercises."""
        if path is None:
            # Create a synthetic test image with various patterns
            img = np.zeros((256, 256), dtype=np.uint8)

            # Add some geometric patterns
            cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
            cv2.circle(img, (180, 180), 30, 128, -1)

            # Add some texture
            x, y = np.meshgrid(np.arange(256), np.arange(256))
            sinusoid = 50 * np.sin(2 * np.pi * x / 20) * np.sin(2 * np.pi * y / 30)
            img = np.clip(img.astype(float) + sinusoid, 0, 255).astype(np.uint8)

            return img
        else:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# =============================================================================
# PART 1: SAMPLING AND ALIASING
# =============================================================================


class SamplingAnalysis:
    """Class for exploring sampling theory and aliasing effects."""

    def create_sine_grating(
        self, size: Tuple[int, int], frequency: float, orientation: float = 0
    ) -> np.ndarray:
        """
        Create a sinusoidal grating pattern.

        Args:
            size: (height, width) of the image
            frequency: Spatial frequency in cycles per pixel
            orientation: Orientation in degrees (0 = horizontal)

        Returns:
            Grayscale image with sinusoidal pattern
        """
        h, w = size
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Implement sinusoidal grating creation
        # Hint: Use np.sin() with appropriate phase calculation
        # Consider orientation using rotation matrix
        # The grating should have values in [0, 255] as uint8
        # Formula: grating = 0.5 + 0.5 * sin(2π * frequency * x_rotated)
        # where x_rotated accounts for the orientation

        theta = np.deg2rad(orientation)
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        grating = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * x_rot)
        grating = np.clip(grating * 255, 0, 255).astype(np.uint8)

        return grating

    def demonstrate_aliasing(self, frequency: float, downsample_factor: int) -> dict:
        """
        Demonstrate aliasing effects when downsampling.

        Args:
            frequency: Original spatial frequency
            downsample_factor: Factor by which to downsample

        Returns:
            Dictionary containing original, aliased, and anti-aliased results
        """
        # Create high-resolution grating
        filePath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "lecCode",
            "input.jpg",
        )

        original = cv2.imread(filePath)

        # Implement naive downsampling (aliased)
        # Simply take every downsample_factor-th pixel
        # Example: aliased = original[::downsample_factor, ::downsample_factor]
        aliased = original[::downsample_factor, ::downsample_factor]

        # Implement anti-aliased downsampling
        # Apply Gaussian filter before downsampling
        # Rule of thumb: sigma ≈ downsample_factor / 2
        # Use cv2.GaussianBlur() then downsample
        sigma = downsample_factor / 2.0

        # Your anti-aliasing implementation here
        # Apply Gaussian blur to original
        filtered = cv2.GaussianBlur(original, (0, 0), sigma)
        # Then downsample the filtered image
        anti_aliased = filtered[::downsample_factor, ::downsample_factor]

        # Calculate theoretical aliased frequency
        theoretical_alias_freq = abs(
            frequency - round(frequency * downsample_factor) / downsample_factor
        )

        return {
            "original": original,
            "aliased": aliased,
            "anti_aliased": anti_aliased,
            "theoretical_alias_freq": theoretical_alias_freq,
        }

    def analyze_frequency_content(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze frequency content of an image using FFT.

        Args:
            image: Input grayscale image

        Returns:
            Log magnitude spectrum (shifted to center)
        """
        # Implement FFT analysis
        # 1. Compute 2D FFT using np.fft.fft2()

        ftt = np.fft.fft2(image)

        # 2. Shift zero frequency to center using np.fft.fftshift()

        ftt_shifted = np.fft.fftshift(ftt)

        # 3. Compute log magnitude spectrum: log(abs(fft) + 1)
        # This should return the frequency analysis for visualization

        log_magnitude = np.log(np.abs(ftt_shifted) + 1)

        return log_magnitude


# =============================================================================
# PART 2: COLOR SPACE CONVERSIONS
# =============================================================================


class ColorSpaceConverter:
    """Class for implementing color space conversions."""

    def rgb_to_hsv_manual(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to HSV manually (following the algorithm from slides).

        Args:
            rgb_image: RGB image with values in [0, 255], shape (H, W, 3)

        Returns:
            HSV image with H in [0, 179], S and V in [0, 255] (OpenCV format)
        """
        # Normalize RGB to [0, 1]
        rgb_norm = rgb_image.astype(np.float32) / 255.0
        R, G, B = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

        # TODO: Implement RGB to HSV conversion following the algorithm from slides
        # 1. Compute V = max(R, G, B)
        # 2. Compute delta = V - min(R, G, B)
        # 3. Compute S = delta / V (handle V = 0 case)
        # 4. Compute H based on which channel is maximum:
        #    - If R is max: H = 60 * ((G-B)/delta) mod 6
        #    - If G is max: H = 60 * ((B-R)/delta + 2)
        #    - If B is max: H = 60 * ((R-G)/delta + 4)
        # 5. Convert to OpenCV ranges: H [0, 360] -> [0, 179], S,V [0, 1] -> [0, 255]

        # Initialize output arrays
        H = np.zeros_like(R)
        S = np.zeros_like(R)
        V = np.zeros_like(R)

        # Your implementation here

        # Convert to OpenCV ranges and return
        H_out = None  # Convert H to [0, 179]
        S_out = None  # Convert S to [0, 255]
        V_out = None  # Convert V to [0, 255]

        return np.stack([H_out, S_out, V_out], axis=2)

    def compare_hsv_implementations(self, rgb_image: np.ndarray) -> dict:
        """
        Compare manual HSV implementation with OpenCV.

        Args:
            rgb_image: RGB image

        Returns:
            Dictionary with manual result, OpenCV result, and difference analysis
        """
        # Convert BGR to RGB for OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Manual implementation
        hsv_manual = self.rgb_to_hsv_manual(rgb_image)

        # OpenCV implementation
        hsv_opencv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # TODO: Compute difference metrics
        # Calculate absolute differences between manual and OpenCV results
        # Compute mean and maximum errors for each channel
        # Analyze the sources of differences

        diff = None  # Compute absolute difference
        mean_diff = None  # Mean error per channel
        max_diff = None  # Maximum error per channel

        return {
            "manual": hsv_manual,
            "opencv": hsv_opencv,
            "difference": diff,
            "mean_error": mean_diff,
            "max_error": max_diff,
        }


# =============================================================================
# PART 3: IMAGE ENHANCEMENT
# =============================================================================


class ImageEnhancement:
    """Class for implementing image enhancement techniques."""

    def histogram_equalization(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement histogram equalization from scratch.

        Args:
            image: Grayscale image

        Returns:
            Tuple of (equalized_image, transformation_function)
        """
        # TODO: Implement histogram equalization algorithm from slides
        # 1. Compute histogram using np.histogram() or cv2.calcHist()
        # 2. Compute cumulative distribution function (CDF)
        # 3. Find minimum non-zero CDF value (cdf_min)
        # 4. Apply transformation: T[k] = round((cdf[k] - cdf_min) / (N - cdf_min) * 255)
        # 5. Apply transformation to image using advanced indexing

        hist = None  # Compute histogram
        cdf = None  # Compute CDF
        cdf_min = None  # Find minimum non-zero CDF
        transform = None  # Create transformation LUT
        equalized = None  # Apply transformation

        return equalized, transform

    def clahe_implementation(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Simplified CLAHE implementation.

        Args:
            image: Grayscale image
            clip_limit: Clipping threshold for histogram
            tile_size: Size of contextual regions

        Returns:
            CLAHE enhanced image
        """
        # TODO: Implement basic CLAHE or use OpenCV's implementation
        # For educational purposes, you can use cv2.createCLAHE() but understand:
        # 1. How tiles are processed independently
        # 2. How clipping prevents noise amplification
        # 3. How interpolation smooths tile boundaries
        # Try to implement at least the clipping concept manually

        pass  # Your implementation here

    def compare_enhancement_methods(self, image: np.ndarray) -> dict:
        """
        Compare different enhancement methods.

        Args:
            image: Input grayscale image

        Returns:
            Dictionary with results from different methods
        """
        # TODO: Apply all enhancement methods and compare
        # 1. Compute original histogram
        # 2. Apply manual histogram equalization
        # 3. Apply OpenCV histogram equalization
        # 4. Apply CLAHE
        # 5. Compute histograms of all results
        # 6. Return comprehensive comparison

        results = {
            "original": image,
            "he_manual": None,  # Manual histogram equalization
            "he_opencv": None,  # OpenCV histogram equalization
            "clahe": None,  # CLAHE result
            "histograms": {
                "original": None,  # Original histogram
                "he_manual": None,  # HE histogram
                "clahe": None,  # CLAHE histogram
            },
            "transform_function": None,
        }

        return results


# =============================================================================
# PART 4: CONVOLUTION AND FILTERING
# =============================================================================


class ConvolutionImplementations:
    """Class for implementing and comparing different convolution methods."""

    def direct_convolution(
        self, image: np.ndarray, kernel: np.ndarray, mode: str = "valid"
    ) -> np.ndarray:
        """
        Implement direct convolution from scratch.

        Args:
            image: Input image
            kernel: Convolution kernel
            mode: 'valid', 'same', or 'full'

        Returns:
            Convolved image
        """
        # TODO: Implement direct convolution
        # 1. Handle padding based on mode ('valid', 'same', 'full')
        # 2. Flip kernel for true convolution (vs correlation)
        # 3. Slide kernel over image computing weighted sums
        # 4. Handle boundary conditions appropriately

        # Key steps:
        # - Determine output size based on mode
        # - Pad image if necessary
        # - Use nested loops to slide kernel
        # - Compute weighted sum at each position

        pass  # Your implementation here

    def separable_convolution(
        self, image: np.ndarray, kernel_1d_v: np.ndarray, kernel_1d_h: np.ndarray
    ) -> np.ndarray:
        """
        Implement separable convolution.

        Args:
            image: Input image
            kernel_1d_v: Vertical 1D kernel
            kernel_1d_h: Horizontal 1D kernel

        Returns:
            Convolved image
        """
        # TODO: Implement separable convolution
        # 1. Apply vertical 1D convolution first
        # 2. Apply horizontal 1D convolution to result
        # This should be much faster than 2D convolution
        # You can use scipy.ndimage.convolve1d or implement manually

        pass  # Your implementation here

    def fft_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Implement FFT-based convolution.

        Args:
            image: Input image
            kernel: Convolution kernel

        Returns:
            Convolved image
        """
        # TODO: Implement FFT-based convolution
        # 1. Pad both image and kernel to prevent circular convolution artifacts
        # 2. Compute FFT of both using np.fft.fft2()
        # 3. Multiply in frequency domain (element-wise)
        # 4. Inverse FFT using np.fft.ifft2() and take real part
        # 5. Extract valid region to match expected output size

        pass  # Your implementation here

    def benchmark_convolution_methods(
        self, image_sizes: list, kernel_sizes: list
    ) -> dict:
        """
        Benchmark different convolution implementations.

        Args:
            image_sizes: List of image sizes to test
            kernel_sizes: List of kernel sizes to test

        Returns:
            Dictionary with timing results
        """
        # TODO: Implement timing for each method
        # 1. Create test data for each size combination
        # 2. Time each convolution method using time.time()
        # 3. Store results for analysis
        # 4. Compare with OpenCV's optimized implementation
        # 5. Analyze crossover points where different methods are optimal

        results = {"direct": [], "separable": [], "fft": [], "opencv": []}

        # Your benchmarking implementation here

        return results


# =============================================================================
# PART 5: EDGE DETECTION
# =============================================================================


class EdgeDetection:
    """Class for implementing edge detection algorithms."""

    def sobel_operator(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Implement Sobel edge detection.

        Args:
            image: Grayscale input image

        Returns:
            Tuple of (magnitude, direction, gradients_x_y)
        """
        # TODO: Implement Sobel operator from slides
        # 1. Define Sobel kernels for x and y directions:
        #    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        #    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        # 2. Convolve image with both kernels using cv2.filter2D()
        # 3. Compute magnitude: sqrt(Gx^2 + Gy^2)
        # 4. Compute direction: arctan2(Gy, Gx) in degrees

        magnitude = None
        direction = None
        grad_x = None
        grad_y = None

        return magnitude, direction, (grad_x, grad_y)

    def non_maximum_suppression(
        self, magnitude: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """
        Implement non-maximum suppression for edge thinning.

        Args:
            magnitude: Edge magnitude image
            direction: Edge direction image (in degrees)

        Returns:
            Thinned edge magnitude
        """
        # TODO: Implement non-maximum suppression from slides
        # 1. Quantize directions to nearest: 0°, 45°, 90°, 135°
        # 2. For each pixel, compare magnitude with two neighbors along gradient direction
        # 3. Keep pixel only if it's a local maximum along that direction
        # 4. Set non-maximum pixels to zero

        # This requires careful indexing and boundary handling

        pass  # Your implementation here

    def double_threshold(
        self, image: np.ndarray, low_threshold: float, high_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply double thresholding for edge classification.

        Args:
            image: Input edge magnitude image
            low_threshold: Low threshold value
            high_threshold: High threshold value

        Returns:
            Tuple of (strong_edges, weak_edges)
        """
        # TODO: Implement double thresholding
        # 1. Create strong edge mask: magnitude >= high_threshold
        # 2. Create weak edge mask: low_threshold <= magnitude < high_threshold
        # 3. Return both masks as binary images

        strong_edges = None
        weak_edges = None

        return strong_edges, weak_edges

    def hysteresis_tracking(
        self, strong_edges: np.ndarray, weak_edges: np.ndarray
    ) -> np.ndarray:
        """
        Implement hysteresis edge tracking.

        Args:
            strong_edges: Strong edge pixels (binary)
            weak_edges: Weak edge pixels (binary)

        Returns:
            Final edge map
        """
        # TODO: Implement hysteresis tracking
        # 1. Start with strong edges as seeds
        # 2. Use BFS or DFS to trace connected weak edges
        # 3. Add weak edges that connect to strong edges
        # 4. Use 8-connectivity for neighbor checking

        # Algorithm:
        # - Initialize final edges with strong edges
        # - For each strong edge pixel, explore 8-connected neighbors
        # - If neighbor is weak edge, add to final and continue exploring
        # - Continue until no more weak edges can be connected

        pass  # Your implementation here

    def canny_edge_detection(
        self,
        image: np.ndarray,
        sigma: float = 1.0,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
    ) -> dict:
        """
        Complete Canny edge detection implementation.

        Args:
            image: Input grayscale image
            sigma: Gaussian smoothing parameter
            low_threshold: Low threshold (as fraction of max gradient)
            high_threshold: High threshold (as fraction of max gradient)

        Returns:
            Dictionary with intermediate and final results
        """
        # TODO: Implement complete Canny algorithm from slides
        # Stage 1: Gaussian smoothing
        # Stage 2: Gradient computation (Sobel)
        # Stage 3: Non-maximum suppression
        # Stage 4: Double thresholding
        # Stage 5: Hysteresis tracking

        # Return all intermediate results for analysis

        results = {
            "smoothed": None,
            "magnitude": None,
            "direction": None,
            "suppressed": None,
            "strong_edges": None,
            "weak_edges": None,
            "final_edges": None,
            "gradients": None,
        }

        return results


# =============================================================================
# PART 6: NON-LINEAR FILTERING AND MORPHOLOGY
# =============================================================================


class NonLinearFiltering:
    """Class for non-linear filtering operations."""

    def median_filter(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Implement median filtering.

        Args:
            image: Input image
            kernel_size: Size of the median filter kernel (odd number)

        Returns:
            Median filtered image
        """
        # TODO: Implement median filtering
        # 1. For each pixel, extract neighborhood window
        # 2. Compute median of neighborhood values
        # 3. Replace center pixel with median value
        # 4. Handle boundary conditions (padding or reflection)

        pass  # Your implementation here

    def bilateral_filter(
        self, image: np.ndarray, d: int, sigma_color: float, sigma_space: float
    ) -> np.ndarray:
        """
        Implement bilateral filtering.

        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space

        Returns:
            Bilateral filtered image
        """
        # TODO: Implement bilateral filtering
        # This is complex - for learning, implement a simplified version:
        # 1. For each pixel, define neighborhood
        # 2. Compute spatial weights: exp(-(spatial_distance^2)/(2*sigma_space^2))
        # 3. Compute intensity weights: exp(-(intensity_difference^2)/(2*sigma_color^2))
        # 4. Combine weights and compute weighted average
        # 5. Normalize by sum of weights

        pass  # Your implementation here


class MorphologicalOperations:
    """Class for morphological operations."""

    def erosion(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Implement morphological erosion.

        Args:
            image: Binary input image (0s and 255s)
            kernel: Structuring element (0s and 1s)

        Returns:
            Eroded image
        """
        # TODO: Implement erosion operation
        # 1. For each pixel position, check if structuring element fits entirely in foreground
        # 2. Set output pixel to 255 only if all kernel positions are foreground in input
        # 3. Handle boundaries appropriately

        pass  # Your implementation here

    def dilation(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Implement morphological dilation.

        Args:
            image: Binary input image (0s and 255s)
            kernel: Structuring element (0s and 1s)

        Returns:
            Dilated image
        """
        # TODO: Implement dilation operation
        # 1. For each foreground pixel in input
        # 2. Place structuring element centered at that pixel
        # 3. Set all positions covered by kernel to foreground in output

        pass  # Your implementation here

    def opening(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Implement morphological opening (erosion followed by dilation).

        Args:
            image: Binary input image
            kernel: Structuring element

        Returns:
            Opened image
        """
        # TODO: Implement opening operation
        # Opening = Dilation(Erosion(image, kernel), kernel)

        pass  # Your implementation here

    def closing(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Implement morphological closing (dilation followed by erosion).

        Args:
            image: Binary input image
            kernel: Structuring element

        Returns:
            Closed image
        """
        # TODO: Implement closing operation
        # Closing = Erosion(Dilation(image, kernel), kernel)

        pass  # Your implementation here


# =============================================================================
# PART 7: MODERN TECHNIQUES WITH PYTORCH
# =============================================================================


class ModernTechniques:
    """Class for exploring modern deep learning approaches to filtering."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_learnable_filter(
        self, kernel_size: int, in_channels: int = 1, out_channels: int = 1
    ) -> nn.Module:
        """
        Create a learnable convolution layer.

        Args:
            kernel_size: Size of the convolution kernel
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            PyTorch convolution layer
        """
        # TODO: Create a learnable convolution layer
        # 1. Use nn.Conv2d to create the layer
        # 2. Set appropriate padding to maintain image size
        # 3. Initialize weights with a meaningful pattern (e.g., edge detector)

        pass  # Your implementation here

    def visualize_cnn_filters(self, model_name: str = "resnet18") -> dict:
        """
        Visualize filters from a pre-trained CNN.

        Args:
            model_name: Name of the pre-trained model

        Returns:
            Dictionary with filter visualizations
        """
        # TODO: Load a pre-trained model and visualize its first layer filters
        # 1. Load pre-trained ResNet18 using torchvision.models
        # 2. Extract first convolutional layer weights
        # 3. Normalize filters for visualization
        # 4. Return filters and metadata

        # This demonstrates what modern CNNs learn automatically

        pass  # Your implementation here

    def demonstrate_learned_vs_classical(self, image: np.ndarray) -> dict:
        """
        Compare classical edge detection with learned filters.

        Args:
            image: Input image

        Returns:
            Dictionary comparing results
        """
        # TODO: Compare classical Sobel with learned filters
        # 1. Convert image to PyTorch tensor
        # 2. Apply classical Sobel operators using F.conv2d
        # 3. Create and apply learnable filter
        # 4. Compare results and analyze differences

        pass  # Your implementation here


# =============================================================================
# MAIN LAB EXECUTION AND VISUALIZATION
# =============================================================================


def run_lab_exercises():
    """
    Main function to run all lab exercises with visualizations.
    Students should complete this function to test their implementations.
    """
    # Initialize lab components
    lab = ImageProcessingLab()
    sampling = SamplingAnalysis()
    color_converter = ColorSpaceConverter()
    enhancement = ImageEnhancement()
    convolution = ConvolutionImplementations()
    edge_detection = EdgeDetection()
    nonlinear = NonLinearFiltering()
    morphology = MorphologicalOperations()
    modern = ModernTechniques()

    # Load test image
    test_image = lab.load_test_image()

    print("Computer Vision Lab 2: Image Processing and Filtering")
    print("=" * 60)

    # TODO: Students implement test calls for each exercise
    # Exercise 1: Sampling and Aliasing
    print("\n1. Testing Sampling and Aliasing Analysis")
    print("-" * 40)
    # Your test code here
    # Example:

    # Exercise 2: Color Space Conversion
    print("\n2. Testing Color Space Conversion")
    print("-" * 40)
    # Your test code here

    # Exercise 3: Image Enhancement
    print("\n3. Testing Image Enhancement")
    print("-" * 40)
    # Your test code here

    # Exercise 4: Convolution Benchmarking
    print("\n4. Testing Convolution Performance")
    print("-" * 40)
    # Your test code here

    # Exercise 5: Edge Detection
    print("\n5. Testing Edge Detection")
    print("-" * 40)
    # Your test code here

    # Exercise 6: Non-linear Filtering
    print("\n6. Testing Non-linear Filtering")
    print("-" * 40)
    # Your test code here

    # Exercise 7: Morphological Operations
    print("\n7. Testing Morphological Operations")
    print("-" * 40)
    # Your test code here

    # Exercise 8: Modern Techniques
    print("\n8. Testing Modern Techniques")
    print("-" * 40)
    # Your test code here

    print("\n" + "=" * 60)
    print("Implement all TODO methods and complete the testing!")


def visualize_all_results():
    """
    Create comprehensive visualizations of all lab results.
    Students should implement this function to display their results.
    """
    # TODO: Students implement comprehensive visualization
    # Should include:
    # - Aliasing demonstration plots

    sampling = SamplingAnalysis()

    # Visualize results
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    while True:
        for i in range(128):
            aliasing_results = sampling.demonstrate_aliasing(
                frequency=0.01, downsample_factor=i + 1
            )

            fig.suptitle(f"Downsample Factor: {i + 1}", fontsize=16)

            axes[0].imshow(cv2.cvtColor(aliasing_results["aliased"], cv2.COLOR_BGR2RGB))
            axes[0].set_title("Aliased (Naive Downsample)")
            axes[0].axis("off")

            axes[1].imshow(
                cv2.cvtColor(aliasing_results["anti_aliased"], cv2.COLOR_BGR2RGB)
            )
            axes[1].set_title("Anti-Aliased (Gaussian + Downsample)")
            axes[1].axis("off")

            plt.pause(0.1)

    # - Color space conversion comparisons
    # - Enhancement method comparisons
    # - Convolution performance charts
    # - Edge detection step-by-step results
    # - Filter comparisons
    # - Morphological operation effects
    # - CNN filter visualizations

    # plt.show()


def performance_analysis():
    """
    Analyze and report on performance characteristics of different methods.
    Students should implement detailed performance analysis.
    """
    # TODO: Students implement performance analysis
    # Should include:
    # - Computational complexity analysis
    # - Memory usage comparison
    # - Accuracy vs speed trade-offs
    # - Recommendations for different use cases
    pass


def write_lab_report():
    """
    Template for lab report structure.
    Students should fill in their findings and analysis.
    """
    report_template = """
    # Computer Vision Lab 2 Report
    
    ## 1. Sampling and Aliasing
    - Analysis of aliasing effects
    - Effectiveness of anti-aliasing
    - Theoretical vs practical results
    
    ## 2. Color Space Conversions
    - Accuracy of manual HSV implementation
    - Comparison with OpenCV
    - Applications and use cases
    
    ## 3. Image Enhancement
    - Histogram equalization effectiveness
    - CLAHE vs global methods
    - Quality metrics comparison
    
    ## 4. Convolution Methods
    - Performance benchmarking results
    - Cross-over points for different methods
    - Memory vs speed trade-offs
    
    ## 5. Edge Detection
    - Canny algorithm step-by-step analysis
    - Parameter sensitivity study
    - Comparison with simple gradient methods
    
    ## 6. Non-linear Filtering
    - Noise removal effectiveness
    - Edge preservation comparison
    - Computational requirements
    
    ## 7. Morphological Operations
    - Effect of different structuring elements
    - Applications to shape analysis
    - Binary vs grayscale morphology
    
    ## 8. Modern Techniques
    - Learned vs hand-crafted filters
    - Analysis of CNN first layer
    - Future directions discussion
    
    ## Conclusions
    - Key findings and insights
    - Practical recommendations
    - Areas for further study
    """

    print(report_template)


if __name__ == "__main__":
    # Run the lab exercises
    run_lab_exercises()

    # Students should also implement and call:
    visualize_all_results()
    # performance_analysis()
    # write_lab_report()
