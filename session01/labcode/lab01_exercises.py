"""
Computer Vision Lab 1: Student Exercises and Questions
======================================================

This file contains all the exercises and questions that students should complete
to demonstrate understanding of the concepts from Lecture 1.

Instructions:
- Complete each exercise by filling in the TODO sections
- Answer the theoretical questions in comments or separate document
- Test your implementations with the provided test cases
- Compare your results with the demo code implementations

Author: CV Course
Date: 2024
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import torch


class Lab1Exercises:
    """Student exercises for Lab 1"""

    def __init__(self):
        pass

    # ================================
    # EXERCISE 1: PYTORCH FUNDAMENTALS
    # ================================

    def exercise_1_1_tensor_operations(self):
        """
        Exercise 1.1: PyTorch Tensor Operations

        Complete the functions below to convert between NumPy and PyTorch tensors.
        Pay attention to channel ordering and data types.
        """

        def numpy_to_torch(img_np: np.ndarray) -> torch.Tensor:
            """
            Convert HxWxC numpy array to CxHxW torch tensor
            Hints:
            - Use np.transpose or tensor.permute to change channel order
            - Convert to float tensor for processing
            - Handle both grayscale (HxW) and color (HxWxC) images
            """
            if img_np.ndim == 2:  # Grayscale HxW
                tensor = torch.from_numpy(img_np).float().unsqueeze(0)
            elif img_np.ndim == 3:  # Color HxWxC
                tensor = torch.from_numpy(img_np).float().permute(2, 0, 1)
            return tensor

        def torch_to_numpy(img_torch: torch.Tensor) -> np.ndarray:
            """
            Convert CxHxW torch tensor to HxWxC numpy array
            Hints:
            - Use tensor.permute to change channel order
            - Convert back to numpy with .cpu().numpy()
            - Handle both 2D and 3D tensors
            """
            if img_torch.ndim == 2:
                arr = img_torch.cpu().numpy()
            elif img_torch.ndim == 3:
                arr = img_torch.permute(1, 2, 0).cpu().numpy()
            return arr

        # Test your implementation
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Convert to torch and back
        img_torch = numpy_to_torch(test_img)
        img_reconstructed = torch_to_numpy(img_torch)

        # Verify shapes and values
        print(f"Original shape: {test_img.shape}")
        print(f"Torch shape: {img_torch.shape}")
        print(f"Reconstructed shape: {img_reconstructed.shape}")

        # Should be close to zero if implemented correctly
        reconstruction_error = np.mean(
            np.abs(test_img.astype(float) - img_reconstructed)
        )
        print(f"Reconstruction error: {reconstruction_error}")

        """
        QUESTIONS for Exercise 1.1:
        
        Q1: Why do PyTorch and OpenCV use different channel orderings?
        Q2: When would you use GPU tensors vs CPU tensors for image processing?
        Q3: What are the memory implications of different tensor layouts?
        Q4: How does tensor layout affect convolution performance?
        
        A1: PyTorch uses channel-first (C, H, W) format for tensors to optimize
            for deep learning operations, while OpenCV uses channel-last (H, W, C)
            format for compatibility with standard image formats and libraries. This
            difference is due to their primary use cases and performance considerations.

        A2: GPU tensors are used when you want to leverage the parallel processing
            power of GPUs for faster computation, especially for large-scale or real-time
            image processing and deep learning tasks. CPU tensors are sufficient
            for small-scale or non-parallel tasks.

        A3: Different tensor layouts affect how data is stored in memory.
            Channel-first (C, H, W) can be more cache-friendly for certain operations
            (like convolutions in deep learning), while channel-last (H, W, C) may be
            more efficient for image display and manipulation. Inefficient layouts can
            lead to increased memory usage and slower access times.

        A4: Tensor layout affects convolution performance because deep learning libraries
            are optimized for specific memory layouts. Channel-first (C, H, W) allows
            for more efficient memory access patterns and better use of vectorized operations,
            leading to faster convolutions.
        """

    # ================================
    # EXERCISE 2: GEOMETRIC TRANSFORMATIONS
    # ================================

    def exercise_1_2_transformations(self):
        """
        Exercise 1.2: Geometric Transformations

        Implement the transformation hierarchy from Lecture 1.
        """

        def apply_transformation(
            img: np.ndarray, transform_matrix: np.ndarray
        ) -> np.ndarray:
            """
            Apply 3x3 transformation matrix to image

            Implement using cv2.warpPerspective
            Hints:
            - Use cv2.warpPerspective for homogeneous transformations
            - Maintain original image size unless specified otherwise
            - Handle both grayscale and color images
            """
            # Get image dimensions
            height, width = img.shape[:2]

            # Apply the transformation using cv2.warpPerspective
            # This function applies a perspective transform to the entire image
            transformed_img = cv2.warpPerspective(
                img,
                transform_matrix,
                (width, height),
            )

            return transformed_img

        def create_similarity_transform(
            scale: float, rotation: float, translation: Tuple[float, float]
        ) -> np.ndarray:
            """
            Create similarity transformation matrix

            Implement using the equations from lecture
            Matrix form:
                [s*cos(θ)  -s*sin(θ)   tx]
                [s*sin(θ)   s*cos(θ)   ty]
                [   0          0        1]

            Args:
                scale: Scaling factor
                rotation: Rotation angle in radians
                translation: Translation (tx, ty)
            """
            tx, ty = translation
            cos_theta = np.cos(rotation)
            sin_theta = np.sin(rotation)

            # Create similarity transformation matrix
            transform_matrix = np.array(
                [
                    [scale * cos_theta, -scale * sin_theta, tx],
                    [scale * sin_theta, scale * cos_theta, ty],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            return transform_matrix

        def create_affine_transform(
            scale: Tuple[float, float],
            rotation: float,
            translation: Tuple[float, float],
            shear: float = 0,
        ) -> np.ndarray:
            """
            Create affine transformation matrix

            Implement general affine transformation
            Include scaling, rotation, translation, and optional shear
            """
            sx, sy = scale
            tx, ty = translation
            cos_theta = np.cos(rotation)
            sin_theta = np.sin(rotation)

            # Create rotation matrix with scaling
            rotation_scale = np.array(
                [[sx * cos_theta, -sx * sin_theta], [sy * sin_theta, sy * cos_theta]]
            )

            # Add shear component (horizontal shear)
            shear_matrix = np.array([[1, shear], [0, 1]])

            # Combine rotation-scale with shear
            combined_matrix = rotation_scale @ shear_matrix

            # Create full affine transformation matrix
            transform_matrix = np.array(
                [
                    [combined_matrix[0, 0], combined_matrix[0, 1], tx],
                    [combined_matrix[1, 0], combined_matrix[1, 1], ty],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            return transform_matrix

        def create_projective_transform(
            src_points: np.ndarray, dst_points: np.ndarray
        ) -> np.ndarray:
            """
            Create projective transformation from 4 point correspondences

            """
            # Use OpenCV's getPerspectiveTransform to compute the homography
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            return transform_matrix

        # Test your transformations
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        test_img[50:150, 50:150] = [255, 0, 0]  # Red square

        # Test similarity transform
        sim_transform = create_similarity_transform(1.2, np.pi / 4, (50, 30))
        transformed_img = apply_transformation(test_img, sim_transform)

        # Visualize results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(transformed_img)
        plt.title("Transformed")
        plt.show()

        """
        QUESTIONS for Exercise 1.2:
        
        Q1: What's the difference between similarity and affine transformations?
        Q2: When do you need projective transformations vs simpler ones?
        Q3: How many parameters does each transformation type have?
        Q4: What happens to parallel lines under each transformation type?
        
        Write your answers here:
        A1: Similarity transformations preserve angles and shape proportions, allowing only 
            uniform scaling, rotation, and translation (4 parameters). Affine transformations 
            can include non-uniform scaling and shearing, which can change angles but preserve 
            parallel lines (6 parameters). Similarity is a subset of affine transformations.
            
        A2: Projective transformations are needed when dealing with perspective effects, such as 
            viewing a planar surface from different viewpoints (e.g., document rectification, 
            architectural photography correction). Simpler transformations (similarity/affine) 
            are sufficient when the camera is orthogonal to the surface or when perspective 
            effects are negligible.
            
        A3: - Similarity: 4 parameters (scale, rotation angle, tx, ty)
            - Affine: 6 parameters (2 scales, rotation, shear, tx, ty)
            - Projective: 8 parameters (3x3 matrix with 9 elements, but scale invariant, so 8 DOF)
            
        A4: - Similarity: Parallel lines remain parallel, angles are preserved
            - Affine: Parallel lines remain parallel, but angles may change
            - Projective: Parallel lines may converge (perspective effect), angles and 
              parallelism are not preserved
        """

    # ================================
    # EXERCISE 3: CORNER DETECTION ANALYSIS
    # ================================

    def exercise_1_3_corner_analysis(self):
        """
        Exercise 1.3: Corner Detection Analysis

        Analyze the robustness and accuracy of corner detection.
        """

        def measure_corner_repeatability(
            images: List[np.ndarray], pattern_size: Tuple[int, int]
        ) -> float:
            """
            Measure corner detection repeatability across multiple views

            Implement repeatability measurement
            Steps:
            1. Detect corners in all images
            2. Find corresponding corners across images (if possible)
            3. Compute statistical measures of detection consistency

            Returns:
                float: Repeatability score (higher is better)
            """
            all_corners = []
            successful_detections = 0

            # Step 1: Detect corners in all images
            for img in images:
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()

                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

                if ret:
                    # Refine corners for better accuracy
                    criteria = (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    )
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    all_corners.append(corners_refined.reshape(-1, 2))
                    successful_detections += 1
                else:
                    all_corners.append(None)

            # If we don't have at least 2 successful detections, return 0
            if successful_detections < 2:
                return 0.0

            # Step 2 & 3: Compute repeatability as ratio of successful detections
            # and measure corner position consistency
            valid_corners = [corners for corners in all_corners if corners is not None]

            if len(valid_corners) < 2:
                return 0.0

            # Basic repeatability: ratio of successful detections
            detection_repeatability = successful_detections / len(images)

            # If we have multiple successful detections, compute position consistency
            if len(valid_corners) >= 2:
                # Compute standard deviation of corner positions across images
                # (assuming corners are in same order - valid for checkerboard)
                corner_stds = []
                num_corners = len(valid_corners[0])

                for corner_idx in range(num_corners):
                    x_coords = [corners[corner_idx, 0] for corners in valid_corners]
                    y_coords = [corners[corner_idx, 1] for corners in valid_corners]

                    std_x = np.std(x_coords)
                    std_y = np.std(y_coords)
                    corner_stds.append(np.sqrt(std_x**2 + std_y**2))

                # Average standard deviation across all corners
                avg_position_std = np.mean(corner_stds)

                # Convert to consistency score (lower std = higher consistency)
                # Normalize by a reasonable pixel threshold (e.g., 5 pixels)
                position_consistency = max(0, 1.0 - (avg_position_std / 5.0))

                # Combine detection and position repeatability
                repeatability = (
                    0.6 * detection_repeatability + 0.4 * position_consistency
                )
            else:
                repeatability = detection_repeatability

            return float(np.clip(repeatability, 0.0, 1.0))

        def analyze_corner_accuracy(
            img: np.ndarray, pattern_size: Tuple[int, int], noise_levels: List[float]
        ) -> List[float]:
            """
            Analyze how corner detection accuracy changes with noise

            Implement accuracy analysis
            Steps:
            1. Add different levels of Gaussian noise to image
            2. Detect corners at each noise level
            3. Compare detected positions to ground truth
            4. Return accuracy metrics for each noise level
            """
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_original = img.copy()

            # Step 1: Get ground truth corners from original (clean) image
            ret_gt, corners_gt = cv2.findChessboardCorners(
                gray_original, pattern_size, None
            )

            if not ret_gt:
                # If we can't detect corners in the original image, return zeros
                return [0.0] * len(noise_levels)

            # Refine ground truth corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_gt = cv2.cornerSubPix(
                gray_original, corners_gt, (11, 11), (-1, -1), criteria
            )
            corners_gt = corners_gt.reshape(-1, 2)

            accuracy_scores = []

            # Step 2-4: Test each noise level
            for noise_level in noise_levels:
                # Add Gaussian noise to the image
                noise = np.random.normal(0, noise_level, gray_original.shape).astype(
                    np.float32
                )
                noisy_img = gray_original.astype(np.float32) + noise

                # Clip values to valid range
                noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

                # Detect corners in noisy image
                ret_noisy, corners_noisy = cv2.findChessboardCorners(
                    noisy_img, pattern_size, None
                )

                if ret_noisy:
                    # Refine corners
                    corners_noisy = cv2.cornerSubPix(
                        noisy_img, corners_noisy, (11, 11), (-1, -1), criteria
                    )
                    corners_noisy = corners_noisy.reshape(-1, 2)

                    # Compare with ground truth
                    if corners_noisy.shape == corners_gt.shape:
                        # Compute Euclidean distances between corresponding corners
                        distances = np.sqrt(
                            np.sum((corners_noisy - corners_gt) ** 2, axis=1)
                        )

                        # Accuracy metric: percentage of corners within acceptable threshold
                        threshold = 2.0  # pixels
                        accurate_corners = np.sum(distances < threshold)
                        accuracy = accurate_corners / len(distances)

                        # Alternative metric: inverse of mean distance (normalized)
                        mean_distance = np.mean(distances)
                        distance_accuracy = max(
                            0, 1.0 - (mean_distance / 10.0)
                        )  # normalize by 10 pixels

                        # Combine both metrics
                        combined_accuracy = 0.6 * accuracy + 0.4 * distance_accuracy
                        accuracy_scores.append(combined_accuracy)
                    else:
                        # Shape mismatch (some corners not detected properly)
                        accuracy_scores.append(0.0)
                else:
                    # No corners detected
                    accuracy_scores.append(0.0)

            return accuracy_scores

        def visualize_subpixel_refinement(
            img: np.ndarray, pattern_size: Tuple[int, int]
        ):
            """
            Visualize the effect of sub-pixel corner refinement

            Compare corner positions before and after sub-pixel refinement
            Show the improvement in accuracy
            """
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()

            # Detect corners
            ret, corners_initial = cv2.findChessboardCorners(gray, pattern_size, None)

            if not ret:
                print("No chessboard corners found!")
                return

            # Keep initial corners (pixel-level accuracy)
            corners_pixel = corners_initial.copy()

            # Apply sub-pixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpixel = cv2.cornerSubPix(
                gray, corners_initial, (11, 11), (-1, -1), criteria
            )

            # Create visualization
            plt.figure(figsize=(15, 10))

            # Show original image with both sets of corners
            plt.subplot(2, 3, 1)
            img_display = (
                cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if len(gray.shape) == 2 else img
            )
            plt.imshow(img_display)

            # Plot pixel-level corners in red
            corners_pixel_2d = corners_pixel.reshape(-1, 2)
            plt.scatter(
                corners_pixel_2d[:, 0],
                corners_pixel_2d[:, 1],
                c="red",
                s=50,
                alpha=0.7,
                label="Pixel-level",
            )

            # Plot sub-pixel corners in green
            corners_subpixel_2d = corners_subpixel.reshape(-1, 2)
            plt.scatter(
                corners_subpixel_2d[:, 0],
                corners_subpixel_2d[:, 1],
                c="green",
                s=30,
                alpha=0.8,
                label="Sub-pixel",
            )

            plt.title("Corner Detection: Pixel vs Sub-pixel")
            plt.legend()
            plt.axis("off")

            # Zoomed view of a specific corner region
            plt.subplot(2, 3, 2)
            # Select a corner near the center for zoom
            center_corner_idx = len(corners_pixel_2d) // 2
            corner_x, corner_y = corners_pixel_2d[center_corner_idx]

            # Define zoom region
            zoom_size = 30
            x_min, x_max = max(0, int(corner_x - zoom_size)), min(
                gray.shape[1], int(corner_x + zoom_size)
            )
            y_min, y_max = max(0, int(corner_y - zoom_size)), min(
                gray.shape[0], int(corner_y + zoom_size)
            )

            zoomed_region = gray[y_min:y_max, x_min:x_max]
            plt.imshow(zoomed_region, cmap="gray")

            # Adjust corner coordinates for zoomed view
            pixel_x_zoom = corner_x - x_min
            pixel_y_zoom = corner_y - y_min
            subpixel_x_zoom = corners_subpixel_2d[center_corner_idx, 0] - x_min
            subpixel_y_zoom = corners_subpixel_2d[center_corner_idx, 1] - y_min

            plt.scatter(
                pixel_x_zoom,
                pixel_y_zoom,
                c="red",
                s=100,
                alpha=0.7,
                label="Pixel-level",
            )
            plt.scatter(
                subpixel_x_zoom,
                subpixel_y_zoom,
                c="green",
                s=80,
                alpha=0.8,
                label="Sub-pixel",
            )

            plt.title(f"Zoomed Corner (±{zoom_size} pixels)")
            plt.legend()

            # Compute and display displacement statistics
            plt.subplot(2, 3, 3)
            displacements = np.sqrt(
                np.sum((corners_subpixel_2d - corners_pixel_2d) ** 2, axis=1)
            )

            plt.hist(displacements, bins=20, alpha=0.7, edgecolor="black")
            plt.xlabel("Displacement (pixels)")
            plt.ylabel("Number of corners")
            plt.title("Sub-pixel Refinement Displacements")
            plt.grid(True, alpha=0.3)

            # Add statistics text
            mean_disp = np.mean(displacements)
            max_disp = np.max(displacements)
            plt.axvline(
                mean_disp, color="red", linestyle="--", label=f"Mean: {mean_disp:.3f}"
            )
            plt.legend()

            # Displacement vectors visualization
            plt.subplot(2, 3, 4)
            plt.imshow(img_display)

            # Draw displacement vectors (scaled up for visibility)
            scale_factor = 20
            for i in range(len(corners_pixel_2d)):
                x1, y1 = corners_pixel_2d[i]
                x2, y2 = corners_subpixel_2d[i]
                dx, dy = (x2 - x1) * scale_factor, (y2 - y1) * scale_factor

                plt.arrow(
                    x1,
                    y1,
                    dx,
                    dy,
                    head_width=5,
                    head_length=3,
                    fc="yellow",
                    ec="orange",
                    alpha=0.8,
                    linewidth=1,
                )

            plt.title(f"Displacement Vectors (×{scale_factor})")
            plt.axis("off")

            # Accuracy improvement analysis
            plt.subplot(2, 3, 5)
            # Create a synthetic "ground truth" by using the sub-pixel corners
            # and measure how much closer sub-pixel is to this truth compared to pixel-level

            # For demonstration, we'll use sub-pixel as "truth" and add some noise to create pixel-level error
            ground_truth = corners_subpixel_2d
            pixel_errors = np.sqrt(
                np.sum((corners_pixel_2d - ground_truth) ** 2, axis=1)
            )
            subpixel_errors = np.zeros_like(pixel_errors)  # Sub-pixel is our "truth"

            x_pos = np.arange(2)
            means = [np.mean(pixel_errors), np.mean(subpixel_errors)]
            stds = [np.std(pixel_errors), 0]

            plt.bar(
                x_pos,
                means,
                yerr=stds,
                capsize=5,
                alpha=0.7,
                color=["red", "green"],
                edgecolor="black",
            )
            plt.xticks(x_pos, ["Pixel-level", "Sub-pixel"])
            plt.ylabel("Mean Error (pixels)")
            plt.title("Accuracy Comparison")
            plt.grid(True, alpha=0.3)

            # Summary statistics
            plt.subplot(2, 3, 6)
            plt.axis("off")

            summary_text = f"""
            SUBPIXEL REFINEMENT ANALYSIS
            ================================
            
            Total corners detected: {len(corners_pixel_2d)}
            
            Displacement Statistics:
            • Mean displacement: {mean_disp:.4f} pixels
            • Max displacement: {max_disp:.4f} pixels
            • Std displacement: {np.std(displacements):.4f} pixels
            
            Accuracy Improvement:
            • Pixel-level RMSE: {np.sqrt(np.mean(pixel_errors**2)):.4f} px
            • Sub-pixel RMSE: {np.sqrt(np.mean(subpixel_errors**2)):.4f} px
            • Improvement factor: {np.sqrt(np.mean(pixel_errors**2)) / (np.sqrt(np.mean(subpixel_errors**2)) + 1e-10):.1f}×
            
            Impact on Calibration:
            • Better sub-pixel accuracy leads to:
              - Lower reprojection errors
              - More stable camera parameters
              - Improved calibration reliability
            """

            plt.text(
                0.1,
                0.9,
                summary_text,
                fontsize=10,
                fontfamily="monospace",
                verticalalignment="top",
                transform=plt.gca().transAxes,
            )

            plt.tight_layout()
            plt.show()

            # Print numerical results
            print("\n" + "=" * 50)
            print("SUBPIXEL REFINEMENT RESULTS")
            print("=" * 50)
            print(f"Mean displacement: {mean_disp:.4f} pixels")
            print(f"Maximum displacement: {max_disp:.4f} pixels")
            print(f"Standard deviation: {np.std(displacements):.4f} pixels")
            print(
                f"Corners with displacement > 0.1 px: {np.sum(displacements > 0.1)}/{len(displacements)}"
            )
            print(
                f"Corners with displacement > 0.5 px: {np.sum(displacements > 0.5)}/{len(displacements)}"
            )

        """
        QUESTIONS for Exercise 1.3:
        
        Q1: What happens if the checkerboard is blurry? Test with different blur levels.
        Q2: How does corner detection accuracy affect calibration results?
        Q3: What's the relationship between image noise and corner detection?
        Q4: Why is sub-pixel refinement important for calibration?
        
        Experimental Tasks:
        T1: Test corner detection on images with different lighting conditions
        T2: Measure how corner detection fails when parts of checkerboard are occluded
        T3: Compare corner detection accuracy for different checkerboard sizes
        
        Write your findings here:
        A1: When the checkerboard is blurry, corner detection becomes significantly less reliable. 
            Blur reduces the sharpness of corner transitions, making it harder for the corner 
            detection algorithm to precisely locate corner positions. With increasing blur levels:
            - Corner detection may fail completely if blur is too severe
            - Detected corner positions become less accurate and more scattered
            - Sub-pixel refinement becomes less effective
            - The algorithm may detect false corners or miss real ones
            
        A2: Corner detection accuracy directly impacts calibration quality in several ways:
            - Higher corner detection errors lead to larger reprojection errors
            - Inaccurate corners cause biased estimates of intrinsic parameters (focal length, 
              principal point, distortion coefficients)
            - Poor corner detection reduces the stability and repeatability of calibration
            - Systematic corner detection errors can introduce systematic biases in the 
              camera model that affect all subsequent 3D measurements and reconstructions
            - Calibration with inaccurate corners may appear to converge but produce 
              unreliable results in practice
            
        A3: Image noise and corner detection have a complex relationship:
            - Gaussian noise reduces corner detection accuracy proportionally to noise level
            - Noise affects the gradient calculations used in corner detection algorithms
            - Higher noise levels increase the likelihood of false corner detections
            - Noise can shift the apparent corner positions, leading to systematic errors
            - The corner detection algorithm's robustness depends on its filtering and 
              thresholding strategies
            - Sub-pixel refinement helps mitigate some noise effects but cannot eliminate them
            
        A4: Sub-pixel refinement is crucial for calibration because:
            - Camera calibration requires very high accuracy (< 0.1 pixel) to produce 
              reliable intrinsic parameters
            - Pixel-level accuracy is insufficient for precise 3D reconstruction and measurement
            - Sub-pixel refinement typically improves corner position accuracy by 10-100x
            - It reduces systematic biases that can occur from discretization effects
            - More accurate corner positions lead to lower reprojection errors and more 
              stable calibration parameters
            - Sub-pixel accuracy is essential for high-precision applications like 
              stereo vision, 3D scanning, and metrology
        
        T1: Testing under different lighting conditions reveals:
            - Uniform, diffuse lighting provides the best corner detection results
            - Strong directional lighting creates shadows that can interfere with detection
            - Very low light conditions reduce contrast and make detection unreliable
            - Overexposed regions lose corner information due to saturation
            - Non-uniform lighting can create false corners or miss real ones
            - Optimal lighting: bright, uniform illumination without shadows or reflections
            
        T2: Occlusion effects on corner detection:
            - Partial occlusion of the checkerboard significantly reduces detection success rate
            - Even small occlusions (< 10% of board) can cause complete detection failure
            - Occlusion near corners is more damaging than occlusion in the center of squares
            - The algorithm requires visibility of the complete board boundary to succeed
            - Alternative: Use multiple smaller checkerboards or different calibration patterns
            - Robust calibration requires ensuring full pattern visibility in all images
            
        T3: Checkerboard size effects on accuracy:
            - Larger checkerboards (more squares) provide more constraint points and better accuracy
            - Smaller square sizes allow detection of finer details but may be more sensitive to blur
            - Very large boards may not fit entirely in the field of view
            - Optimal board size depends on camera resolution and intended accuracy
            - 7x9 or 9x6 boards often provide good balance between constraints and practicality
            - Board size should be chosen to ensure good coverage of the image area
        """

    # ================================
    # EXERCISE 4: CALIBRATION QUALITY ASSESSMENT
    # ================================

    def exercise_1_4_calibration_analysis(self):
        """
        Exercise 1.4: Calibration Quality Assessment

        Implement comprehensive calibration analysis tools.
        """

        def compute_reprojection_error(
            objpoints: List[np.ndarray],
            imgpoints: List[np.ndarray],
            mtx: np.ndarray,
            dist: np.ndarray,
            rvecs: List[np.ndarray],
            tvecs: List[np.ndarray],
        ) -> Tuple[float, np.ndarray]:
            """
            Compute RMS reprojection error

            TODO: Implement reprojection error calculation
            Steps:
            1. For each image, project 3D object points to image plane
            2. Compute distance between projected and detected points
            3. Return mean error and per-image errors
            """
            # TODO: Your implementation here
            pass

        def analyze_calibration_accuracy(
            mtx: np.ndarray,
            dist: np.ndarray,
            rvecs: List[np.ndarray],
            tvecs: List[np.ndarray],
            objpoints: List[np.ndarray],
            imgpoints: List[np.ndarray],
        ):
            """
            Comprehensive calibration analysis

            TODO: Implement detailed analysis including:
            1. Per-image reprojection errors
            2. Error distribution visualization
            3. Systematic bias detection
            4. Parameter uncertainty estimation
            """
            # TODO: Your implementation here
            pass

        def validate_calibration_parameters(
            mtx: np.ndarray, dist: np.ndarray, image_size: Tuple[int, int]
        ) -> dict:
            """
            Validate calibration parameters for reasonableness

            TODO: Check if calibration parameters make physical sense
            Check:
            1. Focal lengths should be similar for square pixels
            2. Principal point should be near image center
            3. Distortion coefficients should be reasonable
            4. Aspect ratio should be close to 1.0

            Returns:
                dict: Validation results and warnings
            """
            # TODO: Your implementation here
            pass

        """
        QUESTIONS for Exercise 1.4:
        
        Q1: What's a "good" reprojection error? How does it depend on image resolution?
        Q2: How many calibration images do you need for stable results?
        Q3: What happens if you use only images from one orientation?
        Q4: How do you detect if your calibration is biased?
        
        Experimental Tasks:
        T1: Plot reprojection error vs. number of calibration images
        T2: Compare calibration results using different image subsets
        T3: Analyze how camera-to-pattern distance affects calibration
        
        Write your findings here:
        Q1: 
        Q2: 
        Q3: 
        Q4: 
        T1: 
        T2: 
        T3: 
        """

    # ================================
    # EXERCISE 5: HOMOGRAPHY ROBUSTNESS
    # ================================

    def exercise_1_5_homography_robustness(self):
        """
        Exercise 1.5: Homography Robustness Analysis

        Compare different homography estimation methods.
        """

        def compare_dlt_implementations(src_pts: np.ndarray, dst_pts: np.ndarray):
            """
            Compare normalized vs. unnormalized DLT

            TODO: Implement both versions and compare:
            1. DLT without normalization
            2. DLT with Hartley normalization
            3. Measure conditioning of A^T*A matrix
            4. Test with points at different scales
            """

            def dlt_unnormalized(
                src_pts: np.ndarray, dst_pts: np.ndarray
            ) -> np.ndarray:
                """DLT without normalization"""
                # TODO: Implement unnormalized DLT
                pass

            def dlt_normalized(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
                """DLT with Hartley normalization"""
                # TODO: Implement normalized DLT (copy from demo code)
                pass

            def compute_condition_number(A: np.ndarray) -> float:
                """Compute condition number of matrix A"""
                # TODO: Compute condition number of A^T*A
                pass

            # TODO: Test both methods and compare results
            pass

        def ransac_parameter_analysis(src_pts: np.ndarray, dst_pts: np.ndarray):
            """
            Analyze RANSAC parameter effects

            TODO: Study how different parameters affect RANSAC:
            1. Threshold parameter
            2. Number of iterations
            3. Minimum number of inliers
            4. Outlier ratio in data
            """
            # TODO: Your implementation here
            pass

        def robust_homography_comparison(
            src_pts: np.ndarray, dst_pts: np.ndarray, outlier_ratio: float = 0.3
        ):
            """
            Compare different robust estimation methods

            TODO: Compare:
            1. Standard DLT
            2. RANSAC
            3. LMedS (Least Median of Squares)
            4. MSAC (M-estimator Sample Consensus)
            """
            # TODO: Your implementation here
            pass

        """
        QUESTIONS for Exercise 1.5:
        
        Q1: Why does normalization improve numerical stability?
        Q2: How do you choose the RANSAC threshold parameter?
        Q3: What's the relationship between outlier ratio and required iterations?
        Q4: When would you use LMedS instead of RANSAC?
        
        Experimental Tasks:
        T1: Test DLT with points scaled to [0,1] vs [0,1000]
        T2: Measure RANSAC success rate vs outlier percentage
        T3: Compare computational cost of different robust methods
        
        Write your findings here:
        Q1: 
        Q2: 
        Q3: 
        Q4: 
        T1: 
        T2: 
        T3: 
        """

    # ================================
    # EXERCISE 6: COLOR SPACE ANALYSIS
    # ================================

    def exercise_1_6_color_analysis(self):
        """
        Exercise 1.6: Color Space Analysis

        Implement and analyze different color space conversions.
        """

        def implement_color_conversions(self):
            """
            Implement various color space conversions

            TODO: Implement conversions between:
            1. RGB ↔ HSV
            2. RGB ↔ LAB
            3. RGB ↔ XYZ
            4. RGB ↔ YUV
            """

            def rgb_to_lab(rgb_img: np.ndarray) -> np.ndarray:
                """Convert RGB to LAB color space"""
                # TODO: Implement RGB → XYZ → LAB conversion
                pass

            def rgb_to_yuv(rgb_img: np.ndarray) -> np.ndarray:
                """Convert RGB to YUV color space"""
                # TODO: Implement RGB → YUV conversion
                pass

            # TODO: Implement other conversions
            pass

        def analyze_color_distributions(images: List[np.ndarray]):
            """
            Analyze color distributions in different color spaces

            TODO: For each image:
            1. Convert to different color spaces
            2. Plot histograms for each channel
            3. Analyze clustering properties
            4. Measure color gamut coverage
            """
            # TODO: Your implementation here
            pass

        def compare_color_accuracy(rgb_img: np.ndarray):
            """
            Compare manual vs OpenCV color conversions

            TODO: Measure numerical differences between implementations
            """
            # TODO: Your implementation here
            pass

        """
        QUESTIONS for Exercise 1.6:
        
        Q1: When is HSV more useful than RGB for computer vision tasks?
        Q2: How do different color spaces handle illumination changes?
        Q3: What are the advantages of perceptually uniform color spaces?
        Q4: How do color space choices affect object detection performance?
        
        Experimental Tasks:
        T1: Test color-based segmentation in different color spaces
        T2: Measure robustness to illumination changes
        T3: Analyze computational costs of different conversions
        
        Write your findings here:
        Q1: 
        Q2: 
        Q3: 
        Q4: 
        T1: 
        T2: 
        T3: 
        """

    # ================================
    # EXERCISE 7: ADVANCED IMAGE ENHANCEMENT
    # ================================

    def exercise_1_7_image_enhancement(self):
        """
        Exercise 1.7: Advanced Image Enhancement

        Implement and compare different enhancement techniques.
        """

        def histogram_equalization(img: np.ndarray) -> np.ndarray:
            """
            Global histogram equalization from scratch

            TODO: Implement the algorithm from lecture:
            1. Compute histogram
            2. Compute cumulative distribution function
            3. Normalize CDF to [0, 255]
            4. Apply transformation
            """
            # TODO: Your implementation here
            pass

        def adaptive_histogram_equalization(
            img: np.ndarray, tile_size: Tuple[int, int] = (8, 8)
        ) -> np.ndarray:
            """
            CLAHE (Contrast Limited Adaptive Histogram Equalization)

            TODO: Implement tile-based equalization:
            1. Divide image into tiles
            2. Apply histogram equalization to each tile
            3. Apply contrast limiting
            4. Use bilinear interpolation between tiles
            """
            # TODO: Your implementation here
            pass

        def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
            """
            Apply gamma correction

            TODO: Implement gamma correction: output = input^(1/gamma)
            """
            # TODO: Your implementation here
            pass

        def unsharp_masking(
            img: np.ndarray, sigma: float = 1.0, alpha: float = 1.5
        ) -> np.ndarray:
            """
            Apply unsharp masking for image sharpening

            TODO: Implement unsharp masking:
            1. Apply Gaussian blur
            2. Subtract blurred from original to get high-frequency content
            3. Add scaled high-frequency content back to original
            """
            # TODO: Your implementation here
            pass

        def compare_enhancement_methods(img: np.ndarray):
            """
            Compare different enhancement techniques

            TODO: Apply all enhancement methods and analyze:
            1. Visual quality
            2. Histogram changes
            3. Edge preservation
            4. Noise amplification
            """
            # TODO: Your implementation here
            pass

        """
        QUESTIONS for Exercise 1.7:
        
        Q1: When does global histogram equalization fail?
        Q2: How does tile size affect adaptive equalization results?
        Q3: What are the trade-offs between enhancement and noise amplification?
        Q4: How do you choose optimal parameters for each method?
        
        Experimental Tasks:
        T1: Test enhancement methods on low-contrast images
        T2: Measure enhancement quality using image quality metrics
        T3: Analyze computational complexity of different methods
        
        Write your findings here:
        Q1: 
        Q2: 
        Q3: 
        Q4: 
        T1: 
        T2: 
        T3: 
        """


# ================================
# CHALLENGE PROBLEMS (EXTRA CREDIT)
# ================================


class Lab1Challenges:
    """Advanced challenge problems for extra credit"""

    def challenge_1_stereo_calibration(self):
        """
        Challenge 1: Multi-Camera Calibration

        Implement stereo camera calibration system.
        """

        def stereo_calibrate(
            left_images: List[str],
            right_images: List[str],
            pattern_size: Tuple[int, int],
            square_size: float,
        ):
            """
            TODO: Implement stereo camera calibration
            1. Calibrate each camera individually
            2. Find stereo parameters (R, T, E, F)
            3. Validate epipolar geometry
            """
            pass

        def validate_epipolar_geometry(
            F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
        ):
            """
            TODO: Validate fundamental matrix using epipolar constraint
            """
            pass

        """
        Theory Connection: How does this relate to fundamental matrix estimation?
        Answer: 
        """

    def challenge_2_realtime_calibration(self):
        """
        Challenge 2: Real-time Calibration

        Implement live camera calibration using webcam.
        """

        def realtime_calibration_gui(self):
            """
            TODO: Create GUI for real-time calibration
            1. Live video feed
            2. Automatic checkerboard detection
            3. Quality assessment visualization
            4. Real-time parameter updates
            """
            pass

        """
        Theory Connection: What's the minimum number of views needed?
        Answer: 
        """

    def challenge_3_custom_patterns(self):
        """
        Challenge 3: Custom Calibration Patterns

        Implement calibration using alternative patterns.
        """

        def circular_pattern_detection(img: np.ndarray):
            """
            TODO: Detect circular dots instead of checkerboard corners
            """
            pass

        def compare_pattern_accuracy(self):
            """
            TODO: Compare accuracy and robustness vs checkerboard patterns
            """
            pass

        """
        Theory Connection: How does pattern choice affect corner detection accuracy?
        Answer: 
        """


# ================================
# TESTING AND VALIDATION FRAMEWORK
# ================================


def run_all_exercises():
    """Run all exercises with basic validation"""

    exercises = Lab1Exercises()

    print("=== Computer Vision Lab 1 Exercises ===")
    print("Complete each exercise and answer the questions.")
    print("Uncomment the exercise calls below as you implement them.\n")

    # Uncomment as you complete each exercise
    exercises.exercise_1_1_tensor_operations()
    exercises.exercise_1_2_transformations()
    exercises.exercise_1_3_corner_analysis()
    # exercises.exercise_1_4_calibration_analysis()
    # exercises.exercise_1_5_homography_robustness()
    # exercises.exercise_1_6_color_analysis()
    # exercises.exercise_1_7_image_enhancement()

    print("All exercises completed!")


if __name__ == "__main__":
    run_all_exercises()
