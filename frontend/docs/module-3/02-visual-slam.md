---
sidebar_position: 2
---

# Visual SLAM

Visual Simultaneous Localization and Mapping (SLAM) represents one of the most significant achievements in mobile robotics, enabling robots to build maps of unknown environments while simultaneously determining their location within those maps. Unlike traditional localization methods that rely on pre-existing maps or external positioning systems, visual SLAM uses only visual observations from onboard cameras, making it applicable to a wide range of scenarios from indoor navigation to planetary exploration. This chapter provides a comprehensive treatment of visual SLAM, covering fundamental concepts, algorithmic approaches, implementation details, and practical considerations for humanoid robotics applications.

## The SLAM Problem and Its Significance

The simultaneous localization and mapping problem has been called the "chicken and egg" problem of robotics because it presents a circular dependency: to build an accurate map, the robot needs to know its location precisely, but to determine its location accurately, the robot needs a detailed map of its environment. This fundamental challenge captured the attention of robotics researchers for decades and the solutions that emerged have transformed what robots can accomplish autonomously.

Visual SLAM specifically uses camera images as the primary sensing modality for addressing this problem. The appeal of vision-based approaches lies in their versatility and cost-effectiveness. Cameras are inexpensive, low-power sensors that can provide rich information about the environment including colors, textures, and structural details. For humanoid robots, which operate in human environments designed for visual perception, cameras are particularly well-suited as the primary sensing modality.

The SLAM problem can be formally defined as follows: given a sequence of sensor measurements collected as a robot moves through an unknown environment, simultaneously estimate the robot's trajectory (positions and orientations over time) and construct a map of the environment. This estimation must be performed online, in real-time, as the robot explores, without access to ground truth information about either the map or the trajectory.

For humanoid robots, visual SLAM provides the foundational localization capability that enables higher-level tasks such as path planning, manipulation planning, and human-robot interaction. A humanoid robot entering an unknown room must quickly build a mental model of the space—where are the walls, where is the furniture, where are the obstacles—while simultaneously tracking its own position relative to these landmarks. This capability is essential for tasks ranging from household chores to search and rescue operations.

## Fundamental Concepts in Visual SLAM

Understanding visual SLAM requires grasping several interconnected concepts from computer vision, optimization theory, and robotics. These concepts form the theoretical foundation upon which all visual SLAM systems are built.

### The Geometry of Camera Projection

Cameras create a projection from the 3D world onto a 2D image plane. This projection can be modeled using the pinhole camera model, which describes how 3D points in the world are mapped to 2D pixel coordinates. The projection equation involves the camera's intrinsic parameters (focal lengths and principal point) and extrinsic parameters (rotation and translation relative to a world coordinate frame).

The intrinsic parameters are typically represented by a 3×3 camera calibration matrix K that transforms normalized image coordinates to pixel coordinates. These parameters are fixed for a given camera and can be determined through a calibration process using known calibration patterns. The extrinsic parameters describe the camera's pose in the world, consisting of a 3×3 rotation matrix R and a 3×1 translation vector t that together form a 4×4 homogeneous transformation matrix.

Real cameras deviate from the ideal pinhole model due to lens distortion, particularly radial distortion that causes straight lines to appear curved near image edges. Accurate visual SLAM requires calibrating these distortion parameters and applying correction transforms to input images before processing. The distortion model typically includes parameters k1, k2, k3 for radial distortion and p1, p2 for tangential distortion.

### Feature Detection and Description

Visual SLAM systems rely on identifying distinctive visual features in images that can be reliably matched across multiple views. These features should be repeatable, meaning the same physical point in the world should be detected consistently across different viewing conditions, and distinctive, meaning features can be distinguished from one another to establish correct correspondences.

The classical approach to visual features involves detector-descriptor pairs. Feature detectors identify locations in the image that are likely to be repeatable, such as corners, blobs, or edge junctions. Feature descriptors compute a compact representation of the local image patch around each detected feature point that enables matching. The most widely used feature detection and description algorithms include:

**Harris Corner Detector** identifies corners by analyzing the local autocorrelation matrix of image intensities. At corners, the image intensity changes significantly in multiple directions, resulting in large eigenvalues in the autocorrelation matrix. The Harris response function combines these eigenvalues to produce a corner response that is high at corner locations and low at uniform regions and edges.

**FAST (Features from Accelerated Segment Test)** provides a computationally efficient corner detector that examines a circular pattern of pixels around a candidate location. A pixel is considered a corner if a contiguous arc of pixels around it are all significantly brighter or darker than the center pixel. FAST achieves real-time performance by avoiding expensive gradient computations.

**SIFT (Scale-Invariant Feature Transform)** detects features across multiple scales using a scale-space pyramid and identifies keypoints as local extrema in the Difference of Gaussians. SIFT descriptors are computed as histograms of gradient orientations in a local patch, made invariant to rotation through orientation normalization. While highly effective, SIFT was historically limited by patent restrictions.

**ORB (Oriented FAST and Rotated BRIEF)** addresses the computational efficiency and licensing limitations of earlier methods. ORB uses FAST for detection with added orientation measurement and rotation invariance, and BRIEF for binary descriptor computation with learning-based improvements for robustness. The combination provides good performance with minimal computational cost, making it popular for real-time applications.

### Epipolar Geometry and Fundamental Matrix

When two cameras observe the same scene from different viewpoints, the geometric relationship between the images is governed by epipolar geometry. The fundamental matrix F is a 3×3 rank-2 matrix that encodes this relationship, relating corresponding points in the two images through the epipolar constraint equation x'^T F x = 0.

The epipolar constraint states that for any point correspondence (x, x') between two views, the point x' must lie on the line defined by the epipole e' in the second image, where e' is the projection of the first camera center into the second image. This constraint provides a powerful geometric prior that can be used to verify the correctness of feature correspondences and to recover the relative camera motion up to scale.

The fundamental matrix can be estimated from a set of point correspondences using the eight-point algorithm or its more robust variants such as RANSAC (Random Sample Consensus). Given the fundamental matrix and the camera calibration matrix K, the essential matrix E can be computed, which encodes the relative rotation and translation between cameras in normalized coordinates.

### Bundle Adjustment and Optimization

Bundle adjustment is the process of jointly refining 3D point positions and camera parameters to minimize the reprojection error—the discrepancy between observed feature positions in images and the predicted positions based on the current estimate of 3D structure and camera poses. This optimization problem is typically formulated as a nonlinear least-squares problem and solved using Levenberg-Marquardt or similar iterative algorithms.

The mathematical formulation of bundle adjustment considers N 3D points observed by M cameras, with each observation providing a constraint that the image of the point should coincide with the observed feature location. The optimization minimizes the sum of squared reprojection errors over all observations, weighted by measurement uncertainties. The optimization variables include the 3D coordinates of each point and the pose parameters of each camera.

In visual SLAM, bundle adjustment serves multiple purposes. During mapping, it refines the map to achieve maximum accuracy by jointly optimizing all landmark positions and camera positions. It also provides a natural framework for loop closure detection, where revisiting a previously mapped area creates additional constraints that can dramatically improve map consistency.

## Visual SLAM Algorithmic Approaches

Visual SLAM systems can be broadly categorized into three approaches based on how they process visual information: feature-based methods, direct methods, and hybrid approaches. Each category has distinct advantages and trade-offs that make it suitable for different applications.

### Feature-Based Visual SLAM

Feature-based visual SLAM systems operate on a sparse set of feature points detected in each frame. The typical pipeline consists of feature extraction, feature matching, motion estimation, map maintenance, and loop closure detection. This approach has been extensively developed and refined, with systems like PTAM, ORB-SLAM, and VINS-Mono representing significant milestones.

The PTAM (Parallel Tracking and Mapping) system, introduced in 2007, was revolutionary in demonstrating that real-time visual SLAM was achievable on consumer hardware. PTAM introduced the keyframe-based approach that separates tracking and mapping into parallel threads, enabling more computationally intensive optimization on the mapping thread while maintaining real-time tracking on the tracking thread. This architecture remains influential in modern visual SLAM systems.

ORB-SLAM represents the current state-of-the-art in feature-based visual SLAM, offering a complete solution with tracking, mapping, relocalization, and loop closing capabilities. The system uses ORB features throughout, providing good computational efficiency while maintaining robustness. Key innovations include a covisibility graph for efficient map management, an essential graph for fast loop closure, and a system for generating and using bags of visual words for place recognition.

The algorithmic structure of a feature-based visual SLAM system proceeds as follows. When a new frame arrives, the tracking thread extracts features and attempts to match them with features from the previous frame using descriptor distance. Initial motion is estimated using a perspective-n-point algorithm that solves for camera pose given 3D-2D correspondences. The estimated motion is refined by minimizing reprojection error using motion-only bundle adjustment.

The mapping thread processes keyframes selected by the tracking thread. For each new keyframe, new features are triangulated to create new 3D map points. The map is maintained through local bundle adjustment that optimizes the poses of the keyframe and its neighbors along with the associated map points. Periodically, global bundle adjustment refines the entire map to eliminate accumulated drift.

Loop closure detection provides long-term consistency by recognizing previously visited locations. This is typically implemented using a place recognition module based on bags of visual words—a vector quantization of feature descriptors into a discrete vocabulary. When a loop closure is detected, a similarity transformation is computed to align the two map segments, and pose graph optimization corrects the accumulated drift.

### Direct Visual SLAM

Direct methods bypass the feature detection and description step, instead using all or most of the pixels in each image for motion estimation. This approach has several potential advantages: it can operate in texture-poor environments where few features can be detected, it avoids the information loss inherent in the feature abstraction, and it can potentially achieve higher accuracy by using more measurement information.

**Direct Sparse Odometry (DSO)** represents a highly influential direct visual odometry system. DSO formulates the problem as minimizing the photometric error—the difference between observed image intensities and intensities predicted by warping a reference frame according to the estimated motion. The system optimizes over a sparse set of points selected based on photometric novelty, maintaining a window of active frames for the optimization.

The photometric error model in direct methods explicitly accounts for camera exposure time and lens vignetting, providing better calibrated error terms than the reprojection error used in feature-based methods. However, direct methods are more sensitive to lighting changes and rolling shutter effects, and they typically require good initial estimates to avoid local minima.

**SVO (Semi-Direct Visual Odometry)** combines elements of both approaches, using direct methods for fast motion estimation while maintaining a feature-based map for relocalization and loop closure. The system first estimates motion by directly minimizing photometric error on a set of sparse patches, then refines feature positions and camera poses through bundle adjustment. This hybrid approach achieves high speed while maintaining the advantages of feature-based map representation.

### Deep Learning Approaches

Recent years have seen significant interest in applying deep learning to visual SLAM, with approaches ranging from learning individual components to learning end-to-end systems. These methods leverage the pattern recognition capabilities of neural networks to address challenges that are difficult to handle with traditional methods.

Learning-based place recognition has proven particularly successful, with networks trained to produce compact descriptors that are robust to appearance changes. These descriptors can be compared efficiently for loop closure detection, and the learned representations are often more robust to lighting and seasonal variations than hand-crafted features.

End-to-end visual odometry networks have been trained to predict camera motion directly from image sequences. These systems use convolutional and recurrent neural networks to learn the relationship between visual input and camera motion, often achieving competitive performance without explicit geometric computations. However, these systems typically struggle with generalization to new environments and may not provide the dense map outputs needed for many applications.

## Implementation with ROS 2

Implementing visual SLAM in ROS 2 requires understanding the ROS 2 communication patterns and integrating with existing SLAM frameworks. The rtabmap_ros package provides one of the most complete visual SLAM solutions available, supporting stereo, RGB-D, and monocular cameras with loop closure and graph optimization.

### rtabmap_ros Configuration

The rtabmap_ros package implements RTAB-Map (Real-Time Appearance-Based Mapping), a graph-based SLAM system that uses appearance-based loop closure detection. The system maintains a database of visual features from keyframes that enables efficient place recognition even in large-scale environments.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

class VisualSLAMNode(Node):
    """
    Visual SLAM node using rtabmap_ros.
    Integrates camera input with odometry for complete SLAM.
    """

    def __init__(self):
        super().__init__('visual_slam_node')

        # Parameters
        self.declare_parameter('subscribe_rgb', True)
        self.declare_parameter('subscribe_depth', True)
        self.declare_parameter('subscribe_rgbd', False)
        self.declare_parameter('subscribe_stereo', False)
        self.declare_parameter('subscribe_odom', True)
        self.declare_parameter('approx_sync', True)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('publish_tf', True)

        # Get parameters
        self.subscribe_rgb = self.get_parameter('subscribe_rgb').value
        self.subscribe_depth = self.get_parameter('subscribe_depth').value
        self.subscribe_odom = self.get_parameter('subscribe_odom').value
        self.frame_id = self.get_parameter('frame_id').value

        # TF broadcaster for camera pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribers
        if self.subscribe_rgb and self.subscribe_depth:
            self.rgb_sub = self.create_subscription(
                Image, '/camera/color/image_raw', self.rgb_callback, 10)
            self.depth_sub = self.create_subscription(
                Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        elif self.subscribe_odom:
            self.odom_sub = self.create_subscription(
                Odometry, '/odom', self.odom_callback, 10)

        # Publisher for SLAM map
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map', 10)

        # Internal state
        self.last_odom = None
        self.initialized = False

    def rgb_callback(self, msg):
        """Handle RGB image data."""
        if not self.initialized:
            self.initialize_slam()
            self.initialized = True

        # Process image through rtabmap
        self.process_image(msg)

    def depth_callback(self, msg):
        """Handle depth image data."""
        # Combine with RGB for RGBD SLAM
        pass

    def odom_callback(self, msg):
        """Handle odometry data for sensor fusion."""
        self.last_odom = msg

        # Broadcast transform
        self.broadcast_transform(msg)

    def broadcast_transform(self, odom_msg):
        """Broadcast camera transform from odometry."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = odom_msg.header.frame_id
        transform.child_frame_id = self.frame_id

        transform.transform.translation.x = odom_msg.pose.pose.position.x
        transform.transform.translation.y = odom_msg.pose.pose.position.y
        transform.transform.translation.z = odom_msg.pose.pose.position.z

        transform.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(transform)

    def initialize_slam(self):
        """Initialize SLAM system."""
        # Configure rtabmap parameters
        self.get_logger().info('Initializing visual SLAM system...')

    def process_image(self, image_msg):
        """Process image through SLAM algorithm."""
        # Feature extraction
        features = self.extract_features(image_msg)

        # Match with previous frame
        matches = self.match_features(features)

        # Estimate motion
        motion = self.estimate_motion(matches)

        # Update map
        self.update_map(features, motion)

        # Check for loop closure
        self.check_loop_closure()

    def extract_features(self, image_msg):
        """Extract visual features from image."""
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=2000)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        return {
            'image': cv_image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'timestamp': image_msg.header.stamp
        }

    def match_features(self, current_features):
        """Match features with previous frame."""
        if self.last_features is None:
            return []

        # Use BFMatcher with Hamming distance for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # K nearest neighbor matching
        matches = bf.knnMatch(
            self.last_features['descriptors'],
            current_features['descriptors'],
            k=2
        )

        # Apply ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches

    def estimate_motion(self, matches):
        """Estimate camera motion from feature matches."""
        if len(matches) < 10:
            return None

        # Get corresponding points
        src_pts = np.float32([
            self.last_features['keypoints'][m.queryIdx].pt
            for m in matches
        ])
        dst_pts = np.float32([
            self.current_features['keypoints'][m.trainIdx].pt
            for m in matches
        ])

        # Estimate essential matrix
        try:
            E, mask = cv2.findEssentialMat(
                src_pts, dst_pts,
                cameraMatrix=self.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            # Recover rotation and translation
            _, R, t, mask = cv2.recoverPose(
                E, src_pts, dst_pts,
                cameraMatrix=self.K
            )

            return {
                'rotation': R,
                'translation': t,
                'inliers': np.sum(mask)
            }
        except Exception as e:
            self.get_logger().warning(f'Motion estimation failed: {e}')
            return None
```

### Camera Calibration and Configuration

Proper camera calibration is essential for accurate visual SLAM. The camera must be calibrated to obtain intrinsic parameters (focal lengths, principal point, distortion coefficients) and, for stereo systems, extrinsic parameters describing the relative poses of the cameras.

```python
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class CameraCalibrator:
    """
    Camera calibration utilities for visual SLAM.
    Supports checkerboard, Charuco, and AprilTag calibration patterns.
    """

    def __init__(self, camera_model='pinhole'):
        self.camera_model = camera_model
        self.calibration_flags = (
            cv2.CALIB_FIX_K3 |  # Fix radial distortion coefficients
            cv2.CALIB_ZERO_TANGENT_DIST  # No tangential distortion
        )

        # Calibration results
        self.K = None  # Intrinsic matrix
        self.D = None  # Distortion coefficients
        self.R = None  # Rotation (for stereo)
        self.T = None  # Translation (for stereo)
        self.reprojection_error = None

    def calibrate_monocular(self, images, pattern_size, square_size):
        """
        Perform monocular camera calibration using checkerboard.

        Args:
            images: List of calibration images
            pattern_size: Tuple (cols, rows) of inner corners
            square_size: Size of square in meters
        """
        # Prepare object points
        obj_points = []
        img_points = []

        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:pattern_size[0],
            0:pattern_size[1]
        ].T.reshape(-1, 2) * square_size

        for idx, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, pattern_size, None
            )

            if ret:
                # Refine corners
                criteria = (
                    cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001
                )
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )

                img_points.append(corners_refined)
                obj_points.append(objp)

        # Perform calibration
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points,
            gray.shape[::-1],
            None, None,
            flags=self.calibration_flags
        )

        self.K = K
        self.D = D.flatten()
        self.reprojection_error = ret

        return {
            'intrinsic_matrix': K,
            'distortion_coefficients': D.flatten(),
            'reprojection_error': ret
        }

    def undistort(self, image):
        """
        Apply distortion correction to image.

        Args:
            image: Distorted input image

        Returns:
            Undistorted image
        """
        if self.K is None or self.D is None:
            raise ValueError('Calibration not performed')

        h, w = image.shape[:2]
        new_K = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(image, self.K, self.D, None, new_K)

        return undistorted

    def generate_remap(self, image_size):
        """
        Generate remap tables for efficient undistortion.

        Args:
            image_size: Tuple (width, height) of image

        Returns:
            map1, map2: Remap tables for cv2.remap
        """
        new_K = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, image_size, 1, image_size
        )

        map1, map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, new_K,
            image_size, cv2.CV_32FC1
        )

        return map1, map2
```

### Visual-Inertial Odometry Integration

Combining visual odometry with inertial measurements from an IMU significantly improves robustness and accuracy, particularly in scenarios with rapid motion or texture-poor environments where visual tracking may fail.

```python
import numpy as np
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter

class VisualInertialOdometry:
    """
    Visual-inertial odometry using extended Kalman filter fusion.
    """

    def __init__(self, params):
        # State dimension: position (3), velocity (3), orientation (3),
        # gyro bias (3), accel bias (3) = 15 states
        self.state_dim = 15
        self.meas_dim = 6  # 3D position + 3D orientation

        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(dim_x=self.state_dim, dim_z=self.meas_dim)

        # State indices
        self.IDX_POS = (0, 1, 2)
        self.IDX_VEL = (3, 4, 5)
        self.IDX_ORI = (6, 7, 8)  # Euler angles
        self.IDX_BG = (9, 10, 11)  # Gyro bias
        self.IDX_BA = (12, 13, 14)  # Accel bias

        # Initialize state
        self.initialize_state(params)

    def initialize_state(self, params):
        """Initialize filter state and covariance."""
        # Initial position at origin
        self.ekf.x[self.IDX_POS] = [0, 0, 0]

        # Initial velocity (assumed zero)
        self.ekf.x[self.IDX_VEL] = [0, 0, 0]

        # Initial orientation (identity rotation)
        r = Rotation.identity()
        euler = r.as_euler('xyz')
        self.ekf.x[self.IDX_ORI] = euler

        # Initial biases
        self.ekf.x[self.IDX_BG] = params.get('initial_gyro_bias', [0, 0, 0])
        self.ekf.x[self.IDX_BA] = params.get('initial_accel_bias', [0, 0, 0])

        # Initial covariance
        P = np.eye(self.state_dim) * 0.1
        self.ekf.P = P

        # Process noise
        self.ekf.Q = self.compute_process_noise(params)

        # Measurement noise
        self.ekf.R = np.eye(self.meas_dim) * 0.01

    def compute_process_noise(self, params):
        """Compute process noise covariance matrix."""
        dt = params['imu_dt']
        Q = np.zeros((self.state_dim, self.state_dim))

        # Position noise (integrated from velocity)
        Q[0:3, 0:3] = np.eye(3) * params['sigma_a'] * dt**3 / 3
        Q[0:3, 3:6] = np.eye(3) * params['sigma_a'] * dt**2 / 2
        Q[3:6, 0:3] = np.eye(3) * params['sigma_a'] * dt**2 / 2
        Q[3:6, 3:6] = np.eye(3) * params['sigma_a'] * dt

        # Orientation noise (from gyro)
        Q[6:9, 6:9] = np.eye(3) * params['sigma_g'] * dt

        # Bias random walk
        Q[9:12, 9:12] = np.eye(3) * params['sigma_bg'] * dt
        Q[12:15, 12:15] = np.eye(3) * params['sigma_ba'] * dt

        return Q

    def predict(self, imu_measurement, dt):
        """
        Predict state using IMU measurements.

        Args:
            imu_measurement: Dictionary with 'accel' and 'gyro' arrays
            dt: Time step in seconds
        """
        # Remove bias estimates
        accel = imu_measurement['accel'] - self.ekf.x[self.IDX_BA]
        gyro = imu_measurement['gyro'] - self.ekf.x[self.IDX_BG]

        # Get current orientation as rotation matrix
        euler = self.ekf.x[self.IDX_ORI]
        R = Rotation.from_euler('xyz', euler).as_matrix()

        # Gravity in body frame
        g = np.array([0, 0, -9.81])
        g_body = R.T @ g

        # Update velocity
        self.ekf.x[self.IDX_VEL] += (accel + g_body) * dt

        # Update position
        self.ekf.x[self.IDX_POS] += self.ekf.x[self.IDX_VEL] * dt

        # Update orientation using Rodrigues' formula
        omega = gyro * dt
        omega_norm = np.linalg.norm(omega)

        if omega_norm > 1e-10:
            omega_skew = self.skew_symmetric(omega)
            R_delta = np.eye(3) + np.sin(omega_norm) / omega_norm * omega_skew + \
                      (1 - np.cos(omega_norm)) / omega_norm**2 * omega_skew @ omega_skew
            R = R_delta @ R

        # Convert back to Euler
        new_euler = Rotation.from_matrix(R).as_euler('xyz')
        self.ekf.x[self.IDX_ORI] = new_euler

        # Predict covariance
        self.ekf.predict()

    def update(self, visual_measurement):
        """
        Update state with visual measurement.

        Args:
            visual_measurement: Dictionary with 'position' and 'orientation'
        """
        # Build measurement vector
        z = np.concatenate([
            visual_measurement['position'],
            visual_measurement['orientation']
        ])

        # Compute measurement Jacobian
        H = np.zeros((self.meas_dim, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position directly observable
        H[3:6, 6:9] = np.eye(3)  # Orientation directly observable

        # Update filter
        self.ekf.update(z, H=H)

    @staticmethod
    def skew_symmetric(v):
        """Create skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def get_pose(self):
        """Get current pose as transformation matrix."""
        position = self.ekf.x[self.IDX_POS]
        euler = self.ekf.x[self.IDX_ORI]
        rotation = Rotation.from_euler('xyz', euler).as_matrix()

        T = np.eye(4)
        T[0:3, 0:3] = rotation
        T[0:3, 3] = position

        return T
```

## Performance Evaluation and Metrics

Evaluating visual SLAM systems requires appropriate metrics that capture both accuracy and robustness. The research community has established standard evaluation protocols and metrics that enable meaningful comparison across different systems.

### Trajectory Evaluation Metrics

The most common metrics for evaluating visual SLAM trajectories are the Absolute Trajectory Error (ATE) and Relative Pose Error (RPE). These metrics capture different aspects of trajectory quality and should be used together for comprehensive evaluation.

Absolute Trajectory Error measures the discrepancy between the estimated and ground truth trajectories at corresponding time stamps. For each time stamp, the Euclidean distance between the estimated and ground truth positions is computed, and the mean, standard deviation, and root mean square of these distances provide aggregate measures of accuracy. ATE is particularly useful for evaluating systems with loop closure, as it captures the global consistency of the trajectory.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def compute_ate(estimated_poses, ground_truth_poses):
    """
    Compute Absolute Trajectory Error between estimated and ground truth trajectories.

    Args:
        estimated_poses: List of 4x4 transformation matrices (estimated)
        ground_truth_poses: List of 4x4 transformation matrices (ground truth)

    Returns:
        ate_stats: Dictionary with mean, std, rmse, max ATE
    """
    # Align trajectories using Umeyama algorithm
    estimated_positions = np.array([p[0:3, 3] for p in estimated_poses])
    gt_positions = np.array([p[0:3, 3] for p in ground_truth_poses])

    # Compute alignment
    scale, R, t = align_trajectories(estimated_positions, gt_positions)

    # Apply alignment
    aligned_positions = scale * (R @ estimated_positions.T).T + t

    # Compute errors
    errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)

    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'max': np.max(errors),
        'errors': errors
    }

def align_trajectories(est, gt):
    """
    Align trajectories using Umeyama algorithm.
    Returns scale, rotation, and translation.
    """
    n = est.shape[0]

    # Compute means
    est_mean = np.mean(est, axis=0)
    gt_mean = np.mean(gt, axis=0)

    # Center trajectories
    est_centered = est - est_mean
    gt_centered = gt - gt_mean

    # Compute variances
    est_var = np.mean(np.sum(est_centered**2, axis=1))
    gt_var = np.mean(np.sum(gt_centered**2, axis=1))

    # Compute covariance matrix
    cov = est_centered.T @ gt_centered / n

    # SVD for rotation
    U, S, Vt = np.linalg.svd(cov)

    # Handle reflection case
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])

    # Compute alignment
    R = U @ D @ Vt
    scale = np.sqrt(gt_var / est_var) if est_var > 0 else 1.0
    t = gt_mean - scale * (R @ est_mean)

    return scale, R, t
```

Relative Pose Error measures the error in incremental motions over a fixed time interval. This metric is particularly useful for evaluating visual odometry systems without loop closure, as it captures the drift rate independent of global consistency. RPE is typically computed for multiple time intervals and reported as mean translational and rotational error per meter or per second.

### Map Quality Assessment

Beyond trajectory accuracy, the quality of the reconstructed map is important for many applications. Map quality can be assessed through several dimensions:

**Geometric accuracy** can be evaluated by comparing reconstructed 3D points with ground truth measurements of the same points or surfaces. This requires a separate ground truth model of the environment, which may be obtained through high-precision surveying or structure-from-motion on carefully controlled image sets.

**Density and completeness** measure how much of the environment has been captured and how densely the reconstruction samples the surfaces. For some applications, sparse reconstructions with a few thousand points may be sufficient, while other applications may require dense surface reconstructions.

**Temporal consistency** evaluates whether the map remains stable as the robot revisits areas or as new information is incorporated. Map deformation over time indicates problems with the SLAM system's optimization or data association.

## Challenges and Advanced Solutions

Visual SLAM systems face several challenges in real-world deployment. Understanding these challenges and the techniques developed to address them is essential for building robust systems.

### Handling Dynamic Environments

Real-world environments contain moving objects—people, vehicles, animals—that can confuse visual SLAM systems. Several strategies address this challenge:

**Motion segmentation** identifies and excludes pixels belonging to moving objects from the SLAM computation. This can be achieved through optical flow analysis, background subtraction, or learning-based object detection. The detected moving objects can be tracked separately if needed for the application.

**Robust optimization** techniques such as M-estimators reduce the influence of outlier measurements that may correspond to dynamic objects. Rather than minimizing the sum of squared errors, M-estimators use robust loss functions that saturate for large errors.

**Multi-motion estimation** explicitly models and estimates the motions of independently moving objects alongside the camera motion. This approach is more complex but can handle environments with significant dynamic content.

### Illumination and Appearance Changes

Lighting conditions can vary dramatically across a robot's operating day—from bright sunlight to dim indoor lighting to artificial lighting at night. Visual SLAM systems must be robust to these changes:

**Appearance-invariant features** such as edge-based or gradient-based features are less sensitive to absolute intensity values than direct photometric matching. Similarly, learned feature descriptors trained on diverse datasets can be robust to appearance changes.

**Adaptive gain control** normalizes image intensities before processing, reducing the sensitivity to overall brightness changes. Histogram equalization and other contrast enhancement techniques can also improve feature detection under difficult lighting.

**Robust data association** uses multiple criteria for feature matching and tracks the quality of matches over time, discarding those that become unreliable due to appearance changes.

### Scale Estimation in Monocular Systems

Monocular visual SLAM systems inherently suffer from scale ambiguity—the reconstruction can only be recovered up to an unknown scale factor. This presents challenges for applications that require metric navigation:

**Scale from motion** estimates scale by analyzing the relationship between camera motion and feature parallax. Faster motion and closer features produce larger parallax, providing scale information.

**Scale from external sensors** combines visual SLAM with other sensors that provide metric information, such as IMU integration or known object sizes. The external sensor measurements fix the scale of the visual reconstruction.

**Scale drift correction** monitors the consistency of scale estimates over time and corrects for systematic scale drift that may occur in unconstrained monocular SLAM.

## Key Takeaways

Visual SLAM provides humanoid robots with the fundamental capability to understand their environment and track their position within it. The combination of theoretical foundations, algorithmic approaches, and practical implementation techniques covered in this chapter provides a comprehensive foundation for building visual SLAM systems.

- **The SLAM problem** requires simultaneously estimating robot trajectory and environment map
- **Geometric foundations** include camera projection, feature detection, epipolar geometry, and bundle adjustment
- **Algorithmic approaches** include feature-based, direct, and learning-based methods
- **ROS 2 integration** enables modular visual SLAM with rtabmap_ros
- **Performance evaluation** uses ATE and RPE metrics for trajectory accuracy
- **Real-world challenges** include dynamic environments, illumination changes, and scale ambiguity

With visual SLAM capabilities established, we can now explore navigation systems that use this localization information to plan and execute robot motion through environments.
