---
sidebar_position: 4
---

# Sensor Simulation in Robotics

Realistic sensor simulation is essential for developing Physical AI systems that transfer from simulation to reality. This chapter covers simulating cameras, LiDAR, IMUs, and other sensors that enable robots to perceive their environment.

## Camera Sensor Simulation

### RGB Camera Model

Camera simulation models the complete imaging pipeline:

```xml
<!-- Gazebo camera sensor plugin -->
<sensor name="head_camera" type="camera" >
    <camera>
        <horizontal_fov>1.396</horizontal_fov>
        <image>
            <width>1280</width>
            <height>720</height>
            <format>R8G8B8</format>
        </image>
        <clip>
            <near>0.1</near>
            <far>100</far>
        </clip>
        <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.01</stddev>
        </noise>
        <distortion>
            <k1>-0.15</k1>
            <k2>0.05</k2>
            <p1>0.0</p1>
            <p2>0.0</p2>
            <center>0.5 0.5</center>
        </distortion>
    </camera>
    <pose>0.15 0 0.1 0 0 0</pose>
</sensor>
```

The key parameters are:

**Intrinsics**: Focal length (via FOV), principal point (via distortion center), and resolution determine the projection matrix.

**Noise Model**: Gaussian noise simulates sensor read noise and photon shot noise. Real cameras have noise that varies with ISO and exposure.

**Distortion**: Radial (k1, k2) and tangential (p1, p2) distortion model lens imperfections. Most real cameras require lens distortion correction.

### Depth Camera Simulation

```xml
<sensor name="depth_camera" type="depth" >
    <camera>
        <horizontal_fov>1.396</horizontal_fov>
        <image>
            <width>640</width>
            <height>480</height>
        </image>
        <clip>
            <near>0.1</near>
            <far>10.0</far>
        </clip>
        <depth_camera>
            <depth_noise>
                <type>gaussian</type>
                <mean>0.0</mean>
                <stddev>0.02</stddev>
            </depth_noise>
        </depth_camera>
    </camera>
</sensor>
```

### Camera Noise Models

```python
import numpy as np

class CameraNoiseModel:
    """
    Models realistic camera noise including:
    - Gaussian read noise
    - Shot noise (Poisson)
    - Fixed pattern noise
    - Salt-and-pepper noise
    """

    def __init__(self, config):
        self.read_noise_std = config.get('read_noise_std', 0.01)
        self.dark_current = config.get('dark_current', 0.01)  # e-/pixel/s
        self.full_well = config.get('full_well', 10000)  # e-
        self.gain = config.get('gain', 1.0)  # e-/DN
        self.bit_depth = config.get('bit_depth', 12)

    def apply(self, image, exposure_time=1/60.0):
        """Apply noise model to clean image."""
        # Convert to electrons
        image_electrons = image * self.gain * self.full_well

        # Shot noise (Poisson)
        shot_noise = np.random.poisson(image_electrons * exposure_time)
        shot_noise = shot_noise / exposure_time / self.full_well

        # Read noise (Gaussian)
        read_noise = np.random.normal(0, self.read_noise_std, image.shape)

        # Combine
        noisy = image + shot_noise + read_noise

        # Clip and quantize
        noisy = np.clip(noisy, 0, 1)
        noisy = np.round(noisy * (2**self.bit_depth - 1))
        noisy = noisy / (2**self.bit_depth - 1)

        return noisy
```

## LiDAR Simulation

### 3D LiDAR Model

```xml
<sensor name="velodyne" type="lidar" >
    <lidar>
        <ray>
            <samples>128</samples>
            <resolution>0.4</resolution>
            <min_angle>-2.356</min_angle>
            <max_angle>2.356</max_angle>
            <min_range>0.1</min_range>
            <max_range>100</max_range>
        </ray>
        <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.01</stddev>
        </noise>
    </lidar>
    <pose>0 0 0.05 0 0 0</pose>
</sensor>
```

### LiDAR Point Cloud Generation

```python
import numpy as np
from scipy.spatial.transform import Rotation

class LiDARSimulator:
    """
    Simulates 3D LiDAR point cloud generation.
    Models:
    - Ray casting
    - Multiple returns
    - Motion distortion
    - Noise and dropouts
    """

    def __init__(self, config):
        self.num_beams = config.get('num_beams', 128)
        self.vertical_fov = config.get('vertical_fov', 40)  # degrees
        self.min_range = config.get('min_range', 0.1)
        self.max_range = config.get('max_range', 100)
        self.noise_std = config.get('noise_std', 0.01)

    def generate_pointcloud(self, scene_mesh, robot_pose, timestamp):
        """
        Generate point cloud from scene geometry.
        """
        # Compute beam directions
        beam_angles = self._compute_beam_directions()

        # For each beam, cast ray and find intersection
        points = []
        for beam_dir in beam_angles:
            # Transform to world frame
            world_dir = robot_pose.R @ beam_dir

            # Ray cast
            intersection = self._ray_cast(
                origin=robot_pose.position,
                direction=world_dir,
                scene=scene_mesh
            )

            if intersection is not None:
                distance = np.linalg.norm(intersection - robot_pose.position)
                if self.min_range < distance < self.max_range:
                    # Add noise
                    noise = np.random.normal(0, self.noise_std, 3)
                    points.append(intersection + noise)

        return np.array(points)

    def apply_motion_distortion(self, points, velocities, timestamps):
        """
        Apply motion distortion based on robot movement during scan.
        """
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        points = points[sorted_indices]
        timestamps = timestamps[sorted_indices]

        # For each point, estimate pose at capture time
        distorted = []
        for point, ts in zip(points, timestamps):
            capture_pose = self._interpolate_pose(velocities, ts)
            # Transform point to capture frame
            local_point = capture_pose.inv @ point
            distorted.append(local_point)

        return np.array(distorted)
```

## IMU Simulation

```xml
<sensor name="imu_sensor" type="imu" >
    <imu>
        <noise>
            <type>gaussian</type>
            <rate>
                <mean>0.0</mean>
                <stddev>0.002</stddev>
            </rate>
            <accel>
                <mean>0.0</mean>
                <stddev>0.01</stddev>
            </accel>
        </imu>
        <orientation>
            <x>
                <noise type="gaussian">
                    <mean>0.0</mean>
                    <stddev>0.001</stddev>
                </noise>
            </x>
            <!-- y, z components similar -->
        </orientation>
    </imu>
    <pose>0 0 0 0 0 0</pose>
</sensor>
```

### IMU Noise and Calibration

```python
class IMUErrorModel:
    """
    Models IMU errors including:
    - Accelerometer bias and scale factor
    - Gyroscope bias and drift
    - Axis misalignment
    - Temperature effects
    """

    def __init__(self, config):
        # Accelerometer parameters
        self.accel_bias = np.array(config.get('accel_bias', [0, 0, 0]))
        self.accel_scale = np.diag(config.get('accel_scale', [1, 1, 1]))
        self.accel_noise_std = config.get('accel_noise_std', 0.01)

        # Gyroscope parameters
        self.gyro_bias = np.array(config.get('gyro_bias', [0, 0, 0]))
        self.gyro_scale = np.diag(config.get('gyro_scale', [1, 1, 1]))
        self.gyro_noise_std = config.get('gyro_noise_std', 0.002)
        self.gyro_random_walk = config.get('gyro_random_walk', 0.001)

        # Axis misalignment matrix
        self.misalignment = np.array(config.get('misalignment', np.eye(3)))

    def measure(self, true_accel, true_angular_vel, dt, temperature=25):
        """Convert true motion to IMU measurements."""
        # Update bias with random walk
        self.gyro_bias += np.random.normal(0, self.gyro_random_walk, 3) * dt
        self.accel_bias += np.random.normal(0, 0.0001, 3) * dt

        # Apply scale and misalignment
        measured_accel = self.misalignment @ self.accel_scale @ true_accel
        measured_gyro = self.misalignment @ self.gyro_scale @ true_angular_vel

        # Add bias
        measured_accel += self.accel_bias
        measured_gyro += self.gyro_bias

        # Add noise
        measured_accel += np.random.normal(0, self.accel_noise_std, 3)
        measured_gyro += np.random.normal(0, self.gyro_noise_std, 3)

        return measured_accel, measured_gyro
```

## Sensor Validation

Validating sensor simulation against real data:

```python
class SensorValidator:
    """
    Validates simulated sensor data against real sensor data.
    """

    def validate_camera(self, sim_image, real_image):
        """Compare simulated and real camera images."""
        metrics = {
            'psnr': self._compute_psnr(sim_image, real_image),
            'ssim': self._compute_ssim(sim_image, real_image),
            'histogram_chi2': self._histogram_comparison(sim_image, real_image)
        }
        return metrics

    def validate_lidar(self, sim_cloud, real_cloud):
        """Compare simulated and real point clouds."""
        # Subsample to common density
        sim_downsampled = self._voxel_downsample(sim_cloud, 0.02)
        real_downsampled = self._voxel_downsample(real_cloud, 0.02)

        # Find correspondence using nearest neighbor
        errors = []
        for point in sim_downsampled:
            distances = np.linalg.norm(real_downsampled - point, axis=1)
            errors.append(np.min(distances))

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'percentile_95': np.percentile(errors, 95)
        }

    def validate_imu(self, sim_measurements, real_measurements):
        """Compare IMU time series."""
        return {
            'accel_mae': np.mean(np.abs(
                sim_measurements['accel'] - real_measurements['accel']
            )),
            'gyro_mae': np.mean(np.abs(
                sim_measurements['gyro'] - real_measurements['gyro']
            )),
            'allan_variance_ratio': self._compute_allan_comparison(
                sim_measurements, real_measurements
            )
        }
```

## Key Takeaways

Realistic sensor simulation is critical for sim-to-real transfer:

- **Camera models** must include noise, distortion, and exposure effects
- **LiDAR simulation** requires accurate ray casting and motion distortion
- **IMU error models** capture bias, scale, and noise characteristics
- **Validation** ensures simulation matches real sensor behavior

With sensors simulated realistically, robots can be trained and tested before deployment on real hardware.
