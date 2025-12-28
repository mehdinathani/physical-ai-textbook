---
sidebar_position: 3
---

# Nav2 Path Planning

Navigation 2 (Nav2) represents the next generation of the Navigation Stack for ROS 2, providing a comprehensive framework for autonomous robot navigation. Built upon the lessons learned from the original ROS navigation stack, Nav2 offers improved modularity, extensibility, and support for diverse robot platforms including humanoid robots. This chapter covers the Nav2 architecture, path planning algorithms, costmap configuration, and practical implementation strategies for humanoid navigation systems.

## Understanding the Nav2 Architecture

The Navigation 2 project was developed to address limitations in the original ROS navigation stack while maintaining compatibility with modern ROS 2 paradigms. The architecture embraces the concept of behavior trees for orchestrating navigation behaviors, provides plugin-based interfaces for algorithm selection, and implements lifecycle management for robust system operation.

### Lifecycle Management System

Nav2 introduces a lifecycle management system based on ROS 2's lifecycle nodes, which provide controlled state transitions for system components. This approach ensures that navigation nodes initialize properly, can be gracefully shutdown, and provide health information for system monitoring. Each navigation component transitions through states including unconfigured, inactive, active, and finalized, with explicit transitions between these states triggered by lifecycle clients.

The lifecycle system enables several important capabilities for humanoid robotics applications. Components can be configured before activation, allowing parameter loading without starting computation. The active state enables normal operation, while deactivation allows reconfiguration without full shutdown. Error handling in the finalized state ensures clean resource release. This structured approach is particularly important for safety-critical humanoid applications where predictable startup and shutdown behavior is essential.

### Server and Action Architecture

Nav2 organizes functionality into server nodes that provide services and actions for navigation tasks. The primary servers include the planning server, which generates paths from current location to goal; the controller server, which generates velocity commands to follow the planned path; and the recovery server, which executes recovery behaviors when the robot becomes stuck. Each server exposes actions that clients can invoke to request navigation operations.

The behavior tree navigator serves as the primary interface for high-level navigation commands. It composes the various servers and their actions into behavior trees that define the navigation logic. Default behavior trees provide standard navigation behaviors including navigating to goals, following paths, and executing recovery sequences. These behavior trees can be extended or replaced to implement custom navigation strategies for specific humanoid applications.

## Costmap Configuration and Management

Costmaps form the foundation of navigation planning, representing the robot's understanding of traversable and obstructed spaces. Nav2 uses layered costmaps that combine information from multiple sources into a single representation used for path planning.

### Static Map Layer

The static map layer incorporates occupancy grid information from SLAM or pre-built maps. This layer provides the base representation of the environment, distinguishing between known free space and known obstacles. For humanoid robots operating in human environments, the static map layer captures permanent structures such as walls, floors, and fixed furniture.

```yaml
# Static map layer configuration
plugin: nav2_costmap_2d::StaticLayer
enabled: true
map_subscribe_transient_local: true
transform_tolerance: 0.1
```

The static layer subscribes to map topics published by SLAM systems or map servers. When a map is received, it populates the costmap with obstacle information from the occupancy grid. Cells with unknown occupancy are not marked as obstacles, allowing the robot to explore unmapped areas. For humanoid applications, static maps should be generated at an appropriate resolution to capture human-scale obstacles such as chair legs, door thresholds, and small furniture.

### Obstacle Layer

The obstacle layer processes sensor data in real-time to identify and track obstacles in the robot's vicinity. This layer is essential for responding to dynamic objects such as people, moving furniture, or other robots that were not present when the static map was created.

```yaml
# Obstacle layer configuration
plugin: nav2_costmap_2d::ObstacleLayer
enabled: true
observation_sources: lidar_scan point_cloud camera
lidar_scan:
  topic: /scan
  sensor_frame: lidar_link
  observation_persistence: 0.5
  expected_update_rate: 10.0
  min_obstacle_height: 0.1
  max_obstacle_height: 2.0
  clearing: true
  marking: true
point_cloud:
  topic: /depth_registered/points
  sensor_frame: head_camera_link
  observation_persistence: 0.5
  min_obstacle_height: 0.1
  max_obstacle_height: 1.5
  clearing: true
  marking: true
```

The obstacle layer maintains an observation buffer for each sensor, storing recent sensor readings to handle sensor occlusions and provide temporal filtering. The observation persistence parameter controls how long observations remain valid, while expected update rate determines when observations are considered stale. For humanoid robots with multiple sensors, the obstacle layer fuses information from cameras, LiDAR, and ultrasonic sensors to create a comprehensive obstacle representation.

### Inflation Layer

The inflation layer expands obstacle cells into the costmap to create safety margins around obstacles. This expansion ensures that the robot maintains a safe distance from obstacles while navigating, preventing collisions even with imperfect path execution.

```yaml
# Inflation layer configuration
plugin: nav2_costmap_2d::InflationLayer
enabled: true
inflation_radius: 0.5
cost_scaling_factor: 2.0
```

The inflation radius defines how far from obstacles the inflation extends, measured in meters. For humanoid robots with significant footprint, the inflation radius should account for the robot's size and error in localization. The cost scaling factor controls how quickly the cost increases as cells approach obstacles, with higher values creating sharper gradients that prefer paths further from obstacles.

### Voxel Layer for 3D Obstacles

For humanoid robots operating in complex 3D environments, the voxel layer provides 3D obstacle representation. This layer processes 3D point clouds to track obstacles at multiple heights, enabling navigation under tables, around hanging objects, and through spaces with vertical clearance constraints.

```yaml
# Voxel layer configuration
plugin: nav2_costmap_2d::VoxelLayer
enabled: true
origin_z: 0.0
resolution: 0.05
z_voxels: 16
max_obstacle_height: 2.0
unknown_threshold: 15
mark_threshold: 0
observation_sources: point_cloud
```

The voxel layer maintains a 3D grid of voxels, marking those that contain obstacles and propagating obstacle information to 2D costmap layers. The z_voxels parameter determines the vertical resolution of the voxel grid, with more voxels providing finer vertical discrimination but requiring more memory and computation.

## Global Path Planning Algorithms

Global path planning generates a high-level path from the robot's current location to a goal position, considering the complete map of the environment. Nav2 provides multiple global planners that can be selected based on the application requirements.

### A* Search Algorithm

The A* (A-star) algorithm combines the advantages of Dijkstra's algorithm's completeness with heuristic guidance to efficiently find optimal paths. The algorithm maintains a priority queue ordered by the sum of the path cost from the start and the estimated cost to the goal. For grid-based costmaps, the heuristic is typically the Euclidean or Manhattan distance to the goal.

```python
import heapq
import numpy as np

class AStarPlanner:
    """
    A* path planner for 2D grid-based navigation.
    """

    def __init__(self, costmap, inflation_radius=1):
        self.costmap = costmap
        self.inflation_radius = inflation_radius
        self.resolution = costmap.resolution
        self.origin = costmap.origin

    def plan(self, start, goal):
        """
        Find path from start to goal using A* algorithm.

        Args:
            start: (x, y) in world coordinates
            goal: (x, y) in world coordinates

        Returns:
            List of (x, y) waypoints
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Initialize open set with start node
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            # Get node with lowest f_score
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                # Check if obstacle
                if self.is_obstacle(neighbor):
                    continue

                tentative_g = g_score[current] + self.cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def cost(self, a, b):
        """Cost of moving from a to b."""
        # Diagonal movement costs more
        if a[0] != b[0] and a[1] != b[1]:
            return 1.414  # sqrt(2)
        return 1

    def get_neighbors(self, node):
        """Get valid neighboring nodes."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (node[0] + dx, node[1] + dy)
                if self.is_valid(neighbor):
                    neighbors.append(neighbor)
        return neighbors

    def is_valid(self, node):
        """Check if node is within bounds and not an obstacle."""
        if not (0 <= node[0] < self.costmap.width and
                0 <= node[1] < self.costmap.height):
            return False
        return not self.is_obstacle(node)

    def is_obstacle(self, node):
        """Check if node is an obstacle in the costmap."""
        cost = self.costmap.get_cost(node[0], node[1])
        return cost >= self.costmap.LETHAL_OBSTACLE

    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid indices."""
        x = int((world_pos[0] - self.origin[0]) / self.resolution)
        y = int((world_pos[1] - self.origin[1]) / self.resolution)
        return (x, y)

    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates."""
        x = grid_pos[0] * self.resolution + self.origin[0]
        y = grid_pos[1] * self.resolution + self.origin[1]
        return (x, y)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary."""
        path = [self.grid_to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.grid_to_world(current))
        return path[::-1]
```

### Theta* Any-Angle Planning

Theta* improves upon A* by allowing any-angle paths rather than restricting movement to grid neighbors. This produces paths that follow the geometry of the environment more closely, with fewer unnecessary waypoints. The algorithm achieves this by checking if the parent of a node can see the current node directly, bypassing intermediate grid cells.

```python
class ThetaStarPlanner(AStarPlanner):
    """
    Theta* any-angle path planner.
    """

    def plan(self, start, goal):
        """Find any-angle path using Theta* algorithm."""
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                if self.is_obstacle(neighbor):
                    continue

                # Try to update from parent's parent (Theta* logic)
                if current != start_grid and self.line_of_sight(
                    came_from.get(current, start_grid), neighbor
                ):
                    parent = came_from[current]
                    tentative_g = g_score[parent] + self.euclidean_dist(parent, neighbor)
                else:
                    tentative_g = g_score[current] + self.cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

    def line_of_sight(self, a, b):
        """Check if there's line of sight between two cells."""
        # Simplified implementation
        return True  # Full implementation would use ray casting
```

### Global Planner Plugin Configuration

Nav2's global planner interface enables plugin-based algorithm selection. Custom planners can be registered as plugins and configured through YAML files:

```yaml
# Global planner configuration
planner_server:
  ros__parameters:
    expected_planner_frequency: 1.0
    use_sim_time: true
    planner_plugins: ['GridBased']
    GridBased:
      plugin: nav2_theta_star_planner::ThetaStarPlanner
      tolerance: 0.5
      heuristic: 2
      downsampling_factor: 1
      occupancy_threshold: 100
```

## Local Path Planning and Control

Local path planning generates velocity commands to follow the global path while reacting to local obstacles. This reactive planning is essential for humanoid robots navigating in dynamic environments where obstacles may appear unexpectedly.

### Dynamic Window Approach

The Dynamic Window Approach (DWA) samples velocity commands in the robot's velocity space and selects those that lead to trajectories staying within the free space while progressing toward the goal. The algorithm considers the robot's dynamic constraints, limiting velocities to those achievable within the next time step given acceleration limits.

```python
import numpy as np

class DWA:
    """
    Dynamic Window Approach for local path planning.
    """

    def __init__(self, config):
        # Robot configuration
        self.max_linear_velocity = config.get('max_linear_velocity', 0.5)
        self.max_angular_velocity = config.get('max_angular_velocity', 1.0)
        self.max_accel = config.get('max_accel', 0.5)
        self.max_angular_accel = config.get('max_angular_accel', 1.0)

        # Velocity resolution
        self.vx_resolution = config.get('vx_resolution', 0.05)
        self.vw_resolution = config.get('vw_resolution', 0.1)

        # Prediction time
        self.time_horizon = config.get('time_horizon', 1.5)

        # Cost weights
        self.alpha_heading = config.get('alpha_heading', 1.0)
        self.alpha_distance = config.get('alpha_distance', 1.0)
        self.alpha_velocity = config.get('alpha_velocity', 1.0)

    def compute_velocity(self, robot_pose, goal, obstacles, current_vel):
        """
        Compute optimal velocity command.

        Args:
            robot_pose: (x, y, theta) current pose
            goal: (x, y) goal position
            obstacles: List of obstacle positions
            current_vel: (vx, vw) current velocity

        Returns:
            (vx, vw) optimal velocity command
        """
        # Compute dynamic window
        dw = self.dynamic_window(current_vel)

        # Generate and score trajectories
        best_score = float('-inf')
        best_vel = (0, 0)

        for vx in np.arange(dw['vx_min'], dw['vx_max'], self.vx_resolution):
            for vw in np.arange(dw['vw_min'], dw['vw_max'], self.vw_resolution):
                # Check velocity limits
                if abs(vx) > self.max_linear_velocity:
                    continue
                if abs(vw) > self.max_angular_velocity:
                    continue

                # Simulate trajectory
                trajectory = self.simulate_trajectory(
                    robot_pose, vx, vw, obstacles
                )

                # Check for collision
                if self.check_collision(trajectory, obstacles):
                    continue

                # Score trajectory
                score = self.score_trajectory(trajectory, goal, vx)

                if score > best_score:
                    best_score = score
                    best_vel = (vx, vw)

        return best_vel

    def dynamic_window(self, current_vel):
        """Compute dynamic window of achievable velocities."""
        vx_min = max(-self.max_linear_velocity,
                     current_vel[0] - self.max_accel * self.time_horizon)
        vx_max = min(self.max_linear_velocity,
                     current_vel[0] + self.max_accel * self.time_horizon)

        vw_min = max(-self.max_angular_velocity,
                     current_vel[1] - self.max_angular_accel * self.time_horizon)
        vw_max = min(self.max_angular_velocity,
                     current_vel[1] + self.max_angular_accel * self.time_horizon)

        return {'vx_min': vx_min, 'vx_max': vx_max,
                'vw_min': vw_min, 'vw_max': vw_max}

    def simulate_trajectory(self, pose, vx, vw, obstacles):
        """Simulate robot trajectory given velocity command."""
        trajectory = [pose]
        x, y, theta = pose

        for t in np.arange(0, self.time_horizon, 0.1):
            # Update position using differential drive kinematics
            x += vx * 0.1 * np.cos(theta)
            y += vx * 0.1 * np.sin(theta)
            theta += vw * 0.1
            trajectory.append((x, y, theta))

        return trajectory

    def score_trajectory(self, trajectory, goal, vx):
        """Score a trajectory based on multiple criteria."""
        final_pose = trajectory[-1]

        # Heading score: how well aligned with goal
        angle_to_goal = np.arctan2(
            goal[1] - final_pose[1],
            goal[0] - final_pose[0]
        )
        heading_score = np.pi - abs(angle_to_goal - final_pose[2])

        # Distance score: proximity to goal
        distance_score = -np.sqrt(
            (goal[0] - final_pose[0])**2 +
            (goal[1] - final_pose[1])**2
        )

        # Velocity score: favor faster progress
        velocity_score = abs(vx)

        return (self.alpha_heading * heading_score +
                self.alpha_distance * distance_score +
                self.alpha_velocity * velocity_score)

    def check_collision(self, trajectory, obstacles):
        """Check if trajectory collides with any obstacle."""
        for pose in trajectory:
            for obstacle in obstacles:
                dist = np.sqrt(
                    (pose[0] - obstacle[0])**2 +
                    (pose[1] - obstacle[1])**2
                )
                if dist < 0.3:  # Robot radius
                    return True
        return False
```

### Timed Elastic Bands

Timed Elastic Bands (TEB) optimizes a trajectory represented as a sequence of timed poses, balancing goal achievement, obstacle avoidance, and trajectory smoothness. The algorithm formulates trajectory optimization as a sparse constrained optimization problem that can be solved efficiently online.

```python
import numpy as np
from scipy.optimize import minimize

class TimedElasticBands:
    """
    Timed Elastic Bands local planner.
    """

    def __init__(self, config):
        # Robot parameters
        self.max_vel_x = config.get('max_vel_x', 0.5)
        self.max_vel_theta = config.get('max_vel_theta', 1.0)
        self.accel_x = config.get('accel_x', 0.5)
        self.accel_theta = config.get('accel_theta', 1.0)

        # TEB parameters
        self.dt = config.get('dt', 0.1)
        self.horizon = config.get('horizon', 10)  # Number of poses
        self.weight_obstacle = config.get('weight_obstacle', 50.0)
        self.weight_velocity = config.get('weight_velocity', 1.0)
        self.weight_time = config.get('weight_time', 1.0)
        self.weight_shortest = config.get('weight_shortest', 0.0)

    def optimize(self, start_pose, goal_pose, obstacles):
        """
        Optimize trajectory from start to goal.

        Args:
            start_pose: (x, y, theta) initial pose
            goal_pose: (x, y, theta) goal pose
            obstacles: List of (x, y) obstacle positions

        Returns:
            List of (x, y, theta) optimized poses
        """
        # Initialize trajectory with linear interpolation
        trajectory = self.initialize_trajectory(start_pose, goal_pose)

        # Optimize trajectory
        for _ in range(5):  # Number of optimization iterations
            # Compute cost and gradient
            cost = self.compute_cost(trajectory, obstacles)
            gradient = self.compute_gradient(trajectory, obstacles)

            # Update trajectory
            trajectory = self.update_trajectory(trajectory, gradient)

        return trajectory

    def initialize_trajectory(self, start_pose, goal_pose):
        """Initialize trajectory with linear interpolation."""
        trajectory = []
        for i in range(self.horizon + 1):
            alpha = i / self.horizon
            x = start_pose[0] + alpha * (goal_pose[0] - start_pose[0])
            y = start_pose[1] + alpha * (goal_pose[1] - start_pose[1])
            theta = start_pose[2] + alpha * (goal_pose[2] - start_pose[2])
            trajectory.append([x, y, theta])
        return trajectory

    def compute_cost(self, trajectory, obstacles):
        """Compute total cost of trajectory."""
        cost = 0

        # Obstacle cost
        for pose in trajectory:
            for obstacle in obstacles:
                dist = np.sqrt((pose[0] - obstacle[0])**2 +
                              (pose[1] - obstacle[1])**2)
                if dist < 0.5:
                    cost += self.weight_obstacle / dist

        # Velocity cost (penalize deviations from max speed)
        total_time = len(trajectory) * self.dt
        avg_velocity = np.sqrt(
            (trajectory[-1][0] - trajectory[0][0])**2 +
            (trajectory[-1][1] - trajectory[0][1])**2
        ) / total_time
        cost += self.weight_velocity * max(0, self.max_vel_x - avg_velocity)

        # Time cost
        cost += self.weight_time * len(trajectory)

        # Shortest path cost
        path_length = sum(
            np.sqrt((trajectory[i+1][0] - trajectory[i][0])**2 +
                   (trajectory[i+1][1] - trajectory[i][1])**2)
            for i in range(len(trajectory) - 1)
        )
        direct_dist = np.sqrt(
            (trajectory[-1][0] - trajectory[0][0])**2 +
            (trajectory[-1][1] - trajectory[0][1])**2
        )
        cost += self.weight_shortest * (path_length - direct_dist)

        return cost

    def compute_gradient(self, trajectory, obstacles):
        """Compute gradient of cost function."""
        gradient = []
        for i in range(1, len(trajectory) - 1):
            grad_x = 0
            grad_y = 0
            grad_theta = 0

            # Numerical gradient for obstacles
            eps = 0.01
            for obstacle in obstacles:
                dist = np.sqrt((trajectory[i][0] - obstacle[0])**2 +
                              (trajectory[i][1] - obstacle[1])**2)
                if dist < 0.5:
                    grad_x += -self.weight_obstacle / dist**2 * (
                        trajectory[i][0] - obstacle[0]) / dist
                    grad_y += -self.weight_obstacle / dist**2 * (
                        trajectory[i][1] - obstacle[1]) / dist

            gradient.append([grad_x, grad_y, grad_theta])
        return gradient

    def update_trajectory(self, trajectory, gradient, step_size=0.1):
        """Update trajectory using gradient descent."""
        for i in range(1, len(trajectory) - 1):
            trajectory[i][0] += gradient[i-1][0] * step_size
            trajectory[i][1] += gradient[i-1][1] * step_size
        return trajectory
```

## Navigation Behavior Trees

Nav2 uses behavior trees to orchestrate complex navigation behaviors. Behavior trees provide a hierarchical, modular approach to defining navigation logic that is more flexible than finite state machines.

```xml
<!-- Navigation behavior tree example -->
<root main_tree_to_execute="NavigateToPose">
    <BehaviorTree ID="NavigateToPose">
        <RecoveryNode number_of_retries="3" name="NavigateRecovery">
            <PipelineSequence name="NavigateAndRecover">
                <NavigateToPose goal="goal"/>
                <ComputePathToPose goal="goal" planner_id="GridBased"/>
                <FollowPath path="path" controller_id="FollowPath"/>
                <CancelAllTasks name="CancelNavigation"/>
            </PipelineSequence>
            <RoundRobin name="RecoveryActions">
                <Sequence name="ClearCostmap">
                    <ClearEntireCostmap service_name="/global_costmap/clear_costmap"/>
                    <ClearEntireCostmap service_name="/local_costmap/clear_costmap"/>
                </Sequence>
                <Sequence name="BackUp">
                    <BackUp backup_dist="0.5" backup_speed="0.1"/>
                </Sequence>
                <Sequence name="Spin">
                    <Spin spin_dist="1.57" time_allowance="10.0"/>
                </Sequence>
            </RoundRobin>
        </RecoveryNode>
    </BehaviorTree>
</root>
```

## Humanoid-Specific Navigation Considerations

Humanoid robots present unique navigation challenges due to their bipedal locomotion, variable height, and operation in human-scale environments. These considerations require modifications to standard navigation approaches.

### Step Planning and Terrain Traversability

Unlike wheeled robots that can traverse continuous surfaces, humanoid robots must plan paths that are traversable with bipedal steps. This requires analyzing terrain for step locations, identifying obstacles that must be stepped over or around, and considering the robot's dynamic stability constraints.

```python
class HumanoidNavigationPlanner:
    """
    Navigation planner for humanoid robots considering bipedal constraints.
    """

    def __init__(self, config):
        self.step_width = config.get('step_width', 0.2)
        self.max_step_height = config.get('max_step_height', 0.15)
        self.max_step_length = config.get('max_step_length', 0.35)
        self.min_step_length = config.get('min_step_length', 0.1)
        self.clearance_margin = config.get('clearance_margin', 0.15)

    def plan_path(self, start_pose, goal_pose, terrain_map):
        """
        Plan navigable path considering humanoid constraints.

        Args:
            start_pose: (x, y, theta) starting pose
            goal_pose: (x, y, theta) goal pose
            terrain_map: 2D grid with traversability information

        Returns:
            List of (x, y) waypoints
        """
        # Analyze terrain for step locations
        traversable_regions = self.analyze_traversability(terrain_map)

        # Plan path through traversable regions
        base_path = self.plan_base_path(start_pose, goal_pose, traversable_regions)

        # Refine path for bipedal navigation
        step_path = self.refine_for_steps(base_path, terrain_map)

        return step_path

    def analyze_traversability(self, terrain_map):
        """
        Analyze terrain map to identify traversable regions.
        Returns traversability scores for each grid cell.
        """
        traversability = np.zeros_like(terrain_map)

        for y in range(terrain_map.shape[0]):
            for x in range(terrain_map.shape[1]):
                cell = terrain_map[y, x]

                # Check for traversability
                if cell.obstacle:
                    traversability[y, x] = 0.0
                elif cell.height_variance > 0.05:
                    # Uneven terrain - may require careful stepping
                    traversability[y, x] = 0.3
                elif cell.slope > 0.2:
                    # Steep slope
                    traversability[y, x] = 0.2
                else:
                    traversability[y, x] = 1.0

        return traversability

    def refine_for_steps(self, path, terrain_map):
        """
        Refine path to ensure it follows humanoid step constraints.
        """
        refined_path = []
        for i in range(len(path)):
            waypoint = path[i]
            terrain_info = terrain_map.get_terrain_info(waypoint)

            # Add step offsets
            if i > 0:
                prev_waypoint = path[i-1]
                dx = waypoint[0] - prev_waypoint[0]
                dy = waypoint[1] - prev_waypoint[1]
                distance = np.sqrt(dx**2 + dy**2)

                # Insert intermediate steps for long segments
                if distance > self.max_step_length:
                    num_steps = int(distance / self.max_step_length)
                    for step_i in range(1, num_steps):
                        alpha = step_i / num_steps
                        step_x = prev_waypoint[0] + alpha * dx
                        step_y = prev_waypoint[1] + alpha * dy
                        refined_path.append((step_x, step_y))

            refined_path.append(waypoint)

        return refined_path

    def compute_traversability_cost(self, position, terrain_map):
        """
        Compute traversability cost for a position.
        Higher cost means less traversable.
        """
        terrain_info = terrain_map.get_terrain_info(position)

        cost = 0

        # Distance from obstacles
        min_obstacle_dist = terrain_info.min_obstacle_distance
        if min_obstacle_dist < self.clearance_margin:
            cost += 10 * (self.clearance_margin - min_obstacle_dist)

        # Height variance
        cost += terrain_info.height_variance * 100

        # Slope
        cost += terrain_info.slope * 50

        return cost
```

### Variable Height Considerations

Humanoid robots may need to navigate under tables, through doorways at different heights, and in spaces with vertical clearance constraints. The navigation system must consider the robot's current posture and any carried objects when planning paths.

```python
class ClearanceAnalyzer:
    """
    Analyzes vertical clearance for humanoid navigation.
    """

    def __init__(self):
        self.robot_heights = {
            'standing': 1.7,
            'crouching': 1.2,
            'sitting': 0.9
        }

    def check_clearance(self, position, posture, clearance_map):
        """
        Check if robot can pass through position in given posture.

        Args:
            position: (x, y) position
            posture: Current robot posture
            clearance_map: Map of vertical clearances

        Returns:
            Tuple of (is_clear, clearance_height)
        """
        robot_height = self.robot_heights.get(posture, 1.7)
        clearance = clearance_map.get_clearance(position)

        is_clear = clearance > robot_height
        return is_clear, clearance
```

## Parameter Tuning and Optimization

Nav2 performance depends heavily on parameter tuning for the specific robot and environment. Systematic parameter optimization ensures reliable navigation across diverse scenarios.

### Costmap Parameter Tuning

Costmap parameters control obstacle representation and should be tuned based on sensor characteristics, robot size, and environment type. Key parameters include the costmap resolution, which affects obstacle representation precision and memory usage, and the obstacle layer parameters that control observation persistence and filtering.

For humanoid robots operating in cluttered indoor environments, a costmap resolution of 0.05 meters provides a good balance between precision and computational requirements. The inflation radius should be set to approximately half the robot's footprint diameter plus an additional safety margin for localization error.

### Planner Parameter Tuning

Global planner parameters control path planning behavior. The tolerance parameter determines when a goal is considered reached, with smaller values requiring more precise navigation. For humanoid robots navigating near furniture and other obstacles, a tolerance of 0.3 to 0.5 meters is typically appropriate.

The local planner parameters significantly affect navigation smoothness and obstacle avoidance aggressiveness. Velocity and acceleration limits should match the robot's actual capabilities. Overshooting these limits causes oscillation and poor tracking. The trajectory scoring weights should be tuned to achieve the desired balance between path adherence and obstacle avoidance.

## Key Takeaways

Nav2 provides a comprehensive navigation framework suitable for humanoid robotics applications. The plugin-based architecture enables customization while the lifecycle management system ensures reliable operation.

- **Architecture** uses lifecycle nodes, behavior trees, and action servers for modular navigation
- **Costmaps** combine static maps, dynamic obstacles, and inflation layers for environment representation
- **Global planners** include A*, Theta*, and custom plugins for path generation
- **Local planners** such as DWA and TEB generate velocity commands for path following
- **Behavior trees** compose navigation behaviors with recovery mechanisms
- **Humanoid considerations** include step planning, terrain traversability, and variable height navigation
- **Parameter tuning** is essential for matching navigation performance to robot capabilities

With navigation capabilities established, we can now explore how to transfer skills learned in simulation to real-world robot deployment.
