---
sidebar_position: 3
---

# Humanoid Locomotion

Humanoid locomotion represents one of the most challenging problems in robotics, requiring the coordination of dozens of joints to maintain balance while moving through complex environments. Unlike wheeled robots that simply need to avoid obstacles, bipedal robots must continuously balance against gravity, plan footsteps that maintain stability, and adapt their movements to terrain variations. This chapter covers the fundamental principles of humanoid locomotion, from theoretical foundations of balance and walking to practical implementation of locomotion controllers.

## Theoretical Foundations of Bipedal Locomotion

Understanding humanoid locomotion requires grasping several key theoretical concepts that govern how bipedal robots maintain balance and move efficiently. These concepts form the foundation upon which all locomotion controllers are built.

### Center of Mass and Balance

The center of mass (CoM) represents the weighted average position of all mass in the robot's body. For a humanoid robot with mass distributed across multiple links, the CoM position depends on the configuration of all joints. Maintaining the CoM over the support polygon—the convex hull of contact points with the ground—is essential for static stability. However, dynamic walking involves moments where the CoM is outside the support polygon, requiring momentum-based balance.

The Zero Moment Point (ZMP) provides a more practical criterion for dynamic balance. The ZMP is the point on the ground where the net moment from gravity and inertial forces has zero moment about the horizontal axes. For stable walking, the ZMP must remain within the support polygon defined by the foot or feet in contact with the ground. This criterion is widely used because it can be computed from known joint trajectories and provides a simple condition for stability.

The Capture Point (CP) concept provides another perspective on balance, representing the point on the ground where the robot must step to come to a complete stop. The capture point dynamics capture how perturbations cause the robot to fall, enabling predictive balance control. When the capture point moves outside the support polygon, the robot cannot stop without taking a step.

### The Linear Inverted Pendulum Model

The Linear Inverted Pendulum Mode (LIPM) provides a simplified but powerful model for humanoid walking. In this model, the robot's mass is concentrated at the CoM, which is connected to a point foot by a massless leg of variable length. The key insight is that by constraining the CoM height to remain constant, the dynamics become linear and analytically tractable.

The LIPM equations describe how the CoM moves under gravity. For a constant CoM height h, the equation of motion is d²x/dt² = (g/h) * x, where x is the horizontal CoM position and g is gravitational acceleration. This equation has solutions that are linear combinations of hyperbolic sine and cosine functions, which can be used to predict future CoM positions given current state and control inputs.

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class LIPMState:
    """State of the Linear Inverted Pendulum Model."""
    x: float  # X position of CoM
    y: float  # Y position of CoM
    dx: float  # X velocity of CoM
    dy: float  # Y velocity of CoM

class LinearInvertedPendulum:
    """
    Implements the Linear Inverted Pendulum Model for walking control.
    """

    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)  # Natural frequency

    def compute_next_state(self, state: LIPMState,
                          control_input: Tuple[float, float],
                          dt: float) -> LIPMState:
        """
        Compute next state using LIPM dynamics.

        The control input is the position where the pendulum touches
        the ground (often the ankle position).
        """
        ux, uy = control_input

        # State transition matrix for LIPM
        A = np.array([
            [np.cosh(self.omega * dt),
             np.sinh(self.omega * dt) / self.omega],
            [self.omega * np.sinh(self.omega * dt),
             np.cosh(self.omega * dt)]
        ])

        # Input transition matrix
        B = np.array([
            [1 - np.cosh(self.omega * dt),
             np.sinh(self.omega * dt) / self.omega - dt],
            [self.omega * np.sinh(self.omega * dt) - self.omega * dt,
             1 - np.cosh(self.omega * dt)]
        ])

        # Apply control input to support point
        current_state = np.array([state.x, state.dx])
        control = np.array([ux, uy])

        # State update
        new_state = A @ current_state + B @ control

        return LIPMState(
            x=new_state[0],
            y=0,  # Would compute Y similarly
            dx=new_state[1],
            dy=0
        )

    def compute_capture_point(self, state: LIPMState) -> Tuple[float, float]:
        """
        Compute the capture point from current state.

        The capture point is where the robot must step to come to a stop.
        """
        cp_x = state.x + state.dx / self.omega
        cp_y = state.y + state.dy / self.omega
        return cp_x, cp_y

    def generate_walking_trajectory(self, target_x: float, target_y: float,
                                     step_time: float, step_length: float,
                                     num_steps: int) -> np.ndarray:
        """
        Generate a CoM trajectory for walking to a target.

        Uses preview control to generate a stable walking pattern.
        """
        trajectory = []
        current_x, current_y = 0, 0
        current_dx, current_dy = 0, 0

        # Time parameters
        t = np.linspace(0, step_time, 100)

        for step in range(num_steps):
            # Target for this step
            step_target_x = current_x + step_length
            step_target_y = current_y + (target_y - current_y) / (num_steps - step)

            # Compute foot placement using preview control
            foot_x = self._compute_foot_placement(
                current_x, current_dx, step_target_x, step_time
            )

            # Generate CoM trajectory for this step
            for ti in t:
                # Fraction through step
                alpha = ti / step_time

                # Polynomial interpolation for smooth motion
                # CoM moves from current position to mirror of next step
                next_foot_x = foot_x + step_length

                # Simple model: CoM moves toward future foot position
                target_x = current_x + (next_foot_x - current_x) * 0.5

                # Interpolate
                x_t = current_x + (target_x - current_x) * alpha
                y_t = current_y + (step_target_y - current_y) * alpha

                trajectory.append([x_t, y_t, ti + step * step_time])

            current_x = step_target_x
            current_y = step_target_y

        return np.array(trajectory)

    def _compute_foot_placement(self, x: float, dx: float,
                                 target_x: float, step_time: float) -> float:
        """
        Compute optimal foot placement to reach target.

        Uses the capture point concept to determine where to place the foot.
        """
        # Time to next step
        T = step_time

        # Optimal foot placement to land at target
        # Based on LIPM dynamics
        foot_x = target_x - np.exp(-self.omega * T) * (
            x + dx / self.omega - target_x
        )

        return foot_x
```

### Walking Pattern Generation

Walking pattern generation creates reference trajectories for the robot's joints to follow. These trajectories must satisfy kinematic constraints (joint limits, velocity limits), dynamic constraints (ZMP within support, contact forces), and task requirements (reaching target positions).

The Preview Control approach computes optimal CoM trajectories by considering a future preview window of reference ZMP positions. By optimizing over a horizon of future steps, preview control produces smooth, stable walking patterns that naturally account for momentum and balance.

```python
import numpy as np
from scipy.optimize import minimize

class WalkingPatternGenerator:
    """
    Generates walking patterns using preview control.
    """

    def __init__(self, robot_config):
        # Robot parameters
        self.com_height = robot_config['com_height']
        self.foot_length = robot_config['foot_length']
        self.foot_width = robot_config['foot_width']
        self.step_time = robot_config['step_time']
        self.dt = robot_config['dt']

        # LIPM parameters
        self.omega = np.sqrt(9.81 / self.com_height)

        # Preview horizon (in steps)
        self.preview_steps = 10

    def generate_straight_line_walk(self, distance: float,
                                     num_steps: int) -> Dict:
        """
        Generate walking pattern for straight-line walking.
        """
        # Generate reference ZMP trajectory
        zmp_trajectory = self._generate_zmp_trajectory(num_steps)

        # Generate CoM trajectory using preview control
        com_trajectory = self._preview_control(zmp_trajectory)

        # Generate foot trajectories
        foot_trajectories = self._generate_foot_trajectories(num_steps)

        return {
            'zmp': zmp_trajectory,
            'com': com_trajectory,
            'left_foot': foot_trajectories['left'],
            'right_foot': foot_trajectories['right'],
            'time': np.arange(len(com_trajectory)) * self.dt
        }

    def _generate_zmp_trajectory(self, num_steps: int) -> np.ndarray:
        """
        Generate reference ZMP trajectory.

        For straight walking, ZMP moves from center to swing foot and back.
        """
        zmp_trajectory = []
        steps_per_phase = int(self.step_time / self.dt)

        for step in range(num_steps):
            is_left_support = step % 2 == 0

            for t in range(steps_per_phase):
                # Normalized time within step
                alpha = t / steps_per_phase

                if is_left_support:
                    # Left foot support: ZMP near left foot
                    # Shift from center to left foot and back
                    zmp_x = -self.foot_length * 0.3 * np.sin(2 * np.pi * alpha)
                    zmp_y = -self.foot_width * 0.5 * (1 - np.cos(2 * np.pi * alpha))
                else:
                    # Right foot support: ZMP near right foot
                    zmp_x = self.foot_length * 0.3 * np.sin(2 * np.pi * alpha)
                    zmp_y = self.foot_width * 0.5 * (1 - np.cos(2 * np.pi * alpha))

                zmp_trajectory.append([zmp_x, zmp_y])

        return np.array(zmp_trajectory)

    def _preview_control(self, zmp_reference: np.ndarray) -> np.ndarray:
        """
        Generate CoM trajectory using preview control.

        Minimizes tracking error of ZMP over a preview horizon.
        """
        n_samples = len(zmp_reference)

        # Initial CoM state (at origin with zero velocity)
        com_x = np.zeros(n_samples)
        com_y = np.zeros(n_samples)
        com_dx = np.zeros(n_samples)
        com_dy = np.zeros(n_samples)

        # Preview control gain (tuned for stability)
        K = 1.0

        for t in range(n_samples - 1):
            # Preview ZMP over horizon
            zmp_future = zmp_reference[t:min(t + self.preview_steps, n_samples)]

            # Compute feedforward correction based on future ZMP
            if len(zmp_future) > 0:
                # Integral of future ZMP deviation
                zmp_correction = K * np.mean(zmp_reference[t] - zmp_future[0])
            else:
                zmp_correction = 0

            # LIPM dynamics
            com_dx[t + 1] = np.exp(-self.omega * self.dt) * com_dx[t]
            com_dx[t + 1] += (1 - np.exp(-self.omega * self.dt)) * zmp_reference[t, 0]
            com_dx[t + 1] += zmp_correction

            com_x[t + 1] = com_x[t] + com_dx[t] * self.dt

            # Y-axis (similar computation)
            com_dy[t + 1] = np.exp(-self.omega * self.dt) * com_dy[t]
            com_dy[t + 1] += (1 - np.exp(-self.omega * self.dt)) * zmp_reference[t, 1]
            com_dy[t + 1] += zmp_correction

            com_y[t + 1] = com_y[t] + com_dy[t] * self.dt

        return np.column_stack([com_x, com_y, com_dx, com_dy])

    def _generate_foot_trajectories(self, num_steps: int) -> Dict:
        """
        Generate swing foot trajectories.

        Feet follow parabolic trajectories during swing phase.
        """
        steps_per_phase = int(self.step_time / self.dt)

        left_foot = np.zeros((num_steps * steps_per_phase, 3))
        right_foot = np.zeros((num_steps * steps_per_phase, 3))

        for step in range(num_steps):
            is_left_support = step % 2 == 0
            swing_start = step * steps_per_phase
            swing_end = (step + 1) * steps_per_phase

            for t in range(steps_per_phase):
                alpha = t / steps_per_phase

                if is_left_support:
                    # Right foot swings
                    swing_x = self.foot_length * alpha  # Move forward
                    swing_z = 4 * 0.05 * alpha * (1 - alpha)  # Parabolic arc
                    right_foot[swing_start + t] = [swing_x, self.foot_width, swing_z]

                    # Left foot stays in place
                    left_foot[swing_start + t] = [0, -self.foot_width, 0]
                else:
                    # Left foot swings
                    swing_x = self.foot_length * alpha
                    swing_z = 4 * 0.05 * alpha * (1 - alpha)
                    left_foot[swing_start + t] = [swing_x, -self.foot_width, swing_z]

                    # Right foot stays in place
                    right_foot[swing_start + t] = [self.foot_length, self.foot_width, 0]

        return {'left': left_foot, 'right': right_foot}
```

## Balance Control Systems

Balance control maintains the robot's stability during walking and disturbance rejection. Modern humanoid robots use hierarchical control architectures where high-level planners generate reference trajectories and low-level controllers track them while maintaining balance.

### Model Predictive Control for Balance

Model Predictive Control (MPC) provides a powerful framework for balance control by optimizing control inputs over a future horizon while respecting constraints. For humanoid robots, MPC can directly optimize for foot placement, ground reaction forces, or joint torques.

```python
import numpy as np
from scipy.optimize import minimize

class BalanceMPC:
    """
    Model Predictive Controller for humanoid balance.
    """

    def __init__(self, config):
        self.horizon = config['mpc_horizon']
        self.dt = config['mpc_dt']
        self.com_height = config['com_height']
        self.omega = np.sqrt(9.81 / self.com_height)

        # State dimension: [x, dx, y, dy]
        self.state_dim = 4
        # Control dimension: [fx, fy] (contact point offset)
        self.control_dim = 2

    def compute_control(self, state: np.ndarray, disturbance: np.ndarray = None) -> np.ndarray:
        """
        Compute optimal control using MPC.

        Args:
            state: Current state [x, dx, y, dy]
            disturbance: Optional external disturbance force

        Returns:
            Optimal control inputs [fx, fy] for current step
        """
        def cost_function(control_sequence):
            """Cost function for MPC optimization."""
            total_cost = 0.0
            current_state = state.copy()

            for i in range(self.horizon):
                # Compute dynamics
                next_state = self._dynamics(current_state, control_sequence[i])

                # Tracking cost (return to equilibrium)
                tracking_cost = np.sum(next_state**2)

                # Control effort cost
                control_cost = 0.01 * np.sum(control_sequence[i]**2)

                total_cost += tracking_cost + control_cost
                current_state = next_state

            return total_cost

        # Initial guess: zero control
        initial_control = np.zeros((self.horizon, self.control_dim))

        # Bounds: control limits
        bounds = [(-0.2, 0.2), (-0.2, 0.2)] * self.horizon

        # Optimize
        result = minimize(
            cost_function,
            initial_control,
            method='SLSQP',
            bounds=bounds
        )

        if result.success:
            return result.x[:self.control_dim]
        else:
            return np.zeros(self.control_dim)

    def _dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        LIPM dynamics with control input.

        state: [x, dx, y, dy]
        control: [fx, fy] - contact point offset
        """
        x, dx, y, dy = state
        fx, fy = control

        # Discretize LIPM dynamics
        dt = self.dt

        # X dynamics
        dx_new = np.exp(-self.omega * dt) * dx + \
                 (1 - np.exp(-self.omega * dt)) * fx
        x_new = x + dx * dt

        # Y dynamics
        dy_new = np.exp(-self.omega * dt) * dy + \
                 (1 - np.exp(-self.omega * dt)) * fy
        y_new = y + dy * dt

        return np.array([x_new, dx_new, y_new, dy_new])

    def compute_zmp_position(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Compute ZMP position from state and control.
        """
        # ZMP is at the contact point when using LIPM
        return control
```

### Walking Controller Implementation

A complete walking controller integrates multiple components: trajectory generation, balance control, and joint tracking. The following implementation shows how these pieces fit together.

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class WalkingCommand:
    """Command for the walking controller."""
    target_velocity: Tuple[float, float]  # vx, vy
    target_rotation: float  # angular velocity
    step_length: float
    step_width: float
    step_time: float

@dataclass
class WalkingState:
    """Internal state of the walking controller."""
    phase: str  # 'left_support', 'right_support', 'double_support'
    phase_time: float
    current_step: int
    left_foot_pose: np.ndarray  # 4x4 homogeneous transform
    right_foot_pose: np.ndarray
    com_state: np.ndarray  # [x, y, z, dx, dy, dz]

class WalkingController:
    """
    Complete walking controller for humanoid robots.
    """

    def __init__(self, robot_config):
        self.config = robot_config

        # Components
        self.wpg = WalkingPatternGenerator(robot_config)
        self.mpc = BalanceMPC(robot_config)

        # State
        self.state = WalkingState(
            phase='left_support',
            phase_time=0.0,
            current_step=0,
            left_foot_pose=np.eye(4),
            right_foot_pose=np.eye(4),
            com_state=np.zeros(6)
        )

        # Parameters
        self.step_time = robot_config['step_time']
        self.double_support_ratio = 0.1  # Fraction of step in double support

    def step(self, command: WalkingCommand, sensor_data: Dict) -> Dict:
        """
        Execute one control cycle of the walking controller.

        Args:
            command: Walking command from high-level planner
            sensor_data: Sensor measurements (IMU, joint states)

        Returns:
            Joint commands for the robot
        """
        # Update phase
        self._update_phase(sensor_data)

        # Generate reference trajectories for this step
        if self.state.phase == 'left_support':
            target_x = self.state.current_step * command.step_length
            target_y = 0
        else:
            target_x = (self.state.current_step + 0.5) * command.step_length
            target_y = 0

        # Generate trajectories
        trajectories = self.wpg.generate_straight_line_walk(
            distance=10,  # Walk 10 meters
            num_steps=20
        )

        # Compute balance correction using MPC
        current_state = self._estimate_com_state(sensor_data)
        balance_correction = self.mpc.compute_control(current_state)

        # Generate joint commands
        joint_commands = self._generate_joint_commands(
            trajectories, balance_correction, sensor_data
        )

        # Check for step transition
        if self._should_step(command):
            self._initiate_step(command)

        return joint_commands

    def _update_phase(self, sensor_data: Dict):
        """Update the walking phase."""
        self.state.phase_time += self.config['control_dt']

        # Check for phase transition
        if self.state.phase == 'left_support':
            if self.state.phase_time >= self.step_time * (1 - self.double_support_ratio):
                self.state.phase = 'double_support'
        elif self.state.phase == 'double_support':
            if self.state.phase_time >= self.step_time:
                self.state.phase = 'right_support'
                self.state.phase_time = 0
        elif self.state.phase == 'right_support':
            if self.state.phase_time >= self.step_time * (1 - self.double_support_ratio):
                self.state.phase = 'double_support'
        elif self.state.phase == 'double_support':
            if self.state.phase_time >= self.step_time:
                self.state.phase = 'left_support'
                self.state.phase_time = 0
                self.state.current_step += 1

    def _should_step(self, command: WalkingCommand) -> bool:
        """Determine if it's time to initiate a new step."""
        # Step if we've completed current step's single support phase
        if self.state.phase in ['left_support', 'right_support']:
            return self.state.phase_time >= self.step_time * (1 - self.double_support_ratio)
        return False

    def _initiate_step(self, command: WalkingCommand):
        """Initiate a new step."""
        # Planning would go here - determine step target based on command
        pass

    def _estimate_com_state(self, sensor_data: Dict) -> np.ndarray:
        """Estimate current CoM state from sensors."""
        # Use IMU for velocity estimation, joint states for position
        return np.zeros(4)  # Simplified

    def _generate_joint_commands(self, trajectories: Dict,
                                  balance_correction: np.ndarray,
                                  sensor_data: Dict) -> Dict:
        """Generate joint commands from trajectories and balance correction."""
        # This would implement inverse kinematics to convert
        # foot and CoM trajectories to joint angles
        return {'joint_positions': {}, 'joint_velocities': {}}

class ReinforcementLearningLocomotion:
    """
    Reinforcement learning-based locomotion controller.
    """

    def __init__(self, config):
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']

        # Neural network policy (simplified)
        self.policy_network = self._create_policy_network()

    def _create_policy_network(self):
        """Create the policy network."""
        import torch
        import torch.nn as nn

        class PolicyNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim=256):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Tanh()
                )

            def forward(self, state):
                return self.network(state)

        return PolicyNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=256
        )

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from policy network."""
        import torch

        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = self.policy_network(state_tensor)
            return action.numpy()

    def compute_reward(self, state: np.ndarray, action: np.ndarray,
                       next_state: np.ndarray, done: bool) -> float:
        """Compute reward for RL training."""
        # Reward for forward velocity
        velocity_reward = next_state[0]  # x velocity

        # Penalty for large joint accelerations
        acceleration_penalty = -0.01 * np.sum(np.abs(action))

        # Penalty for being off-balance
        com_x = next_state[1]  # CoM x position
        balance_penalty = -0.1 * abs(com_x) if abs(com_x) > 0.1 else 0

        # Bonus for not falling
        alive_bonus = 1.0 if not done else 0

        return velocity_reward + acceleration_penalty + balance_penalty + alive_bonus
```

## Terrain Adaptation and Dynamic Locomotion

Humanoid robots must adapt their locomotion to varying terrain, including slopes, stairs, rough ground, and obstacles. This requires perception-based adaptation and robust control strategies.

```python
class TerrainAdaptiveWalking:
    """
    Walking controller that adapts to terrain.
    """

    def __init__(self, robot_config):
        self.config = robot_config
        self.foot_placement_planner = FootPlacementPlanner(robot_config)

    def adapt_to_terrain(self, terrain_map: Dict, current_pose: np.ndarray) -> Dict:
        """
        Adapt walking parameters to terrain.

        Args:
            terrain_map: Local terrain height map
            current_pose: Current robot pose

        Returns:
            Adapted walking parameters
        """
        # Analyze terrain
        terrain_analysis = self._analyze_terrain(terrain_map)

        # Adjust step parameters based on terrain
        if terrain_analysis['roughness'] > 0.05:
            # Rough terrain: shorter steps, higher foot clearance
            step_length = self.config['nominal_step_length'] * 0.7
            step_height = 0.08  # Higher foot lift
        else:
            step_length = self.config['nominal_step_length']
            step_height = 0.05

        # Adjust for slope
        slope = terrain_analysis['mean_slope']
        if abs(slope) > 0.1:
            # Uphill/downhill: adjust step length and height
            step_length *= (1 - abs(slope) * 0.5)
            step_height += abs(slope) * 0.02

        # Check for obstacles
        obstacle_avoidance = self._plan_obstacle_avoidance(
            terrain_analysis['obstacles'], current_pose
        )

        return {
            'step_length': step_length,
            'step_height': step_height,
            'step_width': self._compute_step_width(terrain_analysis),
            'foot_positions': obstacle_avoidance
        }

    def _analyze_terrain(self, terrain_map: Dict) -> Dict:
        """Analyze terrain characteristics."""
        height_map = terrain_map['height_map']

        # Compute roughness (standard deviation of heights)
        roughness = np.std(height_map)

        # Compute mean slope
        gradient_x, gradient_y = np.gradient(height_map)
        mean_slope = np.mean(np.sqrt(gradient_x**2 + gradient_y**2))

        # Detect obstacles (regions of high gradient)
        obstacle_mask = np.abs(gradient_x) > 0.1 | np.abs(gradient_y) > 0.1

        return {
            'roughness': roughness,
            'mean_slope': mean_slope,
            'obstacles': obstacle_mask,
            'traversable': roughness < 0.1 and mean_slope < 0.2
        }

    def _plan_obstacle_avoidance(self, obstacle_mask: np.ndarray,
                                   current_pose: np.ndarray) -> Dict:
        """Plan foot placements to avoid obstacles."""
        # Use A* or similar to find paths around obstacles
        # Simplified: shift foot positions away from obstacles
        left_foot_pos = [0, self.config['foot_width'] / 2]
        right_foot_pos = [0, -self.config['foot_width'] / 2]

        return {
            'left': left_foot_pos,
            'right': right_foot_pos
        }

    def _compute_step_width(self, terrain_analysis: Dict) -> float:
        """Compute appropriate step width based on terrain."""
        base_width = self.config['nominal_step_width']

        # Wider stance on rough terrain
        if terrain_analysis['roughness'] > 0.05:
            return base_width * 1.2

        return base_width
```

## Key Takeaways

Humanoid locomotion integrates mechanics, control theory, and real-time computation to enable bipedal movement. The combination of theoretical foundations and practical implementations enables robots to walk stably and adapt to challenging environments.

- **Balance fundamentals** include center of mass, ZMP, and capture point concepts
- **LIPM model** provides tractable dynamics for walking pattern generation
- **Preview control** generates smooth, stable CoM trajectories
- **Model predictive control** enables reactive balance correction
- **Reinforcement learning** offers data-driven locomotion approaches
- **Terrain adaptation** requires perception and adaptive control strategies
- **Complete controllers** integrate trajectory generation, balance, and tracking

With locomotion established, the capstone pipeline integrates all textbook concepts into a complete Physical AI system.
