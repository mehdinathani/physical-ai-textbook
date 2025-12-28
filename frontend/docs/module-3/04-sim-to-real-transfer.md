---
sidebar_position: 4
---

# Sim-to-Real Transfer

The sim-to-real gap, often called the "reality gap," represents one of the most significant challenges in Physical AI development. Policies trained in simulation often fail when deployed on physical robots due to discrepancies between simulated and real-world physics, sensor characteristics, and environmental conditions. This chapter provides a comprehensive treatment of sim-to-real transfer techniques, from understanding the sources of the reality gap to implementing practical solutions for humanoid robotics applications.

## Understanding the Reality Gap

The reality gap encompasses all differences between simulation and reality that can cause a trained policy to fail on a physical robot. These differences manifest across multiple dimensions: the physics simulation may not perfectly match real-world dynamics, sensor observations may differ in noise characteristics and artifacts, and the environment itself may behave unpredictably compared to controlled simulation conditions.

### Sources of Simulation Discrepancy

Understanding the sources of the reality gap is essential for developing effective solutions. The discrepancies can be categorized into several types, each requiring different mitigation strategies.

**Physics Discrepancies** arise from imperfections in the physics simulation. Rigid body dynamics simulations rely on approximations that break down in edge cases such as high-speed collisions, frictional contact, and multi-body interactions. For humanoid robots, the complexity of bipedal locomotion makes physics discrepancies particularly challenging, as subtle differences in contact modeling can lead to dramatically different balance behaviors.

Frictional contact is notoriously difficult to simulate accurately. The Coulomb friction model used in most simulators is an approximation of actual contact mechanics, which involves deformation, hysteresis, and other complex phenomena. For humanoid robots that constantly interact with the ground through their feet, accurate friction modeling is essential but difficult to achieve. Similarly, restitution (bounciness) is often poorly calibrated, affecting how robots respond to impacts.

The dynamics of articulated bodies present additional challenges. Joint friction, motor dynamics, and gearbox behavior are all simplified in simulation. A humanoid robot's hip joint, for example, involves motors, gears, bearings, and structural compliance that together create a complex dynamic response that simulations may not fully capture.

**Sensor Discrepancies** occur when simulated sensors do not perfectly replicate the characteristics of real sensors. Camera images in simulation lack the noise, lens artifacts, motion blur, and exposure variations of real cameras. LiDAR measurements in simulation may have different noise characteristics and miss certain surface types that cause issues in real sensors. IMU measurements in simulation often lack the bias drift, scale factors, and temperature dependencies of real inertial units.

**Domain Discrepancies** encompass broader differences between simulation and reality, including environmental factors such as lighting conditions, material appearance, and object configurations that differ from those seen during training. These discrepancies are particularly important for perception-based policies, where the visual appearance of objects in simulation may differ substantially from their real appearance.

### Quantifying the Reality Gap

Before attempting to bridge the reality gap, it is essential to quantify its magnitude. This quantification guides the selection of appropriate mitigation strategies and provides a baseline for measuring improvement.

The performance gap can be quantified by comparing success rates, task completion times, or other task-specific metrics between simulation and real-world deployment. For a humanoid locomotion task, for example, one might compare the distance walked before falling in simulation versus on the real robot.

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class TransferMetrics:
    """Metrics for quantifying sim-to-real transfer quality."""
    sim_success_rate: float
    real_success_rate: float
    sim_avg_return: float
    real_avg_return: float
    performance_ratio: float
    failure_modes: Dict[str, float]

class RealityGapAnalyzer:
    """
    Analyzes and quantifies the reality gap between simulation and reality.
    """

    def __init__(self):
        self.sim_results = []
        self.real_results = []

    def add_simulation_result(self, result):
        """Add a result from simulation evaluation."""
        self.sim_results.append(result)

    def add_real_result(self, result):
        """Add a result from real-world evaluation."""
        self.real_results.append(result)

    def compute_transfer_metrics(self) -> TransferMetrics:
        """Compute metrics characterizing the reality gap."""
        sim_success = np.mean([r.success for r in self.sim_results])
        real_success = np.mean([r.success for r in self.real_results])

        sim_returns = np.mean([r.return_value for r in self.sim_results])
        real_returns = np.mean([r.return_value for r in self.real_results])

        # Identify common failure modes
        failure_modes = self._identify_failure_modes()

        return TransferMetrics(
            sim_success_rate=sim_success,
            real_success_rate=real_success,
            sim_avg_return=sim_returns,
            real_avg_return=real_returns,
            performance_ratio=real_returns / sim_returns if sim_returns > 0 else 0,
            failure_modes=failure_modes
        )

    def _identify_failure_modes(self) -> Dict[str, float]:
        """Identify and count failure modes from real experiments."""
        failures = {}
        for result in self.real_results:
            if not result.success:
                failure_type = self._classify_failure(result)
                failures[failure_type] = failures.get(failure_type, 0) + 1

        # Normalize by total failures
        total_failures = sum(failures.values())
        for key in failures:
            failures[key] /= total_failures if total_failures > 0 else 1

        return failures

    def _classify_failure(self, result) -> str:
        """Classify the type of failure observed."""
        if result.fell:
            return "balance_failure"
        elif result.timed_out:
            return "timeout"
        elif result.collided:
            return "collision"
        else:
            return "unknown"
```

## Domain Randomization

Domain randomization is a simple yet effective approach to sim-to-real transfer that exposes the policy to a wide range of variations during training, making it robust to the specific conditions encountered during deployment.

### Randomization Parameters

Effective domain randomization requires identifying the parameters that most significantly affect policy performance and randomizing them within realistic ranges. For humanoid robots, key randomization parameters include:

**Dynamics Parameters**: Mass properties, friction coefficients, joint damping, and motor characteristics can all be randomized to make the policy robust to modeling errors. The ranges should be chosen based on uncertainty estimates for each parameter.

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable

@dataclass
class RandomizationConfig:
    """Configuration for domain randomization."""
    # Mass randomization
    mass_scale_range: tuple = (0.8, 1.2)  # +/- 20%
    link_mass_range: tuple = (0.9, 1.1)

    # Friction randomization
    friction_scale_range: tuple = (0.5, 1.5)

    # Joint dynamics randomization
    kp_scale_range: tuple = (0.8, 1.2)
    kd_scale_range: tuple = (0.8, 1.2)

    # Observation noise
    imu_noise_std_range: tuple = (0.5, 2.0)  # Multiplier on base noise
    joint_noise_std_range: tuple = (0.001, 0.01)

class DomainRandomizer:
    """
    Applies domain randomization to simulation environment.
    """

    def __init__(self, config: RandomizationConfig):
        self.config = config
        self.ranges = self._build_ranges()

    def randomize(self, env, mode='train'):
        """
        Apply domain randomization to environment.

        Args:
            env: The simulation environment
            mode: 'train' for training randomization, 'test' for minimal randomization
        """
        if mode == 'train':
            self._randomize_dynamics(env)
            self._randomize_observations(env)
            self._randomize_environment(env)
        else:
            # Minimal randomization for evaluation
            self._apply_nominal(env)

    def _build_ranges(self):
        """Build parameter ranges for randomization."""
        return {
            'mass_scale': np.linspace(
                self.config.mass_scale_range[0],
                self.config.mass_scale_range[1],
                5
            ),
            'friction_scale': np.linspace(
                self.config.friction_scale_range[0],
                self.config.friction_scale_range[1],
                5
            ),
            'kp_scale': np.linspace(
                self.config.kp_scale_range[0],
                self.config.kp_scale_range[1],
                5
            ),
            'kd_scale': np.linspace(
                self.config.kd_scale_range[0],
                self.config.kd_scale_range[1],
                5
            ),
            'imu_noise_mult': np.linspace(
                self.config.imu_noise_std_range[0],
                self.config.imu_noise_std_range[1],
                5
            )
        }

    def _randomize_dynamics(self, env):
        """Randomize dynamics parameters."""
        # Random mass scaling
        mass_scale = np.random.uniform(
            self.config.mass_scale_range[0],
            self.config.mass_scale_range[1]
        )
        self._scale_link_masses(env, mass_scale)

        # Random friction
        friction_scale = np.random.uniform(
            self.config.friction_scale_range[0],
            self.config.friction_scale_range[1]
        )
        self._scale_friction(env, friction_scale)

        # Random control gains
        kp_scale = np.random.uniform(
            self.config.kp_scale_range[0],
            self.config.kp_scale_range[1]
        )
        kd_scale = np.random.uniform(
            self.config.kd_scale_range[0],
            self.config.kd_scale_range[1]
        )
        self._scale_control_gains(env, kp_scale, kd_scale)

    def _scale_link_masses(self, env, scale):
        """Scale masses of all robot links."""
        for link in env.robot.get_links():
            original_mass = link.get_mass()
            new_mass = original_mass * scale
            link.set_mass(new_mass)

    def _scale_friction(self, env, scale):
        """Scale friction coefficients."""
        for material in env.robot.get_materials():
            original_friction = material.get_friction()
            new_friction = original_friction * scale
            material.set_friction(np.clip(new_friction, 0.1, 2.0))

    def _scale_control_gains(self, env, kp_scale, kd_scale):
        """Scale proportional and derivative control gains."""
        for joint in env.robot.get_joints():
            joint.set_kp(joint.get_kp() * kp_scale)
            joint.set_kd(joint.get_kd() * kd_scale)

    def _randomize_observations(self, env):
        """Randomize sensor observation characteristics."""
        # IMU noise scaling
        imu_noise_mult = np.random.uniform(
            self.config.imu_noise_std_range[0],
            self.config.imu_noise_std_range[1]
        )
        self._set_imu_noise(env, imu_noise_mult)

        # Joint position/velocity noise
        joint_noise_std = np.random.uniform(
            self.config.joint_noise_std_range[0],
            self.config.joint_noise_std_range[1]
        )
        env.joint_noise_std = joint_noise_std

    def _set_imu_noise(self, env, multiplier):
        """Set IMU noise level based on multiplier."""
        base_noise = 0.01  # Base IMU noise
        env.imu_noise_std = base_noise * multiplier

    def _randomize_environment(self, env):
        """Randomize environmental factors."""
        # Randomize gravity slightly (tilting the floor)
        gravity_jitter = np.random.uniform(-0.05, 0.05)
        current_gravity = env.physics.get_gravity()
        env.physics.set_gravity([
            current_gravity[0],
            current_gravity[1],
            current_gravity[2] + gravity_jitter
        ])

        # Randomize time step slightly
        dt_jitter = np.random.uniform(0.99, 1.01)
        env.physics.set_dt(env.base_dt * dt_jitter)

    def _apply_nominal(self, env):
        """Apply nominal (non-randomized) parameters."""
        # Set all parameters to their nominal values
        self._scale_link_masses(env, 1.0)
        self._scale_friction(env, 1.0)
        self._scale_control_gains(env, 1.0, 1.0)
        env.imu_noise_std = 0.01
        env.joint_noise_std = 0.001
```

### Texture and Visual Randomization

For policies that use visual observations, randomization of textures, lighting, and camera parameters helps bridge the visual domain gap:

```python
class VisualRandomizer:
    """
    Randomizes visual properties for domain randomization.
    """

    def __init__(self, config):
        self.config = config

    def randomize(self, env):
        """Apply visual randomization."""
        self._randomize_textures(env)
        self._randomize_lighting(env)
        self._randomize_camera(env)

    def _randomize_textures(self, env):
        """Randomize material textures and colors."""
        for material in env.scene.get_materials():
            # Randomize color
            color = np.random.uniform(0.3, 0.9, 3)
            material.set_albedo(color)

            # Randomize roughness
            roughness = np.random.uniform(0.1, 0.9)
            material.set_roughness(roughness)

            # Randomize metallic
            metallic = np.random.uniform(0.0, 0.5)
            material.set_metallic(metallic)

    def _randomize_lighting(self, env):
        """Randomize lighting conditions."""
        # Randomize light intensity
        light = env.scene.get_main_light()
        intensity_scale = np.random.uniform(0.5, 1.5)
        light.set_intensity(light.get_intensity() * intensity_scale)

        # Randomize light position
        pos_offset = np.random.uniform(-0.5, 0.5, 3)
        light.set_position(light.get_position() + pos_offset)

        # Randomize ambient light
        ambient_scale = np.random.uniform(0.3, 0.7)
        env.scene.set_ambient_light(ambient_scale)

    def _randomize_camera(self, env):
        """Randomize camera parameters."""
        for camera in env.cameras:
            # Randomize extrinsic (camera position relative to robot)
            if hasattr(camera, 'randomize_pose'):
                camera.randomize_pose(
                    position_std=0.02,
                    rotation_std=0.02
                )

            # Randomize intrinsic (if applicable)
            if hasattr(camera, 'randomize_exposure'):
                camera.randomize_exposure(
                    exposure_min=0.001,
                    exposure_max=0.1
                )
```

## System Identification

System identification takes the opposite approach from domain randomization: rather than making the policy robust to variations, it attempts to accurately model the real system and then trains in simulation that matches reality as closely as possible.

### Parameter Estimation

Systematic parameter estimation identifies the values of physical parameters that best explain real-world observations:

```python
import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class SysIdConfig:
    """Configuration for system identification."""
    # Parameter bounds
    mass_bounds: tuple = (0.8, 1.2)  # Multipliers on nominal mass
    friction_bounds: tuple = (0.5, 1.5)
    kp_bounds: tuple = (0.8, 1.2)
    kd_bounds: tuple = (0.8, 1.2)

    # Optimization settings
    n_iterations: int = 100
    population_size: int = 20

class SystemIdentifier:
    """
    Identifies system parameters to match simulation to real robot.
    """

    def __init__(self, config: SysIdConfig, simulator_factory: Callable):
        self.config = config
        self.simulator_factory = simulator_factory

        # Nominal parameters (from robot model)
        self.nominal_params = {
            'mass_scale': 1.0,
            'friction_scale': 1.0,
            'kp_scale': 1.0,
            'kd_scale': 1.0
        }

    def identify_parameters(self, real_trajectories, actions):
        """
        Identify system parameters that best explain real trajectories.

        Args:
            real_trajectories: List of observed state trajectories
            actions: Corresponding action sequences

        Returns:
            Dict of identified parameter values
        """
        def objective(params):
            """Objective function: sum of squared prediction errors."""
            scales = {
                'mass_scale': params[0],
                'friction_scale': params[1],
                'kp_scale': params[2],
                'kd_scale': params[3]
            }

            total_error = 0
            for traj, act in zip(real_trajectories, actions):
                sim_traj = self._simulate_trajectory(scales, act)
                error = self._compute_trajectory_error(traj, sim_traj)
                total_error += error

            return total_error

        # Define bounds for each parameter
        bounds = [
            self.config.mass_bounds,
            self.config.friction_bounds,
            self.config.kp_bounds,
            self.config.kd_bounds
        ]

        # Use differential evolution for global optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.config.n_iterations,
            popsize=self.config.population_size,
            seed=42
        )

        return {
            'mass_scale': result.x[0],
            'friction_scale': result.x[1],
            'kp_scale': result.x[2],
            'kd_scale': result.x[3],
            'optimization_success': result.success
        }

    def _simulate_trajectory(self, scales, actions):
        """Simulate trajectory with given parameter scales."""
        # Create simulator with specified parameters
        sim = self.simulator_factory(scales)

        # Run simulation
        trajectory = []
        state = sim.reset()
        trajectory.append(state.copy())

        for action in actions:
            state, _, done, _ = sim.step(action)
            trajectory.append(state.copy())
            if done:
                break

        return np.array(trajectory)

    def _compute_trajectory_error(self, real_traj, sim_traj):
        """Compute error between real and simulated trajectory."""
        # Interpolate to same length if needed
        min_len = min(len(real_traj), len(sim_traj))
        real_traj = real_traj[:min_len]
        sim_traj = sim_traj[:min_len]

        # Compute weighted state error
        # Position states weighted more heavily than velocity
        weights = np.ones_like(real_traj[0])
        weights[0:3] = 10.0  # Base position
        weights[3:6] = 1.0   # Base velocity
        weights[6:] = 0.5    # Joint states

        # Sum of squared errors
        errors = real_traj - sim_traj
        weighted_errors = errors * weights
        return np.sum(weighted_errors**2)
```

### Bayesian System Identification

For more robust parameter estimation, Bayesian approaches provide uncertainty estimates alongside parameter values:

```python
import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple

class BayesianSystemIdentifier:
    """
    Bayesian system identification using particle filtering or MCMC.
    """

    def __init__(self, prior_bounds, simulator_factory):
        self.prior_bounds = prior_bounds
        self.simulator_factory = simulator_factory
        self.n_particles = 100

    def identify_mcmc(self, real_trajectory, actions, n_samples=1000):
        """
        Use MCMC to sample from posterior distribution of parameters.
        """
        def log_likelihood(params):
            """Log likelihood of parameters given real trajectory."""
            scales = self._params_to_scales(params)
            sim_traj = self._simulate_trajectory(scales, actions)
            error = self._compute_error(real_traj, sim_traj)
            return -0.5 * error  # Gaussian likelihood

        def log_prior(params):
            """Log prior over parameters."""
            for i, (param, (low, high)) in enumerate(
                zip(params, self.prior_bounds)
            ):
                if param < low or param > high:
                    return -np.inf
            return 0  # Uniform prior

        def log_posterior(params):
            """Log posterior = log likelihood + log prior."""
            lp = log_prior(params)
            if not np.isfinite(lp):
                return lp
            return lp + log_likelihood(params)

        # Use Metropolis-Hastings MCMC
        samples = self._metropolis_hastings(
            log_posterior,
            n_samples,
            proposal_std=0.05
        )

        return {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'samples': samples,
            'acceptance_rate': self._get_acceptance_rate()
        }

    def _metropolis_hastings(self, log_posterior, n_samples, proposal_std):
        """Run Metropolis-Hastings MCMC."""
        # Initialize at prior mean
        current = np.array([
            (low + high) / 2 for low, high in self.prior_bounds
        ])
        current_density = log_posterior(current)

        samples = []
        n_accepted = 0

        for _ in range(n_samples):
            # Propose new sample
            proposal = current + np.random.normal(0, proposal_std)
            proposal_density = log_posterior(proposal)

            # Accept or reject
            if np.log(np.random.random()) < proposal_density - current_density:
                current = proposal
                current_density = proposal_density
                n_accepted += 1

            samples.append(current.copy())

        self._acceptance_rate = n_accepted / n_samples
        return np.array(samples)

    def _params_to_scales(self, params):
        """Convert parameter vector to scale dictionary."""
        return {
            'mass_scale': params[0],
            'friction_scale': params[1],
            'kp_scale': params[2],
            'kd_scale': params[3]
        }

    def _simulate_trajectory(self, scales, actions):
        """Simulate trajectory (see previous implementation)."""
        sim = self.simulator_factory(scales)
        trajectory = []
        state = sim.reset()
        trajectory.append(state.copy())

        for action in actions:
            state, _, done, _ = sim.step(action)
            trajectory.append(state.copy())
            if done:
                break

        return np.array(trajectory)

    def _compute_error(self, real_traj, sim_traj):
        """Compute error between trajectories."""
        min_len = min(len(real_traj), len(sim_traj))
        return np.sum((real_traj[:min_len] - sim_traj[:min_len])**2)

    def _get_acceptance_rate(self):
        """Get MCMC acceptance rate."""
        return getattr(self, '_acceptance_rate', 0.0)
```

## Domain Adaptation

Domain adaptation techniques aim to learn a transformation that maps between simulation and real domains, enabling policies trained in simulation to work on real data.

### Adversarial Domain Adaptation

Adversarial domain adaptation uses domain classifiers to learn domain-invariant representations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdaptivePolicy(nn.Module):
    """
    Policy with domain adaptation for sim-to-real transfer.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Feature extractor (shared between domains)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state, alpha=1.0):
        """
        Forward pass with domain adaptation.

        Args:
            state: Input state
            alpha: Domain adaptation strength

        Returns:
            action: Policy action
            domain_pred: Domain prediction (for adversarial loss)
        """
        features = self.feature_extractor(state)

        # Get policy action
        action = self.policy_head(features)

        # Domain classification (with gradient reversal)
        domain_pred = self.domain_classifier(
            GradientReversalFunction.apply(features, alpha)
        )

        return action, domain_pred

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal function for domain adaptation.
    Flips gradients during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Flip the gradient
        return grad_output.neg() * ctx.lambda_, None

class DomainAdaptationTrainer:
    """
    Trainer for domain-adaptive policies.
    """

    def __init__(self, policy, lr=3e-4, domain_weight=0.5):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.domain_weight = domain_weight

    def train_step(self, sim_batch, real_batch):
        """
        Single training step with domain adaptation.

        Args:
            sim_batch: Batch of simulation transitions
            real_batch: Batch of real-world transitions
        """
        # Unpack batches
        sim_states, sim_actions, sim_rewards, sim_next_states = sim_batch
        real_states, real_actions, real_rewards, real_next_states = real_batch

        # Combine batches
        all_states = torch.cat([sim_states, real_states], dim=0)

        # Forward pass
        actions, domain_preds = self.policy(all_states, alpha=1.0)

        # Split predictions
        sim_domain_preds = domain_preds[:len(sim_states)]
        real_domain_preds = domain_preds[len(sim_states):]

        # Policy loss (only on simulation data)
        sim_action_loss = F.mse_loss(
            actions[:len(sim_states)],
            sim_actions
        )
        sim_reward_loss = -sim_rewards.mean()

        # Domain classification loss
        # Simulation should be classified as domain 0, real as domain 1
        sim_domain_loss = F.binary_cross_entropy(
            sim_domain_preds,
            torch.zeros_like(sim_domain_preds)
        )
        real_domain_loss = F.binary_cross_entropy(
            real_domain_preds,
            torch.ones_like(real_domain_preds)
        )
        domain_loss = sim_domain_loss + real_domain_loss

        # Total loss
        policy_loss = sim_action_loss + sim_reward_loss
        total_loss = policy_loss + self.domain_weight * domain_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item()
        }
```

## Progressive and Iterative Transfer

Rather than attempting to transfer directly from simulation to reality, progressive and iterative approaches gradually bridge the gap through intermediate steps.

### Progressive Networks

Progressive networks maintain columns for each domain and transfer knowledge progressively:

```python
import torch
import torch.nn as nn

class ProgressiveColumn(nn.Module):
    """
    Single column of a progressive network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.hidden = nn.Linear(input_dim, hidden_dim)

        # Lateral connections from previous columns
        self.lateral_connections = nn.ModuleList()

        self.output = nn.Linear(hidden_dim, output_dim)

    def init_lateral(self, n_columns):
        """Initialize lateral connections from all previous columns."""
        self.lateral_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_columns)
        ])

class ProgressiveNetwork(nn.Module):
    """
    Progressive network for sim-to-real transfer.
    Each column represents a different domain (simulation, real, etc.)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.columns = nn.ModuleList()
        self.hidden_dim = hidden_dim

        # Create columns for different domains
        # Column 0: Simulation
        self.columns.append(
            ProgressiveColumn(state_dim, hidden_dim, action_dim)
        )

        # Column 1: Real robot (will connect to column 0)
        self.columns.append(
            ProgressiveColumn(state_dim, hidden_dim, action_dim)
        )

        # Initialize lateral connections for column 1
        self.columns[1].init_lateral(1)

    def forward(self, state, column_idx):
        """
        Forward pass through specified column.

        Args:
            state: Input state
            column_idx: Which column to use (0=sim, 1=real)

        Returns:
            action: Output action
        """
        x = torch.relu(self.columns[column_idx].hidden(state))

        # Add lateral connections from previous columns
        if column_idx > 0:
            for i in range(column_idx):
                lateral_out = torch.relu(
                    self.columns[column_idx].lateral_connections[i](
                        self.columns[i].hidden(state)
                    )
                )
                x = x + lateral_out  # Residual connection

        action = self.columns[column_idx].output(x)
        return action.tanh()
```

### Iterative Online Adaptation

Iterative approaches alternate between simulation training and real-world refinement:

```python
class IterativeTransfer:
    """
    Iterative sim-to-real transfer with online adaptation.
    """

    def __init__(self, sim_env, real_env, policy_class, config):
        self.sim_env = sim_env
        self.real_env = real_env
        self.config = config

        # Initialize policy
        self.policy = policy_class(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        # Replay buffers
        self.sim_buffer = ReplayBuffer(config.buffer_size)
        self.real_buffer = ReplayBuffer(config.buffer_size)

    def train_iteration(self, n_sim_episodes=100, n_real_episodes=10):
        """
        Single iteration of iterative training.
        """
        # Phase 1: Collect simulation data
        print("Collecting simulation data...")
        for _ in range(n_sim_episodes):
            trajectory = self._collect_trajectory(self.sim_env, self.policy)
            self.sim_buffer.add_trajectory(trajectory)

        # Phase 2: Train on simulation data
        print("Training on simulation data...")
        for _ in range(self.config.n_gradient_steps):
            batch = self.sim_buffer.sample(self.config.batch_size)
            self._update_policy(batch)

        # Phase 3: Collect real data (with current policy)
        print("Collecting real data...")
        for _ in range(n_real_episodes):
            trajectory = self._collect_trajectory(self.real_env, self.policy)
            self.real_buffer.add_trajectory(trajectory)

        # Phase 4: Update policy with real data (if enough collected)
        if len(self.real_buffer) > self.config.min_real_samples:
            print("Adapting to real data...")
            for _ in range(self.config.n_adaptation_steps):
                # Mix simulation and real data
                sim_batch = self.sim_buffer.sample(self.config.batch_size // 2)
                real_batch = self.real_buffer.sample(self.config.batch_size // 2)
                self._adapt_policy(sim_batch, real_batch)

    def _collect_trajectory(self, env, policy):
        """Collect trajectory using current policy."""
        trajectory = []
        state = env.reset()

        for _ in range(self.config.max_episode_length):
            action = policy.select_action(state)
            next_state, reward, done, info = env.step(action)

            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            state = next_state
            if done:
                break

        return trajectory

    def _update_policy(self, batch):
        """Update policy with simulation data."""
        # Standard RL update (e.g., PPO, SAC)
        pass

    def _adapt_policy(self, sim_batch, real_batch):
        """
        Adapt policy using both simulation and real data.
        Uses importance weighting to account for distribution shift.
        """
        # Compute importance weights
        sim_states = torch.tensor(
            np.array([b['state'] for b in sim_batch])
        )
        real_states = torch.tensor(
            np.array([b['state'] for b in real_batch])
        )

        # Domain classifier for importance weights
        with torch.no_grad():
            _, sim_domain = self.policy(sim_states, alpha=1.0)
            _, real_domain = self.policy(real_states, alpha=1.0)

        # Importance weights
        sim_weights = sim_domain / (1 - sim_domain + 1e-8)
        real_weights = (1 - real_domain) / (real_domain + 1e-8)

        # Weighted update (implementation depends on RL algorithm)
        pass
```

## Practical Transfer Pipeline

A complete sim-to-real transfer pipeline combines multiple techniques for robust transfer:

```python
class SimToRealPipeline:
    """
    Complete sim-to-real transfer pipeline.
    Combines domain randomization, system identification, and adaptation.
    """

    def __init__(self, config):
        self.config = config

        # Components
        self.domain_randomizer = DomainRandomizer(config.randomization_config)
        self.sysid = SystemIdentifier(config.sysid_config, self._create_simulator)
        self.adaptive_policy = self._create_policy()

        # Data collection
        self.real_trajectories = []
        self.sim_trajectories = []

    def run_transfer(self, n_iterations=10):
        """
        Run complete transfer pipeline.

        Args:
            n_iterations: Number of iterations of the pipeline
        """
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")

            # Step 1: Collect real data with current policy
            print("  Collecting real data...")
            for _ in range(self.config.n_real_rollouts):
                traj = self._collect_real_trajectory()
                self.real_trajectories.append(traj)

            # Step 2: System identification
            if iteration == 0:
                print("  Running system identification...")
                sysid_result = self.sysid.identify_parameters(
                    self.real_trajectories,
                    self._extract_actions(self.real_trajectories)
                )
                print(f"  Identified scales: {sysid_result}")

            # Step 3: Train in simulation with randomization
            print("  Training in simulation...")
            self._train_simulation(sysid_result)

            # Step 4: Online adaptation
            print("  Online adaptation...")
            self._adapt_online()

    def _collect_real_trajectory(self):
        """Collect trajectory on real robot."""
        trajectory = []
        state = self.real_env.reset()

        for _ in range(self.config.max_rollout_length):
            action = self.adaptive_policy.select_action(state)
            next_state, reward, done, info = self.real_env.step(action)

            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            state = next_state
            if done:
                break

        return trajectory

    def _train_simulation(self, sysid_result):
        """Train policy in simulation with identified parameters."""
        # Set simulation parameters from system identification
        self._configure_simulation(sysid_result)

        # Training loop with domain randomization
        for epoch in range(self.config.n_epochs):
            # Randomize environment
            self.domain_randomizer.randomize(self.sim_env, mode='train')

            # Collect and train
            for _ in range(self.config.n_envs_per_epoch):
                traj = self._collect_sim_trajectory()
                self._train_on_trajectory(traj)

    def _adapt_online(self):
        """Adapt policy online using real robot data."""
        recent_trajectories = self.real_trajectories[-self.config.n_recent_trajectories:]

        for traj in recent_trajectories:
            # Compute importance weights based on domain classifier
            weights = self._compute_importance_weights(traj)

            # Update policy with weighted losses
            self._adapt_policy(traj, weights)

    def _compute_importance_weights(self, trajectory):
        """Compute importance weights for trajectory adaptation."""
        states = torch.tensor(np.array([t['state'] for t in trajectory]))

        with torch.no_grad():
            _, domain_pred = self.adaptive_policy(states, alpha=1.0)

        # Weight real data by inverse of domain classifier output
        weights = 1.0 / (domain_pred + 1e-8)
        weights = weights / weights.mean()  # Normalize

        return weights.numpy()

    def _create_policy(self):
        """Create the policy network."""
        return DomainAdaptivePolicy(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        )

    def _create_simulator(self, scales):
        """Create simulator with given scales."""
        # Implementation depends on simulator
        pass

    def _configure_simulation(self, scales):
        """Configure simulation with identified parameters."""
        # Set masses, friction, etc. based on scales
        pass

    def _collect_sim_trajectory(self):
        """Collect trajectory in simulation."""
        pass

    def _train_on_trajectory(self, trajectory):
        """Train policy on collected trajectory."""
        pass

    def _extract_actions(self, trajectories):
        """Extract action sequences from trajectories."""
        return [
            np.array([t['action'] for t in traj])
            for traj in trajectories
        ]
```

## Key Takeaways

Sim-to-real transfer remains one of the most challenging problems in Physical AI, but multiple techniques have proven effective for bridging the reality gap. The choice of technique depends on the specific application and the nature of the reality gap.

- **Understanding the gap** requires identifying sources of discrepancy in physics, sensors, and environment
- **Domain randomization** makes policies robust by exposing them to varied conditions during training
- **System identification** tunes simulation parameters to match the real system
- **Domain adaptation** learns transformations between simulation and real domains
- **Iterative transfer** progressively bridges the gap through multiple adaptation stages
- **Combining techniques** often provides the most robust transfer for complex systems

With sim-to-real transfer techniques mastered, we can now explore Vision-Language-Action models that enable robots to understand and execute natural language instructions.
