---
sidebar_label: 'NVIDIA Isaac Gym / RL Examples'
sidebar_position: 4
---

# NVIDIA Isaac Gym / RL Examples

## Overview

NVIDIA Isaac Gym is a GPU-accelerated robotics simulation framework that enables parallel training of reinforcement learning agents for robotic tasks. Built on NVIDIA's PhysX physics engine and leveraging GPU acceleration, Isaac Gym can simulate thousands of robot environments in parallel, dramatically reducing training time for complex robotic behaviors. This chapter explores Isaac Gym's capabilities and provides practical examples for humanoid robotics applications.

## Introduction to NVIDIA Isaac Gym

### What is Isaac Gym?

NVIDIA Isaac Gym is:
- A GPU-accelerated physics simulation framework
- A reinforcement learning environment specifically designed for robotics
- Part of the Isaac ecosystem for robotics development
- Capable of parallelizing thousands of robot simulations simultaneously
- Integrated with NVIDIA's RTX rendering for computer vision training

### Key Features

#### GPU Acceleration
- Parallel physics simulation across thousands of environments
- Real-time rendering for computer vision applications
- CUDA-accelerated compute operations
- Efficient memory management for large-scale training

#### Robotics-Focused Design
- Realistic physics simulation with PhysX 5.0
- Support for complex robot articulations
- Integrated sensor simulation
- ROS/ROS2 bridge for real robot deployment

#### RL Integration
- Native support for popular RL algorithms
- Parallel training and inference
- Curriculum learning capabilities
- Pre-built environments for common robotic tasks

### Architecture

```
Application Layer (RL Algorithms)
       ↓
Environment Manager (Parallel Environments)
       ↓
Physics Engine (PhysX 5.0)
       ↓
GPU Compute (CUDA/RTX)
```

## Installing and Setting Up Isaac Gym

### Prerequisites

Before installing Isaac Gym, ensure your system meets the requirements:

```bash
# System requirements
- NVIDIA GPU with compute capability 6.0 or higher (Pascal or newer)
- CUDA 11.8 or later
- Linux (Ubuntu 20.04/22.04) or Windows 10/11
- Python 3.8-3.10

# Install Isaac Gym
pip install isaacgym

# For reinforcement learning tools
pip install rlgpu  # NVIDIA's RL GPU tools
```

### Basic Environment Setup

```python
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np

# Initialize Isaac Gym
gym = gymapi.acquire_gym()

# Configure simulation parameters
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# Choose physics engine
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.max_gpu_contact_pairs = 2**23
sim_params.physx.max_gpu_deleted_pairs = 2**23
sim_params.physx.use_gpu = True  # Enable GPU physics

# Create simulation
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
```

## Creating a Basic Robot Environment

### Robot Asset Loading

```python
# Load robot asset
asset_root = "path/to/robot/assets"
asset_file = "humanoid.urdf"  # or .usd, .obj

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False  # Robot should be free to move
asset_options.enable_self_collision = True  # Enable self-collision
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = False

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Get robot properties
robot_dof_props = gym.get_asset_dof_properties(sim, asset)
robot_rigid_shape_props = gym.get_asset_rigid_shape_properties(sim, asset)
```

### Environment Creation

```python
# Configure environments
num_envs = 4096  # Number of parallel environments
spacing = 2.5

# Create environment
env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

print("Creating %d environments" % num_envs)
num_per_row = int(np.sqrt(num_envs))

envs = []
for i in range(num_envs):
    # Create environment
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)

    # Add robot to environment
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # Start position
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Start orientation

    actor_handle = gym.create_actor(env, asset, pose, "robot", i, 1, 0)

    # Set DOF drive properties
    gym.set_actor_dof_properties(env, actor_handle, robot_dof_props)

    # Set rigid body properties
    gym.set_actor_rigid_body_properties(env, actor_handle, robot_rigid_body_props)

    envs.append(env)
```

## Implementing RL Agents with Isaac Gym

### RL Environment Class

```python
import torch
import torch.nn as nn
import numpy as np
from rl_games.common.env_configurations import get_env_info
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import isaacgym

class IsaacGymEnv:
    def __init__(self, gym, sim, envs, num_actor_obs, num_actor_actions):
        self.gym = gym
        self.sim = sim
        self.envs = envs

        # Action and observation dimensions
        self.num_actions = num_actor_actions
        self.num_obs = num_actor_obs

        # Get gym tensors
        self.obs_tensor = gym.acquire_env_tensors(sim, 'obs')
        self.rew_tensor = gym.acquire_env_tensors(sim, 'rew')
        self.reset_tensor = gym.acquire_env_tensors(sim, 'reset')

        # Actor handles
        self.actor_handles = []
        for env in envs:
            actor_handle = gym.get_actor_handle(env, 0)
            self.actor_handles.append(actor_handle)

    def reset(self):
        """Reset the environment and return initial observations"""
        # Reset simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Fetch observations
        obs = gymtorch.wrap_tensor(self.obs_tensor)
        return obs

    def step(self, actions):
        """Execute actions and return (obs, reward, done, info)"""
        # Apply actions to simulation
        actions_tensor = torch.from_numpy(actions).float()
        gymtorch.set_dof_position_targets_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Get results
        obs = gymtorch.wrap_tensor(self.obs_tensor)
        rew = gymtorch.wrap_tensor(self.rew_tensor)
        done = gymtorch.wrap_tensor(self.reset_tensor)

        return obs, rew, done, {}
```

### Deep Deterministic Policy Gradient (DDPG) for Robot Control

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = deque(maxlen=1000000)
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=100):
        # Sample replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, not_done = map(torch.FloatTensor, zip(*batch))
        state = state.cuda()
        action = action.cuda()
        next_state = next_state.cuda()
        reward = reward.cuda()
        not_done = not_done.cuda()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## Humanoid Locomotion Example

### Bipedal Walker Environment

```python
import torch
import torch.nn as nn
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch

class BipedalWalkerEnv:
    def __init__(self, num_envs=4096):
        self.num_envs = num_envs
        self.num_obs = 48  # Observation dimension
        self.num_actions = 10  # 5 joints per leg = 10 total

        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()

        # Create simulation
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 2**23
        sim_params.physx.use_gpu = True

        self.sim = gymapi.create_sim(self.gym, 0, 0, gymapi.SIM_PHYSX, sim_params)

        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        self.gym.add_ground(self.sim, plane_params)

        # Load robot asset
        asset_root = "path/to/humanoid/assets"
        asset_file = "bipedal_walker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.enable_self_collision = True
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Create environments
        self._create_envs()

        # Get gym tensors
        self._setup_tensors()

    def _create_envs(self):
        spacing = 2.5
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        num_per_row = int(np.sqrt(self.num_envs))

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # Add robot to environment
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            actor_handle = self.gym.create_actor(env, self.robot_asset, pose, "robot", i, 1, 0)

            # Set DOF properties
            robot_dof_props = self.gym.get_asset_dof_properties(self.sim, self.robot_asset)
            for j in range(robot_dof_props.num_dofs):
                robot_dof_props.stiffness[j] = 800.0
                robot_dof_props.damping[j] = 50.0
                robot_dof_props.armature[j] = 0.01

            self.gym.set_actor_dof_properties(env, actor_handle, robot_dof_props)
            self.envs.append(env)

    def _setup_tensors(self):
        self.obs_tensor = self.gym.acquire_env_tensors(self.sim, 'obs')
        self.rew_tensor = self.gym.acquire_env_tensors(self.sim, 'rew')
        self.reset_tensor = self.gym.acquire_env_tensors(self.sim, 'reset')
        self.state_tensor = self.gym.acquire_env_tensors(self.sim, 'state')

        self.obs_buf = gymtorch.wrap_tensor(self.obs_tensor).clone()
        self.rew_buf = gymtorch.wrap_tensor(self.rew_tensor).clone()
        self.reset_buf = gymtorch.wrap_tensor(self.reset_tensor).clone()
        self.state_buf = gymtorch.wrap_tensor(self.state_tensor).clone()

    def reset(self):
        """Reset all environments and return initial observations"""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Reset joint positions and velocities
        joint_pos = gymtorch.get_actor_dof_position_tensor(self.sim)
        joint_vel = gymtorch.get_actor_dof_velocity_tensor(self.sim)

        # Add small randomization
        joint_pos[:] = torch.randn_like(joint_pos) * 0.1
        joint_vel[:] = torch.randn_like(joint_vel) * 0.01

        obs = gymtorch.wrap_tensor(self.obs_tensor)
        return obs

    def step(self, actions):
        """Execute actions and return step results"""
        # Apply actions to simulation
        actions_tensor = torch.from_numpy(actions).float().cuda()
        gymtorch.set_dof_position_targets_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Get results
        obs = gymtorch.wrap_tensor(self.obs_tensor)
        rew = gymtorch.wrap_tensor(self.rew_tensor)
        done = gymtorch.wrap_tensor(self.reset_tensor)

        # Reset environments that need resetting
        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        return obs, rew, done, {}

    def reset_idx(self, env_ids):
        """Reset specific environments"""
        # Reset joint positions and velocities for specific environments
        joint_pos = gymtorch.get_actor_dof_position_tensor(self.sim)
        joint_vel = gymtorch.get_actor_dof_velocity_tensor(self.sim)

        # Randomize starting positions
        joint_pos[env_ids] = torch.randn_like(joint_pos[env_ids]) * 0.1
        joint_vel[env_ids] = torch.randn_like(joint_vel[env_ids]) * 0.01
```

### Reward Function for Walking

```python
def walking_reward(robot_state, actions, dt=1/60):
    """Calculate reward for bipedal walking task"""
    # Robot state includes position, velocity, orientation, joint angles, etc.
    pos = robot_state['position']
    lin_vel = robot_state['linear_velocity']
    ang_vel = robot_state['angular_velocity']
    joint_pos = robot_state['joint_positions']
    joint_vel = robot_state['joint_velocities']

    # Forward velocity reward (encourage moving forward)
    forward_vel_reward = max(0, lin_vel[0])  # x-axis is forward

    # Balance reward (maintain upright orientation)
    up_vec = robot_state['up_vector']
    target_up = torch.tensor([0, 0, 1]).float()
    balance_reward = torch.dot(up_vec, target_up)

    # Energy efficiency (penalize large joint torques/velocities)
    energy_penalty = -torch.mean(torch.abs(joint_vel))

    # Stay alive bonus
    alive_bonus = 0.1

    # Penalty for falling
    fall_penalty = 0
    if robot_state['height'] < 0.5:  # Robot is on the ground
        fall_penalty = -1.0

    total_reward = (0.5 * forward_vel_reward +
                   0.3 * balance_reward +
                   0.1 * energy_penalty +
                   0.05 * alive_bonus +
                   0.05 * fall_penalty)

    return total_reward
```

## Advanced Isaac Gym Features

### Domain Randomization

```python
class DomainRandomizedBipedalEnv(BipedalWalkerEnv):
    def __init__(self, num_envs=4096, randomization_params=None):
        super().__init__(num_envs)

        # Randomization parameters
        self.randomization_params = randomization_params or {
            'mass_scale_range': [0.8, 1.2],
            'friction_range': [0.5, 1.5],
            'restitution_range': [0.0, 0.5],
            'motor_strength_range': [0.8, 1.2],
            'gravity_range': [-10.8, -8.8]
        }

        # Store original properties for randomization
        self.original_mass_props = None
        self.original_dof_props = None

    def randomize_env(self, env_ids):
        """Randomize properties for specific environments"""
        for env_id in env_ids:
            # Randomize robot mass
            mass_scale = np.random.uniform(
                self.randomization_params['mass_scale_range'][0],
                self.randomization_params['mass_scale_range'][1]
            )

            # Randomize DOF properties
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)
            for i in range(len(dof_props.stiffness)):
                dof_props.stiffness[i] *= np.random.uniform(
                    self.randomization_params['motor_strength_range'][0],
                    self.randomization_params['motor_strength_range'][1]
                )
                dof_props.damping[i] *= np.random.uniform(
                    self.randomization_params['motor_strength_range'][0],
                    self.randomization_params['motor_strength_range'][1]
                )

            self.gym.set_actor_dof_properties(self.envs[env_id], 0, dof_props)

    def reset(self):
        """Reset environments with randomization"""
        reset_env_ids = torch.arange(self.num_envs, device='cuda', dtype=torch.long)
        self.randomize_env(reset_env_ids)

        return super().reset()
```

### Curriculum Learning

```python
class CurriculumBipedalEnv(BipedalWalkerEnv):
    def __init__(self, num_envs=4096):
        super().__init__(num_envs)

        # Curriculum parameters
        self.curriculum_levels = [
            {'obstacle_height': 0.0, 'slope_angle': 0.0, 'reward_scaling': 1.0},
            {'obstacle_height': 0.05, 'slope_angle': 0.05, 'reward_scaling': 1.2},
            {'obstacle_height': 0.1, 'slope_angle': 0.1, 'reward_scaling': 1.5},
            {'obstacle_height': 0.15, 'slope_angle': 0.15, 'reward_scaling': 2.0}
        ]

        self.current_level = 0
        self.performance_threshold = 0.8  # Success rate threshold
        self.level_progress = torch.zeros(self.num_envs, device='cuda')

    def update_curriculum(self, episode_rewards):
        """Update curriculum level based on performance"""
        success_rate = torch.mean((episode_rewards > self.performance_threshold).float())

        if success_rate > 0.9 and self.current_level < len(self.curriculum_levels) - 1:
            self.current_level += 1
            print(f"Curriculum advanced to level {self.current_level}")

        return success_rate
```

### Sensor Simulation

```python
class SensorizedBipedalEnv(BipedalWalkerEnv):
    def __init__(self, num_envs=4096):
        super().__init__(num_envs)

        # Add camera sensors
        self.camera_handles = []

        for i, env in enumerate(self.envs):
            # Create camera sensor
            camera_props = gymapi.CameraProperties()
            camera_props.width = 640
            camera_props.height = 480
            camera_props.enable_tensors = True

            camera_handle = self.gym.create_camera_sensor(env, 0, camera_props)

            # Set camera position (mounted on robot head)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(0, 0, 0.5)  # 0.5m above ground
            local_transform.r = gymapi.Quat(0, 0, 0, 1)

            self.gym.set_camera_local_transform(env, camera_handle, local_transform)
            self.camera_handles.append(camera_handle)

    def get_camera_observations(self):
        """Get camera observations from all environments"""
        camera_obs = []

        for i, camera_handle in enumerate(self.camera_handles):
            # Get camera tensor
            camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], camera_handle, gymapi.IMAGE_COLOR
            )

            # Convert to PyTorch tensor
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            camera_obs.append(torch_camera_tensor)

        return torch.stack(camera_obs)
```

## Training with Isaac Gym

### PPO Implementation for Isaac Gym

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(PPOActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value network)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.feature_extractor(obs)

        # Actor
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Critic
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action_and_value(self, obs, action=None):
        action_mean, action_std, value = self.forward(obs)

        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(1, keepdim=True)
        entropy = dist.entropy().sum(1, keepdim=True)

        return action, log_prob, entropy, value

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, clip_eps=0.2):
        self.actor_critic = PPOActorCritic(obs_dim, action_dim).cuda()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5

    def update(self, obs, actions, log_probs, returns, advantages):
        # Convert to tensors
        obs = torch.FloatTensor(obs).cuda()
        actions = torch.FloatTensor(actions).cuda()
        old_log_probs = torch.FloatTensor(log_probs).cuda()
        returns = torch.FloatTensor(returns).cuda().unsqueeze(1)
        advantages = torch.FloatTensor(advantages).cuda().unsqueeze(1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get new action and value
        _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(obs, actions)

        # Calculate ratios
        ratios = torch.exp(new_log_probs - old_log_probs)

        # PPO surrogate losses
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratings, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = (new_values - returns).pow(2).mean()

        # Total loss
        loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), value_loss.item(), entropy.mean().item()
```

### Training Loop

```python
def train_isaac_gym_agent():
    """Complete training loop for Isaac Gym agent"""

    # Initialize environment
    env = BipedalWalkerEnv(num_envs=4096)

    # Initialize agent
    agent = PPOAgent(obs_dim=env.num_obs, action_dim=env.num_actions)

    # Training parameters
    num_iterations = 10000
    num_steps = 32  # Steps per update
    gamma = 0.99
    gae_lambda = 0.95

    # Initialize buffers
    obs = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device='cuda')
    episode_lengths = torch.zeros(env.num_envs, device='cuda')
    episode_count = torch.zeros(env.num_envs, device='cuda')

    for iteration in range(num_iterations):
        # Collect trajectories
        obs_buf = torch.zeros((num_steps, env.num_envs, env.num_obs), device='cuda')
        action_buf = torch.zeros((num_steps, env.num_envs, env.num_actions), device='cuda')
        log_prob_buf = torch.zeros((num_steps, env.num_envs, 1), device='cuda')
        reward_buf = torch.zeros((num_steps, env.num_envs), device='cuda')
        done_buf = torch.zeros((num_steps, env.num_envs), device='cuda')
        value_buf = torch.zeros((num_steps, env.num_envs), device='cuda')

        for step in range(num_steps):
            # Get action from agent
            with torch.no_grad():
                action, log_prob, entropy, value = agent.actor_critic.get_action_and_value(obs)

            # Store data
            obs_buf[step] = obs
            action_buf[step] = action
            log_prob_buf[step] = log_prob
            value_buf[step] = value.squeeze(1)

            # Step environment
            obs, reward, done, info = env.step(action.cpu().numpy())

            reward_buf[step] = reward
            done_buf[step] = done

            # Update episode statistics
            episode_rewards += reward
            episode_lengths += 1

            # Reset episode statistics for done environments
            episode_count += done
            episode_rewards = episode_rewards * (1 - done)
            episode_lengths = episode_lengths * (1 - done)

        # Calculate returns and advantages using GAE
        with torch.no_grad():
            next_value = agent.actor_critic.get_action_and_value(obs)[3].squeeze(1)
            returns = torch.zeros_like(reward_buf)
            advantages = torch.zeros_like(reward_buf)

            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - done_buf[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t+1]
                    nextvalues = value_buf[t+1]

                delta = reward_buf[t] + gamma * nextvalues * nextnonterminal - value_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + value_buf

        # Flatten buffers for training
        b_obs = obs_buf.reshape((-1,) + (env.num_obs,))
        b_action = action_buf.reshape((-1,) + (env.num_actions,))
        b_log_prob = log_prob_buf.reshape(-1, 1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # Update agent
        actor_loss, critic_loss, entropy = agent.update(
            b_obs, b_action, b_log_prob, b_returns, b_advantages
        )

        # Log progress
        if iteration % 10 == 0:
            avg_reward = episode_rewards.mean().item()
            avg_length = episode_lengths.mean().item()
            avg_episodes = episode_count.mean().item()

            print(f"Iteration {iteration}: "
                  f"Reward={avg_reward:.2f}, "
                  f"Length={avg_length:.2f}, "
                  f"Episodes={avg_episodes:.2f}, "
                  f"Actor Loss={actor_loss:.4f}, "
                  f"Critic Loss={critic_loss:.4f}")

    return agent
```

## Deployment to Real Robots

### Transfer from Simulation to Reality

```python
def deploy_to_real_robot(agent, real_robot_interface):
    """Deploy trained Isaac Gym policy to real robot"""

    # Set agent to evaluation mode
    agent.actor_critic.eval()

    # Initialize real robot
    real_robot_interface.initialize()

    # Main control loop
    try:
        obs = real_robot_interface.get_observation()

        while True:
            # Preprocess observation (may need normalization)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).cuda()

            # Get action from trained policy
            with torch.no_grad():
                action_mean, action_std, value = agent.actor_critic(obs_tensor)
                action = torch.tanh(action_mean).cpu().numpy()[0]  # Apply tanh and convert to numpy

            # Scale action to robot's action space
            scaled_action = scale_action_to_robot(action)

            # Send action to robot
            real_robot_interface.send_action(scaled_action)

            # Get next observation
            obs = real_robot_interface.get_observation()

            # Check for termination conditions
            if real_robot_interface.should_terminate():
                break

    except KeyboardInterrupt:
        print("Deployment interrupted by user")
    finally:
        real_robot_interface.shutdown()

def scale_action_to_robot(robot_action):
    """Scale normalized action (-1 to 1) to robot-specific ranges"""
    # Define robot-specific joint limits (example)
    joint_limits = {
        'hip': (-0.5, 0.5),
        'knee': (0, 1.5),
        'ankle': (-0.3, 0.3)
    }

    scaled_action = np.copy(robot_action)

    # Scale each joint based on its specific limits
    for i, joint_name in enumerate(joint_limits.keys()):
        min_val, max_val = joint_limits[joint_name]
        # From [-1, 1] to [min_val, max_val]
        scaled_action[i] = min_val + (robot_action[i] + 1) * (max_val - min_val) / 2

    return scaled_action
```

### Safe Deployment Strategies

```python
class SafeDeploymentManager:
    def __init__(self, real_robot_interface):
        self.robot = real_robot_interface
        self.safety_monitor = SafetyMonitor()
        self.fallback_controller = FallbackController()

    def deploy_with_safety(self, trained_policy):
        """Deploy policy with safety checks"""

        print("Starting safe deployment...")

        # Initialize safety systems
        self.safety_monitor.start_monitoring()
        self.fallback_controller.activate()

        try:
            obs = self.robot.get_observation()

            while True:
                # Check safety conditions
                if not self.safety_monitor.is_safe():
                    print("Safety violation detected! Activating fallback...")
                    self.fallback_controller.take_control()
                    continue

                # Get policy action
                action = trained_policy.get_action(obs)

                # Verify action is within safe limits
                safe_action = self.safety_monitor.clamp_action(action)

                # Send to robot
                self.robot.send_action(safe_action)

                # Get next observation
                obs = self.robot.get_observation()

                # Check for termination
                if self.robot.emergency_stop_requested():
                    break

        except Exception as e:
            print(f"Error during deployment: {e}")
            self.fallback_controller.emergency_stop()
        finally:
            self.safety_monitor.stop_monitoring()
            self.fallback_controller.deactivate()
```

## Best Practices and Optimization

### Performance Optimization

```python
def optimize_isaac_gym_training():
    """Best practices for optimizing Isaac Gym training"""

    # 1. Optimize simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # Reduce solver iterations if possible
    sim_params.physx.num_position_iterations = 4  # Start with lower value
    sim_params.physx.num_velocity_iterations = 1

    # Use GPU for physics
    sim_params.physx.use_gpu = True
    sim_params.physx.max_gpu_contact_pairs = 2**23

    # 2. Optimize environment spacing
    spacing = 2.0  # Minimize space while avoiding interference

    # 3. Batch operations when possible
    def batch_reset(env_ids):
        """Reset multiple environments efficiently"""
        # Set all reset flags at once
        reset_tensor = torch.zeros(len(env_ids), device='cuda', dtype=torch.bool)
        return reset_tensor

    # 4. Use appropriate network architecture
    # For high-dimensional observations, consider using CNNs for vision
    # For joint-based observations, MLPs often work well
```

### Memory Management

```python
class EfficientMemoryManager:
    def __init__(self, max_memory=1000000):
        self.max_memory = max_memory
        self.replay_buffer = deque(maxlen=max_memory)

        # Pre-allocate tensors to avoid memory fragmentation
        self.obs_tensor = torch.empty((1000, obs_dim), device='cuda')
        self.action_tensor = torch.empty((1000, action_dim), device='cuda')

    def add_experience(self, obs, action, reward, next_obs, done):
        """Add experience to replay buffer efficiently"""
        self.replay_buffer.append((obs, action, reward, next_obs, done))

    def sample_batch(self, batch_size):
        """Sample batch with efficient tensor operations"""
        batch = random.sample(self.replay_buffer, batch_size)

        # Convert to pre-allocated tensors to avoid memory allocation
        obs_batch = torch.stack([exp[0] for exp in batch]).cuda()
        action_batch = torch.stack([exp[1] for exp in batch]).cuda()
        reward_batch = torch.tensor([exp[2] for exp in batch], device='cuda').float()
        next_obs_batch = torch.stack([exp[3] for exp in batch]).cuda()
        done_batch = torch.tensor([exp[4] for exp in batch], device='cuda').float()

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
```

## Troubleshooting Common Issues

### Physics Instability
```python
# Issue: Robot joints becoming unstable
# Solution: Adjust physics parameters

def stabilize_physics():
    """Stabilize physics simulation"""
    # Reduce simulation time step
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0/120.0  # Use smaller time step

    # Increase solver iterations for better accuracy
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2

    # Add joint damping to reduce oscillations
    dof_props = gym.get_asset_dof_properties(sim, asset)
    for i in range(len(dof_props.damping)):
        dof_props.damping[i] = 50.0  # Increase damping
```

### GPU Memory Issues
```python
def manage_gpu_memory():
    """Manage GPU memory for large-scale training"""

    # Reduce number of environments if hitting memory limits
    max_envs = torch.cuda.get_device_properties(0).total_memory // (256 * 1024 * 1024)  # Approximate
    num_envs = min(max_envs, desired_num_envs)

    # Use mixed precision training
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    # Clear GPU cache periodically
    if iteration % 100 == 0:
        torch.cuda.empty_cache()
```

## Summary

NVIDIA Isaac Gym provides a powerful platform for training reinforcement learning agents for robotic tasks. Its GPU acceleration enables parallel simulation of thousands of environments, dramatically reducing training time for complex behaviors like humanoid locomotion. The combination of realistic physics simulation, flexible environment creation, and integration with popular RL frameworks makes Isaac Gym an ideal choice for developing AI brains for humanoid robots. Success requires careful attention to simulation design, reward engineering, and safe deployment strategies to bridge the sim-to-real gap.