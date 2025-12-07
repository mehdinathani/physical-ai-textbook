import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Physical AI & Humanoid Robotics Textbook
  physaiTextbook: [
    {
      type: 'category',
      label: 'Module 0: Foundations',
      items: [
        'module-0/physical-ai-concepts',
        'module-0/hardware-stack-jetson-rtx',
        'module-0/lab-architecture',
        'module-0/humanoid-landscape',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 (Nervous System)',
      items: [
        'module-1/ros2-architecture',
        'module-1/python-rclpy-agents',
        'module-1/urdf-robot-description',
        'module-1/launch-systems',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Simulation)',
      items: [
        'module-2/gazebo-physics',
        'module-2/world-building-sdf',
        'module-2/unity-visualization',
        'module-2/sensor-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Isaac AI (The Brain)',
      items: [
        'module-3/isaac-sim-intro',
        'module-3/visual-slam',
        'module-3/nav2-path-planning',
        'module-3/sim-to-real-transfer',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA (Vision-Language-Action)',
      items: [
        'module-4/voice-to-action-whisper',
        'module-4/llm-cognitive-planning',
        'module-4/humanoid-locomotion',
        'module-4/capstone-pipeline',
      ],
    },
  ],
};

export default sidebars;