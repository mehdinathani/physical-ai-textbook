import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs',
    component: ComponentCreator('/docs', '958'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '313'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'd84'),
            routes: [
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', 'e84'),
                exact: true
              },
              {
                path: '/docs/module-0/hardware-stack-jetson-rtx',
                component: ComponentCreator('/docs/module-0/hardware-stack-jetson-rtx', '0ac'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-0/humanoid-landscape',
                component: ComponentCreator('/docs/module-0/humanoid-landscape', '4c7'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-0/lab-architecture',
                component: ComponentCreator('/docs/module-0/lab-architecture', '555'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-0/physical-ai-concepts',
                component: ComponentCreator('/docs/module-0/physical-ai-concepts', '6c6'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-1/launch-systems',
                component: ComponentCreator('/docs/module-1/launch-systems', '5f8'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-1/python-rclpy-agents',
                component: ComponentCreator('/docs/module-1/python-rclpy-agents', 'b9f'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-1/ros2-architecture',
                component: ComponentCreator('/docs/module-1/ros2-architecture', '0f3'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-1/urdf-robot-description',
                component: ComponentCreator('/docs/module-1/urdf-robot-description', '004'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-2/gazebo-physics',
                component: ComponentCreator('/docs/module-2/gazebo-physics', '553'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-2/sensor-simulation',
                component: ComponentCreator('/docs/module-2/sensor-simulation', '331'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-2/unity-visualization',
                component: ComponentCreator('/docs/module-2/unity-visualization', '2b5'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-2/world-building-sdf',
                component: ComponentCreator('/docs/module-2/world-building-sdf', 'd20'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-3/isaac-sim-intro',
                component: ComponentCreator('/docs/module-3/isaac-sim-intro', 'b1a'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-3/nav2-path-planning',
                component: ComponentCreator('/docs/module-3/nav2-path-planning', '458'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-3/sim-to-real-transfer',
                component: ComponentCreator('/docs/module-3/sim-to-real-transfer', '10d'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-3/visual-slam',
                component: ComponentCreator('/docs/module-3/visual-slam', '590'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-4/capstone-pipeline',
                component: ComponentCreator('/docs/module-4/capstone-pipeline', '52b'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-4/humanoid-locomotion',
                component: ComponentCreator('/docs/module-4/humanoid-locomotion', 'e7c'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-4/llm-cognitive-planning',
                component: ComponentCreator('/docs/module-4/llm-cognitive-planning', 'cd4'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/docs/module-4/voice-to-action-whisper',
                component: ComponentCreator('/docs/module-4/voice-to-action-whisper', '50b'),
                exact: true,
                sidebar: "physaiTextbook"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', 'c92'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
