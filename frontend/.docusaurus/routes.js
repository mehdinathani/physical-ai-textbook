import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/physical-ai-textbook/docs',
    component: ComponentCreator('/physical-ai-textbook/docs', '709'),
    routes: [
      {
        path: '/physical-ai-textbook/docs',
        component: ComponentCreator('/physical-ai-textbook/docs', 'c94'),
        routes: [
          {
            path: '/physical-ai-textbook/docs',
            component: ComponentCreator('/physical-ai-textbook/docs', '6f9'),
            routes: [
              {
                path: '/physical-ai-textbook/docs/intro',
                component: ComponentCreator('/physical-ai-textbook/docs/intro', '920'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/docs/module-0/hardware-stack-jetson-rtx',
                component: ComponentCreator('/physical-ai-textbook/docs/module-0/hardware-stack-jetson-rtx', 'dff'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-0/humanoid-landscape',
                component: ComponentCreator('/physical-ai-textbook/docs/module-0/humanoid-landscape', 'be4'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-0/lab-architecture',
                component: ComponentCreator('/physical-ai-textbook/docs/module-0/lab-architecture', '83a'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-0/physical-ai-concepts',
                component: ComponentCreator('/physical-ai-textbook/docs/module-0/physical-ai-concepts', 'e82'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/launch-systems',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/launch-systems', '132'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/python-rclpy-agents',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/python-rclpy-agents', 'd84'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/ros2-architecture',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/ros2-architecture', 'ed1'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/urdf-robot-description',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/urdf-robot-description', '948'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/gazebo-physics',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/gazebo-physics', '4ea'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/sensor-simulation',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/sensor-simulation', '06f'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/unity-visualization',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/unity-visualization', 'a57'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/world-building-sdf',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/world-building-sdf', '1e6'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/isaac-sim-intro',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/isaac-sim-intro', '556'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/nav2-path-planning',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/nav2-path-planning', 'dee'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/sim-to-real-transfer',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/sim-to-real-transfer', '572'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/visual-slam',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/visual-slam', '857'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/capstone-pipeline',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/capstone-pipeline', '029'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/humanoid-locomotion',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/humanoid-locomotion', '183'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/llm-cognitive-planning',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/llm-cognitive-planning', '2bc'),
                exact: true,
                sidebar: "physaiTextbook"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/voice-to-action-whisper',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/voice-to-action-whisper', '9f7'),
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
    path: '/physical-ai-textbook/',
    component: ComponentCreator('/physical-ai-textbook/', '4db'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
