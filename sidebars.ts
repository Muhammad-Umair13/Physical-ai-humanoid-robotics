import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

/**
 * Sidebar configuration for Physical AI & Humanoid Robotics textbook
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',

    // Module 1: ROS 2
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      items: [
        'module1-ros2/introduction-to-ros2',
        'module1-ros2/ros2-architecture',
        'module1-ros2/nodes-topics-services-actions',
        'module1-ros2/launch-files-parameters',
        {
          type: 'category',
          label: 'Labs',
          items: [
            'module1-ros2/labs/lab1-ros2-node-sensor-integration',
          ],
        },
      ],
    },

    // Module 2: Digital Twin
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      items: [
        'module2-digital-twin/introduction-to-digital-twin',
        'module2-digital-twin/gazebo-simulation-fundamentals',
        'module2-digital-twin/connecting-ros2-with-unity',
        'module2-digital-twin/creating-a-gazebo-world',
        'module2-digital-twin/nvidia-isaac-sim-basics',
        'module2-digital-twin/unity-robotics-hub',
        {
          type: 'category',
          label: 'Labs',
          items: [
            'module2-digital-twin/labs/lab1-gazebo-world',
            'module2-digital-twin/labs/lab2-unity-ros2-bridge',
            'module2-digital-twin/labs/lab3-isaac-sim-simple-robot',
          ],
        },
      ],
    },

    // Module 3: AI-Robot Brain
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain',
      items: [
        'module3-ai-robot-brain/introduction-to-robot-ai-brains',
        'module3-ai-robot-brain/behavior-trees',
        'module3-ai-robot-brain/nvidia-isaac-gym-rl-examples',
        'module3-ai-robot-brain/reinforcement-learning-for-robotics',
        // Labs removed because they do not exist yet
      ],
    },

    // Module 4: VLA (Vision-Language-Action)
    {
      type: 'category',
      label: 'Module 4: VLA (Vision-Language-Action)',
      items: [
        'module4-vla/introduction-to-vla',
        'module4-vla/language-models-for-robotics',
        'module4-vla/vision-models-for-robotics',
        'module4-vla/action-policy-generation',
        {
          type: 'category',
          label: 'Labs',
          items: [
            'module4-vla/labs/lab1-object-detection',
            'module4-vla/labs/lab2-vision-language-pick-and-place',
          ],
        },
      ],
    },
  ],
};

export default sidebars;
