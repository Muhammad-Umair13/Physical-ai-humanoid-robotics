import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'Comprehensive guide to Physical AI and Humanoid Robotics with integrated RAG chatbot',
  favicon: 'img/favicon.ico',

  // Docusaurus v4 future compatibility
  future: {
    v4: true,
  },

  // ✅ Correct for Vercel deployment
  url: 'https://physical-ai-humanoid-robotics-kappa-seven.vercel.app',
  baseUrl: '/',

  // ❗ These are optional & do nothing on Vercel, but safe to keep or remove
  organizationName: 'your-organization',
  projectName: 'physical-ai-humanoid-robotics',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/your-username/physical-ai-humanoid-robotics/tree/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/your-username/physical-ai-humanoid-robotics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Module 1: ROS 2',
              to: '/docs/module1-ros2/01-introduction-to-ros2',
            },
            {
              label: 'Module 2: Digital Twin',
              to: '/docs/module2-digital-twin/01-introduction-to-digital-twin',
            },
            {
              label: 'Module 3: AI-Robot Brain',
              to: '/docs/module3-ai-robot-brain/01-introduction-ai-robot-brain',
            },
            {
              label: 'Module 4: VLA',
              to: '/docs/module4-vla/01-introduction-vla',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/ros2',
            },
            {
              label: 'Robotics Stack Exchange',
              href: 'https://robotics.stackexchange.com/',
            },
            {
              label: 'ROS Answers',
              href: 'https://answers.ros.org/questions/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/your-username/physical-ai-humanoid-robotics',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
