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

  // ✅ Correct Vercel deployment URL
  url: 'https://physical-ai-humanoid-robotics-textbook.vercel.app',
  baseUrl: '/',  // Vercel requires "/" always

  // (Optional) For GitHub links only - safe to keep
  organizationName: 'your-username',
  projectName: 'physical-ai-humanoid-robotics-textbook',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

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
            'https://github.com/your-username/physical-ai-humanoid-robotics-textbook/tree/main/',
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
        alt: 'Physical AI Logo',
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
          href: 'https://github.com/your-username/physical-ai-humanoid-robotics-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
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
            { label: 'Stack Overflow', href: 'https://stackoverflow.com/' },
            { label: 'Robotics Stack Exchange', href: 'https://robotics.stackexchange.com/' },
            { label: 'ROS Answers', href: 'https://answers.ros.org/' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'GitHub Repo', href: 'https://github.com/your-username/physical-ai-humanoid-robotics-textbook' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
