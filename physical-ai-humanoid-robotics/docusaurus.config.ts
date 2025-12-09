import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'Comprehensive guide to Physical AI and Humanoid Robotics with integrated RAG chatbot',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // GitHub Pages URL
  url: 'https://Muhammad-Umair13.github.io',

  // GitHub Pages base path
  baseUrl: '/Physical-ai-humanoid-robotics/',

  // Disable broken links so build never fails
  onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'ignore',

  organizationName: 'Muhammad-Umair13',
  projectName: 'Physical-ai-humanoid-robotics',

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
            'https://github.com/Muhammad-Umair13/Physical-ai-humanoid-robotics/edit/main/',
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
          href: 'https://github.com/Muhammad-Umair13/Physical-ai-humanoid-robotics',
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
            { label: 'Module 1: ROS 2', to: 'docs/module1-ros2/01-introduction-to-ros2' },
            { label: 'Module 2: Digital Twin', to: 'docs/module2-digital-twin/01-introduction-to-digital-twin' },
            { label: 'Module 3: AI-Robot Brain', to: 'docs/module3-ai-robot-brain/01-introduction-ai-robot-brain' },
            { label: 'Module 4: VLA', to: 'docs/module4-vla/01-introduction-vla' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Stack Overflow', href: 'https://stackoverflow.com/questions/tagged/ros2' },
            { label: 'Robotics Stack Exchange', href: 'https://robotics.stackexchange.com/' },
            { label: 'ROS Answers', href: 'https://answers.ros.org/questions/' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'GitHub', href: 'https://github.com/Muhammad-Umair13/Physical-ai-humanoid-robotics' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
