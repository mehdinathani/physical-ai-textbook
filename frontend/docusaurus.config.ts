import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Load environment variables
const CHATKIT_BACKEND_URL = process.env.VITE_CHATKIT_BACKEND_URL || 'http://localhost:8000/chatkit';
const CHATKIT_DOMAIN_KEY = process.env.VITE_CHATKIT_DOMAIN_KEY || 'localhost';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'From digital AI to embodied intelligence',
  favicon: 'img/favicon.ico',

  // Custom fields to make env vars available to React components
  customFields: {
    VITE_CHATKIT_BACKEND_URL: CHATKIT_BACKEND_URL,
    VITE_CHATKIT_DOMAIN_KEY: CHATKIT_DOMAIN_KEY,
  },

  // Set the production url of your site here
  url: 'https://physai-foundations.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'mehdinathani', // Usually your GitHub org/user name.
  projectName: 'physical-ai-textbook', // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
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
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/physai-foundations/physai-foundations/tree/main/',
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-ideal-image',
      {
        quality: 70,
        max: 1030, // max resized image's size.
        min: 640, // min resized image's size. if original is not bigger, no resize is done.
        steps: 2, // the max number of images generated between min and max (inclusive)
        disableInDev: false,
      },
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'PhysAI Textbook',
      logo: {
        alt: 'Physical AI & Humanoid Robotics',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'physaiTextbook',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/physai-foundations/physai-foundations',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Physical AI Concepts',
              to: '/docs/module-0/physical-ai-concepts',
            },
            {
              label: 'ROS 2 Architecture',
              to: '/docs/module-1/ros2-architecture',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/mehdinathani/physai-foundations',
            },
            {
              label: 'ROS 2 Documentation',
              href: 'https://docs.ros.org/en/humble/',
            },
            {
              label: 'NVIDIA Isaac Sim',
              href: 'https://developer.nvidia.com/isaac-sim',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'ROS Discourse',
              href: 'https://discourse.ros.org/',
            },
            {
              label: 'NVIDIA Forums',
              href: 'https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-sim/',
            },
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/ros2',
            },
          ],
        },
        {
          title: 'Connect',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/mehdinathani',
            },
            {
              label: 'LinkedIn',
              href: 'https://linkedin.com/in/mehdinathani',
            },
          ],
        },
      ],
      copyright: `Copyright © 2026 mehdinathani. Built with Docusaurus. Made with ❤️ for the Physical AI community.`,
    },
    prism: {
      theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;