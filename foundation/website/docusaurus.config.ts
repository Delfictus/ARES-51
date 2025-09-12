import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'ARES ChronoFabric',
  tagline: 'Quantum Temporal Correlation System for Distributed Computing',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://ares-chronofabric-docs.netlify.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',

  // GitHub pages deployment config.
  organizationName: '1onlyadvance',
  projectName: 'CSF',

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
          editUrl: 'https://github.com/1onlyadvance/CSF/tree/main/website/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: {
          showReadingTime: true,
          readingTime: ({content, frontMatter, defaultReadingTime}) =>
            defaultReadingTime({content, options: {wordsPerMinute: 300}}),
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/1onlyadvance/CSF/tree/main/website/',
          blogTitle: 'ARES ChronoFabric Development Blog',
          blogDescription: 'Development updates and technical insights for the ARES ChronoFabric system',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
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
        max: 1030,
        min: 640,
        steps: 2,
        disableInDev: false,
      },
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/ares-chronofabric-social-card.jpg',
    metadata: [
      {name: 'keywords', content: 'quantum computing, temporal correlation, distributed systems, rust, high-performance computing'},
      {name: 'description', content: 'ARES ChronoFabric: Revolutionary quantum temporal correlation system for distributed computing with sub-microsecond latency and million+ messages/second throughput'},
    ],
    navbar: {
      title: 'ARES ChronoFabric',
      logo: {
        alt: 'ARES ChronoFabric Logo',
        src: 'img/ares-logo.svg',
        srcDark: 'img/ares-logo-dark.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'API Reference',
        },
        {
          type: 'docSidebar',
          sidebarId: 'adrSidebar',
          position: 'left',
          label: 'ADRs',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/1onlyadvance/CSF',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/intro',
            },
            {
              label: 'System Architecture',
              to: '/docs/architecture',
            },
            {
              label: 'API Reference',
              to: '/docs/api/overview',
            },
          ],
        },
        {
          title: 'Development',
          items: [
            {
              label: 'Contributing Guide',
              to: '/docs/contributing',
            },
            {
              label: 'Strategic Roadmap',
              to: '/docs/roadmap',
            },
            {
              label: 'Architecture Decisions',
              to: '/docs/adr/overview',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Performance Guide',
              to: '/docs/performance',
            },
            {
              label: 'Security Model',
              to: '/docs/security',
            },
            {
              label: 'GitHub Repository',
              href: 'https://github.com/1onlyadvance/CSF',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Ididia Serfaty. ARES ChronoFabric System. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml', 'bash'],
    },
    algolia: {
      // The application ID provided by Algolia
      appId: 'YOUR_APP_ID',
      // Public API key: it is safe to commit it
      apiKey: 'YOUR_SEARCH_API_KEY',
      indexName: 'ares-chronofabric',
      // Optional: see doc section below
      contextualSearch: true,
      // Optional: Specify domains where the navigation should occur through window.location instead on history.push
      externalUrlRegex: 'external\\.com|domain\\.com',
      // Optional: Replace parts of the item URLs from Algolia. Useful when using the same search index for multiple deployments using a different baseUrl
      replaceSearchResultPathname: {
        from: '/docs/', // or as RegExp: /\/docs\//
        to: '/',
      },
      // Optional: Algolia search parameters
      searchParameters: {},
      // Optional: path for search page that enabled by default (`false` to disable it)
      searchPagePath: 'search',
      // Optional: whether the insights feature is enabled or not on Docsearch (`false` by default)
      insights: false,
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    announcementBar: {
      id: 'production_ready',
      content:
        '🚀 <b>Production Ready!</b> ARES ChronoFabric system has achieved production-grade status with full compilation and testing validation.',
      backgroundColor: '#20232a',
      textColor: '#fff',
      isCloseable: false,
    },
  } satisfies Preset.ThemeConfig,

  themes: [
    '@docusaurus/theme-mermaid',
  ],

  markdown: {
    mermaid: true,
  },
};

export default config;