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
  // Main documentation sidebar
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'installation',
        'quick-start',
        'configuration',
      ],
    },
    {
      type: 'category',
      label: 'System Architecture',
      items: [
        'architecture/overview',
        'architecture/comprehensive-system-architecture',
        'architecture/core-components',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/temporal-coherence',
      ],
    },
    {
      type: 'category',
      label: 'Development',
      items: [
        'development/contributing',
        'development/coding-standards',
        'development/testing',
        'development/ci-cd',
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'operations/deployment',
        'operations/observability',
        'operations/performance',
      ],
    },
    {
      type: 'category',
      label: 'Security',
      items: [
        'security/overview',
        'security/threat-model',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/integration-examples',
        'guides/performance-troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/strategic-roadmap',
        'reference/success-metrics',
      ],
    },
  ],

  // API Reference sidebar
  apiSidebar: [
    'api/overview',
    {
      type: 'category',
      label: 'Available APIs',
      items: [
        'api/csf-bus',
        'api/csf-consensus',
        'api/csf-network',
        'api/csf-sil',
        'api/csf-time',
      ],
    },
  ],

  // Architecture Decision Records sidebar
  adrSidebar: [
    'adr/overview',
    'adr/template',
    {
      type: 'category',
      label: 'Decisions',
      items: [
        'adr/one-bus-one-scheduler-one-config',
      ],
    },
  ],
};

export default sidebars;
