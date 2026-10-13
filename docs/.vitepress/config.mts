import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Deep Learning',
  description: 'Deep Learning course notes',
  base: '/EEP596/',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    math: true,
  },
  themeConfig: {
    nav: [
      { text: 'Overview', link: '/overview' },
      { text: 'Lectures', link: '/lectures/lecture-01' },
      { text: 'Labs', link: '/labs/lab-01' },
    ],
    sidebar: [
      { text: 'Overview', link: '/overview' },
      {
        text: 'Lectures',
        items: [
          { text: '1. Introduction & Regression', link: '/lectures/lecture-01' },
          { text: '2. Logistic Regression, Neuron, MLP', link: '/lectures/lecture-02' },
        ],
      },
      {
        text: 'Labs',
        items: [
          { text: '1. Python, NumPy & Ridge Regression', link: '/labs/lab-01' },
          { text: '2. PyTorch Introduction', link: '/labs/lab-02' },
        ],
      },
    ],
    outline: [2, 3],
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/BV003/EEP596' }],
  },
})
