/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'system-ui', 'sans-serif'],
        display: ['Sora', 'Plus Jakarta Sans', 'sans-serif'],
      },
      fontWeight: {
        '600': '600',
        '700': '700',
        '800': '800',
      },
      colors: {
        bg: {
          base:     '#060910',
          surface:  '#0b1120',
          elevated: '#111828',
          hover:    '#16203a',
        },
        border: {
          subtle:  '#1a2540',
          default: '#243060',
          bright:  '#2f4080',
        },
        accent: {
          DEFAULT: '#22c55e',
          bright:  '#4ade80',
          dim:     '#166534',
          muted:   '#0f3a22',
        },
        tx: {
          primary:   '#e2e8f5',
          secondary: '#8896b0',
          muted:     '#4a5878',
        },
        danger: {
          DEFAULT: '#f43f5e',
          dim:     '#4c0519',
          muted:   '#2d0313',
        },
        warn: {
          DEFAULT: '#f59e0b',
          dim:     '#451a03',
        },
      },
      boxShadow: {
        glow:    '0 0 20px rgba(34, 197, 94, 0.15)',
        'glow-lg':'0 0 40px rgba(34, 197, 94, 0.20)',
        danger:  '0 0 30px rgba(244, 63, 94, 0.15)',
        card:    '0 4px 24px rgba(0,0,0,0.4)',
        float:   '0 8px 32px rgba(0,0,0,0.6)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'accent-gradient': 'linear-gradient(135deg, #22c55e, #10b981)',
        'dark-gradient':   'linear-gradient(135deg, #111828, #0b1120)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
        'slide-up':   'slideUp 0.3s ease-out',
        'fade-in':    'fadeIn 0.25s ease-out',
      },
      keyframes: {
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 12px rgba(34, 197, 94, 0.2)' },
          '50%':      { boxShadow: '0 0 24px rgba(34, 197, 94, 0.5)' },
        },
        slideUp: {
          from: { opacity: 0, transform: 'translateY(10px)' },
          to:   { opacity: 1, transform: 'translateY(0)' },
        },
        fadeIn: {
          from: { opacity: 0 },
          to:   { opacity: 1 },
        },
      },
    },
  },
  plugins: [],
}
