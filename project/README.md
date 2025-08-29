# Bolt - React TypeScript Project

A modern React application built with TypeScript, Vite, and Tailwind CSS.

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation
```bash
npm install
```

### Development
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build
```bash
npm run build
```

### Preview Build
```bash
npm run preview
```

## 📁 Project Structure

```
src/
├── components/
│   ├── layout/          # Layout components (Header, Footer)
│   ├── sections/        # Page sections (Hero, Features, etc.)
│   └── ui/             # Reusable UI components
├── pages/               # Page components
│   ├── LandingPage.tsx  # Main landing page
│   ├── SignInPage.tsx   # Sign in page with Google OAuth UI
│   └── DashboardPage.tsx # Dashboard placeholder
├── App.tsx              # Main app with routing
└── index.css            # Global styles with Tailwind
```

## 🛣️ Routes

- `/` - Landing page (main marketing page)
- `/signin` - Sign in page with Google OAuth UI
- `/dashboard` - Dashboard placeholder (future implementation)

## 🎨 Features

- **Modern UI/UX**: Beautiful, responsive design with Tailwind CSS
- **Google Sign-in**: Ready-to-implement Google OAuth UI
- **Routing**: React Router for navigation between pages
- **TypeScript**: Full type safety
- **Responsive**: Mobile-first design approach
- **Animations**: Smooth transitions and micro-interactions

## 🔧 Tech Stack

- **Frontend**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router DOM
- **Icons**: Lucide React
- **Package Manager**: npm

## 🚧 Future Development

- Implement actual Google OAuth authentication
- Build out dashboard functionality
- Add user management features
- Implement backend API integration
- Add more interactive components

## 📝 Notes

- The sign-in page currently has a placeholder Google OAuth button
- Clicking "Continue with Google" will redirect to the dashboard page
- All components are organized in a scalable folder structure
- The landing page maintains all original content and styling
