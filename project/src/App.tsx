import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import SignInPage from './pages/SignInPage';
import DashboardPage from './pages/DashboardPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage key="landing" />} />
        <Route path="/signin" element={<SignInPage key="signin" />} />
        <Route path="/dashboard" element={<DashboardPage key="dashboard" />} />
      </Routes>
    </Router>
  );
}

export default App;