import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Chrome, Car } from 'lucide-react';
import { useScrollToTop } from '../hooks/useScrollToTop';

function SignInPage() {
  useScrollToTop();
  const navigate = useNavigate();
  
  const handleGoogleSignIn = () => {
    // TODO: Implement Google OAuth logic
    console.log('Google sign in clicked');
    // For now, just redirect to dashboard using React Router
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black flex items-center justify-center p-4">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 w-full max-w-md">
        {/* Logo/Brand */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl mb-4">
            <Car className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">Welcome Back</h1>
          <p className="text-gray-400">Sign in to continue to your dashboard</p>
        </div>

        {/* Sign in form */}
        <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 shadow-2xl">
          {/* Google Sign In Button */}
          <button
            onClick={handleGoogleSignIn}
            className="w-full flex items-center justify-center gap-3 bg-white text-gray-900 hover:bg-gray-100 transition-all duration-200 rounded-xl py-6 px-8 font-medium shadow-lg hover:shadow-xl transform hover:scale-[1.02] active:scale-[0.98] text-lg"
          >
            <Chrome className="w-6 h-6" />
            Continue with Google
          </button>

          {/* Simple message */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-400">
              Secure sign-in with your Google account
            </p>
          </div>
        </div>

        {/* Back to landing page */}
        <div className="text-center mt-6">
          <Link
            to="/"
            className="text-gray-400 hover:text-white transition-colors duration-200 text-sm"
          >
            ← Back to landing page
          </Link>
        </div>
      </div>
    </div>
  );
}

export default SignInPage;
