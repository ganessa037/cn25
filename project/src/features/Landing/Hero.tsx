import React from 'react';
import { Link } from 'react-router-dom';

function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500/20 rounded-full blur-3xl animate-float-3d" style={{ animationDuration: '10s' }}></div>
        <div className="absolute bottom-20 right-10 w-72 h-72 bg-purple-500/20 rounded-full blur-3xl animate-float-3d delay-1000" style={{ animationDuration: '12s' }}></div>
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="particle" style={{ left: '15%', animationDelay: '0s', animationDuration: '15s', transform: 'translateZ(-50px) scale(0.8)' }}></div>
        <div className="particle" style={{ left: '30%', animationDelay: '2s', animationDuration: '18s', transform: 'translateZ(20px) scale(1.1)' }}></div>
        <div className="particle" style={{ left: '45%', animationDelay: '4s', animationDuration: '16s', transform: 'translateZ(-30px) scale(0.9)' }}></div>
        <div className="particle" style={{ left: '60%', animationDelay: '6s', animationDuration: '20s', transform: 'translateZ(0px) scale(1.0)' }}></div>
        <div className="particle" style={{ left: '75%', animationDelay: '8s', animationDuration: '17s', transform: 'translateZ(-10px) scale(0.95)' }}></div>
      </div>

      <div className="container mx-auto px-4 text-center relative z-10">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full px-4 py-2 mb-8">
          <span className="text-yellow-400">⭐</span>
          <span className="text-sm text-gray-300">Trusted by 50K+ Malaysian drivers</span>
        </div>

        {/* Main Heading */}
        <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
          Drive your life, not your
          <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-gradient">
            {' '}paperwork
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-4xl mx-auto leading-relaxed">
          KeretaKu tracks every ringgit, stores every document, and renews every policy.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
          <Link
            to="/signin"
            className="group bg-gradient-to-r from-blue-500 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl flex items-center gap-2"
          >
            Let's get started
            <span className="group-hover:translate-x-1 transition-transform duration-200">→</span>
          </Link>
        </div>

        {/* Abstract 3D Panel for data representation */}
        <div className="relative w-full max-w-4xl mx-auto mt-16 perspective-1000">
          <div className="w-full h-64 bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 shadow-2xl relative transform rotate-x-15 rotate-z-5 translate-y-20 translate-z-[-50px] opacity-70">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 opacity-50 blur-xl"></div>
            <div className="relative h-full flex flex-col justify-between">
              <div className="space-y-3">
                <div className="w-3/4 h-4 bg-gray-500/30 rounded-full"></div>
                <div className="w-1/2 h-4 bg-gray-500/30 rounded-full"></div>
              </div>
              <div className="flex justify-between items-center">
                <div className="w-1/3 h-4 bg-blue-400/50 rounded-full"></div>
                <div className="w-1/4 h-4 bg-purple-400/50 rounded-full"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-white/50 rounded-full mt-2 animate-pulse"></div>
        </div>
      </div>
    </section>
  );
}

export default Hero;