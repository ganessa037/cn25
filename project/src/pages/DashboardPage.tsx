import React from 'react';
import { Link } from 'react-router-dom';
import { BarChart3, Settings, User, LogOut, Car } from 'lucide-react';
import { useScrollToTop } from '../hooks/useScrollToTop';

function DashboardPage() {
  useScrollToTop();
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
      {/* Header */}
      <header className="bg-white/5 backdrop-blur-xl border-b border-white/10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Car className="w-5 h-5 text-white" />
              </div>
              <span className="text-white font-bold text-xl">KeretaKu Dashboard</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="text-gray-300 hover:text-white transition-colors duration-200">
                <Settings className="w-5 h-5" />
              </button>
              <button className="text-gray-300 hover:text-white transition-colors duration-200">
                <User className="w-5 h-5" />
              </button>
              <Link
                to="/"
                className="text-gray-300 hover:text-white transition-colors duration-200"
              >
                <LogOut className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-4">Welcome to Your KeretaKu Dashboard</h1>
          <p className="text-gray-400 text-lg mb-8">
            Manage your cars, track expenses, and stay on top of renewals. This is where your car management journey begins.
          </p>
          
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 max-w-2xl mx-auto">
            <BarChart3 className="w-16 h-16 text-blue-400 mx-auto mb-4" />
            <h2 className="text-2xl font-semibold text-white mb-4">Car Management Features Coming Soon</h2>
            <ul className="text-left text-gray-300 space-y-2 mb-6">
              <li>• Vehicle management & tracking</li>
              <li>• Expense tracking & receipts</li>
              <li>• Document vault & storage</li>
              <li>• Smart renewal alerts</li>
            </ul>
            
            <Link
              to="/"
              className="inline-block bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:from-blue-600 hover:to-purple-700 transition-all duration-200"
            >
              Back to Landing Page
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}

export default DashboardPage;
