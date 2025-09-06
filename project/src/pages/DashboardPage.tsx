import React from "react";
import { Link } from "react-router-dom";
import { BarChart3, Settings, User, LogOut, Car } from "lucide-react";
import { useScrollToTop } from "../hooks/useScrollToTop";

export default function DashboardPage() {
  useScrollToTop();
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
      <main className="container mx-auto px-4 py-8">
        <Link
          to="/"
          className="inline-block bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transition-all duration-200 px-4 py-2 rounded-lg"
        >
          Back to Landing Page
        </Link>
      </main>
    </div>
  );
}