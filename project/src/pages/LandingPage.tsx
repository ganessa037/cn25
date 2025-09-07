import React from 'react';
import Header from '../layout/Header';
import { Hero, ProblemSolution, Features, Trust, Testimonial, FAQ, FinalCTA } from '../features/Landing';
import Footer from '../layout/Footer';
import { useScrollToTop } from '../hooks/useScrollToTop';
try { localStorage.removeItem("user"); } catch {}

function LandingPage() {
  useScrollToTop();
  
  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      <Header />
      <main>
        <Hero />
        <ProblemSolution />
        <Features />
        <Trust />
        <Testimonial />
        <FAQ />
        <FinalCTA />
      </main>
      <Footer />
    </div>
  );
}

export default LandingPage;
