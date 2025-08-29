import React from 'react';
import Header from '../components/layout/Header';
import Hero from '../components/sections/Hero';
import ProblemSolution from '../components/sections/ProblemSolution';
import Features from '../components/sections/Features';
import Trust from '../components/sections/Trust';
import Testimonial from '../components/sections/Testimonial';
import FAQ from '../components/sections/FAQ';
import FinalCTA from '../components/sections/FinalCTA';
import Footer from '../components/layout/Footer';
import { useScrollToTop } from '../hooks/useScrollToTop';

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
