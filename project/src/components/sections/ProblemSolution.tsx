import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Receipt, FileX, ArrowRight } from 'lucide-react';

const ProblemSolution = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.2 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const problems = [
    {
      icon: AlertTriangle,
      title: "Missed Renewals",
      solution: "Smart alerts stop late penalties",
      description: "Never miss another road tax or insurance renewal deadline",
      gradient: "from-red-500 to-orange-500"
    },
    {
      icon: Receipt,
      title: "Lost Receipts",
      solution: "Snap and store petrol, repairs, tolls",
      description: "Keep all your car expenses organized and accessible",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      icon: FileX,
      title: "Form Typos",
      solution: "AI plate-OCR + auto-fill ends manual errors",
      description: "Accurate data entry every time with smart automation",
      gradient: "from-purple-500 to-pink-500"
    }
  ];

  return (
    <section ref={sectionRef} className="py-32 bg-gradient-to-b from-black to-gray-900 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <div className={`text-center mb-20 transition-all duration-1000 ${
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
        }`}>
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-extralight text-white mb-8 leading-tight">
            Why Malaysian drivers love{' '}
            <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent font-normal">
              KeretaKu
            </span>
          </h2>
          <p className="text-xl text-white/70 max-w-4xl mx-auto leading-relaxed font-light mb-6">
            You're busy. Road-tax sneaks up, receipts vanish, and insurance forms demand the same data again and again.
          </p>
          <p className="text-2xl text-white font-light">
            KeretaKu puts all that chaos in one tidy vault:
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {problems.map((problem, index) => (
            <div 
              key={index} 
              className={`group relative bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/10 hover:border-white/20 transition-all duration-700 hover:scale-105 hover:-translate-y-2 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
              }`}
              style={{ transitionDelay: `${index * 200}ms` }}
            >
              {/* Glow Effect */}
              <div className={`absolute inset-0 bg-gradient-to-r ${problem.gradient} opacity-0 group-hover:opacity-20 rounded-3xl blur-xl transition-opacity duration-500`}></div>
              
              <div className="relative z-10">
                <div className={`bg-gradient-to-r ${problem.gradient} w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <problem.icon className="h-8 w-8 text-white" />
                </div>
                
                <h3 className="text-2xl font-semibold text-white mb-4">
                  {problem.title}
                </h3>
                
                <div className="flex items-center space-x-3 mb-4 group">
                  <ArrowRight className="h-5 w-5 text-white/50 group-hover:text-blue-400 group-hover:translate-x-1 transition-all duration-300" />
                  <p className="text-blue-400 font-medium text-lg">
                    {problem.solution}
                  </p>
                </div>
                
                <p className="text-white/60 leading-relaxed">
                  {problem.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ProblemSolution;
