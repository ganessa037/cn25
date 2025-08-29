import React, { useEffect, useRef, useState } from 'react';
import { Shield, Lock, Database } from 'lucide-react';

const Trust = () => {
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

  const trustPoints = [
    {
      icon: Shield,
      title: "End-to-end encryption on AWS ap-southeast-1",
      description: "Your data stays secure and in Malaysia",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      icon: Lock,
      title: "Multi-factor log-in for every user role",
      description: "Advanced security for all account access",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      icon: Database,
      title: "Immutable blockchain audit trail for each plate-number validation",
      description: "Transparent and tamper-proof record keeping",
      gradient: "from-green-500 to-emerald-500"
    }
  ];

  return (
    <section ref={sectionRef} id="security" className="py-32 bg-gradient-to-b from-gray-900 to-black relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 right-1/4 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 left-1/4 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1.5s' }}></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <div className={`text-center mb-20 transition-all duration-1000 ${
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
        }`}>
          <h2 className="text-4xl sm:text-5xl font-extralight text-white mb-6">
            Built for{' '}
            <span className="bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent font-normal">
              trust
            </span>
          </h2>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {trustPoints.map((point, index) => (
            <div 
              key={index} 
              className={`group relative text-center transition-all duration-700 hover:scale-105 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
              }`}
              style={{ transitionDelay: `${index * 300}ms` }}
            >
              {/* Card Background */}
              <div className="relative bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/10 group-hover:border-white/20 transition-all duration-500">
                {/* Icon Container */}
                <div className="relative mx-auto mb-6">
                  <div className={`bg-gradient-to-r ${point.gradient} w-20 h-20 rounded-2xl flex items-center justify-center group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 mx-auto`}>
                    <point.icon className="h-10 w-10 text-white" />
                  </div>
                  {/* Glow Effect */}
                  <div className={`absolute inset-0 bg-gradient-to-r ${point.gradient} opacity-0 group-hover:opacity-30 rounded-2xl blur-xl transition-opacity duration-500`}></div>
                </div>
                
                <h3 className="text-xl font-semibold text-white mb-4 leading-tight group-hover:text-blue-400 transition-colors duration-300">
                  {point.title}
                </h3>
                
                <p className="text-white/60 leading-relaxed group-hover:text-white/80 transition-colors duration-300">
                  {point.description}
                </p>

                {/* Floating Particles */}
                <div className="absolute top-4 right-4 w-1 h-1 bg-white/40 rounded-full animate-ping"></div>
                <div className="absolute bottom-6 left-6 w-1 h-1 bg-blue-400/60 rounded-full animate-pulse" style={{ animationDelay: '1s' }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Trust;