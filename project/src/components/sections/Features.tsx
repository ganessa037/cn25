import React, { useEffect, useRef, useState } from 'react';
import { Car, DollarSign, Shield, Bell, Zap } from 'lucide-react';

const Features = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const features = [
    {
      icon: Car,
      title: "Vehicles Hub",
      description: "Add a car once, snap the number plate, and let OCR fill in make, model, year, chassis and engine numbers.",
      gradient: "from-blue-500 to-cyan-500",
      delay: 0
    },
    {
      icon: DollarSign,
      title: "Expense Tracker",
      description: "Point-and-shoot receipts. See real-time spend by fuel, maintenance, insurance and road-tax.",
      gradient: "from-green-500 to-emerald-500",
      delay: 200
    },
    {
      icon: Shield,
      title: "Document Vault",
      description: "Geran, insurance, driving licence—encrypted, searchable, shareable in one tap.",
      gradient: "from-purple-500 to-violet-500",
      delay: 400
    },
    {
      icon: Bell,
      title: "Smart Alerts",
      description: "Push, email, and in-app reminders before any expiry or service due date.",
      gradient: "from-orange-500 to-red-500",
      delay: 600
    },
    {
      icon: Zap,
      title: "Auto-Fill Extension",
      description: "One click populates BJAK and other insurance forms. Zero re-typing, zero typos.",
      gradient: "from-yellow-500 to-orange-500",
      delay: 800
    }
  ];

  return (
    <section ref={sectionRef} id="features" className="py-32 bg-black relative overflow-hidden">
      {/* Background Animation */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-1/3 w-72 h-72 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-1/3 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '3s' }}></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <div className={`text-center mb-20 transition-all duration-1000 ${
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
        }`}>
          <h2 className="text-4xl sm:text-5xl font-extralight text-white mb-6">
            What you get
          </h2>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className={`group relative bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/10 hover:border-white/20 transition-all duration-700 hover:scale-105 cursor-pointer ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
              }`}
              style={{ transitionDelay: `${feature.delay}ms` }}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              {/* Glow Effect */}
              <div className={`absolute inset-0 bg-gradient-to-r ${feature.gradient} opacity-0 group-hover:opacity-20 rounded-3xl blur-xl transition-all duration-500`}></div>
              
              {/* Animated Background Pattern */}
              <div className="absolute inset-0 opacity-5">
                <div className={`h-full w-full transition-transform duration-1000 ${
                  hoveredIndex === index ? 'scale-110 rotate-1' : 'scale-100'
                }`} style={{
                  backgroundImage: `radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 1px, transparent 1px)`,
                  backgroundSize: '20px 20px',
                }}></div>
              </div>
              
              <div className="relative z-10">
                <div className={`bg-gradient-to-r ${feature.gradient} w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 group-hover:rotate-6 transition-all duration-500`}>
                  <feature.icon className="h-8 w-8 text-white" />
                </div>
                
                <h3 className="text-2xl font-semibold text-white mb-4 group-hover:text-blue-400 transition-colors duration-300">
                  {feature.title}
                </h3>
                
                <p className="text-white/70 leading-relaxed group-hover:text-white/90 transition-colors duration-300">
                  {feature.description}
                </p>
              </div>

              {/* Corner Accent */}
              <div className={`absolute top-4 right-4 w-2 h-2 bg-gradient-to-r ${feature.gradient} rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500`}></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;