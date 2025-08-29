import React, { useEffect, useRef, useState } from 'react';
import { Star, Quote } from 'lucide-react';

const Testimonial = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.3 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} className="py-32 bg-gradient-to-br from-black via-gray-900 to-black relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-1/3 left-1/2 w-96 h-96 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full blur-3xl animate-pulse transform -translate-x-1/2"></div>
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6 lg:px-8">
        <div className={`text-center mb-16 transition-all duration-1000 ${
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
        }`}>
          <h2 className="text-4xl sm:text-5xl font-extralight text-white mb-6">
            A day with{' '}
            <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent font-normal">
              KeretaKu
            </span>
          </h2>
        </div>

        <div className={`relative transition-all duration-1200 delay-300 ${
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-16'
        }`}>
          {/* Main Testimonial Card */}
          <div className="relative bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-2xl rounded-3xl p-12 border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-700">
            {/* Quote Icon */}
            <div className="absolute top-8 left-8">
              <Quote className="h-12 w-12 text-blue-400/30 group-hover:text-blue-400/50 transition-colors duration-500" />
            </div>
            
            {/* Stars */}
            <div className="flex justify-center mb-8">
              {[...Array(5)].map((_, i) => (
                <Star 
                  key={i} 
                  className="h-6 w-6 text-yellow-400 fill-current group-hover:scale-110 transition-transform duration-300" 
                  style={{ transitionDelay: `${i * 100}ms` }}
                />
              ))}
            </div>
            
            {/* Testimonial Text */}
            <blockquote className="text-2xl sm:text-3xl text-white mb-12 font-light leading-relaxed text-center group-hover:text-blue-100 transition-colors duration-500">
              "Two weeks before my X70's insurance expired, KeretaKu pinged me, filled the renewal form in seconds, and saved me{' '}
              <span className="bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent font-medium">
                RM 450
              </span>
              {' '}in late fees."
            </blockquote>
            
            {/* Author */}
            <div className="flex items-center justify-center space-x-6">
              <div className="relative">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold text-xl group-hover:scale-110 transition-transform duration-300">
                  A
                </div>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-lg opacity-0 group-hover:opacity-50 transition-opacity duration-500"></div>
              </div>
              <div className="text-left">
                <p className="font-semibold text-white text-xl group-hover:text-blue-400 transition-colors duration-300">Amir</p>
                <p className="text-white/60 group-hover:text-white/80 transition-colors duration-300">Shah Alam</p>
              </div>
            </div>

            {/* Floating Elements */}
            <div className="absolute top-1/4 right-8 w-2 h-2 bg-blue-400/60 rounded-full animate-pulse"></div>
            <div className="absolute bottom-1/4 left-12 w-1 h-1 bg-purple-400/60 rounded-full animate-ping"></div>
            
            {/* Glow Effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonial;