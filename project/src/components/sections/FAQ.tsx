import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, HelpCircle } from 'lucide-react';

const FAQ = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);
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

  const faqs = [
    {
      question: "Does it work with any Malaysian insurer?",
      answer: "Yes—our auto-fill supports all major providers, and you can still export details for others."
    },
    {
      question: "Is my data secure?",
      answer: "Everything is AES-256 encrypted at rest, SSL in transit, and stored only in Malaysia."
    },
    {
      question: "What if I sell my car?",
      answer: "Archive the vehicle in one tap; records remain for your reference."
    }
  ];

  return (
    <section ref={sectionRef} id="faq" className="py-32 bg-gradient-to-b from-black to-gray-900 relative overflow-hidden">
      {/* Background Animation */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-6 lg:px-8">
        <div className={`text-center mb-16 transition-all duration-1000 ${
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'
        }`}>
          <div className="flex items-center justify-center space-x-3 mb-6">
            <HelpCircle className="h-8 w-8 text-blue-400" />
            <h2 className="text-4xl sm:text-5xl font-extralight text-white">
              Frequently Asked
            </h2>
          </div>
        </div>

        <div className="space-y-6">
          {faqs.map((faq, index) => (
            <div 
              key={index} 
              className={`group relative bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-xl rounded-2xl overflow-hidden border border-white/10 hover:border-white/20 transition-all duration-500 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
              }`}
              style={{ transitionDelay: `${index * 200}ms` }}
            >
              {/* Glow Effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              
              <button
                className="relative z-10 w-full px-8 py-6 text-left flex items-center justify-between group-hover:bg-white/5 transition-all duration-300"
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
              >
                <h3 className="text-xl font-medium text-white pr-6 group-hover:text-blue-400 transition-colors duration-300">
                  {faq.question}
                </h3>
                <ChevronDown className={`h-6 w-6 text-white/60 group-hover:text-blue-400 transition-all duration-500 ${
                  openIndex === index ? 'rotate-180 scale-110' : ''
                }`} />
              </button>
              
              <div className={`overflow-hidden transition-all duration-500 ease-out ${
                openIndex === index ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
              }`}>
                <div className="px-8 pb-6">
                  <div className="h-px bg-gradient-to-r from-transparent via-white/20 to-transparent mb-4"></div>
                  <p className="text-white/70 leading-relaxed text-lg">
                    {faq.answer}
                  </p>
                </div>
              </div>

              {/* Corner Accent */}
              <div className="absolute top-4 right-4 w-1 h-1 bg-blue-400/60 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQ;