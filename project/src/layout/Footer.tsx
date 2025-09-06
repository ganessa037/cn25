import React from 'react';
import { Car, Mail, MapPin, Phone, Github, Twitter, Linkedin } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-black border-t border-white/10 py-16 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute bottom-0 left-1/4 w-64 h-64 bg-blue-500/5 rounded-full blur-3xl"></div>
        <div className="absolute top-0 right-1/4 w-48 h-48 bg-purple-500/5 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <div className="grid md:grid-cols-4 gap-12">
          {/* Logo and Description */}
          <div className="col-span-2">
            <div className="flex items-center space-x-3 mb-6 group cursor-pointer">
              <div className="relative">
                <Car className="h-10 w-10 text-white group-hover:scale-110 group-hover:rotate-12 transition-all duration-500" />
                <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              </div>
              <span className="text-2xl font-light tracking-wide text-white group-hover:text-blue-400 transition-colors duration-300">
                KeretaKu
              </span>
            </div>
            <p className="text-white/60 mb-8 leading-relaxed font-light text-lg max-w-md">
              The smart way to manage your car in Malaysia. Track expenses, store documents, and never miss a renewal again.
            </p>
            <div className="flex items-center space-x-3 text-white/50 group hover:text-white/80 transition-colors duration-300">
              <MapPin className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
              <span className="font-medium">Made in Malaysia</span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-white mb-6 text-lg">Product</h4>
            <ul className="space-y-4">
              {['Features', 'Security', 'FAQ'].map((item) => (
                <li key={item}>
                  <a 
                    href={`#${item.toLowerCase()}`} 
                    className="text-white/60 hover:text-white transition-all duration-300 hover:translate-x-1 inline-block font-light"
                  >
                    {item}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="font-semibold text-white mb-6 text-lg">Contact</h4>
            <ul className="space-y-4">
              <li className="flex items-center space-x-3 text-white/60 hover:text-white transition-colors duration-300 group">
                <Mail className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                <span className="font-light">support@keretaku.my</span>
              </li>
              <li className="flex items-center space-x-3 text-white/60 hover:text-white transition-colors duration-300 group">
                <Phone className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                <span className="font-light">+60 3-1234 5678</span>
              </li>
            </ul>

            {/* Social Links */}
            <div className="flex items-center space-x-4 mt-8">
              {[Github, Twitter, Linkedin].map((Icon, index) => (
                <a 
                  key={index}
                  href="#" 
                  className="w-10 h-10 bg-white/5 backdrop-blur-xl rounded-full flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-all duration-300 hover:scale-110 border border-white/10 hover:border-white/20"
                >
                  <Icon className="h-5 w-5" />
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-white/10 mt-16 pt-8">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            <p className="text-white/40 text-sm font-light">
              © 2025 KeretaKu. All rights reserved.
            </p>
            <div className="flex items-center space-x-6 text-sm">
              <a href="#" className="text-white/40 hover:text-white/80 transition-colors duration-300 font-light">Privacy Policy</a>
              <a href="#" className="text-white/40 hover:text-white/80 transition-colors duration-300 font-light">Terms of Service</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;