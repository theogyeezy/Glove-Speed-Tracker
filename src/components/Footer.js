import React from 'react';

function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-8">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-lg font-bold mb-4">Glove Speed Tracker</h3>
            <p className="text-gray-400">
              Advanced computer vision technology to analyze and improve baseball catcher performance.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-bold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><a href="/" className="text-gray-400 hover:text-white transition duration-300">Home</a></li>
              <li><a href="/upload" className="text-gray-400 hover:text-white transition duration-300">Upload Video</a></li>
              <li><a href="/about" className="text-gray-400 hover:text-white transition duration-300">About</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-lg font-bold mb-4">Contact</h3>
            <p className="text-gray-400">
              Have questions or feedback? <br />
              <a href="mailto:contact@glovespeedtracker.com" className="text-blue-400 hover:text-blue-300">
                contact@glovespeedtracker.com
              </a>
            </p>
          </div>
        </div>
        
        <div className="border-t border-gray-700 mt-8 pt-6 text-center text-gray-400">
          <p>&copy; {new Date().getFullYear()} Glove Speed Tracker. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
