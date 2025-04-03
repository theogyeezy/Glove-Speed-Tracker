import React from 'react';
import { Link } from 'react-router-dom';

function Header() {
  return (
    <header className="bg-gray-800 text-white">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="text-xl font-bold">Glove Speed Tracker</Link>
        
        <nav>
          <ul className="flex space-x-6">
            <li><Link to="/" className="hover:text-blue-400 transition duration-300">Home</Link></li>
            <li><Link to="/upload" className="hover:text-blue-400 transition duration-300">Upload</Link></li>
            <li><Link to="/about" className="hover:text-blue-400 transition duration-300">About</Link></li>
          </ul>
        </nav>
      </div>
    </header>
  );
}

export default Header;
