import React from 'react';
import { Link } from 'react-router-dom';

function Header() {
  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-4">
        <div className="flex justify-between items-center">
          <Link to="/" className="text-2xl font-bold text-blue-600">
            Glove Speed Tracker
          </Link>
          
          <nav>
            <ul className="flex space-x-6">
              <li>
                <Link to="/" className="text-gray-700 hover:text-blue-600 transition duration-300">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/upload" className="text-gray-700 hover:text-blue-600 transition duration-300">
                  Upload
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-gray-700 hover:text-blue-600 transition duration-300">
                  About
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
}

export default Header;
