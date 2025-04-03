import React from 'react';
import { Link } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';

function App() {
  return (
    <div className="App">
      <Header />
      
      <main>
        <section className="hero bg-blue-600 text-white py-16">
          <div className="container mx-auto px-4 text-center">
            <h1 className="text-4xl font-bold mb-4">Baseball Catcher's Glove Speed Tracker</h1>
            <p className="text-xl mb-8">
              Advanced computer vision technology to analyze and improve catcher performance
            </p>
            <Link 
              to="/upload" 
              className="bg-white text-blue-600 font-bold py-3 px-6 rounded-lg hover:bg-gray-100 transition duration-300"
            >
              Start Tracking Now
            </Link>
          </div>
        </section>
        
        <section className="features py-16 bg-gray-50">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="feature-card bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition duration-300">
                <h3 className="text-xl font-bold mb-3 text-blue-600">Glove Detection & Tracking</h3>
                <p className="text-gray-700">
                  Advanced computer vision algorithms to accurately detect and track the catcher's glove throughout video footage.
                </p>
              </div>
              
              <div className="feature-card bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition duration-300">
                <h3 className="text-xl font-bold mb-3 text-blue-600">Speed Calculation</h3>
                <p className="text-gray-700">
                  Precise measurement of glove speed in real-world units (mph and m/s) with frame-by-frame analysis.
                </p>
              </div>
              
              <div className="feature-card bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition duration-300">
                <h3 className="text-xl font-bold mb-3 text-blue-600">Performance Analytics</h3>
                <p className="text-gray-700">
                  Comprehensive data analysis with visualizations to identify patterns and improvement opportunities.
                </p>
              </div>
            </div>
          </div>
        </section>
        
        <section className="how-it-works py-16">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-4">1</div>
                <h3 className="text-lg font-bold mb-2">Upload Video</h3>
                <p className="text-gray-700">Upload your baseball catcher footage in various formats</p>
              </div>
              
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-4">2</div>
                <h3 className="text-lg font-bold mb-2">Automated Analysis</h3>
                <p className="text-gray-700">Our system detects and tracks the glove movement</p>
              </div>
              
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-4">3</div>
                <h3 className="text-lg font-bold mb-2">Speed Calculation</h3>
                <p className="text-gray-700">Precise measurements of glove speed and acceleration</p>
              </div>
              
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-4">4</div>
                <h3 className="text-lg font-bold mb-2">View Results</h3>
                <p className="text-gray-700">Get detailed reports and visualizations of performance</p>
              </div>
            </div>
            
            <div className="text-center mt-12">
              <Link 
                to="/upload" 
                className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition duration-300"
              >
                Try It Now
              </Link>
            </div>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
}

export default App;
