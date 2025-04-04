import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import UploadPage from './components/UploadPage';
import ResultsPage from './components/ResultsPage';
import AboutPage from './components/AboutPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col bg-gray-50">
        <Routes>
          <Route path="/" element={
            <div>
              <Header />
              <main className="flex-grow">
                <div className="container mx-auto px-4 py-12">
                  <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold mb-4">Glove Speed Tracker</h1>
                    <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                      Advanced computer vision technology to analyze baseball catcher's glove speed and movement patterns.
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto mb-12">
                    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition duration-300">
                      <h2 className="text-xl font-bold mb-3">Upload Videos</h2>
                      <p className="text-gray-600 mb-4">
                        Upload your baseball catcher videos in various formats for analysis.
                      </p>
                      <a href="/upload" className="text-blue-600 font-semibold hover:text-blue-800">
                        Start Uploading →
                      </a>
                    </div>
                    
                    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition duration-300">
                      <h2 className="text-xl font-bold mb-3">Analyze Performance</h2>
                      <p className="text-gray-600 mb-4">
                        Get detailed metrics on glove speed, acceleration, and movement patterns.
                      </p>
                      <a href="/upload" className="text-blue-600 font-semibold hover:text-blue-800">
                        Analyze Now →
                      </a>
                    </div>
                    
                    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition duration-300">
                      <h2 className="text-xl font-bold mb-3">Improve Technique</h2>
                      <p className="text-gray-600 mb-4">
                        Use data-driven insights to enhance catching technique and reaction time.
                      </p>
                      <a href="/about" className="text-blue-600 font-semibold hover:text-blue-800">
                        Learn More →
                      </a>
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <a 
                      href="/upload" 
                      className="inline-block bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition duration-300"
                    >
                      Get Started
                    </a>
                  </div>
                </div>
              </main>
              <Footer />
            </div>
          } />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/results/:videoId" element={<ResultsPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
