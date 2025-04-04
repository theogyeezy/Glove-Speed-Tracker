import React from 'react';
import { Link } from 'react-router-dom';
import Header from './Header';
import Footer from './Footer';

function AboutPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow py-12">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8 text-center">About Glove Speed Tracker</h1>
          
          <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8 mb-8">
            <h2 className="text-2xl font-bold mb-4">Our Mission</h2>
            <p className="mb-6">
              Glove Speed Tracker is dedicated to helping baseball catchers improve their performance through 
              advanced computer vision technology. Our application analyzes video footage to calculate glove 
              speed, track movement patterns, and provide actionable insights for players and coaches.
            </p>
            
            <h2 className="text-2xl font-bold mb-4">How It Works</h2>
            <div className="mb-6">
              <p className="mb-4">Our technology uses a multi-step process to analyze catcher performance:</p>
              <ol className="list-decimal pl-6 space-y-2">
                <li>Upload your video footage of baseball catchers in action</li>
                <li>Our computer vision algorithms detect and track the catcher's glove throughout the video</li>
                <li>Advanced calculations determine glove speed, acceleration, and movement patterns</li>
                <li>Results are presented in an easy-to-understand dashboard with visualizations</li>
                <li>Data-driven insights help identify strengths and areas for improvement</li>
              </ol>
            </div>
            
            <h2 className="text-2xl font-bold mb-4">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-2">Speed Analysis</h3>
                <p>Accurate measurements of maximum and average glove speed in both mph and m/s.</p>
              </div>
              
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-2">Movement Tracking</h3>
                <p>Detailed tracking of glove movement patterns and positioning throughout the video.</p>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-2">Performance Insights</h3>
                <p>Data-driven recommendations to improve reaction time and catching technique.</p>
              </div>
              
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-2">Visual Reports</h3>
                <p>Comprehensive visual reports that can be shared with coaches and teammates.</p>
              </div>
            </div>
            
            <div className="text-center">
              <Link 
                to="/upload" 
                className="inline-block bg-blue-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-blue-700 transition duration-300"
              >
                Try It Now
              </Link>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default AboutPage;
