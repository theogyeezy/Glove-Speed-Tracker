import React from 'react';
import Header from './components/Header';
import Footer from './components/Footer';

function AboutPage() {
  return (
    <div>
      <Header />
      
      <main className="py-12">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8 text-center">About Glove Speed Tracker</h1>
          
          <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8 mb-8">
            <h2 className="text-2xl font-semibold mb-4">Our Mission</h2>
            <p className="text-gray-700 mb-6">
              Glove Speed Tracker was developed to provide baseball coaches, players, and analysts with advanced 
              tools to measure and analyze catcher performance. By leveraging computer vision and machine learning 
              technologies, we aim to bring professional-level analytics to players at all levels.
            </p>
            
            <h2 className="text-2xl font-semibold mb-4">The Technology</h2>
            <p className="text-gray-700 mb-6">
              Our application uses state-of-the-art computer vision algorithms to detect and track the catcher's 
              glove throughout video footage. By analyzing the movement patterns, we can calculate precise speed 
              measurements, identify performance trends, and provide actionable insights for improvement.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-blue-50 p-6 rounded-lg border border-blue-100">
                <h3 className="text-lg font-semibold mb-2 text-blue-800">Video Processing</h3>
                <p className="text-gray-700">
                  Our system processes various video formats and extracts individual frames for detailed analysis.
                </p>
              </div>
              
              <div className="bg-green-50 p-6 rounded-lg border border-green-100">
                <h3 className="text-lg font-semibold mb-2 text-green-800">Object Detection</h3>
                <p className="text-gray-700">
                  Advanced deep learning models identify and track the catcher's glove with high precision.
                </p>
              </div>
              
              <div className="bg-purple-50 p-6 rounded-lg border border-purple-100">
                <h3 className="text-lg font-semibold mb-2 text-purple-800">Data Analysis</h3>
                <p className="text-gray-700">
                  Sophisticated algorithms analyze movement patterns and calculate speed metrics.
                </p>
              </div>
            </div>
            
            <h2 className="text-2xl font-semibold mb-4">How It Helps</h2>
            <ul className="list-disc pl-6 mb-6 space-y-2 text-gray-700">
              <li>Coaches can identify areas for improvement in catcher mechanics</li>
              <li>Players can track their progress over time with objective measurements</li>
              <li>Teams can compare performance metrics across different catchers</li>
              <li>Analysts can gather detailed data for scouting and player development</li>
            </ul>
            
            <h2 className="text-2xl font-semibold mb-4">Our Team</h2>
            <p className="text-gray-700 mb-6">
              Glove Speed Tracker was developed by a team of baseball enthusiasts, computer vision experts, 
              and data scientists passionate about bringing advanced analytics to the game we love.
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-semibold mb-6 text-center">Frequently Asked Questions</h2>
            
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">What video formats are supported?</h3>
                <p className="text-gray-700">
                  We support most common video formats including MP4, AVI, MOV, and MKV.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">How accurate are the speed measurements?</h3>
                <p className="text-gray-700">
                  Our system has been calibrated to provide measurements within 2-3% of radar gun readings 
                  under optimal conditions. Accuracy depends on video quality and camera positioning.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">Can I use this for other positions besides catcher?</h3>
                <p className="text-gray-700">
                  While our system is optimized for tracking catcher's glove movements, it can also be used 
                  for other positions with varying degrees of accuracy.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">Is there a mobile app available?</h3>
                <p className="text-gray-700">
                  Currently, we offer a web-based application accessible from any device. A dedicated 
                  mobile app is in development and will be released soon.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">How can I provide feedback or report issues?</h3>
                <p className="text-gray-700">
                  We welcome your feedback! Please contact us at support@glovespeedtracker.com with any 
                  questions, suggestions, or issues you encounter.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default AboutPage;
