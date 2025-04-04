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
          
          <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-bold mb-4">Project Overview</h2>
            <p className="mb-6">
              Glove Speed Tracker is a specialized computer vision application designed to analyze baseball catchers' performance by tracking and measuring glove speed. This tool helps coaches, players, and analysts gain insights into catching mechanics and reaction times.
            </p>
            
            <h2 className="text-2xl font-bold mb-4">Key Features</h2>
            <ul className="list-disc pl-5 space-y-2 mb-6">
              <li>Video processing with support for multiple formats (MP4, AVI, MOV)</li>
              <li>Advanced glove detection and tracking using computer vision</li>
              <li>Precise speed and acceleration measurements</li>
              <li>Movement pattern analysis and classification</li>
              <li>Comprehensive visual reports and data export</li>
            </ul>
            
            <h2 className="text-2xl font-bold mb-4">How It Works</h2>
            <ol className="list-decimal pl-5 space-y-2 mb-6">
              <li>Upload a video of a baseball catcher in action</li>
              <li>Our system processes the video using computer vision algorithms</li>
              <li>The catcher's glove is detected and tracked throughout the video</li>
              <li>Speed, acceleration, and movement patterns are calculated</li>
              <li>Results are presented in an easy-to-understand visual format</li>
            </ol>
            
            <h2 className="text-2xl font-bold mb-4">Technology Stack</h2>
            <p className="mb-6">
              Glove Speed Tracker is built using modern technologies including:
            </p>
            <ul className="list-disc pl-5 space-y-2 mb-6">
              <li>React for the frontend user interface</li>
              <li>Python with OpenCV for computer vision processing</li>
              <li>TensorFlow/PyTorch for machine learning models</li>
              <li>Supabase for backend storage and processing</li>
              <li>Netlify for application hosting</li>
            </ul>
            
            <div className="mt-8 text-center">
              <Link 
                to="/" 
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-300"
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
