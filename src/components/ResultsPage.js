import React from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';

function ResultsPage() {
  // Mock data for demonstration
  const analysisResults = {
    videoName: "catcher_practice_session.mp4",
    duration: "00:02:15",
    frameCount: 4050,
    maxSpeed: 42.8,
    avgSpeed: 28.3,
    topAcceleration: 15.2,
    movementPatterns: [
      { type: "Quick Snap", count: 12, avgSpeed: 38.5 },
      { type: "Lateral Movement", count: 8, avgSpeed: 24.7 },
      { type: "Vertical Reach", count: 5, avgSpeed: 31.2 }
    ]
  };

  return (
    <div>
      <Header />
      
      <main className="py-12">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8 text-center">Analysis Results</h1>
          
          <div className="bg-white rounded-lg shadow-md p-8 mb-8">
            <h2 className="text-2xl font-semibold mb-4">Video Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-gray-500 text-sm">Filename</p>
                <p className="font-medium">{analysisResults.videoName}</p>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-gray-500 text-sm">Duration</p>
                <p className="font-medium">{analysisResults.duration}</p>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-gray-500 text-sm">Total Frames</p>
                <p className="font-medium">{analysisResults.frameCount}</p>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-6">
              <div className="flex-1">
                <h3 className="text-xl font-semibold mb-4">Processed Video</h3>
                <div className="bg-gray-200 rounded-lg aspect-video flex items-center justify-center">
                  <p className="text-gray-600">Video player would be displayed here</p>
                </div>
              </div>
              
              <div className="flex-1">
                <h3 className="text-xl font-semibold mb-4">Key Metrics</h3>
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                    <p className="text-gray-700">Maximum Glove Speed</p>
                    <p className="text-3xl font-bold text-blue-600">{analysisResults.maxSpeed} mph</p>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg border border-green-100">
                    <p className="text-gray-700">Average Glove Speed</p>
                    <p className="text-3xl font-bold text-green-600">{analysisResults.avgSpeed} mph</p>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg border border-purple-100">
                    <p className="text-gray-700">Top Acceleration</p>
                    <p className="text-3xl font-bold text-purple-600">{analysisResults.topAcceleration} m/s²</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Speed Over Time</h2>
              <div className="bg-gray-100 rounded-lg aspect-video flex items-center justify-center">
                <p className="text-gray-600">Speed chart would be displayed here</p>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Glove Trajectory</h2>
              <div className="bg-gray-100 rounded-lg aspect-video flex items-center justify-center">
                <p className="text-gray-600">Trajectory visualization would be displayed here</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-semibold mb-4">Movement Pattern Analysis</h2>
            
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white">
                <thead>
                  <tr className="bg-gray-100 text-gray-700">
                    <th className="py-3 px-4 text-left">Movement Type</th>
                    <th className="py-3 px-4 text-left">Count</th>
                    <th className="py-3 px-4 text-left">Average Speed (mph)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {analysisResults.movementPatterns.map((pattern, index) => (
                    <tr key={index}>
                      <td className="py-3 px-4">{pattern.type}</td>
                      <td className="py-3 px-4">{pattern.count}</td>
                      <td className="py-3 px-4">{pattern.avgSpeed}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="mt-8 flex justify-center space-x-4">
              <button className="bg-blue-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-blue-700 transition duration-300">
                Download Report
              </button>
              <button className="bg-gray-200 text-gray-800 font-bold py-2 px-6 rounded-lg hover:bg-gray-300 transition duration-300">
                Share Results
              </button>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default ResultsPage;
