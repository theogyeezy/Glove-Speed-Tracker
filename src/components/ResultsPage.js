import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Header from './Header';
import Footer from './Footer';
import { supabase } from '../supabaseClient';

function ResultsPage() {
  const { videoId } = useParams();
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchResults() {
      try {
        setLoading(true);
        
        // For demo purposes, generate simulated results
        // In a real implementation, this would fetch actual results from Supabase
        setTimeout(() => {
          const simulatedResults = {
            maxSpeed: Math.round(Math.random() * 40 + 60), // 60-100 mph
            avgSpeed: Math.round(Math.random() * 20 + 50), // 50-70 mph
            topAcceleration: Math.round(Math.random() * 30 + 40), // 40-70 ft/s²
            movementPatterns: [
              { name: 'Quick Snap', percentage: Math.round(Math.random() * 40 + 30) },
              { name: 'Lateral Movement', percentage: Math.round(Math.random() * 30 + 20) },
              { name: 'Vertical Extension', percentage: Math.round(Math.random() * 20 + 10) }
            ],
            speedOverTime: Array.from({ length: 10 }, () => Math.round(Math.random() * 40 + 40))
          };
          
          setResults(simulatedResults);
          setLoading(false);
        }, 1500);
        
      } catch (error) {
        console.error('Error fetching results:', error);
        setError('Failed to load analysis results. Please try again.');
        setLoading(false);
      }
    }
    
    fetchResults();
  }, [videoId]);
  
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow py-12">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8 text-center">Analysis Results</h1>
          
          {loading ? (
            <div className="flex justify-center items-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
              <span className="ml-3">Loading analysis results...</span>
            </div>
          ) : error ? (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
              <span className="block sm:inline">{error}</span>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-2xl font-bold mb-4">Speed Metrics</h2>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold">Maximum Glove Speed</h3>
                    <div className="flex items-end">
                      <span className="text-4xl font-bold text-blue-600">{results.maxSpeed}</span>
                      <span className="ml-2 text-gray-600">mph</span>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold">Average Glove Speed</h3>
                    <div className="flex items-end">
                      <span className="text-4xl font-bold text-blue-600">{results.avgSpeed}</span>
                      <span className="ml-2 text-gray-600">mph</span>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold">Top Acceleration</h3>
                    <div className="flex items-end">
                      <span className="text-4xl font-bold text-blue-600">{results.topAcceleration}</span>
                      <span className="ml-2 text-gray-600">ft/s²</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-2xl font-bold mb-4">Movement Patterns</h2>
                
                <div className="space-y-4">
                  {results.movementPatterns.map((pattern, index) => (
                    <div key={index}>
                      <div className="flex justify-between mb-1">
                        <span className="font-semibold">{pattern.name}</span>
                        <span>{pattern.percentage}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${pattern.percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="bg-white rounded-lg shadow-md p-6 md:col-span-2">
                <h2 className="text-2xl font-bold mb-4">Speed Over Time</h2>
                
                <div className="h-64 flex items-end space-x-2">
                  {results.speedOverTime.map((speed, index) => (
                    <div 
                      key={index} 
                      className="bg-blue-600 w-full rounded-t"
                      style={{ height: `${speed}%` }}
                    >
                      <div className="h-full w-full hover:bg-blue-500 transition-colors duration-200"></div>
                    </div>
                  ))}
                </div>
                
                <div className="flex justify-between mt-2 text-sm text-gray-600">
                  <span>Start</span>
                  <span>Time</span>
                  <span>End</span>
                </div>
              </div>
            </div>
          )}
          
          <div className="mt-8 text-center">
            <Link 
              to="/" 
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-300"
            >
              Analyze Another Video
            </Link>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default ResultsPage;
