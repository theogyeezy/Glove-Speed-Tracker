import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { analysisService, storageService } from '../supabase/services';

function ResultsPage() {
  const { videoId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [videoStatus, setVideoStatus] = useState('processing');
  const [videoUrl, setVideoUrl] = useState(null);
  
  useEffect(() => {
    // If no videoId is provided, redirect to upload page
    if (!videoId) {
      navigate('/upload');
      return;
    }
    
    // Check video status and get results if available
    const checkStatus = async () => {
      try {
        // Get video status
        const statusResult = await analysisService.checkAnalysisStatus(videoId);
        
        if (!statusResult.success) {
          throw new Error(statusResult.error || 'Failed to check video status');
        }
        
        setVideoStatus(statusResult.status);
        
        // If video is completed, get analysis results
        if (statusResult.status === 'completed') {
          const resultsData = await analysisService.getAnalysisResults(videoId);
          
          if (!resultsData.success) {
            throw new Error(resultsData.error || 'Failed to get analysis results');
          }
          
          setResults(resultsData.results);
          
          // Get video details to display the video
          const { data: videoData, error: videoError } = await supabase
            .from('videos')
            .select('*')
            .eq('id', videoId)
            .single();
          
          if (videoError) throw videoError;
          
          // Get public URL for the video
          const publicUrl = storageService.getPublicUrl(videoData.file_path);
          setVideoUrl(publicUrl);
        } else if (statusResult.status === 'error') {
          throw new Error(statusResult.errorMessage || 'An error occurred during video processing');
        }
        
        setLoading(false);
      } catch (error) {
        console.error('Error checking video status:', error);
        setError(error.message);
        setLoading(false);
      }
    };
    
    checkStatus();
    
    // If video is still processing, check status every 3 seconds
    if (videoStatus === 'processing') {
      const interval = setInterval(checkStatus, 3000);
      return () => clearInterval(interval);
    }
  }, [videoId, videoStatus, navigate]);
  
  // Format speed value with units
  const formatSpeed = (speed) => {
    return `${speed} mph (${(speed * 0.44704).toFixed(1)} m/s)`;
  };
  
  // Format acceleration value with units
  const formatAcceleration = (acceleration) => {
    return `${acceleration} m/s²`;
  };
  
  return (
    <div>
      <Header />
      
      <main className="py-12">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8 text-center">Analysis Results</h1>
          
          {loading ? (
            <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-8 text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <h2 className="text-xl font-semibold mb-2">Processing Your Video</h2>
              <p className="text-gray-600">
                This may take a few minutes depending on the video length and complexity.
              </p>
            </div>
          ) : error ? (
            <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-8">
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-6">
                <strong className="font-bold">Error: </strong>
                <span className="block sm:inline">{error}</span>
              </div>
              <div className="text-center">
                <button
                  onClick={() => navigate('/upload')}
                  className="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 transition duration-300"
                >
                  Try Again
                </button>
              </div>
            </div>
          ) : results ? (
            <div className="max-w-5xl mx-auto">
              <div className="bg-white rounded-lg shadow-md overflow-hidden mb-8">
                <div className="p-6">
                  <h2 className="text-2xl font-bold mb-4">Glove Speed Analysis</h2>
                  
                  {videoUrl && (
                    <div className="mb-6">
                      <h3 className="text-lg font-semibold mb-2">Analyzed Video</h3>
                      <video 
                        controls 
                        className="w-full rounded-lg"
                        src={videoUrl}
                      >
                        Your browser does not support the video tag.
                      </video>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h3 className="text-lg font-semibold mb-2">Maximum Speed</h3>
                      <p className="text-3xl font-bold text-blue-600">
                        {formatSpeed(results.max_speed)}
                      </p>
                    </div>
                    
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h3 className="text-lg font-semibold mb-2">Average Speed</h3>
                      <p className="text-3xl font-bold text-green-600">
                        {formatSpeed(results.avg_speed)}
                      </p>
                    </div>
                    
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <h3 className="text-lg font-semibold mb-2">Top Acceleration</h3>
                      <p className="text-3xl font-bold text-purple-600">
                        {formatAcceleration(results.top_acceleration)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="mb-8">
                    <h3 className="text-xl font-semibold mb-4">Movement Patterns</h3>
                    
                    <div className="overflow-x-auto">
                      <table className="min-w-full bg-white border border-gray-200">
                        <thead>
                          <tr>
                            <th className="py-3 px-4 bg-gray-100 text-left">Pattern Type</th>
                            <th className="py-3 px-4 bg-gray-100 text-left">Count</th>
                            <th className="py-3 px-4 bg-gray-100 text-left">Average Speed</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.movement_patterns && JSON.parse(results.movement_patterns).map((pattern, index) => (
                            <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : ''}>
                              <td className="py-3 px-4 border-b border-gray-200">{pattern.type}</td>
                              <td className="py-3 px-4 border-b border-gray-200">{pattern.count}</td>
                              <td className="py-3 px-4 border-b border-gray-200">{pattern.avgSpeed} mph</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <button
                      onClick={() => navigate('/upload')}
                      className="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 transition duration-300"
                    >
                      Analyze Another Video
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-bold mb-4">Performance Insights</h2>
                <p className="mb-4">
                  Based on the analysis of your catcher's glove movement, here are some key insights:
                </p>
                <ul className="list-disc pl-5 space-y-2 mb-4">
                  <li>
                    <strong>Speed Performance:</strong> The maximum glove speed of {formatSpeed(results.max_speed)} is 
                    {results.max_speed > 40 ? ' excellent' : results.max_speed > 30 ? ' good' : ' average'} for a catcher.
                  </li>
                  <li>
                    <strong>Consistency:</strong> The average speed is {((results.avg_speed / results.max_speed) * 100).toFixed(0)}% of the maximum speed, indicating 
                    {(results.avg_speed / results.max_speed) > 0.7 ? ' good consistency' : ' room for improvement in consistency'}.
                  </li>
                  <li>
                    <strong>Acceleration:</strong> The top acceleration of {formatAcceleration(results.top_acceleration)} shows 
                    {results.top_acceleration > 15 ? ' excellent' : results.top_acceleration > 10 ? ' good' : ' average'} reaction time.
                  </li>
                </ul>
                <p>
                  For detailed training recommendations based on this analysis, please consult with your coach or trainer.
                </p>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-8 text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <h2 className="text-xl font-semibold mb-2">Processing Your Video</h2>
              <p className="text-gray-600">
                Status: {videoStatus}
              </p>
              <p className="text-gray-600 mt-2">
                This may take a few minutes depending on the video length and complexity.
              </p>
            </div>
          )}
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default ResultsPage;
