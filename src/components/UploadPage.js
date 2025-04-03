import React, { useState } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';

function UploadPage() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!file) {
      alert('Please select a video file first');
      return;
    }
    
    // Simulate upload process
    setIsUploading(true);
    
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          // Redirect to results page after upload completes
          window.location.href = '/results';
          return 100;
        }
        return prev + 10;
      });
    }, 500);
  };
  
  return (
    <div>
      <Header />
      
      <main className="py-12">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8 text-center">Upload Video for Analysis</h1>
          
          <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-8">
            <div className="mb-8">
              <h2 className="text-xl font-semibold mb-4">Instructions</h2>
              <ul className="list-disc pl-5 space-y-2 text-gray-700">
                <li>Upload a video of a baseball catcher in action</li>
                <li>Supported formats: MP4, AVI, MOV (max 100MB)</li>
                <li>For best results, ensure the catcher's glove is clearly visible</li>
                <li>Processing may take a few minutes depending on video length</li>
              </ul>
            </div>
            
            <form onSubmit={handleSubmit}>
              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-2">
                  Select Video File
                </label>
                <input
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime"
                  onChange={handleFileChange}
                  className="w-full border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                {file && (
                  <p className="mt-2 text-sm text-gray-600">
                    Selected: {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                  </p>
                )}
              </div>
              
              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-2">
                  Analysis Options
                </label>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="option-speed"
                      className="mr-2"
                      defaultChecked
                    />
                    <label htmlFor="option-speed">Speed Calculation</label>
                  </div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="option-trajectory"
                      className="mr-2"
                      defaultChecked
                    />
                    <label htmlFor="option-trajectory">Trajectory Analysis</label>
                  </div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="option-advanced"
                      className="mr-2"
                      defaultChecked
                    />
                    <label htmlFor="option-advanced">Advanced Statistics</label>
                  </div>
                </div>
              </div>
              
              {isUploading ? (
                <div className="mb-6">
                  <p className="mb-2 font-semibold">Uploading: {uploadProgress}%</p>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full" 
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                </div>
              ) : (
                <button
                  type="submit"
                  className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition duration-300"
                  disabled={!file}
                >
                  Upload and Analyze
                </button>
              )}
            </form>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default UploadPage;
