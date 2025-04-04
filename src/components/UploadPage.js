import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from './Header';
import Footer from './Footer';
import { supabase } from '../supabaseClient';

function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a video file to upload');
      return;
    }
    
    // Check file type
    const fileType = file.type;
    if (!fileType.includes('video/')) {
      setError('Please upload a video file');
      return;
    }
    
    try {
      setUploading(true);
      setUploadProgress(0);
      
      // Create a unique file name
      const fileExt = file.name.split('.').pop();
      const fileName = `${Math.random().toString(36).substring(2, 15)}.${fileExt}`;
      const filePath = `videos/${fileName}`;
      
      // Upload to Supabase Storage
      const { data, error: uploadError } = await supabase.storage
        .from('videos')
        .upload(filePath, file, {
          cacheControl: '3600',
          upsert: false,
          onUploadProgress: (progress) => {
            const percent = Math.round((progress.loaded / progress.total) * 100);
            setUploadProgress(percent);
          }
        });
      
      if (uploadError) {
        throw uploadError;
      }
      
      // Create a record in the videos table
      const { data: videoData, error: videoError } = await supabase
        .from('videos')
        .insert([
          { 
            file_path: filePath,
            status: 'uploaded'
          }
        ])
        .select();
      
      if (videoError) {
        throw videoError;
      }
      
      // Trigger processing
      const videoId = videoData[0].id;
      
      // Simulate processing delay
      setTimeout(() => {
        // Navigate to results page
        navigate(`/results/${videoId}`);
      }, 2000);
      
    } catch (error) {
      console.error('Error uploading video:', error);
      setError('Error uploading video: ' + error.message);
      setUploading(false);
    }
  };
  
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow py-12">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8 text-center">Upload Video</h1>
          
          <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-8">
            {!uploading ? (
              <form onSubmit={handleSubmit}>
                <div className="mb-6">
                  <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="video">
                    Select Video File
                  </label>
                  <input
                    className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                    id="video"
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    Supported formats: MP4, AVI, MOV, etc.
                  </p>
                </div>
                
                {error && (
                  <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
                    <span className="block sm:inline">{error}</span>
                  </div>
                )}
                
                <div className="flex items-center justify-between">
                  <button
                    className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-300"
                    type="submit"
                  >
                    Upload & Analyze
                  </button>
                  
                  {file && (
                    <span className="text-sm text-gray-600">
                      Selected: {file.name}
                    </span>
                  )}
                </div>
              </form>
            ) : (
              <div>
                <h2 className="text-xl font-semibold mb-4">Uploading Video</h2>
                
                <div className="mb-4">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">
                    {uploadProgress}% Complete
                  </p>
                </div>
                
                <p className="text-gray-700">
                  {uploadProgress < 100 
                    ? 'Uploading your video...' 
                    : 'Processing your video. This may take a few moments...'}
                </p>
              </div>
            )}
          </div>
          
          <div className="max-w-2xl mx-auto mt-8">
            <h2 className="text-xl font-bold mb-4">Tips for Best Results</h2>
            <ul className="list-disc pl-5 space-y-2">
              <li>Use videos with good lighting and clear visibility of the catcher</li>
              <li>Ensure the catcher's glove is visible throughout the video</li>
              <li>Record from a stable position to minimize camera movement</li>
              <li>Videos between 10-60 seconds provide optimal analysis results</li>
              <li>Higher resolution videos (720p or better) yield more accurate measurements</li>
            </ul>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default UploadPage;
