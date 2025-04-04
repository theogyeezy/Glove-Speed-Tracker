import supabase from '../supabaseClient';

/**
 * Storage service for handling video uploads and retrievals
 */
export const storageService = {
  /**
   * Upload a video file to Supabase storage
   * @param {File} file - The video file to upload
   * @param {string} userId - Optional user identifier
   * @returns {Promise<Object>} - Upload result with file path
   */
  async uploadVideo(file, userId = 'anonymous') {
    try {
      const fileExt = file.name.split('.').pop();
      const fileName = `${userId}_${Date.now()}.${fileExt}`;
      const filePath = `${userId}/${fileName}`;
      
      const { data, error } = await supabase.storage
        .from('videos')
        .upload(filePath, file, {
          cacheControl: '3600',
          upsert: false
        });
      
      if (error) throw error;
      
      // Create a record in the videos table
      const { data: videoRecord, error: recordError } = await supabase
        .from('videos')
        .insert([
          { 
            file_name: fileName,
            file_path: filePath,
            user_id: userId,
            status: 'uploaded',
            original_name: file.name,
            file_size: file.size
          }
        ])
        .select();
      
      if (recordError) throw recordError;
      
      return {
        success: true,
        filePath,
        videoId: videoRecord[0].id,
        data
      };
    } catch (error) {
      console.error('Error uploading video:', error);
      return {
        success: false,
        error: error.message
      };
    }
  },
  
  /**
   * Get a public URL for a video file
   * @param {string} filePath - Path to the file in storage
   * @returns {string} - Public URL for the file
   */
  getPublicUrl(filePath) {
    const { data } = supabase.storage
      .from('videos')
      .getPublicUrl(filePath);
    
    return data.publicUrl;
  },
  
  /**
   * List all videos for a user
   * @param {string} userId - User identifier
   * @returns {Promise<Array>} - List of videos
   */
  async listUserVideos(userId = 'anonymous') {
    try {
      const { data, error } = await supabase
        .from('videos')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      
      return {
        success: true,
        videos: data
      };
    } catch (error) {
      console.error('Error listing videos:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
};

/**
 * Analysis service for handling video processing and results
 */
export const analysisService = {
  /**
   * Get analysis results for a video
   * @param {string} videoId - Video identifier
   * @returns {Promise<Object>} - Analysis results
   */
  async getAnalysisResults(videoId) {
    try {
      const { data, error } = await supabase
        .from('analysis_results')
        .select('*')
        .eq('video_id', videoId)
        .single();
      
      if (error && error.code !== 'PGRST116') throw error;
      
      if (!data) {
        return {
          success: false,
          status: 'pending',
          message: 'Analysis not completed yet'
        };
      }
      
      return {
        success: true,
        results: data
      };
    } catch (error) {
      console.error('Error getting analysis results:', error);
      return {
        success: false,
        error: error.message
      };
    }
  },
  
  /**
   * Trigger video analysis process
   * @param {string} videoId - Video identifier
   * @returns {Promise<Object>} - Operation result
   */
  async triggerAnalysis(videoId) {
    try {
      // Update video status to 'processing'
      const { error: updateError } = await supabase
        .from('videos')
        .update({ status: 'processing' })
        .eq('id', videoId);
      
      if (updateError) throw updateError;
      
      // Get the video record to access the file path
      const { data: videoData, error: videoError } = await supabase
        .from('videos')
        .select('*')
        .eq('id', videoId)
        .single();
        
      if (videoError) throw videoError;
      
      // Call the Supabase Edge Function to process the video
      // In a real implementation, this would trigger a serverless function
      // that runs the Python video processing code
      const { data: functionData, error: functionError } = await supabase
        .functions
        .invoke('process-video', {
          body: { 
            videoId: videoId,
            filePath: videoData.file_path
          }
        });
      
      if (functionError) throw functionError;
      
      return {
        success: true,
        message: 'Analysis started',
        data: functionData
      };
    } catch (error) {
      console.error('Error triggering analysis:', error);
      
      // Update video status to 'error'
      try {
        await supabase
          .from('videos')
          .update({ 
            status: 'error',
            error_message: error.message
          })
          .eq('id', videoId);
      } catch (updateError) {
        console.error('Failed to update video status:', updateError);
      }
      
      return {
        success: false,
        error: error.message
      };
    }
  },
  
  /**
   * Check the status of a video analysis
   * @param {string} videoId - Video identifier
   * @returns {Promise<Object>} - Status information
   */
  async checkAnalysisStatus(videoId) {
    try {
      const { data, error } = await supabase
        .from('videos')
        .select('status, error_message')
        .eq('id', videoId)
        .single();
      
      if (error) throw error;
      
      return {
        success: true,
        status: data.status,
        errorMessage: data.error_message
      };
    } catch (error) {
      console.error('Error checking analysis status:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
};
