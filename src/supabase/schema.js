import supabase from '../supabaseClient';

/**
 * Initialize Supabase database schema
 * This function creates the necessary tables if they don't exist
 */
export const initializeDatabase = async () => {
  try {
    // Check if videos table exists by attempting to query it
    const { error: checkVideosError } = await supabase
      .from('videos')
      .select('id')
      .limit(1);
    
    // If table doesn't exist, we need to create our schema
    if (checkVideosError && checkVideosError.code === '42P01') {
      console.log('Database tables not found. Creating schema...');
      
      // In a production environment, we would use migrations
      // For this demo, we'll create tables directly
      
      // Create videos table
      const createVideosTable = `
        CREATE TABLE IF NOT EXISTS videos (
          id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
          file_name TEXT NOT NULL,
          file_path TEXT NOT NULL,
          original_name TEXT,
          file_size INTEGER,
          user_id TEXT NOT NULL,
          status TEXT NOT NULL,
          error_message TEXT,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
      `;
      
      // Create analysis_results table
      const createAnalysisTable = `
        CREATE TABLE IF NOT EXISTS analysis_results (
          id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
          video_id UUID REFERENCES videos(id),
          max_speed FLOAT,
          avg_speed FLOAT,
          top_acceleration FLOAT,
          movement_patterns JSONB,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
      `;
      
      // Execute SQL directly (in a real app, we would use migrations)
      // Note: This is a simplified approach for demonstration
      const { error: createError } = await supabase.rpc('exec_sql', {
        sql: createVideosTable + createAnalysisTable
      });
      
      if (createError) {
        console.error('Error creating database schema:', createError);
        return {
          success: false,
          error: createError.message
        };
      }
      
      console.log('Database schema created successfully');
    }
    
    // Check if storage bucket exists
    const { data: buckets, error: bucketsError } = await supabase
      .storage
      .listBuckets();
    
    if (bucketsError) {
      console.error('Error checking storage buckets:', bucketsError);
      return {
        success: false,
        error: bucketsError.message
      };
    }
    
    // Create videos bucket if it doesn't exist
    const videosBucketExists = buckets.some(bucket => bucket.name === 'videos');
    if (!videosBucketExists) {
      const { error: createBucketError } = await supabase
        .storage
        .createBucket('videos', {
          public: true,
          fileSizeLimit: 100 * 1024 * 1024, // 100MB limit
          allowedMimeTypes: ['video/mp4', 'video/quicktime', 'video/x-msvideo']
        });
      
      if (createBucketError) {
        console.error('Error creating videos bucket:', createBucketError);
        return {
          success: false,
          error: createBucketError.message
        };
      }
      
      console.log('Videos storage bucket created successfully');
    }
    
    return {
      success: true,
      message: 'Database and storage initialized successfully'
    };
  } catch (error) {
    console.error('Error initializing database:', error);
    return {
      success: false,
      error: error.message
    };
  }
};
