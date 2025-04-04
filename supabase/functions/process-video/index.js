// process-video.js - Supabase Edge Function
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.5.0'

// Initialize Supabase client
const supabaseUrl = Deno.env.get('SUPABASE_URL')
const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')
const supabase = createClient(supabaseUrl, supabaseKey)

// This function simulates video processing and generates realistic analysis results
Deno.serve(async (req) => {
  try {
    // Parse request body
    const { videoId, filePath } = await req.json()
    
    if (!videoId || !filePath) {
      return new Response(
        JSON.stringify({ error: 'Missing required parameters' }),
        { headers: { 'Content-Type': 'application/json' }, status: 400 }
      )
    }
    
    console.log(`Processing video: ${videoId}, path: ${filePath}`)
    
    // Get video metadata from database
    const { data: videoData, error: videoError } = await supabase
      .from('videos')
      .select('*')
      .eq('id', videoId)
      .single()
    
    if (videoError) {
      throw new Error(`Error fetching video data: ${videoError.message}`)
    }
    
    // Update video status to processing
    const { error: updateError } = await supabase
      .from('videos')
      .update({ status: 'processing' })
      .eq('id', videoId)
    
    if (updateError) {
      throw new Error(`Error updating video status: ${updateError.message}`)
    }
    
    // Simulate processing time based on file size
    // In a real implementation, this would be where we call the Python processing code
    const processingTime = Math.min(Math.floor(videoData.file_size / (1024 * 1024) * 500), 5000)
    await new Promise(resolve => setTimeout(resolve, processingTime))
    
    // Generate realistic analysis results based on file metadata
    // In a real implementation, these would come from the Python CV code
    const fileSize = videoData.file_size
    const fileName = videoData.original_name
    
    // Use file properties to generate somewhat deterministic but realistic values
    const seed = (fileSize % 1000) / 1000
    const maxSpeed = (35 + seed * 15).toFixed(1) // 35-50 mph
    const avgSpeed = (maxSpeed * 0.7).toFixed(1) // ~70% of max speed
    const topAcceleration = (10 + seed * 10).toFixed(1) // 10-20 m/s²
    
    // Generate movement patterns
    const quickSnapCount = Math.floor(5 + seed * 10)
    const lateralCount = Math.floor(3 + seed * 8)
    const verticalCount = Math.floor(2 + seed * 6)
    
    const movementPatterns = [
      { 
        type: "Quick Snap", 
        count: quickSnapCount, 
        avgSpeed: (parseFloat(avgSpeed) * 1.2).toFixed(1) 
      },
      { 
        type: "Lateral Movement", 
        count: lateralCount, 
        avgSpeed: (parseFloat(avgSpeed) * 0.8).toFixed(1) 
      },
      { 
        type: "Vertical Reach", 
        count: verticalCount, 
        avgSpeed: (parseFloat(avgSpeed) * 1.0).toFixed(1) 
      }
    ]
    
    // Insert analysis results into database
    const { data: resultData, error: resultError } = await supabase
      .from('analysis_results')
      .insert([
        {
          video_id: videoId,
          max_speed: parseFloat(maxSpeed),
          avg_speed: parseFloat(avgSpeed),
          top_acceleration: parseFloat(topAcceleration),
          movement_patterns: JSON.stringify(movementPatterns)
        }
      ])
      .select()
    
    if (resultError) {
      throw new Error(`Error inserting analysis results: ${resultError.message}`)
    }
    
    // Update video status to completed
    const { error: completeError } = await supabase
      .from('videos')
      .update({ status: 'completed' })
      .eq('id', videoId)
    
    if (completeError) {
      throw new Error(`Error updating video status: ${completeError.message}`)
    }
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        message: 'Video processed successfully',
        results: {
          max_speed: maxSpeed,
          avg_speed: avgSpeed,
          top_acceleration: topAcceleration,
          movement_patterns: movementPatterns
        }
      }),
      { headers: { 'Content-Type': 'application/json' } }
    )
    
  } catch (error) {
    console.error('Error processing video:', error)
    
    // Try to update video status to error if possible
    try {
      if (req.body && req.body.videoId) {
        const { videoId } = await req.json()
        await supabase
          .from('videos')
          .update({ 
            status: 'error',
            error_message: error.message
          })
          .eq('id', videoId)
      }
    } catch (updateError) {
      console.error('Failed to update video status:', updateError)
    }
    
    return new Response(
      JSON.stringify({ error: error.message }),
      { headers: { 'Content-Type': 'application/json' }, status: 500 }
    )
  }
})
