# Processing Service for Glove Speed Tracker

This directory contains the code for the Python processing service that can be deployed to Google Cloud Run to enable real video analysis instead of simulated results.

## Overview

The processing service is a Flask API that:
1. Receives video processing requests from Supabase
2. Downloads videos from Supabase storage
3. Processes videos using our computer vision algorithms
4. Uploads results back to Supabase

## Directory Structure

```
processing-service/
├── Dockerfile
├── requirements.txt
├── app.py
├── detector.py
├── tracker.py
├── speed_calculator.py
└── utils/
    ├── __init__.py
    ├── supabase_client.py
    └── video_utils.py
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service locally:
```bash
python app.py
```

3. Test with a sample request:
```bash
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{"videoId": "test-id", "filePath": "test-path", "supabaseUrl": "your-url", "supabaseKey": "your-key"}'
```

## Deployment

See the [real-processing-deployment.md](../docs/real-processing-deployment.md) document for detailed deployment instructions.

## Environment Variables

The service requires the following environment variables:
- `PORT`: Port to run the service on (default: 8080)
- `DEBUG`: Enable debug mode (default: False)
- `TEMP_DIR`: Directory for temporary files (default: /tmp)

## API Endpoints

### POST /process

Process a video and store results in Supabase.

**Request Body:**
```json
{
  "videoId": "uuid-of-video",
  "filePath": "path/to/video/in/supabase",
  "supabaseUrl": "https://your-project.supabase.co",
  "supabaseKey": "your-supabase-key"
}
```

**Response:**
```json
{
  "success": true,
  "videoId": "uuid-of-video",
  "results": {
    "max_speed": 45.2,
    "avg_speed": 32.7,
    "top_acceleration": 15.3,
    "movement_patterns": [...]
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```
