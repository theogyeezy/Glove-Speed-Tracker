# Deploying Real Video Processing to Google Cloud Run

This document provides instructions for deploying the Python video processing code to Google Cloud Run to enable real video analysis instead of simulated results.

## Prerequisites

1. Google Cloud Platform account
2. Google Cloud SDK installed locally
3. Docker installed locally
4. Supabase project (already set up)

## Step 1: Set Up Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Cloud Run API
   - Container Registry API
   - Cloud Build API

## Step 2: Create Service Account

1. In Google Cloud Console, go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Name the service account (e.g., "glove-speed-processor")
4. Grant the following roles:
   - Cloud Run Admin
   - Storage Admin
5. Create and download the JSON key file

## Step 3: Build and Deploy the Processing Service

1. Clone the repository locally
2. Navigate to the `processing-service` directory
3. Build the Docker image:

```bash
docker build -t gcr.io/[YOUR_PROJECT_ID]/glove-speed-processor:v1 .
```

4. Push the image to Google Container Registry:

```bash
gcloud auth configure-docker
docker push gcr.io/[YOUR_PROJECT_ID]/glove-speed-processor:v1
```

5. Deploy to Cloud Run:

```bash
gcloud run deploy glove-speed-processor \
  --image gcr.io/[YOUR_PROJECT_ID]/glove-speed-processor:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

6. Note the URL of your deployed service

## Step 4: Update Supabase Edge Function

1. Update the `process-video` Edge Function to call your Cloud Run service:

```javascript
// In process-video/index.js
// Replace the simulated processing with a call to your Cloud Run service

// Call the real processing service
const response = await fetch('https://your-cloud-run-url.run.app/process', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    videoId,
    filePath,
    supabaseUrl,
    supabaseKey
  }),
});

const result = await response.json();
```

2. Deploy the updated Edge Function:

```bash
cd supabase/functions
supabase functions deploy process-video
```

## Step 5: Test the Integration

1. Upload a video through the web interface
2. The video will be stored in Supabase
3. The Edge Function will trigger and call your Cloud Run service
4. The Cloud Run service will process the video using the real computer vision algorithms
5. Results will be stored back in Supabase and displayed in the web interface

## Troubleshooting

- Check Cloud Run logs for processing errors
- Verify Supabase Edge Function logs for connection issues
- Ensure the service account has the necessary permissions
- Check that the video formats are supported by the processing service

## Cost Considerations

- Google Cloud Run charges based on usage (CPU, memory, and request count)
- For processing many videos, consider setting up a queue system
- Monitor your usage to avoid unexpected charges
