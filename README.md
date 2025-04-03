# Baseball Catcher's Glove Speed Tracker

A computer vision application that processes video footage of baseball catchers to calculate and analyze glove speed.

## Overview

This application uses advanced computer vision and machine learning techniques to track a baseball catcher's glove movements in video footage and calculate its speed. The system provides valuable insights for training, performance analysis, and player development.

## Features

- **Video Processing**: Support for various video formats (MP4, AVI, MOV, MKV)
- **Glove Detection & Tracking**: Deep learning-based object detection and tracking
- **Speed Calculation**: Accurate measurement of glove speed in real-world units
- **Data Analysis**: Statistical analysis of movement patterns and performance metrics
- **Visualization**: Real-time speed overlay and graphical representations
- **Reporting**: Comprehensive performance reports and data export
- **User Interface**: Web-based interface for easy video upload and result viewing
- **Cloud Deployment**: Ready for deployment on cloud platforms

## Technology Stack

- **Backend**: Python, OpenCV, TensorFlow
- **Data Science**: NumPy, Pandas, Matplotlib
- **Web Interface**: Flask
- **Deployment**: Docker, Kubernetes

## Project Structure

```
glove_speed_tracker/
├── data/               # Sample data and test videos
├── src/                # Source code
│   ├── config.py       # Configuration settings
│   ├── video_processor.py  # Video processing module
│   ├── detector.py     # Glove detection module
│   ├── tracker.py      # Object tracking module
│   ├── speed_calculator.py   # Speed calculation module
│   ├── data_analyzer.py  # Data analysis module
│   ├── visualizer.py   # Visualization module
│   ├── ui.py           # User interface components
│   ├── app.py          # Web application
│   ├── cloud_deploy.py # Cloud deployment module
│   └── tester.py       # Testing and optimization module
├── models/             # Trained ML models
├── output/             # Output files and results
├── tests/              # Unit and integration tests
├── docs/               # Documentation
├── ui/                 # User interface components
├── kubernetes/         # Kubernetes deployment configurations
├── Dockerfile          # Docker container configuration
├── docker-compose.yml  # Docker Compose configuration
├── cloudbuild.yaml     # Google Cloud Build configuration
├── deploy.sh           # Deployment script
└── requirements.txt    # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- TensorFlow
- Flask

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/theogyeezy/Glove-Speed-Tracker.git
   cd Glove-Speed-Tracker
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/app.py
   ```

## Usage

1. Upload a video of a baseball catcher in action
2. Configure detection parameters if needed
3. Process the video to track the glove movement
4. View the results with speed overlay and analytics
5. Export reports for further analysis

## Development Roadmap

- [x] Project setup and environment configuration
- [x] Video processing module implementation
- [x] Glove detection and tracking module
- [x] Speed calculation algorithms
- [x] Data analysis and model training
- [x] Visualization and reporting features
- [x] User interface development
- [x] Containerization and deployment preparation
- [x] Cloud deployment configuration
- [x] Performance testing and optimization

## Deployment

The application can be deployed using the provided deployment script:

```bash
# For local deployment with Docker
./deploy.sh --target local

# For Google Cloud Run deployment
./deploy.sh --target gcp-run --project YOUR_GCP_PROJECT_ID

# For Kubernetes deployment
./deploy.sh --target gcp-k8s --project YOUR_GCP_PROJECT_ID
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Baseball analytics community
- Computer vision research papers and implementations
- Open-source machine learning frameworks
