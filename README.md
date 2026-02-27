# Audio Intelligence & Human Communication Analytics System

An offline system designed to analyze human speaking behavior from audio recordings without relying on deep learning frameworks or internet access.

## Features
- Voice Activity Detection (WebRTC VAD)
- Audio streaming to support up to 30 min recordings without loading all into memory
- Offline Feature Extraction & Language Classification (Simulated)
- Advanced Analytics: pace, interruptions, dead-air, lecture vs discussion classification.
- Visualizations: Timeline, Heatmap, Piechart.

## Setup Requirements
System dependencies: FFmpeg must be installed and added to PATH for audio conversion.

## How to Run

1. Place your audio files inside `inputs/`. An example `sample.mp3` can be used.
2. Run the `setup.bat` file to initialize the project:
   ```cmd
   setup.bat
   ```
3. Run the main processing script:
   ```cmd
   python main.py inputs/sample.mp3
   ```

Output statistics, segment CSV, and graphs will be populated in the `outputs/` folder.

