# Pedestrian Psychology Machine
Currently working on YOLOv8 + ByteTrack integration

Computer vision pipeline for pedestrian detection and analysis using the JAAD dataset.

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Framework:** PyTorch
- **Video processing:** OpenCV

## Setup

1. Clone the repo:
```bash
   git clone https://github.com/Muhammad-Numan/Pedestrian_psychology_machine.git
   cd Pedestrian_psychology_machine
```

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/Scripts/activate    # Windows (Git Bash)
   source venv/bin/activate        # Linux/Mac
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Dataset

This project uses the JAAD dataset. Download it from the [official JAAD repository](https://github.com/ykotseruba/JAAD) and place the videos in a `JAAD/` folder at the project root.

## Usage

```bash
python count_pedestrians.py
```

## Project Status

Work in progress — Master's research project on pedestrian detection, tracking, and demographic attribute classification.
