
# Video Annotation and Analysis Toolkit

A comprehensive collection of Python scripts for video frame annotation, labeling, analysis, and quality control using YOLO-based object detection models. This toolkit provides both automated and manual annotation capabilities, supporting complete workflows from video processing to dataset analysis for machine learning and computer vision projects.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Detailed Script Documentation](#detailed-script-documentation)
6. [File Formats](#file-formats)
7. [Usage Workflows](#usage-workflows)
8. [Configuration](#configuration)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)
12. [Examples](#examples)

## Features

* **Automated Frame Annotation:** Extract frames from videos and annotate them using pre-trained YOLO models with configurable confidence thresholds
* **GUI-based Manual Annotation:** Intuitive PyQt5 interface for manual bounding box annotation, refinement, and class assignment with undo/redo support
* **Batch Video Processing:** Process multiple videos simultaneously or sequentially with frame-skipping options and custom export dimensions
* **Threading & Performance:** Optimized multi-threaded frame reading and writing for improved processing speed
* **Class Distribution Analysis:** Analyze and visualize class distributions across annotated datasets with statistical summaries
* **Label Management:** Update and manage class name mappings in label files with batch operations
* **Class Weight Calculation:** Calculate weighted distributions for imbalanced datasets to improve model training
* **Video Export:** Generate output videos with annotations overlaid on original footage with configurable frame rates and codecs
* **YOLO Format Support:** Full support for YOLO-format annotations for seamless integration with popular detection frameworks
* **Quality Control:** Validation scripts to ensure annotation consistency and data integrity
* **Logging & Monitoring:** Comprehensive logging for debugging and tracking processing progress

## Prerequisites

### System Requirements

* Python 3.8 or higher
* 4GB+ RAM (8GB+ recommended for batch processing)
* GPU support (CUDA 11.0+) for faster inference (optional but recommended)
* Linux/MacOS/Windows with OpenCV support

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Runtime environment |
| opencv-python | 4.5+ | Video/image processing |
| PyQt5 | 5.15+ | GUI annotation interface |
| PyTorch | 1.9+ | Deep learning framework |
| Ultralytics | 8.0+ | YOLO model inference |
| numpy | 1.19+ | Numerical computations |

## Installation

### Step 1: Clone or Download the Repository

```bash
cd /path/to/annotation-toolkit
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install opencv-python==4.8.0.74 PyQt5==5.15.9 torch==2.0.0 ultralytics==8.0.181 numpy==1.24.3
```

Or using requirements file (if available):

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import cv2, PyQt5, torch, ultralytics; print('All dependencies installed successfully')"
```

## Project Structure

```
Annotation_Script/
├── Annotated_Frames_And_Export_Video.py    # Main automated annotation script
├── Annotation_Project.py                    # GUI annotation tool
├── Test_And_Export_Videos.py                # Video testing & export
├── Class_Distribution_Count_In Annotated_Frames.py  # Class analysis
├── Weights_Of_Each_Class.py                 # Weight calculation
├── Update_Labels_txt_Mapping.py             # Label management
├── readme.md                                # This file
└── [Output directories created during execution]
    ├── annotated_frames/                    # Exported frame images
    ├── labels/                              # Frame labels (YOLO format)
    └── output_videos/                       # Generated annotated videos
```

## Detailed Script Documentation

### 1. Annotated_Frames_And_Export_Video.py

**Purpose:** Extracts frames from video files, performs YOLO object detection inference, and exports annotated frames with corresponding label files.

**Key Features:**
- Multi-threaded frame reading and writing for performance optimization
- Configurable frame skipping to reduce dataset size
- Custom export dimensions for resizing frames
- Real-time progress logging
- Support for multiple video formats (MP4, AVI, MOV, MKV, etc.)
- Automatic directory creation for output

**Usage:**

```bash
python Annotated_Frames_And_Export_Video.py
```

**Configuration Parameters (edit in script):**

```python
VIDEO_PATH = "path/to/your/video.mp4"
MODEL_PATH = "models/best.pt"  # Path to YOLO model weights
OUTPUT_DIR = "output/"
SKIP_FRAMES = 0  # 0 = process all frames, 5 = process every 5th frame
EXPORT_DIMENSION = None  # (640, 480) for custom size, None for original
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
THREAD_COUNT = 4
```

**Input Requirements:**
- Valid video file in supported format
- Pre-trained YOLO weights file (.pt)

**Output Files:**
- `frames_dir/Frame_[number].jpg` - Annotated frame images
- `labels_dir/Frame_[number].txt` - Label files in YOLO format

**Logging Output:**
```
2024-04-06 10:15:23 - INFO - Started reading frames...
2024-04-06 10:15:24 - INFO - Skipped first 0 frames.
2024-04-06 10:15:30 - INFO - Finished reading frames.
2024-04-06 10:15:40 - INFO - Processing complete: 300 frames annotated
```

---

### 2. Annotation_Project.py

**Purpose:** Interactive PyQt5 GUI application for manual annotation, allowing users to draw bounding boxes, assign class labels, and refine annotations with full undo/redo support.

**Key Features:**
- Draw and edit bounding boxes with mouse
- Color-coded boxes for different classes (8 color options)
- Undo/redo functionality (up to 30 actions)
- Class assignment buttons for quick labeling
- Session persistence (saves last session state)
- Support for single image or batch annotation
- Export to YOLO format

**Usage:**

```bash
python Annotation_Project.py
```

**Keyboard Shortcuts:**
- `Left Click & Drag` - Draw bounding box
- `Right Click` - Select/edit box
- `Delete` - Remove selected box
- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo
- `S` - Save annotations
- `N` - Next image
- `P` - Previous image

**GUI Components:**
- **Main Canvas:** Display area for images with bounding boxes
- **Class Buttons:** Quick select buttons for each class (color-coded)
- **Box List:** Shows all boxes on current image with delete options
- **File Browser:** Select input images/videos
- **Session Manager:** Recall previous session data

**Output:**
- Annotations saved in `annotations/` directory
- Format: YOLO txt files with normalized coordinates
- Session state in `last_session.txt`

---

### 3. Test_And_Export_Videos.py

**Purpose:** Tests trained YOLO models on input videos, generates output videos with visual annotations overlaid, and creates frame-by-frame statistics.

**Key Features:**
- Model inference on video frames
- Real-time visualization with bounding boxes
- Configurable output video codec and frame rate
- Statistics generation (detections per frame, class counts)
- Support for multiple model formats
- Progress bar for long videos

**Usage:**

```bash
python Test_And_Export_Videos.py
```

**Configuration:**

```python
VIDEO_PATH = "path/to/test_video.mp4"
MODEL_PATH = "models/best.pt"
OUTPUT_VIDEO = "output_annotated.mp4"
CONFIDENCE = 0.5
IOU = 0.45
FPS = 30
OUTPUT_CODEC = "mp4v"  # or "XVID" for .avi
```

**Output:**
- `output_annotated.mp4` - Video with visual annotations
- `detection_stats.csv` - Frame-by-frame statistics
- Console output with model performance metrics

---

### 4. Class_Distribution_Count_In Annotated_Frames.py

**Purpose:** Analyzes the distribution of object classes across annotated frames, providing statistical insights for dataset balance assessment.

**Key Features:**
- Glob pattern support for processing multiple files
- Calculates total annotations and per-class counts
- Generates statistical summaries (mean, median, standard deviation)
- Visual percentage distribution display
- Supports dynamic or predefined class names

**Usage:**

```bash
python "Class_Distribution_Count_In Annotated_Frames.py"
```

**Configuration:**

```python
LABEL_PATH = "labels/**/*.txt"  # Glob pattern
CLASS_NAMES = {
    0: "car",
    1: "person",
    2: "truck",
    # ... add more classes
}
```

**Output Example:**

```
Class Distribution Analysis
===========================
Total annotations: 1250

Class Breakdown:
- car (ID: 0): 450 (36.0%)
- person (ID: 1): 520 (41.6%)
- truck (ID: 2): 280 (22.4%)

Statistics:
- Mean annotations per frame: 4.17
- Median: 4.0
- Std Dev: 1.23
```

---

### 5. Weights_Of_Each_Class.py

**Purpose:** Calculates class weights for imbalanced datasets, useful for training models with weighted loss functions.

**Key Features:**
- Computes inverse frequency weights
- Generates class balancing weights
- Exports weights in multiple formats (JSON, Python dict, TensorFlow)
- Handles missing classes gracefully

**Usage:**

```bash
python Weights_Of_Each_Class.py
```

**Output Example:**

```python
Class Weights (for training):
{
    "car": 1.23,
    "person": 0.95,
    "truck": 1.82
}

Normalized weights (sum=1):
{
    "car": 0.33,
    "person": 0.26,
    "truck": 0.41
}
```

Use these weights in your model:
```python
class_weights = torch.tensor([0.33, 0.26, 0.41])
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

### 6. Update_Labels_txt_Mapping.py

**Purpose:** Manages and updates class name mappings in existing label files, allowing bulk renaming or remapping of class IDs.

**Key Features:**
- Batch update multiple label files
- Remap class IDs (e.g., 0→2, 1→0)
- Backup original files before changes
- Validate mapping integrity
- Generate change reports

**Usage:**

```bash
python Update_Labels_txt_Mapping.py
```

**Configuration:**

```python
LABEL_DIR = "labels/"
CLASS_MAPPING = {
    0: 1,  # Map old class 0 to new class 1
    1: 0,  # Map old class 1 to new class 0
    2: 2   # Keep class 2 unchanged
}
```

---

## File Formats

### YOLO Annotation Format

Each frame has a corresponding `.txt` file with one annotation per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Details:**
- `class_id`: Integer identifier for the object class (0-indexed)
- `x_center`, `y_center`: Normalized coordinates of bounding box center (0-1 range)
- `width`, `height`: Normalized dimensions of bounding box (0-1 range)

**Example:**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
2 0.8 0.7 0.2 0.25
```

### File Naming Convention

| Type | Format | Example |
|------|--------|---------|
| Frame Image | `Frame_[index].jpg` | `Frame_0.jpg`, `Frame_1.jpg` |
| Label File | `Frame_[index].txt` | `Frame_0.txt`, `Frame_1.txt` |
| Video Output | `[basename]_annotated.mp4` | `video_annotated.mp4` |

---

## Usage Workflows

### Workflow 1: Quick Automated Annotation

1. Prepare your video file and YOLO model weights
2. Update `Annotated_Frames_And_Export_Video.py` configuration
3. Run the script:
   ```bash
   python Annotated_Frames_And_Export_Video.py
   ```
4. Review output frames and labels
5. Analyze class distribution:
   ```bash
   python "Class_Distribution_Count_In Annotated_Frames.py"
   ```

### Workflow 2: Manual Annotation & Refinement

1. Start the GUI annotation tool:
   ```bash
   python Annotation_Project.py
   ```
2. Load images or video frames
3. Draw bounding boxes and assign classes
4. Use undo/redo as needed
5. Save annotations (Ctrl+S)
6. Export to YOLO format

### Workflow 3: Dataset Preparation for Training

1. Annotate all videos using automated or manual methods
2. Analyze class distribution:
   ```bash
   python "Class_Distribution_Count_In Annotated_Frames.py"
   ```
3. Calculate class weights for imbalanced data:
   ```bash
   python Weights_Of_Each_Class.py
   ```
4. Prepare train/val/test splits (external tool)
5. Train YOLO model with calculated weights

### Workflow 4: Model Testing & Evaluation

1. Run model on test video:
   ```bash
   python Test_And_Export_Videos.py
   ```
2. Review output video with annotations
3. Check detection statistics
4. Adjust confidence threshold if needed
5. Compare with ground truth annotations

---

## Configuration

### Environment Variables

```bash
export YOLO_MODEL_PATH="/path/to/models/best.pt"
export VIDEO_INPUT_DIR="/path/to/videos/"
export ANNOTATION_OUTPUT_DIR="/path/to/output/"
export GPU_ID=0  # For multi-GPU systems
```

### Model Selection

Supported YOLO versions:
- YOLOv5 (*.pt files)
- YOLOv8 (recommended)
- Custom trained models

**Model sizes available:**
- `yolov8n` - Nano (fastest, least accurate)
- `yolov8s` - Small
- `yolov8m` - Medium
- `yolov8l` - Large
- `yolov8x` - Extra Large (slowest, most accurate)

---

## Advanced Usage

### Batch Processing Multiple Videos

```python
import glob
from pathlib import Path

video_dir = "videos/"
for video_file in glob.glob(f"{video_dir}/*.mp4"):
    print(f"Processing {video_file}...")
    # Update VIDEO_PATH and run Annotated_Frames_And_Export_Video.py
```

### GPU Acceleration

```python
import torch
# Check available GPUs
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")

# Set specific GPU
torch.cuda.set_device(0)
```

### Custom YOLO Model Training

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640,
    device=0  # GPU ID
)
```

### Memory Optimization

For large videos with limited RAM:

```python
SKIP_FRAMES = 5  # Process every 5th frame
EXPORT_DIMENSION = (480, 360)  # Reduce resolution
THREAD_COUNT = 2  # Reduce threads
```

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | `pip install opencv-python` |
| `CUDA out of memory` | Model too large for GPU | Reduce batch size or use CPU |
| `Video file not found` | Incorrect path | Verify file path and format |
| `GUI won't start` | Display server issue | Set `export DISPLAY=:0` (Linux) |
| `Poor detection quality` | Wrong model or low confidence | Use appropriate model for task, lower threshold |
| `Frames not exported` | Permission issue | Check directory write permissions |
| `Slow processing` | CPU bottleneck | Enable GPU, reduce frame resolution |

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

---

## Performance Optimization

### Speed Improvements

1. **Use GPU Acceleration:**
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = YOLO('best.pt', device=device)
   ```

2. **Optimize Frame Configuration:**
   - Skip frames for faster processing
   - Reduce export dimensions if file size is concern
   - Increase thread count (4-8 optimal)

3. **Model Selection:**
   - Use smaller models (nano/small) for real-time
   - Use larger models only if accuracy critical

4. **Batch Processing:**
   - Process multiple videos in parallel
   - Use queue-based system for memory management

### Typical Processing Times

| Video Duration | FPS | Model | GPU | Time |
|---|---|---|---|---|
| 1 minute | 30 | YOLOv8m | RTX 3060 | ~15 sec |
| 10 minutes | 30 | YOLOv8m | RTX 3060 | ~2 min |
| 1 hour | 30 | YOLOv8n | CPU | ~4 hours |

---

## Examples

### Example 1: Complete Annotation Pipeline

```python
# Step 1: Extract and annotate frames
python Annotated_Frames_And_Export_Video.py

# Step 2: Analyze distribution
python "Class_Distribution_Count_In Annotated_Frames.py"

# Step 3: Calculate weights
python Weights_Of_Each_Class.py

# Step 4: Manual refinement
python Annotation_Project.py

# Step 5: Export annotated video
python Test_And_Export_Videos.py
```

### Example 2: Dataset Configuration (data.yaml)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 3  # number of classes
names: ['car', 'person', 'truck']
```

---

## Support & Contribution

For issues, questions, or contributions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review script logging output
3. Verify all dependencies are correctly installed
4. Test with sample video in minimal configuration

## License

This toolkit is part of the Data Annotation project. See root LICENSE file for details.


