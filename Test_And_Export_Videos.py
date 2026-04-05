"""
PPE Detection Video Processing Script

This script processes video files using YOLOv8 for object detection,
allowing selective annotation of detected classes with multithreaded processing.

Usage:
    python Video_Test.py <input_video> <output_video> <model_path>
    
Or call process_video() function directly with paths.
"""

import cv2
import threading
import queue
import os
import sys
from ultralytics import YOLO


# ==================== Constants ====================
CONF_THRESHOLD = 0.4  # Confidence threshold (40%)
MAX_QUEUE_SIZE = 30


# ==================== Utility Functions ====================

def display_available_classes(model):
    """Display all available model classes to the user."""
    print("\n" + "="*60)
    print("Available Model Classes:")
    print("="*60)
    for cls_id, class_name in model.names.items():
        print(f"  [{cls_id}] {class_name}")


def get_class_selection(model):
    """
    Prompt user to select which classes to annotate.
    
    Args:
        model: YOLO model instance
        
    Returns:
        set: Selected class IDs
    """
    display_available_classes(model)
    
    print("\nEnter class IDs to annotate (comma-separated, or press Enter for all):")
    print("Example: 0,2,3")
    user_input = input("> ").strip()
    
    if user_input:
        try:
            selected_classes = set(int(x.strip()) for x in user_input.split(','))
            # Validate selected classes exist
            invalid_classes = selected_classes - set(model.names.keys())
            if invalid_classes:
                print(f"Warning: Invalid class IDs {invalid_classes} ignored.")
                selected_classes -= invalid_classes
            if not selected_classes:
                print("No valid classes selected. Using all classes.")
                selected_classes = set(model.names.keys())
        except ValueError:
            print("Invalid input. Annotating all classes.")
            selected_classes = set(model.names.keys())
    else:
        selected_classes = set(model.names.keys())
    
    print(f"\nAnnotating classes: {', '.join(model.names[c] for c in sorted(selected_classes))}")
    print("="*60 + "\n")
    
    return selected_classes


def get_video_properties(video_path):
    """
    Extract video properties (FPS, width, height).
    
    Args:
        video_path (str): Path to input video
        
    Returns:
        tuple: (fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if fps == 0 or width == 0 or height == 0:
        raise ValueError(f"Invalid video properties: fps={fps}, width={width}, height={height}")
    
    return fps, width, height


def create_output_directory(output_path):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path (str): Path to output video
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def setup_video_writer(output_path, fps, width, height):
    """
    Setup video writer for output.
    
    Args:
        output_path (str): Path to output video
        fps (int): Frames per second
        width (int): Frame width
        height (int): Frame height
        
    Returns:
        cv2.VideoWriter: Video writer object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for: {output_path}")
    
    return writer


def read_frames(video_path, frame_queue):
    """
    Read frames from video in a separate thread.
    
    Args:
        video_path (str): Path to input video
        frame_queue (queue.Queue): Queue to store frames
    """
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
    finally:
        frame_queue.put(None)  # Signal end of video
        cap.release()


def filter_detections(boxes, selected_classes):
    """
    Filter detection boxes by selected classes.
    
    Args:
        boxes: YOLO detection boxes
        selected_classes (set): Set of class IDs to keep
        
    Returns:
        list: Filtered boxes
    """
    filtered = []
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id in selected_classes:
            filtered.append(box)
    return filtered


def process_detections(results, model, selected_classes, verbose=True):
    """
    Process and filter detections from model results.
    
    Args:
        results: YOLO detection results
        model: YOLO model instance
        selected_classes (set): Set of class IDs to keep
        verbose (bool): Print detection info
    """
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in selected_classes:
            score = float(box.conf[0])
            class_name = model.names[cls_id]
            if verbose:
                print(f"  Class: {class_name}, Score: {score:.2f}")
    
    # Filter boxes by selected classes
    results.boxes = filter_detections(results.boxes, selected_classes)


# ==================== Main Processing Function ====================

def process_video(input_path, output_path, model_path, verbose=True):
    """
    Process video with YOLO object detection and selective annotation.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        model_path (str): Path to YOLO model weights
        verbose (bool): Print detection information
        
    Raises:
        FileNotFoundError: If input video or model doesn't exist
        RuntimeError: If video processing fails
    """
    # Validate inputs
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get user class selection
    selected_classes = get_class_selection(model)
    
    # Get video properties
    fps, width, height = get_video_properties(input_path)
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Setup output
    create_output_directory(output_path)
    writer = setup_video_writer(output_path, fps, width, height)
    
    # Setup frame reading queue and thread
    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    reader_thread = threading.Thread(target=read_frames, args=(input_path, frame_queue), daemon=False)
    reader_thread.start()
    
    # Process frames
    frame_count = 0
    print(f"\nProcessing video...")
    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")
            
            # Run inference
            results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]
            
            # Process and filter detections
            process_detections(results, model, selected_classes, verbose=verbose and frame_count % 30 == 0)
            
            # Write annotated frame
            annotated_frame = results.plot()
            writer.write(annotated_frame)
    
    finally:
        reader_thread.join()
        writer.release()
        print(f"\nCompleted! Processed {frame_count} frames.")
        print(f"Output saved to: {output_path}")


# ==================== Entry Point ====================

def main():
    """Main entry point for script execution."""
    # Example usage
    input_video = 'Architecture/x86/PPE_Detection/Frames_Videos/VIdeos/27.mp4'
    output_video = 'Architecture/x86/PPE_Detection/Frames_Videos/Output_Videos/27032026_yolov8n/27_output.mp4'
    model_path = 'Architecture/x86/PPE_Detection/Models/Yolov8n/27032026_yolov8n/weights/best.pt'
    
    try:
        process_video(input_video, output_video, model_path, verbose=True)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()