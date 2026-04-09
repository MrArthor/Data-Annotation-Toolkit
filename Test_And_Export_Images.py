"""
Image Detection and Annotation Script

This script processes image files using YOLOv8 for object detection,
allowing selective annotation of detected classes with batch processing.

Usage:
    python Test_And_Export_Images.py
    
Or call process_images() function directly with paths.
"""

import cv2
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import json
from datetime import datetime


# ==================== Constants ====================
CONF_THRESHOLD = 0.4  # Confidence threshold (40%)
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


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


def get_image_paths(input_path):
    """
    Get list of image paths from directory or glob pattern.
    
    Args:
        input_path (str): Directory path or glob pattern
        
    Returns:
        list: List of image file paths
    """
    if '*' in input_path:
        # Glob pattern provided
        image_paths = glob.glob(input_path, recursive=True)
    elif os.path.isdir(input_path):
        # Directory provided - search for all supported formats
        image_paths = []
        for ext in SUPPORTED_FORMATS:
            image_paths.extend(glob.glob(os.path.join(input_path, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(input_path, f'**/*{ext}'), recursive=True))
    else:
        # Single file provided
        if os.path.isfile(input_path):
            image_paths = [input_path]
        else:
            image_paths = []
    
    # Remove duplicates and filter existing files
    image_paths = list(set(p for p in image_paths if os.path.isfile(p)))
    image_paths.sort()
    
    return image_paths


def create_output_directory(output_path):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path (str): Path to output directory
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)


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
        
    Returns:
        list: List of detection dictionaries
    """
    detections = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in selected_classes:
            score = float(box.conf[0])
            class_name = model.names[cls_id]
            
            # Get bounding box coordinates
            xyxy = box.xyxy[0].tolist()
            
            detection = {
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': round(score, 4),
                'bbox': {
                    'x_min': round(xyxy[0], 2),
                    'y_min': round(xyxy[1], 2),
                    'x_max': round(xyxy[2], 2),
                    'y_max': round(xyxy[3], 2)
                }
            }
            detections.append(detection)
            
            if verbose:
                print(f"  Class: {class_name}, Confidence: {score:.2f}, BBox: {xyxy}")
    
    # Filter boxes by selected classes
    results.boxes = filter_detections(results.boxes, selected_classes)
    
    return detections


def generate_statistics(all_detections, model, image_count):
    """
    Generate detection statistics.
    
    Args:
        all_detections (dict): Dictionary of detections per image
        model: YOLO model instance
        image_count (int): Total number of images processed
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'total_images': image_count,
        'total_detections': 0,
        'class_distribution': defaultdict(int),
        'average_detections_per_image': 0.0,
        'images_with_detections': 0
    }
    
    for image_data in all_detections.values():
        if image_data['detections']:
            stats['images_with_detections'] += 1
        for detection in image_data['detections']:
            stats['total_detections'] += 1
            class_name = detection['class_name']
            stats['class_distribution'][class_name] += 1
    
    if image_count > 0:
        stats['average_detections_per_image'] = round(stats['total_detections'] / image_count, 2)
    
    # Convert defaultdict to regular dict for JSON serialization
    stats['class_distribution'] = dict(stats['class_distribution'])
    
    return stats


def save_statistics(stats, output_dir):
    """
    Save statistics to JSON file.
    
    Args:
        stats (dict): Statistics dictionary
        output_dir (str): Output directory
    """
    stats_file = os.path.join(output_dir, 'detection_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")


def print_statistics(stats):
    """
    Print formatted statistics.
    
    Args:
        stats (dict): Statistics dictionary
    """
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per image: {stats['average_detections_per_image']}")
    
    if stats['class_distribution']:
        print("\nClass Distribution:")
        for class_name, count in sorted(stats['class_distribution'].items()):
            percentage = (count / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    print("="*60 + "\n")


# ==================== Main Processing Function ====================

def process_images(input_path, output_dir, model_path, save_annotated=True, save_stats=True, verbose=True):
    """
    Process images with YOLO object detection and selective annotation.
    
    Args:
        input_path (str): Path to images directory, glob pattern, or single image
        output_dir (str): Path to output directory for annotated images
        model_path (str): Path to YOLO model weights
        save_annotated (bool): Save annotated images
        save_stats (bool): Save detection statistics
        verbose (bool): Print detection information
        
    Raises:
        FileNotFoundError: If input images or model doesn't exist
        RuntimeError: If image processing fails
    """
    # Validate model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get user class selection
    selected_classes = get_class_selection(model)
    
    # Get image paths
    image_paths = get_image_paths(input_path)
    
    if not image_paths:
        raise FileNotFoundError(f"No images found at: {input_path}")
    
    print(f"Found {len(image_paths)} image(s) to process\n")
    
    # Setup output
    create_output_directory(output_dir)
    
    # Create subdirectories for organized output
    annotated_dir = os.path.join(output_dir, 'annotated') if save_annotated else None
    if annotated_dir:
        create_output_directory(annotated_dir)
    
    # Process images
    image_count = 0
    all_detections = {}
    
    print(f"Processing images...\n")
    
    try:
        for image_path in image_paths:
            image_count += 1
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping...")
                continue
            
            image_name = os.path.basename(image_path)
            print(f"[{image_count}/{len(image_paths)}] Processing: {image_name}")
            
            # Run inference
            results = model(img, verbose=False, conf=CONF_THRESHOLD)[0]
            
            # Process and filter detections
            detections = process_detections(results, model, selected_classes, verbose=verbose)
            
            # Store detection data
            all_detections[image_name] = {
                'image_path': image_path,
                'detections': detections,
                'image_dimensions': {
                    'width': img.shape[1],
                    'height': img.shape[0]
                }
            }
            
            # Save annotated image if enabled
            if save_annotated:
                annotated_frame = results.plot()
                output_image_path = os.path.join(annotated_dir, f"annotated_{image_name}")
                cv2.imwrite(output_image_path, annotated_frame)
                print(f"  → Saved annotated image")
            
            if image_count % 10 == 0 or image_count == len(image_paths):
                print(f"  Progress: {image_count}/{len(image_paths)} completed\n")
        
        # Generate and save statistics
        stats = generate_statistics(all_detections, model, image_count)
        
        if save_stats:
            save_statistics(stats, output_dir)
        
        print_statistics(stats)
        
        # Save detailed detection results
        results_file = os.path.join(output_dir, 'detailed_detections.json')
        with open(results_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        print(f"Detailed results saved to: {results_file}")
        
        print(f"Completed! Processed {image_count} images.")
        if save_annotated:
            print(f"Annotated images saved to: {annotated_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


# ==================== Entry Point ====================

def main():
    """Main entry point for script execution."""
    # Example usage - modify these paths as needed
    input_path = 'Architecture/x86/Helmet_Detection_Model/Frames_And_Videos/Edge_Cases'  # Directory or glob pattern
    output_dir = 'Architecture/x86/Helmet_Detection_Model/Frames_And_Videos/Output_Images/'
    model_path = 'Architecture/x86/Helmet_Detection_Model/Models/Yolov8n/08042026_Yolov8n_Final/weights/best.pt'
    
    try:
        process_images(
            input_path=input_path,
            output_dir=output_dir,
            model_path=model_path,
            save_annotated=True,
            save_stats=False,
            verbose=False
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
