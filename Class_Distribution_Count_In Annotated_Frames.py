import glob
from collections import Counter
from pathlib import Path
import sys

def count_class_distribution(label_path, class_names=None):
    """
    Dynamically count class distribution in YOLO format dataset
    
    Args:
        label_path (str): Path to labels directory (supports glob patterns)
        class_names (dict): Optional mapping of class IDs to names
    
    Returns:
        dict: Counter object with class distributions
    """
    label_files = glob.glob(label_path, recursive=True)
    
    if not label_files:
        print(f"❌ No label files found at: {label_path}")
        return None
    
    counts = Counter()
    total_annotations = 0
    
    for file in label_files:
        try:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls = int(line.split()[0])
                        counts[cls] += 1
                        total_annotations += 1
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")
    
    if total_annotations == 0:
        print("❌ No annotations found in label files")
        return None
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Class Distribution Analysis")
    print(f"{'='*50}")
    print(f"Total Annotations: {total_annotations}")
    print(f"Total Classes Found: {len(counts)}")
    print(f"Label Files: {len(label_files)}\n")
    
    # Sort by class ID
    for cls_id in sorted(counts.keys()):
        count = counts[cls_id]
        percentage = (count / total_annotations) * 100
        class_label = class_names.get(cls_id, f"Class {cls_id}") if class_names else f"Class {cls_id}"
        bar = "█" * int(percentage / 2)
        print(f"{class_label:20s} | {count:6d} | {percentage:6.2f}% | {bar}")
    
    print(f"{'='*50}\n")
    return counts


if __name__ == "__main__":
    # Default path - can be changed
    LABEL_PATH = '/home/vansh/Desktop/Code/Architecture/x86/Helmet_Detection_Model/Frames_And_Videos/Proccessed_Frames/*.txt'
    
    # Optional: Define class names for your model
    CLASS_NAMES = {
        0: "bike",
        1: "helmet",
        2: "no - helmet",
    }
    
    # Override with command line argument if provided
    if len(sys.argv) > 1:
        LABEL_PATH = sys.argv[1]
    
    if len(sys.argv) > 2:
        # Parse class names if provided as comma-separated key:value pairs
        # Usage: python script.py "path" "0:Class0,1:Class1,2:Class2"
        try:
            class_mapping = sys.argv[2].split(',')
            CLASS_NAMES = {int(k): v for k, v in [pair.split(':') for pair in class_mapping]}
        except:
            pass
    
    counts = count_class_distribution(LABEL_PATH, CLASS_NAMES)