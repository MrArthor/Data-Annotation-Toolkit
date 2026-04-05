import os

label_dir = '/home/vansh/Desktop/Code/Architecture/x86/Helmet_Detection_Model/Frames_And_Videos/Dataset/face.v1i.yolov8/train/labels'
mapping = {'0': '2'}

for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        path = os.path.join(label_dir, label_file)
        with open(path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.split()
            if parts[0] in mapping:
                parts[0] = mapping[parts[0]]
            new_lines.append(" ".join(parts) + "\n")
            
        with open(path, 'w') as f:
            f.writelines(new_lines)