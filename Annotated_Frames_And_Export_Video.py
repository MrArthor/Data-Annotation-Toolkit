import cv2
import os
import threading
import queue
import time
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_frames(video_path, input_queue, stop_event, skip_frames=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        stop_event.set()
        return
    
    if skip_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        logging.info(f"Skipped first {skip_frames} frames.")
        
    logging.info("Started reading frames...")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        input_queue.put(frame)
    
    input_queue.put(None)
    cap.release()
    logging.info("Finished reading frames.")

def write_data(output_queue, frames_dir, labels_dir, writer, stop_event, export_dimension=None):
    logging.info("Started writing data to disk...")
    while not stop_event.is_set():
        data = output_queue.get()
        if data is None:
            break
            
        frame_idx, orig_frame, annotated_frame, labels = data
        
        # Resize frame if export_dimension is specified
        frame_to_save = orig_frame
        if export_dimension is not None:
            frame_to_save = cv2.resize(orig_frame, export_dimension)
        
        frame_filename = os.path.join(frames_dir, f"Frame_{frame_idx}.jpg")
        cv2.imwrite(frame_filename, frame_to_save)
        
        label_filename = os.path.join(labels_dir, f"Frame_{frame_idx}.txt")
        with open(label_filename, "w") as f:
            for label in labels:
                f.write(label + "\n")
                
        if writer is not None and annotated_frame is not None:
            writer.write(annotated_frame)
            
    if writer is not None:
        writer.release()
    logging.info("Finished writing data.")

def process_and_save(video_path, output_dir, model_path, skip_frames=0, export_video=False, export_dimension=None, inference_dimension=(1280, 1280)):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    logging.info(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Failed to open video for metadata.")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    logging.info(f"Video Info: {w}x{h} at {fps} FPS, ~{total_frames} frames total.")
    if inference_dimension:
        logging.info(f"Inference will run on resized frames: {inference_dimension[0]}x{inference_dimension[1]}")

    writer = None
    if export_video:
        out_video_path = os.path.join(output_dir, "annotated_video.mp4")
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    input_queue = queue.Queue(maxsize=30)
    output_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()

    reader_thread = threading.Thread(target=read_frames, args=(video_path, input_queue, stop_event, skip_frames))
    writer_thread = threading.Thread(target=write_data, args=(output_queue, frames_dir, labels_dir, writer, stop_event, export_dimension))

    reader_thread.start()
    writer_thread.start()

    frame_idx = skip_frames
    start_time = time.time()
    
    logging.info("Starting inference loop...")
    try:
        while True:
            frame = input_queue.get()
            if frame is None:
                output_queue.put(None)
                break

            # Resize frame for inference if inference_dimension is specified
            frame_for_inference = frame
            if inference_dimension is not None:
                frame_for_inference = cv2.resize(frame, inference_dimension)
            
            results = model(frame_for_inference, verbose=False)[0]
            
            labels = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x_c, y_c, bw, bh = box.xywhn[0].tolist()
                labels.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            annotated_frame = results.plot() if export_video else None
            
            output_queue.put((frame_idx, frame, annotated_frame, labels))
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = (frame_idx - skip_frames) / elapsed
                logging.info(f"Processed {frame_idx}/{total_frames} frames | Speed: {current_fps:.2f} FPS")

    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Shutting down gracefully...")
        stop_event.set()
        while not input_queue.empty(): 
            input_queue.get()
        output_queue.put(None)
        
    reader_thread.join()
    writer_thread.join()
    
    total_time = time.time() - start_time
    processed_count = frame_idx - skip_frames
    logging.info(f"Processing complete! {processed_count} frames processed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    # Ask user for inference dimension
    print("\n" + "="*50)
    print("Inference Dimension Configuration")
    print("="*50)
    inf_dim_input = input("Enter inference dimension (WIDTHxHEIGHT, e.g., 1280x1280) or press Enter for default (1280x1280): ").strip()
    
    inference_dimension = (1280, 1280)  # Default
    if inf_dim_input:
        try:
            width, height = map(int, inf_dim_input.split('x'))
            inference_dimension = (width, height)
            logging.info(f"Inference dimension set to: {inference_dimension}")
        except ValueError:
            logging.warning(f"Invalid format '{inf_dim_input}'. Using default (1280x1280).")
    else:
        logging.info(f"Using default inference dimension: {inference_dimension}")
    
    # Ask user for export dimension
    print("-" * 50)
    print("Export Dimension Configuration")
    print("-" * 50)
    dim_input = input("Enter export dimension (WIDTHxHEIGHT, e.g., 1280x720) or press Enter for original size: ").strip()
    
    export_dimension = None
    if dim_input:
        try:
            width, height = map(int, dim_input.split('x'))
            export_dimension = (width, height)
            logging.info(f"Export dimension set to: {export_dimension}")
        except ValueError:
            logging.warning(f"Invalid format '{dim_input}'. Using original frame dimensions.")
    else:
        logging.info("Using original frame dimensions.")
    
    print("="*50 + "\n")
   
    # process_and_save(
    #     video_path='Architecture/x86/PPE_Detection/Frames_Videos/VIdeos/25.mp4', 
    #     output_dir='/home/vansh/Desktop/Code/Architecture/x86/PPE_Detection/Frames_Videos/Dataset/25_Annotated', 
    #     model_path='Architecture/x86/PPE_Detection/Models/Yolov8n/23032026_Yolov8n/weights/best.pt', 
    #     skip_frames=0, 
    #     export_video=True,
    #     export_dimension=export_dimension,
    #     inference_dimension=inference_dimension
    # )

    process_and_save(
        video_path='Architecture/x86/Helmet_Detection_Model/Frames_And_Videos/Data_Set_Videos/IMG_2367.MOV', 
        output_dir='/home/vansh/Desktop/Code/Architecture/x86/Helmet_Detection_Model/Frames_And_Videos/Dataset/IMG_2367', 
        model_path='Architecture/x86/Helmet_Detection_Model/Models/Yolov8n/01042026_Yolov8n/weights/best.pt', 
        skip_frames=0, 
        export_video=False,
        export_dimension=export_dimension,
        inference_dimension=inference_dimension
    )