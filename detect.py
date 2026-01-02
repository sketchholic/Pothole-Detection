import os
import shutil
from ultralytics import YOLO

# MODEL_PATH_V8 = r"D:\Coding\ML Pothole\runs\yolov8\pothole_yolov8n\weights\best.pt"
# MODEL_PATH_V11 = r"D:\Coding\ML Pothole\runs\yolov11\pothole_yolo11n\weights\best.pt"

MODEL_PATH_V11 = r"D:\Coding\ML Pothole\runs\best_weights\yolo11n_best.pt"
MODEL_PATH_V8 = r"D:\Coding\ML Pothole\runs\best_weights\yolov8n_best.pt"

def detect_single_image(image_path):
    model = YOLO(MODEL_PATH_V8)
    
    results = model.predict(source=image_path, save=True, conf=0.5, iou=0.5, project='temp_results_v8', name='single_run', exist_ok=True)
    
    output_dir = 'temp_results_v8/single_run'
    img_name = os.path.basename(image_path)
    result_img_path = os.path.join(output_dir, img_name)
    
    final_path = os.path.join(output_dir, 'result_v8.jpg')
    if os.path.exists(result_img_path):  
        shutil.copy(result_img_path, final_path)
    
    detections = results[0].boxes
    pothole_count = len(detections)
    
    txt_path = os.path.join(output_dir, 'count.txt')
    with open(txt_path, 'w') as f:
        f.write(str(pothole_count))
    
    print(f"Results saved to {output_dir}")
    print(f"[YOLOv8] Detection Over, Potholes Found: {pothole_count} | Output: result_v8.jpg")
    
    if os.path.exists(result_img_path):
        return result_img_path, txt_path
    elif os.path.exists(final_path):
        return final_path, txt_path
    else:
        return None, None


def detect_single_image_v11(image_path):
    model = YOLO(MODEL_PATH_V11)
    
    results = model.predict(source=image_path, save=True, conf=0.5, iou=0.5, project='temp_results_v11', name='single_run', exist_ok=True)
    
    output_dir = 'temp_results_v11/single_run'
    img_name = os.path.basename(image_path)
    result_img_path = os.path.join(output_dir, img_name)
    
    final_path = os.path.join(output_dir, 'result_v11.jpg')
    if os.path.exists(result_img_path):
        shutil.copy(result_img_path, final_path)
    
    detections = results[0].boxes
    pothole_count = len(detections)
    
    txt_path = os.path.join(output_dir, 'count.txt')
    with open(txt_path, 'w') as f:
        f.write(str(pothole_count))
    
    print(f"Results saved to {output_dir}")
    print(f"[YOLOv11] Detection Over, Potholes Found: {pothole_count} | Output: result_v11.jpg")
    
    if os.path.exists(result_img_path):
        return result_img_path, txt_path
    elif os.path.exists(final_path):
        return final_path, txt_path
    else:
        return None, None