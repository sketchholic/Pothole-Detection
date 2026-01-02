import os
from pathlib import Path
from ultralytics import YOLO

def check_data_exists(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

def check_model_file(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

def find_last_checkpoint(project_dir: str, exp_name: str) -> str | None:
    checkpoint_path = Path(project_dir) / exp_name / "weights" / "last.pt"
    return str(checkpoint_path) if checkpoint_path.exists() else None

def train_multiple_models():
    models = {
        #"yolov8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        "yolov11": ["yolov11n.pt", "yolov11s.pt", "yolov11m.pt","yolo11n.pt"]
    }
    results = {}
    check_data_exists("data.yaml")

    best_folder = Path("runs/best_weights")
    best_folder.mkdir(parents=True, exist_ok=True)

    for version, model_list in models.items():
        project_dir = f"runs/{version}"
        for model_name in model_list:
            exp_name = f"pothole_{model_name.split('.')[0]}"
            print(f"\n===============================")
            print(f"Preparing training for {model_name}")
            print(f"===============================")

            try:
                check_model_file(model_name)

                print(f"Looking for checkpoints for {model_name}...")
                checkpoint = find_last_checkpoint(project_dir=project_dir, exp_name=exp_name)

                if checkpoint:
                    print(f"Found checkpoint → Resuming training from {checkpoint}")
                    model = YOLO(checkpoint)
                    resume_training = True
                else:
                    print(f"No checkpoint found → Starting fresh training for {model_name}")
                    model = YOLO(model_name)
                    resume_training = False

                print(f"Training started for {model_name}...")
                result = model.train(
                    data="data.yaml",
                    epochs=30,
                    patience=20,
                    batch=16,
                    resume=resume_training,
                    name=exp_name,
                    project=project_dir
                )

                results[model_name] = result
                print(f"Training finished for {model_name}")

                best_path = Path(project_dir) / exp_name / "weights" / "best.pt"
                if best_path.exists():
                    target_path = best_folder / f"{model_name.split('.')[0]}_best.pt"
                    target_path.write_bytes(best_path.read_bytes())
                    print(f" best.pt saved to central folder: {target_path}")
                else:
                    print(f"Warning: best.pt not found for {model_name}")

            except Exception as e:
                print(f"Error occurred while training {model_name}: {str(e)}")
                print(" Suggestion: Check dataset, model file, or training parameters.")

    return results

def main():
    print("Starting Multi-Model Training with Resume Support...")
    results = train_multiple_models()
    print("\nTraining completed for all models!")
    print("Check results inside: runs/yolov8/, runs/yolov11/, and runs/best_weights/")

if __name__ == "__main__":
    main()
