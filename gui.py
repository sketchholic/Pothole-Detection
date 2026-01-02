import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

from detect import detect_single_image, detect_single_image_v11

PRIMARY_BG = "#31393C"
ACCENT_BG = "#CCC7BF"
TEXT_FG = "#3E96F4"
HIGHLIGHT_FG = "#FFFFFF"
MAX_IMAGE_SIZE = (600, 400)

class ImagePanel(tk.Label):
    def __init__(self, master):
        super().__init__(master, bg=PRIMARY_BG, relief="ridge", bd=3)
        self.image_path = None
        self.original_image_path = None  
        self.photo = None
        self.config(text="No image loaded", fg=HIGHLIGHT_FG, font=("Arial", 12))
    
    def display_image(self, path, is_original=False):
        self.image_path = path
        if is_original:
            self.original_image_path = path  
        img = Image.open(path)
        img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.config(image=self.photo, text="")
    
    def reload_original(self):
        """Reload the original image without detection boxes"""
        if self.original_image_path:
            self.display_image(self.original_image_path)
            return True
        return False

def open_image(panel, status_label):
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if file_path:
        panel.display_image(file_path, is_original=True)
        status_label.config(text="✅ Image loaded successfully", fg="green")

def clear_detection(panel, status_label):
    """Clear detection results and reload original image"""
    if panel.reload_original():
        status_label.config(text="🔄 Original image restored", fg="blue")
    else:
        status_label.config(text="⚠️ No image to clear!", fg="red")

def run_detection(detect_func, panel, status_label, model_name):
    if not panel.image_path:
        status_label.config(text="⚠️ No image selected!", fg="red")
        return
    
    status_label.config(text=f"🔄 Detecting with {model_name}...", fg="orange")
    
    def task():
        try:
            result = detect_func(panel.original_image_path)
            if isinstance(result, tuple):
                result_path = result[0]  
                if len(result) > 1:
                    txt_path = result[1]
                    try:
                        import os
                        if os.path.exists(txt_path):
                            with open(txt_path, 'r') as f:
                                count = f.read().strip()
                                status_text = f"✅ Detection complete ({model_name}) - Potholes detected: {count}"
                        else:
                            status_text = f"✅ Detection complete ({model_name})"
                    except:
                        status_text = f"✅ Detection complete ({model_name})"
            else:
                result_path = result  
                status_text = f"✅ Detection complete ({model_name})"
           
            if result_path:
                panel.display_image(result_path)
                status_label.config(text=status_text, fg="green")
            else:
                status_label.config(text=f"❌ Detection failed ({model_name})", fg="red")
               
        except Exception as e:
            status_label.config(
                text=f"❌ Error ({model_name}): {str(e)}", fg="red"
            )
    
    threading.Thread(target=task, daemon=True).start()

def main():
    root = tk.Tk()
    root.title("YOLO Pothole Detection")
    root.configure(bg=PRIMARY_BG)
    root.geometry("700x650")
    root.minsize(700, 650)

    title_label = tk.Label(
        root,
        text="Pothole Detection System",
        font=("Arial", 20, "bold"),
        bg=PRIMARY_BG,
        fg=HIGHLIGHT_FG
    )
    title_label.pack(pady=20)

    panel = ImagePanel(root)
    panel.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    status_label = tk.Label(
        root,
        text="ℹ️ Load an image to start",
        fg=TEXT_FG,
        bg=PRIMARY_BG,
        font=("Arial", 10)
    )
    status_label.pack(pady=10)

    button_frame = tk.Frame(root, bg=PRIMARY_BG)
    button_frame.pack(pady=15)

    tk.Button(
        button_frame,
        text="📂 Open Image",
        command=lambda: open_image(panel, status_label),
        bg=ACCENT_BG,
        fg=TEXT_FG,
        font=("Arial", 12, "bold"),
        padx=15,
        pady=8,
        relief="flat"
    ).grid(row=0, column=0, padx=10)

    tk.Button(
        button_frame,
        text="🔍 Detect (YOLOv8)",
        command=lambda: run_detection(detect_single_image, panel, status_label, "YOLOv8"),
        bg=ACCENT_BG,
        fg=TEXT_FG,
        font=("Arial", 12, "bold"),
        padx=15,
        pady=8,
        relief="flat"
    ).grid(row=0, column=1, padx=10)
    
    tk.Button(
        button_frame,
        text="🔍 Detect (YOLOv11)",
        command=lambda: run_detection(detect_single_image_v11, panel, status_label, "YOLOv11"),
        bg=ACCENT_BG,
        fg=TEXT_FG,
        font=("Arial", 12, "bold"),
        padx=15,
        pady=8,
        relief="flat"
    ).grid(row=0, column=2, padx=10)

    tk.Button(
        button_frame,
        text="🔄 Clear",
        command=lambda: clear_detection(panel, status_label),
        bg=ACCENT_BG,
        fg=TEXT_FG,
        font=("Arial", 12, "bold"),
        padx=15,
        pady=8,
        relief="flat"
    ).grid(row=0, column=3, padx=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()