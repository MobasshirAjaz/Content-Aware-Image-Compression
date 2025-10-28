import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import shutil
import time
import threading

# --- Local pipeline modules ---
# It's good practice to handle potential import errors.
try:
    from LLM import run_on_latest
except (ImportError, Exception):
    run_on_latest = None

try:
    import segmentation
except (ImportError, Exception) as e:
    # This will print the true reason the import is failing
    print(f"--- FAILED TO IMPORT SEGMENTATION ---")
    print(f"Error: {e}")
    print(f"------------------------------------")
    segmentation = None

try:
    import make_transparent
except (ImportError, Exception):
    make_transparent = None


class BeforeAfterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Content-Aware Image Compression")

        # Upload button
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        # Slider section
        self.canvas = tk.Canvas(root, width=600, height=400, bg="lightgray")
        self.canvas.pack()

        self.slider = tk.Scale(root, from_=0, to=100, orient="horizontal", command=self.update_slider)
        self.slider.pack(fill="x", padx=20, pady=10)

        # Label for sizes
        self.size_label = tk.Label(root, text="Upload an image to see sizes", font=("Arial", 12))
        self.size_label.pack(pady=5)

        # Placeholders
        self.before_img = None
        self.after_img = None
        self.tk_before = None
        self.tk_after_full = None
        self.canvas_before = None
        self.canvas_after = None

        self.before_size = 0
        self.after_size = 0

    def upload_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not filepath:
            return

        # Ensure input_images directory exists and copy the uploaded image there
        base_dir = os.path.dirname(__file__)
        input_dir = os.path.join(base_dir, "input_images")
        os.makedirs(input_dir, exist_ok=True)

        original_name = os.path.basename(filepath)
        name, ext = os.path.splitext(original_name)
        timestamp = int(time.time())
        stored_name = f"{name}_{timestamp}{ext}"
        stored_path = os.path.join(input_dir, stored_name)
        
        try:
            shutil.copy2(filepath, stored_path)
        except Exception:
            img_tmp = Image.open(filepath)
            img_tmp.save(stored_path)

        # Load before image from the stored copy
        self.before_img = Image.open(stored_path).convert("RGB").resize((600, 400))
        self.before_size = os.path.getsize(stored_path)

        # Placeholder for after image
        self.after_img = self.compute_after_image(self.before_img)
        self.after_img.save("temp_after.jpg", "JPEG")
        self.after_size = os.path.getsize("temp_after.jpg")

        # Convert to Tk
        self.tk_before = ImageTk.PhotoImage(self.before_img)
        self.tk_after_full = ImageTk.PhotoImage(self.after_img)

        # Clear canvas and add images
        self.canvas.delete("all")
        self.canvas_before = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_before)
        self.canvas_after = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_after_full)

        # Start slider
        self.slider.set(50)
        self.update_slider(50)
        self.update_size_label()

        # --- CORRECTED LOGIC ---
        # 1. Get user input in the main thread BEFORE starting the worker.
        subjects = None
        if run_on_latest is not None:
            try:
                subjects = run_on_latest(min_confidence=0.75)
            except Exception:
                subjects = None

        if not subjects:
            # This is now safe because it's called from the main GUI thread.
            prompt = simpledialog.askstring("Subjects", "Enter comma-separated subject names (leave blank for dummy masks):")
            if prompt:
                subjects = [s.strip() for s in prompt.split(",") if s.strip()]
            else:
                # Use an empty list to signal that dummy masks should be used.
                subjects = []

        # 2. Start the background pipeline and pass the subjects to it.
        worker = threading.Thread(
            target=self.run_full_pipeline, 
            args=(stored_path, subjects), # <<< Pass subjects to the thread
            daemon=True
        )
        worker.start()

    # <<< MODIFIED function signature to accept subjects
    def run_full_pipeline(self, stored_path: str, subjects: list):
        """Run segmentation and transparency generation in the background."""
        
        # We now receive the subjects directly. If the list is empty, we use dummy data.
        use_dummy = not subjects
        if use_dummy:
            subjects = ["subject1", "subject2", "subject3"]

        # 1. Run segmentation
        if segmentation is not None:
            try:
                segmentation.generate_masks(stored_path, subjects, use_dummy=use_dummy)
            except Exception as e:
                message = f"Segmentation failed: {e}"
                try:
                    # Use messagebox from the main thread if possible, but print as a fallback.
                    self.root.after(0, lambda: messagebox.showerror("Segmentation Error", message))
                except Exception:
                    print(message)
                return
        else:
            print("Segmentation module not available. Skipping segmentation.")

        # 2. Create transparent images from bw masks
        if make_transparent is not None:
            try:
                make_transparent.process_all_for_image(stored_path)
            except Exception as e:
                message = f"Transparency generation failed: {e}"
                try:
                    self.root.after(0, lambda: messagebox.showerror("Transparency Error", message))
                except Exception:
                    print(message)
                return
        else:
            print("make_transparent module not available. Skipping.")

        # 3. Notify user of completion
        msg = "Pipeline completed"
        if use_dummy:
            msg += " (used dummy segmentation masks)"
        
        try:
            # Schedule the final message box to run on the main thread.
            self.root.after(0, lambda: messagebox.showinfo("Done", msg))
        except Exception:
            print(msg)

    def compute_after_image(self, img: Image.Image) -> Image.Image:
        """Placeholder for your image computation logic."""
        return img.copy()

    def update_slider(self, value):
        if self.before_img is None or self.after_img is None:
            return

        value = int(value)
        width, height = self.before_img.size
        mask_width = int((value / 100) * width)

        # Crop after image
        cropped_after = self.after_img.crop((0, 0, mask_width, height))
        tk_cropped = ImageTk.PhotoImage(cropped_after)

        self.canvas.itemconfig(self.canvas_before, image=self.tk_before)
        self.canvas.itemconfig(self.canvas_after, image=tk_cropped)
        self.tk_after_cropped = tk_cropped # Keep reference

    def update_size_label(self):
        def fmt_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.2f} KB"
            else:
                return f"{size_bytes/1024**2:.2f} MB"

        before_str = fmt_size(self.before_size)
        after_str = fmt_size(self.after_size)
        self.size_label.config(text=f"Before size: {before_str}   →   After size: {after_str}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BeforeAfterApp(root)
    root.mainloop()