import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import shutil
import time
import threading
from tkinter import simpledialog

# local pipeline modules
try:
    from LLM import run_on_latest
except Exception:
    run_on_latest = None

try:
    import segmentation
except Exception:
    segmentation = None

try:
    import make_transparent
except Exception:
    make_transparent = None


class BeforeAfterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Before & After Image Slider")

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
            # fallback: try to save via PIL
            img_tmp = Image.open(filepath)
            img_tmp.save(stored_path)

        # Load before image from the stored copy
        self.before_img = Image.open(stored_path).convert("RGB").resize((600, 400))
        self.before_size = os.path.getsize(stored_path)  # in bytes

        # ---- Computation step (placeholder: same as before) ----
        self.after_img = self.compute_after_image(self.before_img)
        # Save temp after image to measure size (kept in repo root)
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

        # Update size label
        self.update_size_label()

        # Start full pipeline in background to avoid freezing the GUI
        worker = threading.Thread(target=self.run_full_pipeline, args=(stored_path,), daemon=True)
        worker.start()

    def run_full_pipeline(self, stored_path: str):
        """Run LLM -> segmentation -> transparent generation for the stored image.

        The function tries to use the LLM first. If LLM is unavailable, prompts
        the user for a comma-separated subject list. If segmentation real models
        are unavailable, it falls back to dummy masks.
        """
        # 1) Get subject list from LLM if available
        subjects = None
        if run_on_latest is not None:
            try:
                subjects = run_on_latest(min_confidence=0.75)
            except Exception as e:
                # LLM not available or failed; we'll prompt the user below
                subjects = None

        if not subjects:
            # Ask user for subjects; allow empty to use dummy
            prompt = simpledialog.askstring("Subjects", "Enter comma-separated subject names (leave blank to use dummy masks):")
            if prompt:
                subjects = [s.strip() for s in prompt.split(",") if s.strip()]
            else:
                # fallback dummy subjects
                subjects = ["subject1", "subject2", "subject3"]

        # 2) Run segmentation
        used_dummy = False
        if segmentation is not None:
            try:
                segmentation.generate_masks(stored_path, subjects, use_dummy=False)
            except Exception as e:
                # If segmentation pipeline not available or raises, fallback to dummy
                try:
                    segmentation.generate_masks(stored_path, subjects, use_dummy=True)
                    used_dummy = True
                except Exception as e2:
                    # both attempts failed
                    message = f"Segmentation failed: {e}\n{e2}"
                    try:
                        messagebox.showerror("Segmentation error", message)
                    except Exception:
                        print(message)
                    return
        else:
            # segmentation module not importable: nothing to run
            try:
                messagebox.showwarning("Segmentation missing", "Segmentation module not available. Skipping segmentation.")
            except Exception:
                print("Segmentation module not available. Skipping segmentation.")

        # 3) Create transparent images from bw masks
        if make_transparent is not None:
            try:
                make_transparent.process_all_for_image(stored_path)
            except Exception as e:
                try:
                    messagebox.showerror("Transparency generation failed", str(e))
                except Exception:
                    print("Transparency generation failed:", e)
                return
        else:
            try:
                messagebox.showwarning("Transparency missing", "make_transparent module not available. Skipping transparent image generation.")
            except Exception:
                print("make_transparent module not available. Skipping transparent image generation.")

        # Notify user of completion
        try:
            msg = "Pipeline completed"
            if used_dummy:
                msg += " (used dummy segmentation masks)"
            messagebox.showinfo("Done", msg)
        except Exception:
            print("Pipeline completed")

    def compute_after_image(self, img: Image.Image) -> Image.Image:
        """
        Placeholder for your image computation logic.
        Replace this with your actual processing.
        """
        # Example: currently just returns the same image
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

        # Update canvas
        self.canvas.itemconfig(self.canvas_before, image=self.tk_before)
        self.canvas.itemconfig(self.canvas_after, image=tk_cropped)

        # Keep reference
        self.tk_after_cropped = tk_cropped

    def update_size_label(self):
        def fmt_size(size_bytes):
            # Convert bytes to KB/MB for readability
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
