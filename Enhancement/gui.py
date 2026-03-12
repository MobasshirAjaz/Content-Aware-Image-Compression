# gui.py — Enhancement Pipeline GUI
# Mirrors Compression/gui.py but runs the Enhancement pipeline:
# LLM subject detection → segmentation → transparency → background enhancement → merge

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import shutil
import time
import threading
import subprocess
import sys

# --- Dependency imports with error handling ---
try:
    from PIL import Image, ImageTk
except ImportError:
    print(
        "Error: Pillow is not installed.\n"
        "Install it with: pip install pillow"
    )
    sys.exit(1)

# --- Local pipeline modules ---
try:
    from LLM import run_on_latest
except (ImportError, Exception):
    run_on_latest = None

try:
    import segmentation
except (ImportError, Exception) as e:
    print(f"--- FAILED TO IMPORT SEGMENTATION ---")
    print(f"Error: {e}")
    print(f"------------------------------------")
    segmentation = None

try:
    import make_transparent
except (ImportError, Exception):
    make_transparent = None


class EnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Content-Aware Background Enhancement")

        # Upload button
        self.upload_btn = tk.Button(
            root, text="Upload Image", command=self.upload_image
        )
        self.upload_btn.pack(pady=10)

        # Slider section
        self.canvas = tk.Canvas(root, width=600, height=400, bg="lightgray")
        self.canvas.pack()

        self.slider = tk.Scale(
            root, from_=0, to=100, orient="horizontal",
            command=self.update_slider
        )
        self.slider.pack(fill="x", padx=20, pady=10)

        # Label for sizes
        self.size_label = tk.Label(
            root, text="Upload an image to see sizes", font=("Arial", 12)
        )
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

        # Placeholder for after image (initially same as before)
        self.after_img = self.compute_after_image(self.before_img)
        self.after_img.save("temp_after.jpg", "JPEG")
        self.after_size = os.path.getsize("temp_after.jpg")

        # Convert to Tk
        self.tk_before = ImageTk.PhotoImage(self.before_img)
        self.tk_after_full = ImageTk.PhotoImage(self.after_img)

        # Clear canvas and add images
        self.canvas.delete("all")
        self.canvas_before = self.canvas.create_image(
            0, 0, anchor="nw", image=self.tk_before
        )
        self.canvas_after = self.canvas.create_image(
            0, 0, anchor="nw", image=self.tk_after_full
        )

        # Start slider
        self.slider.set(50)
        self.update_slider(50)
        self.update_size_label()

        # --- Get user input in the main thread BEFORE starting the worker ---
        subjects = None
        if run_on_latest is not None:
            try:
                subjects = run_on_latest(min_confidence=0.75)
            except Exception:
                subjects = None

        if not subjects:
            prompt = simpledialog.askstring(
                "Subjects",
                "Enter comma-separated subject names (leave blank for dummy masks):"
            )
            if prompt:
                subjects = [s.strip() for s in prompt.split(",") if s.strip()]
            else:
                subjects = []

        # Start the background pipeline
        worker = threading.Thread(
            target=self.run_full_pipeline,
            args=(stored_path, subjects),
            daemon=True
        )
        worker.start()

    def run_full_pipeline(self, stored_path: str, subjects: list):
        """Run segmentation → transparency → enhancement → merge in background."""

        use_dummy = not subjects
        if use_dummy:
            subjects = ["subject1", "subject2", "subject3"]

        # ---------------------------------------------------------------
        # STEP 1: Segmentation
        # ---------------------------------------------------------------
        if segmentation is not None:
            try:
                segmentation.generate_masks(
                    stored_path, subjects, use_dummy=use_dummy
                )
            except Exception as e:
                message = f"Segmentation failed: {e}"
                try:
                    self.root.after(
                        0, lambda: messagebox.showerror("Segmentation Error", message)
                    )
                except Exception:
                    print(message)
                return
        else:
            print("Segmentation module not available. Skipping segmentation.")

        # ---------------------------------------------------------------
        # STEP 2: Create transparent images from BW masks
        # ---------------------------------------------------------------
        if make_transparent is not None:
            try:
                make_transparent.process_image(stored_path)
            except Exception as e:
                message = f"Transparency generation failed: {e}"
                try:
                    self.root.after(
                        0, lambda: messagebox.showerror("Transparency Error", message)
                    )
                except Exception:
                    print(message)
                return
        else:
            print("make_transparent module not available. Skipping.")

        msg = "Segmentation & transparency completed"
        if use_dummy:
            msg += " (used dummy masks)."
        msg += "\n\nNow starting background enhancement..."

        try:
            self.root.after(
                0, lambda: messagebox.showinfo("Pipeline Step 1/3 Done", msg)
            )
        except Exception:
            print(msg)

        # ---------------------------------------------------------------
        # STEP 3: Enhance background with Real-ESRGAN
        # ---------------------------------------------------------------
        try:
            import enhance_background
            enhance_background.run()
        except Exception as e:
            message = f"Background enhancement failed: {e}"
            try:
                self.root.after(
                    0, lambda: messagebox.showerror("Enhancement Failed", message)
                )
            except Exception:
                print(message)
            return

        enhance_msg = "Background enhancement finished successfully!\n\nNow starting final merge..."
        try:
            self.root.after(
                0, lambda: messagebox.showinfo("Pipeline Step 2/3 Done", enhance_msg)
            )
        except Exception:
            print(enhance_msg)

        # ---------------------------------------------------------------
        # STEP 4: Merge enhanced background with foregrounds
        # ---------------------------------------------------------------
        try:
            import merge_enhanced
            merge_enhanced.run()
        except Exception as e:
            message = f"Merge failed: {e}"
            try:
                self.root.after(
                    0, lambda: messagebox.showerror("Merge Failed", message)
                )
            except Exception:
                print(message)
            return

        # --- Update the after image with the actual result ---
        self._update_after_image()

        success_msg = "Pipeline fully completed!\n\nEnhanced image merged successfully!"
        try:
            self.root.after(
                0, lambda: messagebox.showinfo("Pipeline 3/3 Complete!", success_msg)
            )
        except Exception:
            print(success_msg)

    def _update_after_image(self):
        """Try to load the final enhanced image and update the canvas."""
        try:
            base_dir = os.path.dirname(__file__)
            final_dir = os.path.join(base_dir, "final_enhanced_output")

            if not os.path.isdir(final_dir):
                return

            # Find the latest enhanced JPG
            jpg_files = [
                f for f in os.listdir(final_dir)
                if f.lower().endswith(('_enhanced.jpg', '_enhanced.png'))
            ]
            if not jpg_files:
                return

            latest = max(
                jpg_files,
                key=lambda f: os.path.getmtime(os.path.join(final_dir, f))
            )
            latest_path = os.path.join(final_dir, latest)

            self.after_img = Image.open(latest_path).convert("RGB").resize((600, 400))
            self.after_size = os.path.getsize(latest_path)

            # Schedule the UI update on the main thread
            def _do_update():
                self.tk_after_full = ImageTk.PhotoImage(self.after_img)
                self.canvas.itemconfig(self.canvas_after, image=self.tk_after_full)
                self.update_slider(self.slider.get())
                self.update_size_label()

            self.root.after(0, _do_update)

        except Exception as e:
            print(f"Could not update after image: {e}")

    def compute_after_image(self, img: Image.Image) -> Image.Image:
        """Placeholder — returns a copy until the pipeline produces the real output."""
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
        self.tk_after_cropped = tk_cropped  # Keep reference

    def update_size_label(self):
        def fmt_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 ** 2:
                return f"{size_bytes / 1024:.2f} KB"
            else:
                return f"{size_bytes / 1024 ** 2:.2f} MB"

        before_str = fmt_size(self.before_size)
        after_str = fmt_size(self.after_size)
        self.size_label.config(
            text=f"Before size: {before_str}   →   After size: {after_str}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancementApp(root)
    root.mainloop()
