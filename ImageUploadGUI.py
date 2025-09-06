import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


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

        # Load before image
        self.before_img = Image.open(filepath).convert("RGB").resize((600, 400))
        self.before_size = os.path.getsize(filepath)  # in bytes

        # ---- Computation step (placeholder: same as before) ----
        self.after_img = self.compute_after_image(self.before_img)
        # Save temp after image to measure size
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
