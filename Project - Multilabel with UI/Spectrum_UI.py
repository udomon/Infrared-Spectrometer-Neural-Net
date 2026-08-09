"""Desktop interface for IR functional-group predictions from plotted spectra."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

LABELS = ["carbonyl", "hydroxyl", "amine", "nitrile", "aromatic", "alcohol", "carboxylic acid", "alkene", "C-O"]
GRID = np.linspace(400, 4000, 1000)

class SpectrumApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IR Spectrum Functional-Group Predictor")
        self.geometry("1200x780")
        self.minsize(1000, 650)
        self.image_path = self.source_image = self.photo = None
        self.scale = 1.0
        self.points, self.trace_pixels = [], None
        self.model = None
        self.thresholds = {label: 0.50 for label in LABELS}
        self._build()

    def _build(self):
        controls = ttk.Frame(self, padding=12); controls.pack(side=tk.LEFT, fill=tk.Y)
        viewer = ttk.Frame(self, padding=(0, 12, 12, 12)); viewer.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        ttk.Label(controls, text="IR Spectrum Predictor", font=("Segoe UI", 16, "bold")).pack(anchor=tk.W)
        ttk.Label(controls, text="1. Load a plotted IR spectrum image\n2. Click the four calibration points\n3. Predict functional groups", justify=tk.LEFT).pack(anchor=tk.W, pady=(5, 14))
        ttk.Button(controls, text="Load spectrum image", command=self.load_image).pack(fill=tk.X)
        self.image_name = tk.StringVar(value="No image selected")
        ttk.Label(controls, textvariable=self.image_name, wraplength=270, foreground="#555").pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(controls, text="Reset calibration", command=self.reset_points).pack(fill=tk.X, pady=(7, 0))
        ttk.Separator(controls).pack(fill=tk.X, pady=14)
        ttk.Label(controls, text="Click in this order:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.instructions = tk.StringVar(value="Load an image to begin.")
        ttk.Label(controls, textvariable=self.instructions, wraplength=270, justify=tk.LEFT).pack(anchor=tk.W, pady=(5, 10))
        ttk.Label(controls, text="Dark-pixel threshold (curve extraction)").pack(anchor=tk.W)
        self.pixel_threshold = tk.IntVar(value=160)
        ttk.Scale(controls, from_=30, to=245, variable=self.pixel_threshold, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(controls, text="Lower values select only darker pixels.\nRaise it if the trace is faint.", wraplength=270).pack(anchor=tk.W, pady=(3, 10))
        self.threshold_label = ttk.Label(controls, text="Per-group thresholds load from thresholds.json.\nFallback: 0.50 for every group.", wraplength=270)
        self.threshold_label.pack(anchor=tk.W, pady=(3, 12))
        ttk.Button(controls, text="Digitise and predict", command=self.predict).pack(fill=tk.X, pady=(4, 12))
        ttk.Label(controls, text="Predictions", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        self.results = ttk.Treeview(controls, columns=("group", "probability", "threshold", "result"), show="headings", height=10)
        for name, width in (("group", 110), ("probability", 70), ("threshold", 68), ("result", 52)):
            self.results.heading(name, text=name.title()); self.results.column(name, width=width, anchor=tk.CENTER)
        self.results.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(viewer, background="#f4f4f4", highlightthickness=1, highlightbackground="#bbb")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def load_image(self):
        filename = filedialog.askopenfilename(title="Select IR spectrum image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not filename: return
        try: self.source_image = Image.open(filename).convert("RGB")
        except Exception as exc: messagebox.showerror("Image error", f"Could not open this image:\n{exc}"); return
        self.image_path = Path(filename)
        self.image_name.set(f"Loaded image: {self.image_path.name}")
        self.reset_points(); self.after_idle(self.render_image)

    def on_canvas_resize(self, _event):
        if self.source_image: self.render_image()

    def render_image(self):
        # Always fit to the current canvas: the image grows and shrinks with the window.
        width, height = max(1, self.canvas.winfo_width() - 4), max(1, self.canvas.winfo_height() - 4)
        self.scale = min(width / self.source_image.width, height / self.source_image.height)
        shown = self.source_image.resize((max(1, round(self.source_image.width * self.scale)), max(1, round(self.source_image.height * self.scale))), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(shown); self.canvas.delete("all"); self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.redraw_overlays()

    def redraw_overlays(self):
        for number, (x, y) in enumerate(self.points, 1):
            sx, sy = x * self.scale, y * self.scale
            self.canvas.create_oval(sx-5, sy-5, sx+5, sy+5, fill="#e63946", outline="white", width=2, tags="marker")
            self.canvas.create_text(sx+11, sy-11, text=str(number), fill="#e63946", font=("Segoe UI", 12, "bold"), tags="marker")
        if self.trace_pixels is not None: self.draw_trace(*self.trace_pixels)

    def reset_points(self):
        self.points, self.trace_pixels = [], None
        self.instructions.set("Click 1: 4000 cm-1 on the x-axis (left edge)." if self.source_image else "Load an image to begin.")
        if self.source_image: self.render_image()

    def add_point(self, event):
        if not self.photo or len(self.points) >= 4: return
        self.points.append((event.x / self.scale, event.y / self.scale)); self.redraw_overlays()
        prompts = ["Click 2: 400 cm-1 on the x-axis (right edge).", "Click 3: 100% transmittance on the y-axis (top edge).", "Click 4: 0% transmittance on the y-axis (bottom edge).", "Calibration complete. Click Digitise and predict."]
        self.instructions.set(prompts[len(self.points)-1])

    def digitise(self):
        if not self.image_path or len(self.points) != 4: raise ValueError("Load an image and select all four calibration points first.")
        (left, _), (right, _), (_, first_y), (_, second_y) = self.points
        top, bottom = sorted((first_y, second_y))
        if abs(right-left) < 10 or abs(top-bottom) < 10: raise ValueError("Calibration points are too close together.")
        image = cv2.imread(str(self.image_path))
        if image is None: raise ValueError("OpenCV could not read the selected image.")
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
        x0, x1 = sorted((round(left), round(right))); y0, y1 = sorted((round(top), round(bottom)))
        # Exclude the axes themselves.  In grayscale mode the left y-axis can
        # otherwise become the first "curve" column and derail the tracker.
        x0, x1 = max(0, x0 + 12), min(grey.shape[1], x1 - 12)
        y0, y1 = max(0, y0 + 5), min(grey.shape[0], y1 - 5)
        # Most spectra are plotted in a colour (as in the supplied red trace).  This
        # deliberately ignores black axes, text, ticks and gridlines.  If the image
        # is monochrome, fall back to dark pixels but follow a smooth local path so a
        # vertical gridline cannot pull the trace to the middle of the plot.
        coloured = saturation[y0:y1, x0:x1] >= 60
        # Use colour only when it represents virtually the whole trace.  Some
        # teaching spectra highlight one diagnostic absorption (for example the
        # acetone carbonyl in red) while leaving the actual trace grey.
        coloured_columns = np.count_nonzero(coloured.any(axis=0))
        use_coloured_trace = coloured_columns >= int((x1 - x0) * 0.85)
        mask = coloured if use_coloured_trace else grey[y0:y1, x0:x1] < self.pixel_threshold.get()
        xs, ys, previous_y = [], [], None
        for relative_x in range(mask.shape[1]):
            candidates = np.where(mask[:, relative_x])[0].astype(float) + y0
            if not candidates.size:
                continue
            if previous_y is None:
                selected_y = float(np.median(candidates))
            elif use_coloured_trace:
                selected_y = float(np.median(candidates))
            else:
                # For a black trace choose the candidate nearest to the recent curve.
                selected_y = float(candidates[np.argmin(np.abs(candidates - previous_y))])
            xs.append(x0 + relative_x); ys.append(selected_y); previous_y = selected_y
        if len(xs) < 40: raise ValueError("Too few curve pixels were found. Raise the dark-pixel threshold and try again.")
        xs, ys = np.asarray(xs), np.asarray(ys); self.trace_pixels = (xs, ys); self.draw_trace(xs, ys)
        # A normal transmittance plot is 100% at the top and 0% at the bottom.
        return 4000 + (xs-left)*(400-4000)/(right-left), (bottom-ys)/(bottom-top)

    def draw_trace(self, xs, ys):
        self.canvas.delete("trace"); points = [value for x, y in zip(xs, ys) for value in (x*self.scale, y*self.scale)]
        if len(points) >= 4: self.canvas.create_line(*points, fill="#00a6fb", width=1, tags="trace")

    def load_model(self):
        if self.model is not None: return self.model
        project = Path(__file__).resolve().parent; sys.path.insert(0, str(project)); from Models import SpectrumCNN
        weights = next((p for p in (project/"trained_weights.pt", project.parent/"trained_weights.pt") if p.is_file()), None)
        if weights is None: raise FileNotFoundError("trained_weights.pt was not found next to this app or in its parent folder.")
        self.model = SpectrumCNN(num_labels=len(LABELS)); self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)); self.model.eval(); self.load_thresholds(project)
        return self.model

    def load_thresholds(self, project):
        path = next((p for p in (project/"thresholds.json", project.parent/"thresholds.json") if p.is_file()), None)
        if path is None: self.threshold_label.config(text="No thresholds.json found: using 0.50 for every group."); return
        try:
            with path.open(encoding="utf-8") as file: saved = json.load(file)
            self.thresholds = {label: float(saved.get(label, 0.50)) for label in LABELS}; self.threshold_label.config(text=f"Using per-group validation thresholds from {path.name}.")
        except (OSError, ValueError, json.JSONDecodeError) as exc: self.threshold_label.config(text=f"Could not read thresholds.json ({exc}); using 0.50 fallback.")

    def predict(self):
        try:
            x_data, y_data = self.digitise(); model = self.load_model(); order = np.argsort(x_data); signal = np.interp(GRID, x_data[order], y_data[order]); span = signal.max()-signal.min()
            if span > 0: signal = (signal-signal.min())/span
            with torch.no_grad(): probabilities = torch.sigmoid(model(torch.tensor(signal, dtype=torch.float32).unsqueeze(0))).squeeze().tolist()
            self.results.delete(*self.results.get_children())
            for label, probability in zip(LABELS, probabilities):
                cutoff = self.thresholds[label]; self.results.insert("", tk.END, values=(label, f"{probability:.1%}", f"{cutoff:.3f}", "YES" if probability >= cutoff else "No"))
        except Exception as exc: messagebox.showerror("Prediction could not run", str(exc))

if __name__ == "__main__": SpectrumApp().mainloop()
