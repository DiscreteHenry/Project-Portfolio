"""
Tkinter GUI for the Crystal Counter.
Dropdowns for slide type & magnification, file/folder selection,
an output-folder picker, threaded analysis, an image verification
viewer with boxes/outlines, and CSV export.
"""

import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

import crystal_core as core

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


class CrystalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crystal Counter")
        self.geometry("720x820")
        self.minsize(640, 700)

        self.results = []
        self.overlay_paths = []
        self.current_idx = 0
        self.selected_paths = []
        self.output_dir = core.default_output_dir()
        self._tk_img = None  # keep reference to prevent garbage collection
        self.msg_queue = queue.Queue()

        self._build_ui()
        self.after(100, self._drain_queue)

    # ---------------- UI construction ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # Slide type
        ttk.Label(self, text="Hemocytometer type:").pack(anchor="w", **pad)
        self.slide_var = tk.StringVar(value=list(core.HEMOCYTOMETER_SPECS)[0])
        ttk.Combobox(self, textvariable=self.slide_var,
                     values=list(core.HEMOCYTOMETER_SPECS),
                     state="readonly").pack(fill="x", **pad)

        # Magnification
        ttk.Label(self, text="Magnification:").pack(anchor="w", **pad)
        self.mag_var = tk.StringVar(value=list(core.MAGNIFICATION_DIAMETERS)[0])
        ttk.Combobox(self, textvariable=self.mag_var,
                     values=list(core.MAGNIFICATION_DIAMETERS),
                     state="readonly").pack(fill="x", **pad)

        # Display mode toggles
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", **pad)
        self.box_var = tk.BooleanVar(value=True)
        self.outline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_frame, text="Draw boxes",
                        variable=self.box_var).pack(side="left", padx=6)
        ttk.Checkbutton(mode_frame, text="Draw outlines",
                        variable=self.outline_var).pack(side="left", padx=6)

        # Input file / folder selection
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)
        ttk.Button(btn_frame, text="Choose Image(s)...",
                   command=self.choose_files).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(btn_frame, text="Choose Folder...",
                   command=self.choose_folder).pack(side="left", expand=True, fill="x", padx=4)

        self.path_label = ttk.Label(self, text="No images selected.", foreground="gray")
        self.path_label.pack(anchor="w", **pad)

        # --- Output folder picker ---
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)
        out_frame = ttk.Frame(self)
        out_frame.pack(fill="x", **pad)
        ttk.Label(out_frame, text="Output folder:").pack(side="left")
        ttk.Button(out_frame, text="Change...",
                   command=self.choose_output_dir).pack(side="right")
        self.output_label = ttk.Label(self, text=self.output_dir,
                                       foreground="blue", wraplength=680, justify="left")
        self.output_label.pack(anchor="w", **pad)

        # Run
        self.run_btn = ttk.Button(self, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(fill="x", **pad)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", **pad)

        # Preview area
        self.preview_label = ttk.Label(self, anchor="center",
                                       text="Verification preview will appear here.")
        self.preview_label.pack(fill="both", expand=True, **pad)

        nav = ttk.Frame(self)
        nav.pack(fill="x", **pad)
        self.prev_btn = ttk.Button(nav, text="< Prev", command=self.show_prev, state="disabled")
        self.prev_btn.pack(side="left")
        self.img_counter = ttk.Label(nav, text="")
        self.img_counter.pack(side="left", expand=True)
        self.next_btn = ttk.Button(nav, text="Next >", command=self.show_next, state="disabled")
        self.next_btn.pack(side="right")

        # Log
        self.log = tk.Text(self, height=7, state="disabled")
        self.log.pack(fill="both", expand=False, **pad)

        # Export
        self.export_btn = ttk.Button(self, text="Export CSV...",
                                     command=self.export_csv, state="disabled")
        self.export_btn.pack(fill="x", **pad)

    # ---------------- Logging ----------------
    def _log(self, text):
        self.msg_queue.put(text)

    def _drain_queue(self):
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.after(100, self._drain_queue)

    # ---------------- File selection ----------------
    def choose_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if paths:
            self.selected_paths = list(paths)
            self.path_label.config(text=f"{len(self.selected_paths)} image(s) selected.")

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_paths = [
                os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith(IMAGE_EXTS)
            ]
            self.path_label.config(
                text=f"{len(self.selected_paths)} image(s) found in folder.")

    def choose_output_dir(self):
        folder = filedialog.askdirectory(initialdir=self.output_dir)
        if folder:
            self.output_dir = folder
            self.output_label.config(text=self.output_dir)

    # ---------------- Analysis ----------------
    def run_analysis(self):
        if not self.selected_paths:
            messagebox.showwarning("No images", "Please select image(s) or a folder first.")
            return
        if not self.box_var.get() and not self.outline_var.get():
            messagebox.showwarning("Display mode",
                                   "Enable at least one of 'Draw boxes' or 'Draw outlines'.")
            return
        if not self.output_dir:
            messagebox.showwarning("Output folder", "Please choose an output folder.")
            return

        # Verify we can create/write to the output folder
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Output folder",
                                 f"Cannot create output folder:\n{e}")
            return

        self.run_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")
        self.progress.start()
        self.results = []
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            overlay_folder = os.path.join(self.output_dir, "verification_images")
            labels_folder = os.path.join(self.output_dir, "automated_labels")

            self._log(f"Output folder: {self.output_dir}")
            self._log("Loading model (first run may take a moment)...")
            model = core.get_model(force_cpu=True)
            slide = self.slide_var.get()
            mag = self.mag_var.get()
            draw_boxes = self.box_var.get()
            draw_outlines = self.outline_var.get()

            for path in self.selected_paths:
                self._log(f"Processing {os.path.basename(path)} ...")
                res = core.process_image(
                    path, slide, mag, model,
                    overlay_folder=overlay_folder,
                    labels_folder=labels_folder,
                    draw_boxes=draw_boxes, draw_outlines=draw_outlines)
                if res is None:
                    self._log("  -> Skipped (unreadable or bad config).")
                    continue
                self.results.append(res)
                self._log(f"  -> {res['Crystal_Count']} crystals | "
                          f"{res['Concentration_Crystals_per_mL']:.2e} /mL")

            self._log(f"\nDone. {len(self.results)} image(s) processed.")
            self._log(f"Overlays saved to: {overlay_folder}")
        except Exception as e:
            self._log(f"ERROR: {e}")
        finally:
            self.after(0, self._finish)

    def _finish(self):
        self.progress.stop()
        self.run_btn.config(state="normal")
        if self.results:
            self.export_btn.config(state="normal")
            self.overlay_paths = [r["Overlay_Path"] for r in self.results if r["Overlay_Path"]]
            self.current_idx = 0
            if self.overlay_paths:
                self.prev_btn.config(state="normal")
                self.next_btn.config(state="normal")
                self._show_current()

    # ---------------- Preview viewer ----------------
    def _show_current(self):
        if not self.overlay_paths:
            return
        path = self.overlay_paths[self.current_idx]
        try:
            img = Image.open(path)
        except Exception as e:
            self._log(f"Could not open preview: {e}")
            return
        img.thumbnail((640, 420))
        self._tk_img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._tk_img, text="")
        res = self.results[self.current_idx]
        self.img_counter.config(
            text=f"{self.current_idx + 1}/{len(self.overlay_paths)}  "
                 f"({res['Filename']}: {res['Crystal_Count']} crystals)")

    def show_next(self):
        if self.overlay_paths:
            self.current_idx = (self.current_idx + 1) % len(self.overlay_paths)
            self._show_current()

    def show_prev(self):
        if self.overlay_paths:
            self.current_idx = (self.current_idx - 1) % len(self.overlay_paths)
            self._show_current()

    # ---------------- Export ----------------
    def export_csv(self):
        if not self.results:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir=self.output_dir,
            initialfile="crystal_density_results.csv",
            filetypes=[("CSV", "*.csv")])
        if path:
            core.export_results(self.results, path)
            messagebox.showinfo("Saved", f"Results saved to:\n{path}")


if __name__ == "__main__":
    CrystalApp().mainloop()