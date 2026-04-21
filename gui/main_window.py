import sys
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading
import cv2
import webbrowser
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.yolo_model import YOLOModelManager


class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Marine Lumber Co. - Detector Pro")
        self.root.geometry("1150x800")
        self.root.minsize(950, 650)
        self.root.configure(bg="#4a4a4a")
        
        self.model_manager = YOLOModelManager()
        self.model_path = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.conf_value = tk.DoubleVar(value=0.25)
        self.slice_value = tk.IntVar(value=640)
        self.is_running = False
        
        self._build_ui()
        self.root.after(500, self._auto_load)
    
    def _build_ui(self):
        # HEADER
        header = tk.Frame(self.root, bg="white", height=110)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        logo_frame = tk.Frame(header, bg="white")
        logo_frame.pack(side=tk.LEFT, padx=25, pady=15)
        try:
            from PIL import Image, ImageTk
            logo_path = Path(__file__).parent.parent / "assets" / "Logo.png"
            if logo_path.exists():
                img = Image.open(logo_path)
                img.thumbnail((120, 120), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                tk.Label(logo_frame, image=self.logo_img, bg="white").pack()
            else:
                raise FileNotFoundError
        except:
            tk.Label(logo_frame, text="[MLC]", font=("Arial Black", 22), 
                    bg="white", fg="#003366").pack()
        
        # 右侧占位块，宽度与左侧LOGO区域一致，确保标题绝对居中
        right_spacer = tk.Frame(header, bg="white", width=90)
        right_spacer.pack(side=tk.RIGHT, padx=25, pady=15)
        right_spacer.pack_propagate(False)
        
        center = tk.Frame(header, bg="white")
        center.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        center_inner = tk.Frame(center, bg="white")
        center_inner.pack(expand=True)
        tk.Label(center_inner, text="MARINE LUMBER CO.", font=("Arial", 28, "bold"),
                bg="white", fg="#003366").pack()
        tk.Label(center_inner, text="Industrial Wood Pile Detection System", 
                font=("Arial", 11), bg="white", fg="#333").pack()
        tk.Label(center_inner, text="Professional Edition v2.0", 
                font=("Arial", 9), bg="white", fg="#666").pack()
        
        # MAIN
        main = tk.Frame(self.root, bg="#4a4a4a")
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        
        # LEFT
        left = tk.LabelFrame(main, text=" CONFIGURATION ", bg="#4a4a4a", 
                            fg="white", font=("Arial", 11, "bold"))
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Model
        tk.Label(left, text="YOLO Model File (.pt):", bg="#4a4a4a", fg="white",
                font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=12, pady=(10,3))
        row_m = tk.Frame(left, bg="#4a4a4a")
        row_m.pack(fill=tk.X, padx=12)
        tk.Entry(row_m, textvariable=self.model_path, width=30, 
                font=("Arial", 10), bg="#e0e0e0", relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row_m, text="📂 Import", command=self._import_model,
                 bg="#d0d0d0", fg="black", font=("Arial", 9), relief=tk.RAISED).pack(side=tk.LEFT, padx=6)
        
        self.model_status = tk.Label(left, text="Auto-loading...", bg="#4a4a4a",
                                    fg="orange", font=("Arial", 11, "bold"))
        self.model_status.pack(anchor=tk.W, padx=12, pady=8)
        
        tk.Frame(left, bg="#666", height=2).pack(fill=tk.X, padx=10, pady=6)
        
        # Input
        tk.Label(left, text="Input Images Folder:", bg="#4a4a4a", fg="white",
                font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=12)
        row1 = tk.Frame(left, bg="#4a4a4a")
        row1.pack(fill=tk.X, padx=12)
        tk.Entry(row1, textvariable=self.input_dir, width=30,
                font=("Arial", 10), bg="#e0e0e0", relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row1, text="📂 Browse", command=self._browse_input,
                 bg="#d0d0d0", fg="black", font=("Arial", 9)).pack(side=tk.LEFT, padx=6)
        
        # Output
        tk.Label(left, text="Output Results Folder:", bg="#4a4a4a", fg="white",
                font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=12, pady=(10,0))
        row2 = tk.Frame(left, bg="#4a4a4a")
        row2.pack(fill=tk.X, padx=12)
        tk.Entry(row2, textvariable=self.output_dir, width=30,
                font=("Arial", 10), bg="#e0e0e0", relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row2, text="📂 Browse", command=self._browse_output,
                 bg="#d0d0d0", fg="black", font=("Arial", 9)).pack(side=tk.LEFT, padx=6)
        
        tk.Frame(left, bg="#666", height=2).pack(fill=tk.X, padx=10, pady=8)
        
        # Confidence
        tk.Label(left, text="Confidence Threshold (0.0 - 1.0):", bg="#4a4a4a",
                fg="white", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=12)
        conf_row = tk.Frame(left, bg="#4a4a4a")
        conf_row.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(conf_row, text="Current:", bg="#4a4a4a", fg="white",
                font=("Arial", 9)).pack(side=tk.LEFT)
        self.conf_display = tk.Label(conf_row, text="0.25", bg="#4a4a4a", 
                                    fg="#00CCFF", font=("Arial", 10, "bold"))
        self.conf_display.pack(side=tk.LEFT, padx=4)
        
        tk.Scale(left, from_=0.05, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.conf_value, bg="#4a4a4a", fg="white",
                highlightthickness=0, length=300, command=self._update_conf,
                troughcolor="#666", activebackground="#00CCFF").pack(padx=12, fill=tk.X)
        tk.Label(left, text="Tip: 0.15-0.30 recommended for best results", bg="#4a4a4a", 
                fg="#AAA", font=("Arial", 8)).pack(anchor=tk.W, padx=12)
        
        # Slice Size
        tk.Label(left, text="Slice Size (px):", bg="#4a4a4a", fg="white",
                font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=12, pady=(10,0))
        slice_row = tk.Frame(left, bg="#4a4a4a")
        slice_row.pack(fill=tk.X, padx=12)
        om = tk.OptionMenu(slice_row, self.slice_value, 240, 320, 480, 640, 800, 1280)
        om.config(bg="#e0e0e0", fg="black", font=("Arial", 10), width=8, relief=tk.FLAT)
        om.pack(side=tk.LEFT)
        
        # RIGHT
        right = tk.LabelFrame(main, text=" MONITORING ", bg="#4a4a4a",
                             fg="white", font=("Arial", 11, "bold"))
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status
        status_box = tk.Frame(right, bg="#3a3a3a", bd=1, relief=tk.SUNKEN)
        status_box.pack(fill=tk.X, padx=10, pady=8)
        self.sys_status = tk.Label(status_box, text="Initializing...", bg="#3a3a3a",
                                  fg="#00CCFF", font=("Arial", 16, "bold"))
        self.sys_status.pack(pady=8)
        
        # Progress
        prog_box = tk.Frame(status_box, bg="#3a3a3a")
        prog_box.pack(fill=tk.X, padx=10, pady=4)
        self.progress = ttk.Progressbar(prog_box, orient=tk.HORIZONTAL,
                                       length=380, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pct_label = tk.Label(prog_box, text="0%", bg="#3a3a3a", fg="#00CCFF",
                                 font=("Arial", 12, "bold"))
        self.pct_label.pack(side=tk.RIGHT, padx=(8,0))
        
        # Stats
        stats = tk.Frame(right, bg="#4a4a4a")
        stats.pack(fill=tk.X, padx=10, pady=6)
        
        stat_left = tk.Frame(stats, bg="#4a4a4a")
        stat_left.pack(side=tk.LEFT, fill=tk.Y)
        self.stat_images = tk.Label(stat_left, text="Images: 0", bg="#4a4a4a", 
                                   fg="white", font=("Arial", 10))
        self.stat_images.pack(anchor=tk.W, pady=2)
        self.stat_objects = tk.Label(stat_left, text="Objects: 0", bg="#4a4a4a", 
                                    fg="white", font=("Arial", 10))
        self.stat_objects.pack(anchor=tk.W, pady=2)
        
        stat_right = tk.Frame(stats, bg="#4a4a4a")
        stat_right.pack(side=tk.RIGHT, fill=tk.Y)
        self.stat_conf = tk.Label(stat_right, text="Avg Conf: -", bg="#4a4a4a", 
                                 fg="white", font=("Arial", 10))
        self.stat_conf.pack(anchor=tk.W, pady=2)
        self.stat_rate = tk.Label(stat_right, text="Rate: -", bg="#4a4a4a", 
                                 fg="white", font=("Arial", 10))
        self.stat_rate.pack(anchor=tk.W, pady=2)
        
        # Log
        tk.Label(right, text="Processing Log", bg="#4a4a4a", fg="white",
                font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(8,2))
        log_box = tk.Frame(right, bg="#1A1A1A", bd=1, relief=tk.SUNKEN)
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_box, bg="#1A1A1A", fg="#00FF66", 
                               font=("Courier", 9), wrap=tk.WORD, relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # BOTTOM
        bottom = tk.Frame(self.root, bg="#333333", height=65)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        bottom.pack_propagate(False)
        
        self.start_btn = tk.Button(bottom, text="▶  START DETECTION", command=self._start,
                 bg="#CCCCCC", fg="black", font=("Arial", 11, "bold"),
                 width=20, height=1, bd=2, relief=tk.RAISED)
        self.start_btn.pack(side=tk.LEFT, padx=25, pady=14)
        
        tk.Button(bottom, text="📂  OPEN OUTPUT", command=self._open_out,
                 bg="#CCCCCC", fg="black", font=("Arial", 10),
                 width=16, height=1, bd=2, relief=tk.RAISED).pack(side=tk.LEFT, padx=12, pady=14)
        
        tk.Button(bottom, text="❓  HELP GUIDE", command=self._help,
                 bg="#CCCCCC", fg="black", font=("Arial", 10),
                 width=16, height=1, bd=2, relief=tk.RAISED).pack(side=tk.RIGHT, padx=25, pady=14)
    
    def _update_conf(self, val):
        self.conf_display.config(text=f"{float(val):.2f}")
    
    def _time(self):
        return datetime.datetime.now().strftime("%H:%M:%S")
    
    def _log(self, msg):
        self.log_text.insert(tk.END, f"[{self._time()}] {msg}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def _auto_load(self):
        def load():
            try:
                success = self.model_manager.import_model("assets/best.pt", "default")
                if success:
                    self.model_status.config(text="Model Ready ✅", fg="#00CCFF")
                    self.sys_status.config(text="System Ready")
                    self._log("Model loaded: best.pt")
                else:
                    self.model_status.config(text="Auto-load Failed", fg="red")
                    self.sys_status.config(text="Load Error")
                    self._log("Auto-load failed")
            except Exception as e:
                self.model_status.config(text="Auto-load Failed", fg="red")
                self._log(f"Auto-load error: {e}")
        threading.Thread(target=load, daemon=True).start()
    
    def _import_model(self):
        path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("YOLO", "*.pt")])
        if not path:
            return
        self.model_path.set(path)
        self._log(f"Importing: {path}")
        self.model_status.config(text="Loading...", fg="orange")
        self.root.update_idletasks()
        def cb():
            try:
                success = self.model_manager.import_model(path, "custom")
                if success:
                    self.model_status.config(text="Model Ready ✅", fg="#00CCFF")
                    self.sys_status.config(text="Model Loaded")
                    self._log(f"Model imported: {Path(path).name}")
                else:
                    self.model_status.config(text="Load Failed ❌", fg="red")
                    self._log("Import failed")
            except Exception as e:
                self.model_status.config(text="Load Failed ❌", fg="red")
                self._log(f"Import error: {e}")
        threading.Thread(target=cb, daemon=True).start()
    
    def _browse_input(self):
        d = filedialog.askdirectory()
        if d: 
            self.input_dir.set(d)
            self._log(f"Input folder: {d}")
    
    def _browse_output(self):
        d = filedialog.askdirectory()
        if d: 
            self.output_dir.set(d)
            self._log(f"Output folder: {d}")
    
    def _open_out(self):
        out = self.output_dir.get()
        if out and Path(out).exists():
            os.system(f'open "{out}"')
    
    def _help(self):
        help_path = Path(__file__).parent.parent / "assets" / "help.html"
        if help_path.exists():
            webbrowser.open(f"file://{help_path.absolute()}")
        else:
            messagebox.showwarning("Help", f"help.html not found at {help_path}")
    
    def _start(self):
        if self.is_running:
            return
        if not self.model_manager.is_loaded():
            messagebox.showwarning("Wait", "Model still loading")
            return
        inp = self.input_dir.get()
        out = self.output_dir.get()
        if not inp or not Path(inp).exists():
            messagebox.showerror("Error", "Select input folder")
            return
        if not out:
            messagebox.showerror("Error", "Select output folder")
            return
        
        self.is_running = True
        self.start_btn.config(text="⏳  PROCESSING...", state="disabled")
        self.progress.config(value=0)
        self.pct_label.config(text="0%")
        self._log("=" * 40)
        self._log("Starting detection...")
        
        threading.Thread(target=self._worker, args=(inp, out), daemon=True).start()
    
    def _worker(self, inp, out):
        try:
            files = list(Path(inp).glob("*.png")) + list(Path(inp).glob("*.jpg")) + list(Path(inp).glob("*.jpeg"))
            images = [f for f in files if not f.name.startswith("detected_")]
            total = len(images)
            if total == 0:
                self.root.after(0, lambda: self._log("No images found"))
                return
            
            total_obj = 0
            total_conf = 0
            
            for i, img in enumerate(images):
                self.root.after(0, lambda idx=i+1, t=total, n=img.name: 
                    self._log(f"[{idx}/{t}] Processing: {n}"))
                
                try:
                    img_cv = cv2.imread(str(img))
                    if img_cv is None:
                        self.root.after(0, lambda: self._log("  Failed to read image"))
                        continue
                    
                    slice_sz = self.slice_value.get()
                    
                    results = self.model_manager.predict_sliced(
                        img_cv,
                        slice_height=slice_sz,
                        slice_width=slice_sz,
                        conf_threshold=self.conf_value.get()
                    )
                    
                    count = 0
                    conf_sum = 0
                    
                    if results and isinstance(results, list):
                        for det in results:
                            bbox = det.get("bbox")
                            conf_val = det.get("confidence", 0)
                            if bbox:
                                x1, y1, x2, y2 = map(int, bbox)
                                count += 1
                                conf_sum += conf_val
                                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(img_cv, f"Wood {conf_val:.2f}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    total_obj += count
                    total_conf += conf_sum
                    
                    if img_cv is not None:
                        out_path = Path(out) / f"detected_{img.name}"
                        ext = Path(img.name).suffix.lower()
                        if ext in ['.jpg', '.jpeg']:
                            _, buf = cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        else:
                            _, buf = cv2.imencode('.png', img_cv)
                        out_path.write_bytes(buf.tobytes())
                    
                    self.root.after(0, lambda c=count: self._log(f"  Detected {c} objects"))
                    
                except Exception as e:
                    self.root.after(0, lambda e=str(e): self._log(f"  Error: {e}"))
                
                pct = int((i + 1) / total * 100)
                self.root.after(0, lambda p=pct: self.progress.config(value=p))
                self.root.after(0, lambda p=pct: self.pct_label.config(text=f"{p}%"))
                self.root.after(0, lambda t=total: self.stat_images.config(text=f"Images: {t}"))
                self.root.after(0, lambda o=total_obj: self.stat_objects.config(text=f"Objects: {o}"))
            
            avg_conf = total_conf / total_obj if total_obj > 0 else 0
            self.root.after(0, lambda: self.stat_conf.config(text=f"Avg Conf: {avg_conf:.1%}"))
            self.root.after(0, lambda: self.stat_rate.config(text=f"Rate: {total_obj/total:.1f}" if total > 0 else "Rate: -"))
            self.root.after(0, lambda: self._log("=" * 40))
            self.root.after(0, lambda: self._log("Detection Complete!"))
            self.root.after(0, lambda: messagebox.showinfo("Detection Complete!", 
                f"Images Processed: {total}\nObjects Found: {total_obj}\nAverage Confidence: {avg_conf:.1%}"))
            
        except Exception as e:
            import traceback
            self.root.after(0, lambda: self._log(f"Fatal Error: {e}"))
            self.root.after(0, lambda: self._log(traceback.format_exc()))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(text="▶  START DETECTION", state="normal"))
            self.root.after(0, lambda: self.progress.config(value=0))
    
    def run(self):
        self.root.mainloop()
