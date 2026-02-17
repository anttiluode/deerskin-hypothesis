import gradio as gr
import cv2
import numpy as np
import os
import glob
import time

# ==============================================================================
# 📡 JANUS TOMOGRAPHY: THE HOLOGRAPHIC SCANNER
# ==============================================================================

class TomographyCortex:
    def __init__(self, resolution=512):
        self.res = int(resolution)
        # Complex Field (Hologram)
        self.memory_field = np.zeros((self.res, self.res), dtype=np.complex64)
        self.registry = [] 
        
    def _generate_complex_carrier(self, freq, angle, spin_rad):
        y, x = np.mgrid[0:self.res, 0:self.res]
        c, s = np.cos(angle), np.sin(angle)
        rx = x * c - y * s
        theta = rx * freq * 2 * np.pi / self.res
        return np.exp(1j * (theta + spin_rad))

    def ingest_folder(self, folder_path, limit=3):
        """
        Encodes images at specific ANGLES (0, 60, 120...) 
        so we can find them with the scanner.
        """
        self.memory_field = np.zeros((self.res, self.res), dtype=np.complex64)
        self.registry = []
        
        if not folder_path: return "Invalid Path"
        clean_path = str(folder_path).strip('"').strip("'")
        if not os.path.exists(clean_path): return "Path not found"
        
        files = glob.glob(os.path.join(clean_path, '*.*'))
        # Filter images
        valid = ['.jpg','.png','.jpeg','.bmp']
        files = [f for f in files if os.path.splitext(f)[1].lower() in valid][:int(limit)]
        
        if not files: return "No images."
        
        print(f"Hiding {len(files)} memories at different angles...")
        
        for i, fpath in enumerate(files):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (self.res, self.res))
            
            # Preprocess (High Pass for better locking)
            blur = cv2.GaussianBlur(img, (0,0), 3)
            img_signal = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            img_signal = (img_signal.astype(np.float32) / 255.0) - 0.5
            
            # ENCODING GEOMETRY
            # We space them evenly in rotation space
            target_angle = (i * (180.0 / len(files))) * np.pi / 180.0
            target_freq = 20.0 # Fixed frequency for the "Radar" sweep
            target_spin = 0.0
            
            carrier = self._generate_complex_carrier(target_freq, target_angle, target_spin)
            self.memory_field += img_signal * carrier
            
            self.registry.append({
                "name": os.path.basename(fpath),
                "angle": np.degrees(target_angle)
            })
            
        # Normalize
        max_val = np.max(np.abs(self.memory_field))
        if max_val > 0: self.memory_field /= max_val
            
        return f"Encoded {len(files)} targets at angles: {[f'{x['angle']:.0f}°' for x in self.registry]}"

    def scan(self, current_angle_deg):
        """
        The Lighthouse Beam.
        Probes the field at 'current_angle_deg'.
        Returns:
        1. The Retrieved Image
        2. The Resonance Energy (The "Pulse Vector Length")
        """
        freq = 20.0
        angle_rad = np.radians(current_angle_deg)
        spin_rad = 0.0
        
        # 1. Generate Probe
        ref_wave = self._generate_complex_carrier(freq, angle_rad, spin_rad)
        decoder = np.conj(ref_wave)
        
        # 2. Demodulate
        raw = self.memory_field * decoder
        signal = raw.real 
        
        # 3. Filter
        wavelength = self.res / freq
        sigma = wavelength * 0.5
        k_size = int(sigma * 4) | 1
        recovered = cv2.GaussianBlur(signal, (k_size, k_size), sigma)
        
        # 4. MEASURE RESONANCE (The "Pulse")
        # How much structure is visible? 
        # We measure Variance (Contrast) as the energy metric.
        # Silence = Gray = Low Variance.
        # Memory = Black/White = High Variance.
        resonance_energy = np.var(recovered) * 1000.0
        
        # 5. Gain & Clip
        recovered = (recovered - np.mean(recovered)) * 5.0 + 0.5
        
        return np.clip(recovered, 0, 1), resonance_energy

# ==============================================================================
# 🖥️ GUI
# ==============================================================================

cortex = TomographyCortex(resolution=256)
scan_history = [] # To plot the pulse history

def run_ingest(path, limit):
    msg = cortex.ingest_folder(path, limit)
    return msg

def run_scan_step(angle):
    global scan_history
    
    # 1. Perform Scan
    img, energy = cortex.scan(angle)
    
    # 2. Update History (Rolling buffer)
    scan_history.append(energy)
    if len(scan_history) > 100: scan_history.pop(0)
    
    # 3. Draw the "Pulse Graph" (Oscilloscope)
    h, w = 100, 300
    graph = np.zeros((h, w, 3), dtype=np.uint8)
    
    if len(scan_history) > 1:
        # Normalize graph
        max_e = max(scan_history) + 1e-6
        points = []
        for i, val in enumerate(scan_history):
            x = int((i / 100) * w)
            y = int(h - (val / max_e) * (h - 10))
            points.append((x, y))
            
        # Draw Line
        for i in range(len(points)-1):
            cv2.line(graph, points[i], points[i+1], (0, 255, 255), 2)
            
        # Draw Current Value text
        cv2.putText(graph, f"RESONANCE: {energy:.2f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img, graph

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 📡 Janus Tomography: The Neural Lighthouse")
    gr.Markdown("We hide 3 memories. The Scanner rotates. Watch the Energy Pulse spike when it hits a memory.")
    
    with gr.Row():
        with gr.Column(scale=1):
            path_in = gr.Textbox(label="Folder Path", placeholder="C:/Images")
            limit_in = gr.Number(value=3, label="Memories to Hide")
            btn_load = gr.Button("Hide Memories in Field", variant="primary")
            status = gr.Textbox(label="Log")
            
            gr.Markdown("---")
            gr.Markdown("### 🔄 The Scanner")
            angle_slider = gr.Slider(0, 180, value=0, label="Scanner Angle")
            # Auto-spin checkbox? (Gradio animation is tricky, manual sliding is safer for demo)
            gr.Markdown("**Instruction:** Drag the 'Scanner Angle' slider back and forth slowly.")

        with gr.Column(scale=2):
            with gr.Row():
                # The Perception
                view_screen = gr.Image(label="Scanner View (The Lock)", type="numpy")
            with gr.Row():
                # The Pulse
                pulse_screen = gr.Image(label="Resonance Pulse (The Vector Length)", type="numpy")

    btn_load.click(run_ingest, [path_in, limit_in], status)
    angle_slider.change(run_scan_step, angle_slider, [view_screen, pulse_screen])

if __name__ == "__main__":
    demo.launch()