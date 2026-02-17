import gradio as gr
import cv2
import numpy as np
import os
import glob

# ==============================================================================
# 🧠 JANUS SPIN: THE COMPLEX RESONATOR
# ==============================================================================

class SpinCortex:
    def __init__(self, resolution=512):
        self.res = int(resolution)
        # UPGRADE: The field is now Complex (Real + Imaginary)
        # This gives us the "Spin" dimension naturally.
        self.memory_field = np.zeros((self.res, self.res), dtype=np.complex64)
        self.registry = [] 
        
    def clear(self):
        self.memory_field = np.zeros((self.res, self.res), dtype=np.complex64)
        self.registry = []

    def _generate_complex_carrier(self, freq, angle, spin_rad):
        """
        Generates a Complex Plane Wave (The Spinning Chessboard).
        Math: Psi = exp(i * (k*x + spin))
        """
        y, x = np.mgrid[0:self.res, 0:self.res]
        
        # Coordinate rotation (Spatial Angle)
        c, s = np.cos(angle), np.sin(angle)
        rx = x * c - y * s
        
        # The Wave Argument
        theta = rx * freq * 2 * np.pi / self.res
        
        # THE SPIN: We add the spin phase directly to the wave argument
        # This rotates the wave in the complex plane without moving it spatially.
        complex_wave = np.exp(1j * (theta + spin_rad))
        
        return complex_wave

    def ingest_folder(self, folder_path, limit=50, use_edges=True, progress=gr.Progress()):
        self.clear()
        
        if not folder_path: return "Invalid Path"
        clean_path = str(folder_path).strip('"').strip("'")
        if not os.path.exists(clean_path): return "Path not found"
        
        files = glob.glob(os.path.join(clean_path, '*.*'))
        valid_exts = ['.jpg', '.png', '.jpeg', '.bmp']
        files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts][:int(limit)]
        
        if not files: return "No images found."
        
        print(f"Ingesting {len(files)} memories with Complex Spin...")
        
        for i, fpath in progress.tqdm(enumerate(files), total=len(files), desc="Spinning Memories"):
            try:
                # 1. Load & Preprocess
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (self.res, self.res))
                
                # Edge Encoding (Sharpening)
                if use_edges:
                    blur = cv2.GaussianBlur(img, (0,0), 3)
                    img_signal = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
                else:
                    img_signal = img
                    
                # Center signal
                img_signal = (img_signal.astype(np.float32) / 255.0) - 0.5
                
                # 2. Assign Addresses
                # We now have 3 dimensions to separate memories: Freq, Angle, SPIN.
                
                # Angle: Golden Ratio distribution
                angle = (i * 0.618033 * np.pi * 2) % np.pi 
                
                # Frequency: Standard spacing
                freq = 10.0 + (i % 10) * 4.0 
                
                # SPIN: This is the new "Hidden Dimension"
                # We rotate the phase for each memory.
                spin = (i * np.pi / 2) % (2 * np.pi) # 0, 90, 180, 270 degrees
                
                # 3. Encode (Complex Modulation)
                # Field += Image * ComplexCarrier
                carrier = self._generate_complex_carrier(freq, angle, spin)
                self.memory_field += img_signal * carrier
                
                # Log
                self.registry.append({
                    "id": i, "name": os.path.basename(fpath), 
                    "freq": freq, "angle": angle, "spin": spin,
                    "original": img
                })
                
            except Exception as e:
                print(f"Skipped {fpath}: {e}")
                continue
        
        # Energy Normalization
        max_val = np.max(np.abs(self.memory_field))
        if max_val > 0: self.memory_field /= (max_val + 1e-6)
            
        return f"Encoded {len(self.registry)} memories into Complex Spin Field."

    def retrieve(self, freq, angle, spin_deg):
        # 1. Generate the Key (The Anti-Spin Wave)
        spin_rad = np.radians(spin_deg)
        # Note: To decode, we multiply by the CONJUGATE.
        # This effectively subtracts the spin.
        ref_wave = self._generate_complex_carrier(freq, angle, spin_rad)
        decoder = np.conj(ref_wave)
        
        # 2. Demodulate
        raw_output = self.memory_field * decoder
        
        # 3. Filter (The "Lens")
        # Since we use complex waves, the "DC" component (our image) is Real.
        # The interference is oscillatory complex noise.
        
        # Extract Real component (Project onto our spin axis)
        # We could also use Abs(), but Real() is more phase-sensitive (allows negative image)
        signal = raw_output.real 
        
        # Adaptive Blur (Matched to frequency)
        wavelength = self.res / (freq + 1e-6)
        sigma = wavelength * 0.5
        k_size = int(sigma * 4) | 1
        
        recovered = cv2.GaussianBlur(signal, (k_size, k_size), sigma)
        
        # 4. Perception Gain
        recovered = (recovered - np.mean(recovered)) * 5.0 + 0.5
        
        # For visualization of the probe
        probe_vis = (ref_wave.real + 1) / 2
        
        return np.clip(recovered, 0, 1), probe_vis

# ==============================================================================
# 🖥️ GUI
# ==============================================================================

cortex = SpinCortex(resolution=512)

def run_ingest(path, limit, res, edges):
    global cortex
    if cortex.res != int(res): cortex = SpinCortex(int(res))
    
    msg = cortex.ingest_folder(path, limit, edges)
    choices = [f"{m['id']}: {m['name']}" for m in cortex.registry]
    
    # Visualization: Magnitude of the complex field
    # This shows "Energy Density" regardless of phase
    mag_view = np.abs(cortex.memory_field)
    mag_view = (mag_view - mag_view.min()) / (np.ptp(mag_view) + 1e-6)
    
    return msg, gr.update(choices=choices, value=choices[0] if choices else None), mag_view

def update_view(freq, angle, spin):
    perc, ref = cortex.retrieve(float(freq), np.radians(float(angle)), float(spin))
    return perc, ref

def jump(val):
    if not val: return 10.0, 0.0, 0.0, None
    try:
        idx = int(str(val).split(':')[0])
        t = cortex.registry[idx]
        return float(t['freq']), float(np.degrees(t['angle'])), float(np.degrees(t['spin'])), t['original']
    except:
        return 10.0, 0.0, 0.0, None

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🌀 Janus Spin: Complex Holographic Cortex")
    gr.Markdown("Memories are stored in a Complex Field. Use Frequency, Angle, AND Spin to find them.")
    
    with gr.Row():
        with gr.Column(scale=1):
            path_in = gr.Textbox(label="Folder Path", placeholder="C:/Images")
            with gr.Row():
                limit_in = gr.Number(value=20, label="Limit")
                res_in = gr.Dropdown([256, 512], value=512, label="Resolution")
            edge_check = gr.Checkbox(label="Edge Encoding", value=True)
            btn = gr.Button("Spin Reality into Cortex", variant="primary")
            status = gr.Textbox(label="Log")
            
            gr.Markdown("---")
            gr.Markdown("### 🧠 Frontal Lobe (Tuner)")
            sel = gr.Dropdown(label="Jump to Memory")
            
            # THE TRIAD OF TUNING
            freq = gr.Slider(1, 100, value=10, label="Frequency (Hz)")
            ang = gr.Slider(0, 180, value=0, label="Angle (Deg)")
            spin = gr.Slider(0, 360, value=0, label="🌀 Spin Phase (The Hidden Dimension)", 
                           info="Rotate this to unlock memories hidden at the same frequency.")
            
        with gr.Column(scale=2):
            with gr.Row():
                # Show Magnitude of Complex Field
                raw_v = gr.Image(label="Cortex Energy (Magnitude)", type="numpy")
                ref_v = gr.Image(label="The Probe (Real Part)", type="numpy")
            with gr.Row():
                perc_view = gr.Image(label="Decoded Perception", type="numpy")
                orig_view = gr.Image(label="Ground Truth", type="numpy")

    # WIRING
    btn.click(run_ingest, [path_in, limit_in, res_in, edge_check], [status, sel, raw_v])
    
    inputs = [freq, ang, spin]
    for x in inputs: x.change(update_view, inputs, [perc_view, ref_v])
    
    sel.change(jump, sel, [freq, ang, spin, orig_view])

if __name__ == "__main__":
    demo.launch()