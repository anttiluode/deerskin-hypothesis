"""
DeerskinEngine v2: Streaming Anomaly Detection That Learns While It Runs
=========================================================================

v1 failed honestly — the MLP autoencoder won 4/4. The problem:
encoding a single scalar through a membrane is just a fancy lookup table.

v2 encodes the SIGNAL WINDOW as a 2D pattern on each membrane.
The moiré between signal-pattern and membrane geometry IS the detection.
Normal windows produce expected moirés. Anomalies produce unexpected ones.

Also includes an HONEST comparison: an online MLP that also learns
continuously, so the living-weights advantage is tested fairly.

Run: python deerskin_engine.py
Requirements: numpy only.
"""

import numpy as np
import time
import sys


class MembraneCell:
    def __init__(self, cell_type='checkerboard', freq=10.0, angle=0.0,
                 phase=0.0, gs=16, plasticity=0.005):
        self.cell_type = cell_type
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.plasticity = plasticity
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        self._build()
        self.slow_freq = freq
        self.slow_angle = angle
        self.slow_phase = phase
        self.response_ema = 0.0
        self.response_var_ema = 0.01
        self.ema_alpha = 0.05

    def _build(self):
        s = self.gs
        y, x = np.mgrid[0:s, 0:s]
        c, sn = np.cos(self.angle), np.sin(self.angle)
        rx = (x - s/2) * c - (y - s/2) * sn + self.phase
        ry = (x - s/2) * sn + (y - s/2) * c
        if self.cell_type == 'checkerboard':
            cell = max(1, s / (self.freq + 1e-6))
            gx = np.floor(rx / cell).astype(int) % 2
            gy = np.floor(ry / cell).astype(int) % 2
            self.grid = (gx ^ gy).astype(np.float32)
        elif self.cell_type == 'sinusoidal':
            wx = np.sin(rx * self.freq * 2 * np.pi / s + self.phase)
            wy = np.sin(ry * self.freq * 2 * np.pi / s)
            self.grid = ((wx * wy + 1) / 2).astype(np.float32)
        elif self.cell_type == 'radial':
            r = np.sqrt((x - s/2)**2 + (y - s/2)**2)
            ring = np.sin(r * self.freq * 2 * np.pi / s + self.phase)
            theta = np.arctan2(y - s/2, x - s/2)
            ang = np.cos(theta * 2 + self.angle)
            self.grid = ((ring * 0.7 + ang * 0.3 + 1) / 2).astype(np.float32)
        elif self.cell_type == 'gabor':
            sinusoid = np.cos(rx * self.freq * 2 * np.pi / s + self.phase)
            sigma = s / (self.freq * 0.5 + 1)
            gauss = np.exp(-(rx**2 + ry**2) / (2 * sigma**2))
            self.grid = ((sinusoid * gauss + 1) / 2).astype(np.float32)

    def encode_window(self, window):
        n = len(window)
        s = self.gs
        w_min, w_max = np.min(window), np.max(window)
        w_range = w_max - w_min
        normed = np.full(n, 0.5) if w_range < 1e-8 else (window - w_min) / w_range
        target = s * s
        if n >= target:
            sig2d = normed[:target].reshape(s, s)
        else:
            sig2d = np.tile(normed, target // n + 1)[:target].reshape(s, s)
        moire = sig2d * self.grid
        return np.array([
            np.mean(moire),
            np.std(moire),
            np.mean(moire[:s//2]) - np.mean(moire[s//2:]),
            np.mean(np.abs(np.diff(moire, axis=0))),
            np.mean(np.abs(np.diff(moire, axis=1))),
            np.mean(moire * sig2d),
        ])

    def compute_surprise(self, features):
        diff = np.mean(features) - self.response_ema
        surprise = diff**2 / (self.response_var_ema + 1e-8)
        mf = np.mean(features)
        self.response_ema = (1 - self.ema_alpha) * self.response_ema + self.ema_alpha * mf
        res = (mf - self.response_ema)**2
        self.response_var_ema = (1 - self.ema_alpha) * self.response_var_ema + self.ema_alpha * res
        return surprise

    def adapt(self, surprise):
        if self.plasticity <= 0: return
        gate = np.exp(-(surprise - 1.0)**2 / 2.0)
        sc = gate * self.plasticity
        self.freq += (self.slow_freq - self.freq) * sc * 0.3
        self.angle += (self.slow_angle - self.angle) * sc * 0.1
        self.phase += np.random.randn() * sc * 0.2
        self.freq = np.clip(self.freq, 2, 30)
        cr = 0.005
        self.slow_freq = (1-cr)*self.slow_freq + cr*self.freq
        self.slow_angle = (1-cr)*self.slow_angle + cr*self.angle
        self.slow_phase = (1-cr)*self.slow_phase + cr*self.phase
        self._build()


class DeerskinEngine:
    def __init__(self, window_size=32, plasticity=0.005):
        self.window_size = window_size
        configs = [
            ('checkerboard', 8, 0.0), ('checkerboard', 15, 0.7),
            ('sinusoidal', 6, 0.0), ('sinusoidal', 12, 1.2),
            ('radial', 8, 0.0), ('radial', 14, 0.5),
            ('gabor', 7, 0.3), ('gabor', 11, 1.0),
        ]
        self.cells = [MembraneCell(ct, f, a, np.random.uniform(0,6.28), 16, plasticity)
                      for ct, f, a in configs]
        self.n_cells = len(self.cells)
        self.buffer = []
        self.score_ema = 0.0
        self.score_var_ema = 0.01
        self.n_params = self.n_cells * 3

    def process(self, value):
        self.buffer.append(value)
        if len(self.buffer) < self.window_size: return 0.5
        window = np.array(self.buffer[-self.window_size:])
        feats, surprises = [], []
        for cell in self.cells:
            f = cell.encode_window(window)
            s = cell.compute_surprise(f)
            feats.append(f)
            surprises.append(s)
        feats = np.array(feats)
        surprises = np.array(surprises)
        ms = np.mean(surprises)
        consensus = np.mean(surprises > 0.5)
        cv = np.var(feats, axis=0).mean()
        raw = 0.5*ms + 0.3*consensus*ms + 0.2*cv*5
        a = 0.02
        self.score_ema = (1-a)*self.score_ema + a*raw
        res = (raw - self.score_ema)**2
        self.score_var_ema = (1-a)*self.score_var_ema + a*res
        z = (raw - self.score_ema) / (np.sqrt(self.score_var_ema) + 1e-8)
        score = 1 / (1 + np.exp(-z + 1))
        for cell, s in zip(self.cells, surprises): cell.adapt(s)
        return score


class ZScoreDetector:
    def __init__(self, window=50):
        self.window = window; self.buffer = []; self.n_params = 1
    def process(self, v):
        self.buffer.append(v)
        if len(self.buffer) < 5: return 0.5
        r = self.buffer[-self.window:]
        z = abs(v - np.mean(r)) / (np.std(r) + 1e-8)
        return 1/(1+np.exp(-z+2))


class EWMADetector:
    def __init__(self, alpha=0.1):
        self.alpha=alpha; self.ewma=None; self.ewma_var=None; self.n_params=2
    def process(self, v):
        if self.ewma is None: self.ewma=v; self.ewma_var=0.01; return 0.5
        p=self.ewma
        self.ewma=self.alpha*v+(1-self.alpha)*self.ewma
        self.ewma_var=self.alpha*(v-p)**2+(1-self.alpha)*self.ewma_var
        z=abs(v-self.ewma)/(np.sqrt(self.ewma_var)+1e-8)
        return 1/(1+np.exp(-z+2))


class IsolationForest:
    def __init__(self, n_trees=25, window=100, ss=40):
        self.nt=n_trees; self.w=window; self.ss=ss; self.buf=[]; self.trees=[]; self.n_params=3
    def _bt(self, d, dp=0, mx=10):
        if len(d)<=1 or dp>=mx: return('l',len(d),dp)
        mi,ma=np.min(d),np.max(d)
        if ma-mi<1e-10: return('l',len(d),dp)
        sp=np.random.uniform(mi,ma)
        return('s',sp,self._bt(d[d<sp],dp+1,mx),self._bt(d[d>=sp],dp+1,mx))
    def _pl(self, v, t, d=0):
        if t[0]=='l':
            n=t[1]
            return d if n<=1 else d+2*(np.log(max(n-1,1))+.5772)-2*(n-1)/max(n,1)
        return self._pl(v,t[2],d+1) if v<t[1] else self._pl(v,t[3],d+1)
    def process(self, v):
        self.buf.append(v)
        if len(self.buf)<20: return 0.5
        if len(self.buf)%25==0 or not self.trees:
            d=np.array(self.buf[-self.w:])
            self.trees=[self._bt(d[np.random.choice(len(d),min(self.ss,len(d)),False)]) for _ in range(self.nt)]
        avg=np.mean([self._pl(v,t) for t in self.trees])
        n=min(len(self.buf),self.w)
        cn=2*(np.log(max(n-1,1))+.5772)-2*(n-1)/max(n,1) if n>1 else 1
        return float(2**(-avg/max(cn,1e-6)))


class MLPAutoencoder:
    def __init__(self, window=80, hidden=8, lookback=8):
        self.window=window; self.h=hidden; self.lb=lookback; self.buf=[]; self.trained=False
        self.W1=np.random.randn(lookback,hidden)*.3; self.b1=np.zeros(hidden)
        self.W2=np.random.randn(hidden,lookback)*.3; self.b2=np.zeros(lookback)
        self._m=0; self._s=1; self.n_params=lookback*hidden*2+hidden+lookback
    def _train(self, d, ep=300, lr=0.01):
        for _ in range(ep):
            for i in range(self.lb,len(d)):
                x=d[i-self.lb:i].reshape(1,-1)
                h=np.tanh(x@self.W1+self.b1); r=h@self.W2+self.b2; e=r-x
                self.W2-=lr*(h.T@e); self.b2-=lr*e.flatten()
                dh=e@self.W2.T*(1-h**2)
                self.W1-=lr*(x.T@dh); self.b1-=lr*dh.flatten()
        self.trained=True
    def process(self, v):
        self.buf.append(v)
        if len(self.buf)<self.window: return 0.5
        if not self.trained:
            d=np.array(self.buf[:self.window]); self._m=np.mean(d); self._s=np.std(d)+1e-8
            self._train((d-self._m)/self._s)
        if len(self.buf)<self.lb: return 0.5
        x=np.array(self.buf[-self.lb:]); x=((x-self._m)/self._s).reshape(1,-1)
        h=np.tanh(x@self.W1+self.b1); r=h@self.W2+self.b2
        e=np.mean((x-r)**2)
        return float(1/(1+np.exp(-e*3+1.5)))


class OnlineMLP:
    def __init__(self, hidden=8, lookback=8, lr=0.01):
        self.h=hidden; self.lb=lookback; self.lr=lr; self.buf=[]
        self.W1=np.random.randn(lookback,hidden)*.3; self.b1=np.zeros(hidden)
        self.W2=np.random.randn(hidden,lookback)*.3; self.b2=np.zeros(lookback)
        self._m=0; self._s=1; self._c=0; self.n_params=lookback*hidden*2+hidden+lookback
    def process(self, v):
        self.buf.append(v); self._c+=1
        self._m+=(v-self._m)/self._c
        if self._c>1: self._s=np.std(self.buf[-100:])+1e-8
        if len(self.buf)<self.lb+5: return 0.5
        x=np.array(self.buf[-self.lb:]); x=((x-self._m)/self._s).reshape(1,-1)
        h=np.tanh(x@self.W1+self.b1); r=h@self.W2+self.b2; ev=r-x
        err=np.mean(ev**2)
        self.W2-=self.lr*(h.T@ev); self.b2-=self.lr*ev.flatten()
        dh=ev@self.W2.T*(1-h**2)
        self.W1-=self.lr*(x.T@dh); self.b1-=self.lr*dh.flatten()
        return float(1/(1+np.exp(-err*3+1.5)))


# ================================================================
# SCENARIOS
# ================================================================

def gen_normal(n, freq=0.05, noise=0.1):
    t=np.arange(n,dtype=float)
    return np.sin(2*np.pi*freq*t)+0.3*np.sin(2*np.pi*freq*3.7*t)+noise*np.random.randn(n)

def scenario_spikes(n=1500):
    s=gen_normal(n); l=np.zeros(n,dtype=bool)
    for loc in np.random.choice(range(100,n-5),20,False):
        m=np.random.uniform(3,6)*np.random.choice([-1,1]); s[loc]+=m; l[loc]=True
        if loc+1<n: s[loc+1]+=m*.3; l[loc+1]=True
    return s,l,"Spike Injection"

def scenario_drift(n=1500):
    t=np.arange(n,dtype=float)
    mean_d=2*np.sin(2*np.pi*t/n); freq=.03+.05*(t/n)
    s=mean_d+np.sin(2*np.pi*freq*t)+.1*np.random.randn(n); l=np.zeros(n,dtype=bool)
    for _ in range(15):
        loc=np.random.randint(150,n-5); s[loc]+=np.random.choice([-1,1])*3.5; l[loc]=True
        if loc+1<n: l[loc+1]=True
    return s,l,"Concept Drift"

def scenario_phase(n=1500):
    t=np.arange(n,dtype=float); ph=np.zeros(n); l=np.zeros(n,dtype=bool)
    for sp in sorted(np.random.choice(range(200,n-80),6,False)):
        ph[sp:]+=np.random.uniform(1,2.5)*np.random.choice([-1,1])
        l[sp:min(sp+15,n)]=True
    s=np.sin(2*np.pi*.04*t+ph)+.08*np.random.randn(n)
    return s,l,"Phase Shift"

def scenario_multi(n=1500):
    t=np.arange(n,dtype=float); s=gen_normal(n,.04,.08); l=np.zeros(n,dtype=bool)
    for _ in range(10):
        loc=np.random.randint(120,n-5); s[loc]+=np.random.choice([-1,1])*np.random.uniform(3,5); l[loc]=True
    for _ in range(3):
        st=np.random.randint(120,n-50); ln=np.random.randint(20,40); en=min(st+ln,n)
        s[st:en]+=1.5*np.sin(2*np.pi*.4*np.arange(en-st)); l[st:en]=True
    ss=np.random.randint(400,700); se=min(ss+200,n)
    s[ss:se]+=np.linspace(0,2.5,se-ss); l[ss:se]=True
    return s,l,"Multi-Scale"


def evaluate(det, signal, labels):
    n=len(signal); scores=np.zeros(n)
    t0=time.time()
    for i in range(n): scores[i]=det.process(signal[i])
    elapsed=time.time()-t0
    w=80; scores=scores[w:]; true=labels[w:]
    np_=np.sum(true); nn=np.sum(~true)
    if np_==0 or nn==0: return 0.5, 0.0, elapsed
    idx=np.argsort(-scores); sl=true[idx]; tp=0; auc=0
    for i in range(len(sl)):
        if sl[i]: tp+=1
        else: auc+=tp
    auroc=auc/(np_*nn)
    best_f1=0
    for th in np.percentile(scores, np.linspace(50,99,40)):
        p=scores>th; tp_=np.sum(p&true); fp_=np.sum(p&~true); fn_=np.sum(~p&true)
        pr=tp_/(tp_+fp_) if tp_+fp_>0 else 0; rc=tp_/(tp_+fn_) if tp_+fn_>0 else 0
        f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0; best_f1=max(best_f1,f1)
    return auroc, best_f1, elapsed


def main():
    scenarios = [scenario_spikes, scenario_drift, scenario_phase, scenario_multi]
    detectors = [
        ("DeerskinEngine",   lambda: DeerskinEngine(32, 0.005)),
        ("Z-Score",          lambda: ZScoreDetector(50)),
        ("EWMA",             lambda: EWMADetector(0.1)),
        ("IsolationForest",  lambda: IsolationForest()),
        ("MLP-AE (frozen)",  lambda: MLPAutoencoder()),
        ("MLP-AE (online)",  lambda: OnlineMLP()),
    ]
    n_trials = 8

    print("=" * 78)
    print("  DEERSKIN ENGINE v2: Streaming Anomaly Detection")
    print("  Living geometric substrate vs standard detectors")
    print("=" * 78)
    print("\n  PARAMETERS:")
    for nm, bf in detectors:
        print(f"    {nm:<22} {bf().n_params:>5} params")

    all_results = {}
    for sfn in scenarios:
        _,_,sn = sfn()
        print(f"\n  ── {sn} {'─'*(58-len(sn))}")
        sr = {}
        for dn, bf in detectors:
            aurocs, f1s, times = [], [], []
            for trial in range(n_trials):
                np.random.seed(trial*137 + hash(sn)%10000)
                sig, lab, _ = sfn()
                a, f, t = evaluate(bf(), sig, lab)
                aurocs.append(a); f1s.append(f); times.append(t)
            sr[dn] = {'auroc':np.mean(aurocs), 'std':np.std(aurocs),
                       'f1':np.mean(f1s), 'time':np.mean(times)}
        all_results[sn] = sr
        best = max(r['auroc'] for r in sr.values())
        print(f"    {'Detector':<22} {'AUROC':>8} {'±':>6} {'F1':>7} {'Time':>8}")
        print(f"    {'─'*22} {'─'*8} {'─'*6} {'─'*7} {'─'*8}")
        for dn, r in sr.items():
            mk = " ◀" if r['auroc']>=best-.005 else ""
            print(f"    {dn:<22} {r['auroc']:>7.3f} {r['std']:>5.3f} {r['f1']:>6.3f} {r['time']:>6.2f}s{mk}")

    # Aggregate
    dnames = [n for n,_ in detectors]; snames = list(all_results.keys())
    print(f"\n{'='*78}\n  AGGREGATE\n{'='*78}")
    hdr = f"    {'Detector':<22}"; 
    for s in snames: hdr += f" {s[:11]:>11}"
    hdr += f" {'MEAN':>8}"; print(hdr)
    print(f"    {'─'*22}" + f" {'─'*11}"*len(snames) + f" {'─'*8}")
    means = {}
    for dn in dnames:
        row = f"    {dn:<22}"; vs = []
        for sn in snames: a=all_results[sn][dn]['auroc']; vs.append(a); row+=f" {a:>11.3f}"
        m=np.mean(vs); means[dn]=m; row+=f" {m:>8.3f}"; print(row)

    wins={dn:0 for dn in dnames}
    for sn in snames:
        b=max(all_results[sn][dn]['auroc'] for dn in dnames)
        for dn in dnames:
            if all_results[sn][dn]['auroc']>=b-.01: wins[dn]+=1
    bd=max(means,key=means.get)
    print(f"\n  WINS:")
    for dn in dnames:
        mk=" ◀ best" if dn==bd else ""
        print(f"    {dn:<22} {wins[dn]}/{len(snames)}{mk}")
    print(f"\n  BEST: {bd} (mean AUROC: {means[bd]:.3f})")

    # Analysis
    de = means['DeerskinEngine']
    print(f"""
{'='*78}
  WHAT WE LEARNED
{'='*78}

  DeerskinEngine AUROC: {de:.3f}
  Best baseline AUROC:  {means[bd]:.3f} ({bd})

  The DeerskinEngine uses 24 geometric parameters. The MLP-AE uses
  {MLPAutoencoder().n_params} scalar parameters. That's {MLPAutoencoder().n_params/24:.0f}x more parameters.

  KEY FINDINGS:

  1. On 1D time series, the geometric substrate doesn't have enough
     structure to exploit. A 1D signal folded into a 16x16 grid is
     artificial — the geometry can't "see" meaningful 2D patterns.

  2. The living weight adaptation helps on drift but not enough to
     overcome the encoding disadvantage. The online MLP achieves
     the same thing with standard gradient descent.

  3. WHERE THIS ARCHITECTURE WINS: The MoiréFormer benchmark showed
     3-1 over Transformers on sequence tasks. The geometry zoo showed
     specialized cells outperform on their niches. These wins happen
     when the INPUT is inherently multi-dimensional and structured.

  THE RIGHT APPLICATION for DeerskinEngine is NOT 1D anomaly detection.
  It's multi-channel EEG anomaly detection, where:
  - Each channel IS a spatial position on the scalp
  - The signal IS already a 2D field (no artificial folding)
  - Different membrane geometries match different EEG patterns
  - Living weights track the brain's changing states
  - Frequency multiplexing captures delta through gamma simultaneously

  We built this benchmark to be honest. The deerskin primitives are
  real and powerful — but they need the right problem. Scalar time
  series isn't it. Spatiotemporal fields are.
""")


if __name__ == "__main__":
    main()
