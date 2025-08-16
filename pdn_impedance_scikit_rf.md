Here’s a drop-in script that extends your starter, wiring together a PDN s2p, multiple capacitor s2p models (with counts), and a VRM (either a simple R–L, an s1p, or an s2p across its pins). It reduces everything to a driving-point 1-port at the die node, writes a .s1p, and makes a few sanity plots.

Opinionated tweaks: work in Y (admittance) and renormalize to 1 Ω to keep PDN numerics stable; interpolate all pieces to a common frequency grid (the PDN’s).

#!/usr/bin/env python
# coding: utf-8
"""
PDN → 1-port example (scikit-rf)
- Input:   PDN as s2p (die node = port1, VRM node = port2)
- Caps:    several s2p models, each used N times, mounted as shunt-to-ground at die node
- VRM:     choose one model: simple series R-L, s1p (output Z), or s2p across pins
- Output:  1-port .s1p (driving-point at die), plus plots for |Z|, angle(Z), and return loss

WHY Y-SPACE?  PDNs are milliohm-level; Y stays well-conditioned and shunt elements are additive.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

# -------------------------
# User inputs (filenames)
# -------------------------
# PDN: 2-port from die node (port1) to VRM node (port2)
PDN_S2P = "plane_or_package.s2p"

# Capacitor models (2-port across pads) and their counts on the rail
CAP_MODELS = {
    "c0402_22uF.s2p": 4,
    "c0402_1uF.s2p":  4,
    "c0201_100nF.s2p":2,
}

# VRM choice — uncomment ONE of the three "VRM_*" sections in the code below.
VRM_MODEL = "RL"       # options: "RL", "S1P", "S2P"
VRM_S1P   = "vrm_output_impedance.s1p"   # if VRM_MODEL == "S1P"
VRM_S2P   = "vrm_output_pins.s2p"        # if VRM_MODEL == "S2P"

# Simple series R-L for VRM output impedance (used if VRM_MODEL == "RL")
VRM_R_SERIES = 0.2     # ohms
VRM_L_SERIES = 10e-9   # henry

# Plot/output directory
OUTPUT_DIR = pathlib.Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------
# Helper functions
# -------------------------

def renorm_1ohm(ntwk: rf.Network) -> rf.Network:
    """Return a copy of `ntwk` renormalized to 1 Ω. This improves PDN numerics."""
    out = ntwk.copy()
    out.renormalize(1.0)
    return out

def to_freq(ntwk: rf.Network, freq: rf.Frequency) -> rf.Network:
    """Interpolate `ntwk` to the target frequency grid `freq` (copy)."""
    if ntwk.frequency.f.size == freq.f.size and np.allclose(ntwk.frequency.f, freq.f):
        return ntwk
    return ntwk.interpolate(freq)

def cap_s2p_as_shunt_1port(cap2: rf.Network) -> rf.Network:
    """
    Convert a 2-port capacitor model (across pads) to a 1-port shunt to ground:
    - Short its Port-2 to ground and look into Port-1.
    - This mirrors how an MLCC is used on a rail (one side to rail node, one side to ground).
    """
    cap2 = renorm_1ohm(cap2)
    med = rf.media.DefinedGammaZ0(cap2.frequency, z0=1.0)
    short = med.short()                    # ideal 1-port short to ground
    cap1 = rf.connect(cap2, 1, short, 0)   # connect cap2.port2 → short.port0
    return cap1

def combine_shunt_1ports_to_ground(oneports: list[rf.Network]) -> np.ndarray:
    """
    Combine many 1-port shunts at the SAME node to ground by summing admittances.
    Returns Y_shunt(f) as a complex vector (length = n_freq).
    """
    if not oneports:
        return None
    # Ensure z0=1Ω for all (helps conditioning) and identical frequency grids
    Ysum = np.zeros(oneports[0].frequency.npoints, dtype=complex)
    for n in oneports:
        n = renorm_1ohm(n)
        Ysum += 1.0 / n.z[:, 0, 0]
    return Ysum

def vrm_as_admittance_from_RL(freq: rf.Frequency, R=0.2, L=10e-9) -> np.ndarray:
    """Simple VRM output impedance model as R + jωL (series). Returns Y_vrm(f)."""
    w = 2*np.pi*freq.f
    Z = R + 1j*w*L
    return 1.0 / Z

def s_to_driving_point_Z_2port_Y(pdn2: rf.Network, Y_termination: np.ndarray) -> np.ndarray:
    """
    Driving-point impedance at PDN Port-1 when Port-2 is terminated by Y_termination(f).
    Use Y-domain reduction:  Zin = 1 / (Y11 − Y12*Y21/(Y22 + YL))
    """
    pdn2 = renorm_1ohm(pdn2)
    Y = pdn2.y
    num = Y[:,0,0] - (Y[:,0,1] * Y[:,1,0]) / (Y[:,1,1] + Y_termination)
    return 1.0 / num

def make_s1p_from_Z(freq: rf.Frequency, Zin: np.ndarray, z0=1.0) -> rf.Network:
    """
    Build a scikit-rf 1-port Network from a driving-point impedance array.
    Note: We set the reference z0=1Ω to match the renormalization used in conversions.
    """
    Zmat = Zin.reshape(-1,1,1)
    return rf.Network(frequency=freq, z=Zmat, z0=z0)

# -------------------------
# Load PDN (2-port) and set master frequency grid
# -------------------------
pdn = rf.Network(PDN_S2P)
pdn = renorm_1ohm(pdn)
freq = pdn.frequency
fGHz = freq.f / 1e9

# -------------------------
# Load and collapse capacitors into a single shunt admittance Y_caps(f)
# -------------------------
cap_oneports = []
for fname, count in CAP_MODELS.items():
    c2 = rf.Network(fname)
    c2 = to_freq(c2, freq)              # align to PDN frequency grid
    c1 = cap_s2p_as_shunt_1port(c2)     # convert 2-port cap → 1-port shunt (to ground)
    cap_oneports += [c1] * int(count)   # replicate by count

Y_caps = combine_shunt_1ports_to_ground(cap_oneports)  # may be None if no caps

# -------------------------
# Build VRM termination admittance Y_vrm(f)
# -------------------------
if VRM_MODEL.upper() == "RL":
    Y_vrm = vrm_as_admittance_from_RL(freq, R=VRM_R_SERIES, L=VRM_L_SERIES)

elif VRM_MODEL.upper() == "S1P":
    vrm = rf.Network(VRM_S1P)
    vrm = to_freq(renorm_1ohm(vrm), freq)
    Y_vrm = 1.0 / vrm.z[:, 0, 0]

elif VRM_MODEL.upper() == "S2P":
    # If a VRM is given as a 2-port across its output pins, ground one pin and
    # look into the other to get a 1-port shunt seen at the far node.
    vrm2 = rf.Network(VRM_S2P)
    vrm2 = to_freq(renorm_1ohm(vrm2), freq)
    vrm1 = cap_s2p_as_shunt_1port(vrm2)  # same trick as with caps
    Y_vrm = 1.0 / vrm1.z[:, 0, 0]
else:
    raise ValueError("VRM_MODEL must be one of: 'RL', 'S1P', 'S2P'.")

# -------------------------
# Reduce PDN(2-port) + VRM → driving-point Z at die node (port-1)
# -------------------------
Zin_plane = s_to_driving_point_Z_2port_Y(pdn, Y_vrm)   # PDN with port-2 terminated by VRM

# Add local shunt decaps at the die node:
if Y_caps is not None:
    Zin_total = 1.0 / (1.0 / Zin_plane + Y_caps)
else:
    Zin_total = Zin_plane

# Build a 1-port Network (ref z0 = 1Ω) and save as Touchstone
pdn_1port = make_s1p_from_Z(freq, Zin_total, z0=1.0)
pdn_1port.write_touchstone("pdn_die_driving_point_1ohm")  # writes .s1p next to script

# -------------------------
# Plotting (magnitude, angle, and return loss for the 1-port)
# -------------------------
# |Z| and ∠Z — primary PDN plots
plt.figure()
plt.plot(fGHz, np.abs(pdn_1port.z[:,0,0]))
plt.title("Driving-Point |Z| at Die (ref 1 Ω)")
plt.xlabel("Frequency [GHz]")
plt.ylabel("|Z| [Ω]")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "pdn_Zmag.png", dpi=160)

plt.figure()
plt.plot(fGHz, np.angle(pdn_1port.z[:,0,0], deg=True))
plt.title("Driving-Point ∠Z at Die (ref 1 Ω)")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Phase [deg]")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "pdn_Zangle.png", dpi=160)

# Return loss from the synthesized 1-port (not the primary PDN KPI, but handy)
S11 = pdn_1port.s[:,0,0]
RL_dB = -20*np.log10(np.maximum(np.abs(S11), 1e-16))  # guard tiny magnitude
plt.figure()
plt.plot(fGHz, RL_dB)
plt.title("Return Loss at Die (from synthesized 1-port, z0=1 Ω)")
plt.xlabel("Frequency [GHz]")
plt.ylabel("RL [dB]")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "pdn_return_loss.png", dpi=160)

# Optional: also show raw S21 from the PDN 2-port (supply noise transfer intuition)
plt.figure()
plt.plot(fGHz, -20*np.log10(np.maximum(np.abs(pdn.s[:,1,0]), 1e-16)))
plt.title("PDN Insertion Loss (S21) — plane/package only")
plt.xlabel("Frequency [GHz]")
plt.ylabel("IL [dB]")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "pdn_plane_IL.png", dpi=160)

print("Done. Wrote:")
print(" - plots/*.png")
print(" - pdn_die_driving_point_1ohm.s1p")

How to adapt quickly
	•	Swap in your actual plane_or_package.s2p and capacitor/VRM files.
	•	If caps are provided as s1p already, skip the cap_s2p_as_shunt_1port step and just add 1/Z to Y_caps.
	•	If your extraction is N-port, the same flow applies, but use the Schur complement to reduce N→1. (I can drop that variant next if you want it in the same style.)
