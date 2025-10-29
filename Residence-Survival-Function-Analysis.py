#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residence survival S_res(t) & mean residence time for Na–H2O and Cl–H2O.
- Uses first-minimum rmin from RDF as shell cutoff.
- Computes intermittent and (blip-tolerant) continuous survival.
- Based on **whole water molecule center-of-mass (COM)**, not just OW.
- Optional: z-binned analysis; block-averaged errors.
- Outputs CSV + PNG.
"""

import argparse, sys, math
import numpy as np
import MDAnalysis as mda
import MDAnalysis.lib.distances as mddist
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr", required=True)
    ap.add_argument("--xtc", required=True)
    ap.add_argument("--water-res", default="SOL", help="water residue name")
    # Removed: --water-oxygen
    ap.add_argument("--na-res", default="NA", help="Na+ residue name")
    ap.add_argument("--cl-res", default="CL", help="Cl- residue name")
    ap.add_argument("--rmin-na", type=float, required=True, help="nm, first minimum Na–H2O COM RDF")
    ap.add_argument("--rmin-cl", type=float, required=True, help="nm, first minimum Cl–H2O COM RDF")
    ap.add_argument("--stride", type=int, default=5, help="frame stride")
    ap.add_argument("--blip-ps", type=float, default=2.0, help="blip tolerance for continuous survival (ps)")
    ap.add_argument("--blocks", type=int, default=1, help="number of time blocks for error bars")
    ap.add_argument("--zbins", nargs=3, type=float, metavar=("zmin","zmax","dz"),
                    help="enable z-binned analysis with [zmin zmax dz] in nm")
    ap.add_argument("--prefix", default="residence", help="output prefix")
    return ap.parse_args()

def exponential(t, A, tau):  # single-exponential model
    return A * np.exp(-t / tau)

def survival_intermit(occ):
    """
    occ: (T, N) 0/1 matrix, 1 if in shell
    returns S(t) with S(0)=1
    """
    T, N = occ.shape
    if N == 0 or T == 0:
        return np.zeros(T)
    hmean = occ.mean()
    if hmean == 0:
        return np.zeros(T)
    S = np.zeros(T)
    count = 0
    for i in range(N):
        x = occ[:, i].astype(float)
        if x.mean() < 1e-12:
            continue
        acf = np.correlate(x, x, mode="full")[T-1:]
        norm = np.arange(T, 0, -1)
        acf = acf / norm
        S += acf / x.mean()
        count += 1
    if count == 0:
        return np.zeros(T)
    S /= count
    if S[0] != 0:
        S = S / S[0]
    return S

def apply_blip_filter(occ, blip_frames):
    """Fill short gaps/spikes shorter than blip_frames."""
    from scipy.ndimage import binary_closing, binary_opening
    occb = occ.astype(bool).copy()
    if blip_frames <= 1:
        return occb
    structure = np.ones(blip_frames, dtype=bool)
    for i in range(occb.shape[1]):
        x = occb[:, i]
        x = binary_closing(x, structure=structure)
        x = binary_opening(x, structure=structure)
        occb[:, i] = x
    return occb

def survival_continuous(occb):
    """
    Continuous survival from t0: fraction that never left shell up to t.
    Only considers molecules in shell at t0.
    """
    T, N = occb.shape
    if N == 0 or T == 0:
        return np.zeros(T)
    start_in = occb[0, :]
    idx = np.where(start_in)[0]
    if idx.size == 0:
        return np.zeros(T)
    lengths = np.zeros(idx.size, dtype=int)
    for k, i in enumerate(idx):
        false_idx = np.argmax(~occb[:, i])
        if false_idx == 0 and occb[:, i].all():
            lengths[k] = T
        elif false_idx == 0:
            lengths[k] = 1
        else:
            lengths[k] = false_idx
    hist = np.bincount(lengths, minlength=T+1)
    survivors = np.cumsum(hist[::-1])[::-1][1:]
    S = survivors / survivors[0]
    return S

def block_edges(T, nblocks):
    sizes = [T // nblocks + (1 if x < T % nblocks else 0) for x in range(nblocks)]
    edges = np.cumsum([0] + sizes)
    return [(edges[i], edges[i+1]) for i in range(nblocks)]

def compute_residence(u, water_resname, ions, rmin, stride, blip_ps, blocks, zbinner=None, prefix="Na"):
    """
    Core loop using water molecule COM.
    Returns dict with S_intermit, S_cont, tau_integral, tau_fit±err.
    """
    times_ps = []
    occ = []          # (T, Nw) occupancy: within rmin of ANY ion
    z0 = None         # initial z positions (COM) for z-binning
    water_group = u.select_atoms(f"resname {water_resname}")
    if water_group.n_atoms == 0:
        raise ValueError(f"No water atoms found with resname {water_resname}")
    n_waters = len(water_group.residues)

    print(f"[INFO] Using {n_waters} water molecules (COM) for {prefix} analysis")

    for ts in u.trajectory[::stride]:
        times_ps.append(ts.time)
        # Compute COM for each water molecule
        water_coms = np.array([res.atoms.center_of_mass() for res in water_group.residues])

        if ions.n_atoms > 0:
            d = mddist.distance_array(water_coms, ions.positions, box=u.dimensions)
            min_d = d.min(axis=1)  # min distance from each water COM to any ion
            occ.append((min_d < rmin).astype(np.int8))
        else:
            occ.append(np.zeros(n_waters, dtype=np.int8))

        if z0 is None:
            z0 = water_coms[:, 2].copy()  # store initial z of COM

    times_ps = np.array(times_ps)
    dt_ps = np.median(np.diff(times_ps))
    occ = np.vstack(occ)  # (T, Nw)
    T, Nw = occ.shape

    def summarize_one(occ_mat, tag):
        S_int = survival_intermit(occ_mat)
        blip_frames = max(1, int(round(blip_ps / dt_ps)))
        occb = apply_blip_filter(occ_mat, blip_frames)
        S_con = survival_continuous(occb)

        tau_int = trapezoid(S_int, dx=dt_ps)

        t = np.arange(T) * dt_ps
        t0 = t[int(0.1 * T)]
        t1 = t[int(0.8 * T)]
        mask = (t >= t0) & (t <= t1) & (S_int > 1e-6)
        A, tau = np.nan, np.nan
        if np.any(mask):
            try:
                (A, tau), _ = curve_fit(exponential, t[mask] - t0, S_int[mask], p0=(1.0, max(1.0, tau_int / 2)))
            except Exception as e:
                print(f"Fit failed for {tag}: {e}")

        tau_blocks = []
        if blocks > 1 and T >= blocks:
            edges = block_edges(T, blocks)
            for a, b in edges:
                if b - a < 10:
                    continue
                Sblk = survival_intermit(occ_mat[a:b, :])
                tau_blocks.append(trapezoid(Sblk, dx=dt_ps))
        tau_err = np.std(tau_blocks, ddof=1) if len(tau_blocks) >= 2 else np.nan

        return {
            "t": t, "S_int": S_int, "S_con": S_con,
            "tau_int": tau_int, "tau_err": tau_err,
            "Afit": A, "tau_fit": tau
        }

    results = {}

    if zbinner is None:
        results["all"] = summarize_one(occ, f"{prefix}-all")
    else:
        zmin, zmax, dz = zbinner
        bins = np.arange(zmin, zmax + dz, dz)
        bidx = np.digitize(z0, bins) - 1
        for bi in range(len(bins) - 1):
            sel = (bidx == bi)
            if sel.sum() < 20:
                continue
            results[f"zbin_{bi}"] = summarize_one(occ[:, sel], f"{prefix}-z{bi}")
        results["bins"] = bins

    return times_ps, results

def plot_results(prefix, label, t, S_int, S_con, color_main, color_cont):
    plt.plot(t, S_int, label=f"{label} intermittent", lw=2, color=color_main)
    plt.plot(t, S_con, label=f"{label} continuous (+blip)", lw=1.8, ls="--", color=color_cont)

def main():
    args = parse_args()
    u = mda.Universe(args.tpr, args.xtc)

    # Select water residue group (for COM calculation)
    water_group = u.select_atoms(f"resname {args.water_res}")
    if water_group.n_atoms == 0:
        sys.exit(f"No water atoms found with resname {args.water_res}; check --water-res")

    na = u.select_atoms(f"resname {args.na_res}")
    cl = u.select_atoms(f"resname {args.cl_res}")

    n_waters = len(water_group.residues)
    print(f"[INFO] Waters (COM): {n_waters}, Na: {na.n_atoms}, Cl: {cl.n_atoms}")

    zbinner = tuple(args.zbins) if args.zbins is not None else None

    # Na–H2O (whole molecule)
    tps_na, res_na = compute_residence(u, args.water_res, na, args.rmin_na,
                                       args.stride, args.blip_ps,
                                       args.blocks, zbinner, prefix="Na")

    # Reset universe for Cl analysis
    u = mda.Universe(args.tpr, args.xtc)
    tps_cl, res_cl = compute_residence(u, args.water_res, cl, args.rmin_cl,
                                       args.stride, args.blip_ps,
                                       args.blocks, zbinner, prefix="Cl")

    # ==== Save results ====
    def save_one(tag, resdict, prefix_out):
        if "bins" in resdict:
            bins = resdict["bins"]
            for k, v in resdict.items():
                if not k.startswith("zbin_"):
                    continue
                idx = int(k.split("_")[1])
                t = v["t"]
                S1 = v["S_int"]
                S2 = v["S_con"]
                np.savetxt(f"{prefix_out}_{tag}_z{idx}.csv",
                           np.c_[t, S1, S2],
                           delimiter=",", header="time_ps,S_intermit,S_continuous", comments="")
        else:
            v = resdict["all"]
            t = v["t"]
            S1 = v["S_int"]
            S2 = v["S_con"]
            np.savetxt(f"{prefix_out}_{tag}.csv",
                       np.c_[t, S1, S2],
                       delimiter=",", header="time_ps,S_intermit,S_continuous", comments="")

    save_one("NaH2O", res_na, args.prefix)
    save_one("ClH2O", res_cl, args.prefix)

    # ==== Plotting ====
    plt.figure(figsize=(6.0, 4.3))
    na_color = 'blue'
    cl_color = 'red'
    na_cont_color = 'lightblue'
    cl_cont_color = 'lightcoral'

    if "bins" in res_na:
        for k, v in res_na.items():
            if not k.startswith("zbin_"):
                continue
            plot_results(args.prefix, f"Na–H2O {k}", v["t"], v["S_int"], v["S_con"], na_color, na_cont_color)
        for k, v in res_cl.items():
            if not k.startswith("zbin_"):
                continue
            plot_results(args.prefix, f"Cl–H2O {k}", v["t"], v["S_int"], v["S_con"], cl_color, cl_cont_color)
        plt.title("Hydration survival by z-bin (whole H₂O)")
    else:
        vna = res_na["all"]
        vcl = res_cl["all"]
        plot_results(args.prefix, "Na–H2O", vna["t"], vna["S_int"], vna["S_con"], na_color, na_cont_color)
        plot_results(args.prefix, "Cl–H2O", vcl["t"], vcl["S_int"], vcl["S_con"], cl_color, cl_cont_color)
        plt.title("Hydration survival (Na–H₂O, Cl–H₂O)")

    plt.xlabel("t (ps)")
    plt.ylabel("S_res(t)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}_Sres.png", dpi=300)

    # ==== Print summary ====
    def print_one(name, rd):
        if "bins" in rd:
            print(f"\n[{name}] z-binned τ_res (integral) per bin:")
            for k, v in rd.items():
                if not k.startswith("zbin_"):
                    continue
                tau, err = v["tau_int"], v["tau_err"]
                print(f"  {k}: tau_int={tau:.2f} ps, err≈{(np.nan if np.isnan(err) else round(err,2))}")
        else:
            v = rd["all"]
            print(f"\n[{name}] τ_res (integral) = {v['tau_int']:.2f} ps,   "
                  f"τ_res(fit) ≈ {np.nan if np.isnan(v['tau_fit']) else round(v['tau_fit'],2)} ps,   "
                  f"block-err ≈ {np.nan if np.isnan(v['tau_err']) else round(v['tau_err'],2)}")

    print_one("Na–H2O", res_na)
    print_one("Cl–H2O", res_cl)

if __name__ == "__main__":
    main()
