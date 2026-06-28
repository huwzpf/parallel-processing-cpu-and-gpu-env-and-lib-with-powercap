# EDP SNR diagnostics — plots and how to read them

This directory holds two complementary signal-to-noise (SNR) studies of the
system-wide **EDP** (Energy–Delay Product) that the dynamic power-cap optimizers
try to minimize. Both ask the same underlying question — *can a gradient
optimizer actually see the effect of changing a power cap above the measurement
noise?* — but they perturb the caps differently:

| Study | Script | What is perturbed | "Signal" |
|-------|--------|-------------------|----------|
| **Single-device perturbation** | `snr_perturbation_test.py` → `snr_perturbation_analysis.py` | **one** device's cap by ±eps, others held at base | Δ system EDP from that one device (what coordinate-wise / SPSA gradients estimate) |
| **Uniform cap sweep** | `snr_test.py` | **all** devices' caps together (global level) | Δ system EDP between two uniform cap levels |

Shared vocabulary:

- **EDP estimate** — one `[OPTIMIZER STEP] edp=...` value the library logs per
  sync *window*. It is noisy window-to-window.
- **N** — number of consecutive windows averaged into one estimate
  (`optimizer_step_interval`). Larger N = less noise, but each estimate takes
  longer to produce. Both studies run **once at the largest N** and reconstruct
  smaller N offline by re-binning the per-window series into blocks of N (exact,
  because the library's N-estimate is just the mean of N windows).
- **Signal** — the change in *mean* EDP caused by the cap change.
- **Noise** — the spread of the EDP estimates at a fixed cap.
- **SNR = signal / noise.** SNR ≥ 1 means the cap's effect rises above the
  noise, i.e. the optimizer can act on it. SNR < 1 means it is drowned out —
  either because the true effect is small (e.g. near the EDP optimum, or a device
  that barely moves system EDP) or because the noise is large (small N).

---

## `snr_perturbation_analysis.py` — single-device perturbation

Consumes `perturbation_samples.csv` + `device_topology.csv` produced by
`snr_perturbation_test.py`. Runs purely offline (never launches the app) and
also writes two derived tables, `point_stats.csv` (per-operating-point block
statistics) and `pair_snr.csv` (the central-difference SNR per
cap/device/eps/N).

```
./venv/bin/python snr_perturbation_analysis.py --out-dir cnn_snr_perturbation_results \
    [--n-values 1 5 10 20 40 60 80] [--snr-threshold 1.0] [--noise global|lag1] \
    [--dist-N 20] [--sanity-N 20]
```

Devices are grouped into **types**: `master-GPU` (rank-0 GPU, no MPI latency),
`slave-GPU`, `slave-CPU`. The headline plots average SNR **across device types**
(each type weighted equally) so the answer isn't dominated by whichever type has
the most devices.

**1. `snr_vs_N.png` — SNR vs averaging window N.**
One panel per base power cap, plus an "aggregate (all caps)" panel; one line per
eps; the dashed red line is SNR = 1.
*Read it as:* how much averaging you need to make a single-device perturbation
detectable. A curve crossing SNR = 1 at, say, N = 20 means the optimizer needs
≥ 20-window estimates to trust that cap's gradient. If no curve reaches 1, that
cap/eps is not usable at any tested N. SNR rising ≈ √N indicates clean averaging
(see plot 3).

**2. `min_N_to_detect.png` — smallest N that reaches SNR ≥ threshold.**
A single heatmap, rows = base cap, cols = eps, averaged across device types; the
number in each cell is the minimum N at which SNR clears the threshold, "never"
(gray) if it doesn't within the tested N range.
*Read it as:* the most actionable summary — "for this cap and eps, budget at
least this many windows per optimizer step." Whole rows of "never" flag caps
where the gradient is too weak to chase (often near the EDP optimum).

**3. `noise_cv_vs_N.png` — relative noise vs N (log-log).**
Per device type, the EDP coefficient of variation (std/mean = 1/SNR-per-point)
vs N, with a dotted 1/√N reference. Two curves: *global* CV and *lag-1* CV
(consecutive-block differences, which strip slow drift).
*Read it as:* does averaging actually pay off? If the measured CV tracks the
1/√N line, noise is clean random error and more N keeps helping. If it flattens
above the reference, a noise floor / drift dominates and raising N buys little.
A gap between global and lag-1 CV is the size of the slow drift within a run.

**4. `edp_response_curves.png` — per-device EDP vs its own cap.**
One panel per base cap; each line is one device's mean system EDP as *its* cap is
swept (others held at base), colored by device type.
*Read it as:* which devices actually move the system EDP and in which direction.
A sloped line = a real gradient the optimizer can exploit; a flat line = that
device barely matters here (so low SNR is "small true signal", not just noise).
A line crossing through a minimum marks that device's local EDP optimum.

**5. `sanity_timeseries_*.png` — raw intuition for one representative cell.**
Per-window EDP time series for −eps / baseline / +eps at one auto-selected
(cap, device, eps); faint = raw windows, bold = N-window block means, dashed =
each run's mean. A text box reports signal, per-window (N=1) SNR, and N-block
SNR.
*Read it as:* *why* averaging matters. The raw windows swing wildly and the three
distributions overlap (tiny N=1 SNR), but the bold block means separate the
+eps/−eps levels — the gap between the dashed means is the signal. Also exposes
any within-run drift. Selectors: `--sanity-cap/-device/-eps/-N`.

**6. `edp_distributions_N20.png` — per-device EDP distributions vs cap.**
Grid of violins: one row per device + a final **aggregate row pooling all
devices**, one column per base cap; each violin is the spread of the N-averaged
EDP at one realized cap, with the median drawn as a response line. (N set by
`--dist-N`.)
*Read it as:* SNR made visual. Where violins of adjacent caps **overlap**, the
cap change is not resolvable at that N (low SNR); where they **separate**, it is.
The bottom aggregate row answers "on average across devices, does bumping one
device shift system EDP?" — tight, flat violins there explain a sub-1 aggregate
SNR. Note this row is *count-weighted* (pools all devices), unlike the
type-equal weighting in plots 1–2.

> Realized vs requested caps: at extreme base caps a ±eps step is clamped to the
> device's min/max, so all signals/gradients use the **realized** cap gap
> (`perturbed_cap_realized`), and clamped duplicates landing on the same cap are
> merged.

---

## `snr_test.py` — uniform cap sweep

Runs the library in `EQUAL_SPLIT_EDP_MONITOR` with **all** devices held at the
same cap, at several fixed cap levels, and measures how distinguishable the EDP
is between cap levels. Here "signal" is the EDP difference between two *uniform*
cap levels and "eps" is the **gap between those cap levels** (cap separation),
not a single-device perturbation. It also sweeps `batch_size`. Writes
`edp_estimates_raw.csv`, `per_cap_stats.csv`, `snr_grid.csv`.

```
python snr_test.py [--app cnn|rnn] [--out-dir snr_results] [--nodes 8 4 ...] \
    [--batch-sizes ...] [--base-caps ...] [--n-values ...]
python snr_test.py --plots-only --out-dir <dir>     # replot from CSVs
```

**1. `edp_dist_bs<bs>_N<N>.png` — EDP estimate distributions per cap.**
One histogram figure per (batch_size, N); one overlaid histogram per cap level.
*Read it as:* the raw separability picture. If the per-cap histograms sit on top
of each other, no N/eps will distinguish those caps; if they pull apart, there is
signal to detect. Tighter histograms at larger N show the averaging gain.

**2. `snr_vs_N.png` — SNR vs N, by cap separation.**
One line per eps (= cap gap `cap_hi − cap_lo`), each the mean SNR over all cap
pairs with that gap and all batch sizes; dashed red = SNR = 1.
*Read it as:* how much averaging is needed to tell two cap levels apart, and how
that improves with a wider cap gap. Larger eps (cap separation) → larger signal →
higher SNR at the same N.

**3. `noise_vs_N.png` — relative noise (CV) vs N (log-log).**
One line per (batch_size, cap) with a dotted 1/√N reference anchored at the
smallest N.
*Read it as:* identical meaning to perturbation plot 3 — whether averaging
reduces noise as clean 1/√N or hits a floor (correlated/drifting noise), which
caps the useful N.

**4. `snr_vs_batchsize.png` — SNR vs batch_size.**
One line per (N, cap pair) plus a bold black mean trend; dashed red = SNR = 1.
*Read it as:* how batch size affects detectability. A rising mean trend means
larger batches give a cleaner EDP signal (fewer, larger work units per window);
use it to pick a batch size that keeps SNR ≥ 1 at a practical N.

---

### Picking parameters from these plots

- **N (optimizer_step_interval):** smallest N whose SNR ≥ 1 in `snr_vs_N` /
  `min_N_to_detect` for the cap regime you operate in. If `noise_*_vs_N` has
  already flattened off 1/√N, do not push N higher — it won't help.
- **eps (perturbation / cap step):** large enough for SNR ≥ 1 but not so large it
  clamps or leaves the locally-linear regime (watch the realized-cap note and the
  response curves).
- **Operating cap:** caps that are "never" detectable usually sit near the EDP
  optimum — informative for where the optimizer will naturally stall.
