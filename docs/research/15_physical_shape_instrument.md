# 15 — Physical Shape Instrument v1: Build, Instrument, Falsify

Date: 2026-08-13 · Basis: notes/14 (grounding + sims) · Hardware: hardware/octa_bistable.scad
This is the ecosystem's first physical shape-instrument: an octahedron that predicts its own failure.

---

## 1. Design intent → physical mechanism

The sim-established claims being physically tested:
1. Distortion is a readable gauge of imbalance (linear dose-response).
2. Probe recovery time diverges ~25% of the load range before snap (mechanical CSD).
3. Fluctuation variance jumps ~12% before snap (late corroborating signal).
4. Drill-down: the deformed edge/mode localizes the fault.

**Mechanism**: octahedral frame, 11 rigid edges, 1 bistable edge = **von Mises truss** (two inclined beams meeting at an apex; snap-through between up/down stable states). External compression applied across the antipodal vertex pair via M5 screw. The bistable strut's rest geometry (beam_t thickness, beam_apex height) sets the two stable lengths and the snap force — the printed analog of the sim's quartic well E(l)=a(l−l₁)²(l−l₂)².

## 2. Bill of materials (~$15, no special tooling)
- FDM print: 6 vertex nodes, 10 rigid struts (or 3mm dowel/rod stock cut to s√2 − 2·node_r), 2 snap beams (PETG recommended — PLA is too brittle for repeat snap cycles), 1 apex cap, 2 compression collars.
- M5×80 screw + 2 nuts (compression drive across antipodal nodes).
- Instrumentation (two tiers):
  - **Tier A — phone**: phyphox accelerometer taped to apex cap (uses ecosystem integration I3: phyphox REST/CSV adapter). Phone mic for snap acoustics (I7: ≥150 Hz AGC-off protocol).
  - **Tier B — RP2040**: RP2040 + MPU6050 on the apex cap, streaming timestamped CSV per the I1 contract (`seq,micros,ax,ay,az,crc8`) via serial_csv.py (I2).
- Optional: printed dial-indicator mount or calipers for static apex-height measurement.

## 3. Experiment protocol (ledger-ready; each claim has refutation conditions)

**E-P1 · Gauge law.** Compression screw in 1/8-turn steps; after each, settle 10 s, measure apex height h (calipers) and record. Claim: distortion vs screw displacement is near-linear in the pre-snap range (refuted if R² < 0.9 on log-log or if hysteresis loop area > 30% of range on the down-sweep — real von Mises trusses have hysteresis the sim lacked).

**E-P2 · Critical slowing down (the headline).** At each compression step: flick the apex with a fixed-magnitude impulse (dropped 20 g nut from 30 mm through a guide tube = repeatable kick). Record apex acceleration for 5 s at ≥500 Hz. Recovery time = time for oscillation envelope to decay to 5% of initial (exp fit). Claim: recovery time is flat (±20%) up to ~60% of snap load and diverges before snap — lead ≥ 15% of load range. Refuted if recovery shows no monotone rise or rises only at ≥95% of snap load.

**E-P3 · Variance signature.** Same recordings, quiet 2 s windows: strut-position variance (band-passed 10–100 Hz). Claim: variance rises ≥ 2× baseline before snap. Refuted if flat within noise.

**E-P4 · Flickering.** Near-snap regime: does the strut hop between wells under ambient vibration (table taps)? Claim: flickering onset (dwell times dropping) precedes the static snap point. This is the shape-native signal no scalar load cell gives you.

**E-P5 · Drill-down.** With frame instrumented (tape markers + phone video tracking, or 3 accelerometers): induce the snap, verify modal decomposition of the displacement field localizes to the bistable strut's vertices (6× concentration, per sim).

**E-P6 · Hysteresis loop (the spinodal made physical).** Full up-sweep and down-sweep of compression: the up-snap and down-snap loads bracket the true spinodal gap — a literal, measurable cusp catastrophe fold in a $15 print. Plot load vs apex height: the S-curve with unstable middle branch. Claim: the measured fold geometry fits the cusp normal form (residual < 10%).

## 4. Data → ecosystem wiring
- All recordings: I1 CSV contract → CDT falsification_ledger entries (each E-P claim above with its refutation condition pre-registered; the ledger's `refute()` gate applies to the physical claims).
- Recovery-time series → CDT cascade_regime_audit SignalReads: `critical_slowing_down` (recovery time trend), `variance_inflation`, `flickering` (well-hopping), h_eff = load/snap-load ratio → regime classification STRESSED→CASCADE at the measured spinodal.
- This closes the loop GBCB→CDT→GM: a physical shape instrument whose signals flow through the calibration gate (sensor = GROUND, sim model = PREDICT — the sim is literally the predictive model, tested against the print) into the ledger.

## 5. Known sim-to-print gaps (pre-registered as risk)
- Real von Mises trusses snap asymmetrically and have rate-dependent snap loads (viscoelastic beams) — the sim is rate-free.
- Print anisotropy: layer orientation across the beams matters; print flat (as laid out) so layers run along the beam.
- Beam creep: PETG creeps under sustained load near snap — keep near-snap dwell < 60 s.
- If beam_t=0.8 gives too stiff a snap for hand loading, drop to 0.6; too floppy, raise to 1.0. The well-separation (apex height 6 mm) is the bistability knob; the thickness is the force-scale knob.

## 6. Success criteria (publication-grade)
E-P2 alone, if it replicates the sim (recovery-time divergence with ≥15% lead), is a complete demonstrable result: **a passive mechanical structure whose probe-response times predict its own failure** — critical slowing down made tactile. With the CSV→ledger→audit pipeline attached, it is also the first end-to-end run of the ecosystem on physical data: geometry sensed → signals gated → claims logged → regime called → snap confirmed/refuted.

---

## E-P8 (added 2026-08-14): the snap-latency channel — the instrument's richest signal

Origin: notes/18 §3. Gomez–Moulton–Vella (Nat. Phys. 13, 142, 2017) measured universal
power laws for snap-through delay; Class I neuron spikes (same fold normal form) encode
input amplitude in spike latency. Our E-P2 measures the PRE-snap state; E-P8 measures
the EVENT itself as an analog-to-event-time converter.

**Claim:** under a controlled quasi-static load ramp (constant dε/dt via the M5 collar,
motorized or metronome-paced hand turns), the time-to-snap after ramp start encodes the
strut's initial distance-from-threshold via t_snap ∝ ε_0^(−1/2) (fold law), such that
snap latency is a decodable measurement of the initial condition — the snap is an ADC.

**Protocol:**
1. Set initial compression ε_0 at N=8 levels spanning 0.30–0.45 (below snap at ~0.495).
2. From each ε_0, start a constant-rate ramp (dε/dt ≈ 0.01/s; record audio + phyphox
   for the snap acoustic/transient signature — the snap is loud, timing is easy).
3. Record t_snap. 5 repeats per level. Fit log t_snap vs log ε-distance; extract exponent.
4. Decoder check: withhold 20% of runs, decode ε_0 from t_snap alone via the fitted law;
   report RMSE in compression units.

**Pass:** exponent in [−0.65, −0.35] (fold −1/2 with tolerance for rate effects) AND
decoder RMSE ≤ 0.02 compression (≈ one E-P2 step).
**Refuted if:** exponent outside band (wrong bifurcation or rate-dominated dynamics),
OR decoder no better than the mean-ε_0 baseline.
**Risks:** ramp-rate dependence (run a second rate; the law should collapse when time
is rescaled by rate); threshold wander over repeats (PETG fatigue — randomize level
order; drift shows as monotone residual trend = per se interesting, feed to ledger).

**Why it matters:** E-P2 proves the instrument warns; E-P8 proves the event REPORTS.
Together they close the loop: slow channel warns, fast channel quantifies, the flip
itself is the ledger entry.

---

## Pre-registration arc (2026-08-14, harness runs — this is why the harness exists)
- **v1 (absolute t-test, sequential scan): REFUTED.** Null arm (rigid strut, creep only)
  fired detection at 96%. Raw recovery-time monitoring cannot distinguish CSD from creep.
- **v2 (differential bistable/rigid ratio, sequential scan): REFUTED.** Null 82% —
  scanning 15 steps for any t>1.86 is a multiple-comparisons machine.
- **v3 (differential ratio, ONE pre-committed checkpoint at compression 0.44 = fixed
  18.3% lead, one t-test): SUPPORTED.** 5/5 seeds fire >80%; null FP = 0.05 (nominal);
  robust at 10% timing noise (median 1.00).
**Physical-protocol consequence:** the printed instrument MUST use the two-arm design
(bistable frame + identical rigid frame, ratio as the signal) and a pre-committed
checkpoint — not continuous scanning. If earliest-warning scanning is wanted, it needs
a multiple-comparison correction (O'Brien–Fleming spending) = v4.
Ledger entries: sims/ep2_prereg*/results/*/ledger_entry.jsonl
