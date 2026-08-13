# Hardware Integration Plan — curly-octo-happiness × Geometric-to-Binary-Computational-Bridge

Date: 2026-08-13. Basis: deep-read of both repos + external gap search. Constraint honored throughout: **stdlib-only Python, runs on phone/Pi, parts-scarce off-grid context**.

## 1. What the bridge repo already provides (build on, don't rebuild)

- **Fabrication pipeline** (`fabrication/`): geometric spec → bond-graph IR (6 substrates + couplers) → falsifiable claims (`CLAIM_TABLE.fab.json` with value/tol_frac/measurement/failure/provenance) → emit (KiCad netlist, Marlin g-code, OpenSCAD, STL, SVG/DXF, coil schedules, loom) → verify (phone-mic Farina sweep WAV, unit-suffixed CSV, LCR CSV, vibration CSV) → verdict pass/drift/fail with diagnostics localized to physical cause.
- **Exact-ratio couplers** (heater=1.0, transformer=N₂/N₁, friction=1.0) — zero-free-parameter tests where disagreement *is* a measurement leak. These are the backbone of trust calibration.
- **GEIS 8-state octahedral encoder** — lossless geometric↔binary, 3-bit states; already reads phone accelerometer via Termux.
- **Sensing node framework** (`sensing/`): Pi Zero W + DS18B20 + MCP3008/ADS1115 ADC + MLX90614 + AS7341, solar power sizing, LoRa/KISS packet framing (transmitters are no-op stubs awaiting a `send_bytes` callable).
- **Two working MCU firmware sketches** (Silicon/): RP2040/Teensy ultrasonic ring with DATA-line streaming that *refuses to pretend one ADC is a power meter* — the right epistemic posture, already embodied.
- **Cross-substrate triangulation**: 3 independent measurement chains → pairwise agreement localizes the leak.

## 2. Gaps found (repo-internal + external)

| # | Gap | Evidence |
|---|---|---|
| G1 | No serial/UART ingest path (GPIO/I²C/SPI deliberately unpinned; no host reader) | sensing/ docs |
| G2 | LoRa/KISS transmitters are stubs | sensing/ code |
| G3 | RF/SDR tier specified, not implemented | tier3_rf_sensing.md |
| G4 | No oscilloscope/logic-analyzer/VNA interfaces | ARCHITECTURE.md |
| G5 | No CAS/symbolic engine; no dimensional analysis; ECE calibration defined but never computed | REVIEW.md |
| G6 | drift_gate falsification scoped to fabrication/ only; root CLAIM_TABLE.json has no scope/reference-class fields | REVIEW.md |
| G7 | BME680/VEML6075 on roadmap; mechanical-beam and thermal-PDE predictors "planned" | sensing/fabrication docs |
| G8 | Todo.md files are pasted code dumps, not roadmaps (repo hygiene) | REVIEW.md §4.4 |
| G9 | No control/safety runtime layer for anything actuated | — |
| G10 | No power-model claims (the #1 killer of cold-climate nodes) | — |
| G11 | No dimensional-metrology verification of printed/machined parts | emit has no geometric verify loop |
| G12 | Three conflicting licenses (MIT/CC0/CC-BY) | REVIEW.md |

## 3. Proposed integrations (dependency order)

### Tier 0 — Unifying connective tissue (week 1, all stdlib)
**I1. One measurement schema.** Single timestamped CSV/JSONL schema + one claim-manifest format shared by every adapter below. Fields: `t_iso, channel, value, unit_suffix, source_id, seq, crc8`. Every measurement in either repo becomes the same record; every record can feed the Beta-posterior claim machinery (curly-octo-happiness) and the fab ledger (bridge repo). *This is the actual missing piece between the repos and hardware.*

**I2. `serial_csv.py` host reader (stdlib)** — POSIX `termios`/`os.read`+`selectors` on `/dev/tty*`, 921600 baud USB-CDC; firmware frames `seq,micros,v0..vN,crc8`. Reference firmware: 20-line Arduino C + MicroPython variants for RP2040/ESP32. Trust firmware `micros()` timestamps, not host. Handles AVR DTR-reset 2 s wait. Output feeds I1 schema directly and curly-octo-happiness's Gray-coded encoders.

### Tier 1 — Cheap instruments (the $4–$50 instrument rack)
**I3. phyphox adapter** — phone-as-instrument: poll REST API (`http://phone:8080/get?...`) or parse phyphox CSV exports; BLE sensor XML definitions for scavenged boards. Covers accel/gyro/mag/baro/mic/light. → `fabrication/verify/` input adapter.
**I4. Pico-DAQ** — RP2040 ($4, 12-bit, 500 kS/s stock) running **Scoppy** (phone-native scope + logic analyzer over USB-OTG) or **sigrok-pico** (CSV export). This replaces the absent oscilloscope interface (G4) for nearly free.
**I5. RTL-SDR tier-3 implementation** — `rtl_power` CSV spectral surveys → Gray-coded band channels; `rtl_433` for free scavenged sensor data (weather stations, TPMS). Implements the specified-but-missing `rf_scanner.py` (G3); doubles as EMI pre-compliance check on fabricated boards and an RF "hidden node" stream for HND.
**I6. NanoVNA/LCR adapters** — NanoVNASaver CSV → coil L/Q claims; ring-down Q via step + Pico capture (zero extra hardware); $15 LCR-T4 for scavenged-part sorting. Pitfall encoded in claim templates: apparent-L inflates near SRF — always check SRF.

### Tier 2 — Measurement protocol hardening (phone-mic reality)
**I7. Verify-adapter hardening rules** (encode as claim-scope constraints):
- AGC destroys amplitude ratios → relative measures only (frequency, decay rate, coherence) unless raw input confirmed; log-decrement Q is AGC-immune (time-domain ratio).
- Never claims below ~150 Hz from phone mics (high-pass); WAV PCM only (no AAC); in-situ sample-rate calibration against mains hum 50/60 Hz.
- Farina sweep: harmonic distortion folds to negative time → clean IR even through nonlinear phone chains; keep sweeps ≤ 2 s on Pi Zero (pure-Python FFT cost).
- H1 + coherence gating: bins with γ² > 0.8 become claims; map γ² + segment count → Beta prior parameters (direct bridge into curly-octo-happiness confidence machinery).

### Tier 3 — Fab verification closure (G11)
**I8. Expectation manifests** — every emitter (g-code, OpenSCAD, KiCad, coil) also emits a machine-readable manifest (bbox, mass, L/R/Q, clearances). Verify adapters check both the artifact (stdlib g-code envelope/extrusion audit; `kicad-cli pcb drc --exit-code-violations` report → claims) and the physical part (phone photo + printed scale marker, honest ~0.1–0.3 mm/px → fit/no-fit claims only, stated in templates).
**I9. Cross-validation oracles** — regenerate select systems in OpenModelica/BondGraphTools offline; eigenvalue/transfer-function agreement recorded as claims. Keeps the minimal IR while borrowing heavyweight validation. Semi-implicit/backward-Euler fixed-point step (~30 stdlib lines) for stiff thermal-fluidic couplings.

### Tier 4 — Control, safety, power (G9, G10)
**I10. Simplex safety layer (~200 stdlib lines)** — verified-simple safety controller + decision module + untrusted advanced policy; envelope boundaries *derived from the bond-graph IR as claims* (max current, temp, stress). CBF-lite = one-step-lookahead closed-form clamp on scalar constraints (no QP solver needed). Two-tier watchdog: Pi heartbeat file + independent RP2040/ATTiny dead-man MOSFET cut-off (scavenged parts suffice; avoid MicroPython WDT+lightsleep lockup). This instantiates PLAN_FORWARD Phase 3.1/3.2 in buildable form.
**I11. Power claims in the stewardship simulator** — model Li-ion charge-inhibition < 0 °C, supercap self-discharge, worst-month solar (steep tilt ≥ latitude+15°). Recommended node pattern: Li-SOCl₂ primary (−55 °C rated) or LiFePO₄ + supercap for TX pulses. Battery telemetry → falsifiable claim "survives February" with live Beta posterior.

### Tier 5 — Cross-repo epistemics (G5, G6)
**I12. Port drift_gate + scope fields to root CLAIM_TABLE.json** — extend claim schema with scope/reference_class (matching curly-octo-happiness), wire the fab verdict ladder into DependencyTree propagation; actually compute the ECE calibration defined-but-unimplemented in PREDICTION_PROTOCOL.md.
**I13. Hypothesis-engine wiring** — the GitHub Action built earlier gains a hardware topic family: claims from measured artifacts (fab ledger) are first-class alongside literature claims; hidden-variable scan correlates build-failure residuals against env series (temperature, humidity, supply voltage) — GAE/HND meets the fab ledger.

## 4. Repurposing/stewardship tie-in (the ecosystem's signature move)

The hardware-repurposing tables (diode→conductor, drift→sensor) extend naturally: I4's Pico-DAQ is itself a repurposed MCU; I5's rtl_433 scavenges neighbors' sensors as free telemetry; I6's LCR-T4 sorts scavenged parts into band-tables that feed adaptive percentile encoders. Every scavenged component gets (a) a measured characterization claim, (b) a band assignment, (c) a repurposing fallback — closing the loop from salvage → measurement → calibrated claim → safe reuse.

## 5. Sequencing & effort

| Tier | Items | Effort | Unblocks |
|---|---|---|---|
| 0 | I1, I2 | ~1 wk | everything |
| 1 | I3–I6 | 2–4 wks | real measurements |
| 2 | I7 | 1 wk | trustworthy phone data |
| 3 | I8, I9 | 2 wks | fab closure (G11) |
| 4 | I10, I11 | 2–3 wks | actuation, winter survival |
| 5 | I12, I13 | 1–2 wks | unified epistemics |

First physical milestone: **phone + $4 Pico + one exact-ratio coupler (resistor heater)** — measure V, I, ΔT; the Joule-ratio claim has zero free parameters, so any disagreement localizes the measurement chain. That single loop exercises I1, I2, I3, I7, I10 and the full claim machinery end-to-end.

## 6. Repo hygiene items (cheap, do anytime)
Reconcile licenses (G12); move Todo.md code dumps out of root (G8); implement BME680/VEML6075 drivers (G7); adopt manifest metadata for dynamic-payload bridges per bridge_registry_gap_report.
