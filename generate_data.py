"""
generate_data.py - make the synthetic data pack (Annex B schema), the late data drop and a data card.

    python generate_data.py

No real data and no LLM are used: everything comes from a numpy random generator with seed 42,
so running it again gives exactly the same files.

Writes:
  data/raw/*.csv                    the 4 pack files, 2026-04-01 .. 2026-09-30
  data/raw/data_card.json           how it was made, counts, traps
  data/late_drop/sensor_readings_2026-10.csv
                                    2026-10-01 .. 2026-10-10, one renamed column, one new fault type
  data/hidden/failures_truth.csv    answer key (all true failures). Training never reads it.

Edge cases planted on purpose (guide 4.4):
  missing readings + duplicate timestamps, two drifting vibration sensors (M-022, M-031),
  one stuck pressure sensor (M-017), one changeover logged as a corrective "failure" (M-005),
  repair notes that name the cause (leakage trap), a machine with little history (M-040),
  a part with zero stock and a long lead time, a few negative-RPM rows, work orders in IST (+05:30).
"""
import json

import numpy as np
import pandas as pd

from src.config import HIDDEN_DIR, LATE_DIR, RAW_DIR, SEED

rng = np.random.default_rng(SEED)        # one seeded generator -> reproducible data

# ---------------------------------------------------------------- what to simulate
START = pd.Timestamp("2026-04-01", tz="UTC")
PACK_END = pd.Timestamp("2026-10-01", tz="UTC")     # pack covers readings before this
LATE_END = pd.Timestamp("2026-10-11", tz="UTC")     # late drop covers PACK_END .. LATE_END
PLAN_END = pd.Timestamp("2026-10-20", tz="UTC")     # schedule is planned ahead
N_MACHINES = 40
DRIFT = {"M-022": ("2026-07-10", 0.04), "M-031": ("2026-08-15", 0.05)}   # start, mm/s per day
STUCK = ("M-017", "2026-06-05", "2026-06-12")                # pressure sensor frozen between these dates
SHORT_HISTORY = ("M-040", "2026-09-28")                      # machine installed late -> little history
TRAP_CHANGEOVER_MACHINE = "M-005"                            # one changeover logged as a "failure"
# Failures in the week AFTER the late drop (hidden outcomes we dry-run the late drop against).
LATE_FAILURES = {"M-003": "wear", "M-011": "wear", "M-019": "wear", "M-027": "wear",
                 "M-008": "pressure_leak", "M-034": "pressure_leak"}   # pressure_leak = new fault type
# Repair notes name the cause - written after the failure, so they must never be a feature.
CAUSES = ["Drive-end bearing worn out, replaced bearing", "Bearing seized, replaced bearing and realigned shaft",
          "Motor overheating, replaced cooling fan", "Coupling misalignment caused vibration, realigned",
          "Gearbox bearing failure, replaced"]

# Machine ids M-001..M-040, each with a type (A/B/C, different normal levels) and a criticality (1-3).
machines = [f"M-{i:03d}" for i in range(1, N_MACHINES + 1)]
mtype = {m: t for m, t in zip(machines, rng.choice(["A", "B", "C"], N_MACHINES, p=[0.4, 0.35, 0.25]))}
crit = {m: int(c) for m, c in zip(machines, rng.choice([1, 2, 3], N_MACHINES, p=[0.3, 0.4, 0.3]))}


def make_schedule(m):
    """Production schedule of one machine.

    Runs of 3-8 days separated by 3h changeovers; a 6h planned maintenance about every 30 days.
    Each run gets a random load (0.85-1.15) that changes speed, vibration and temperature.
    """
    rows, t = [], START if m != SHORT_HISTORY[0] else pd.Timestamp(SHORT_HISTORY[1], tz="UTC")
    next_pm = t + pd.Timedelta(days=int(rng.integers(15, 35)))
    while t < PLAN_END:
        end = t + pd.Timedelta(hours=int(rng.integers(3 * 24, 8 * 24)))
        rows.append((m, t, end, "run", float(rng.uniform(0.85, 1.15))))
        if end >= next_pm:
            stop, kind = end + pd.Timedelta(hours=6), "planned_maintenance"
            next_pm = end + pd.Timedelta(days=int(rng.integers(26, 34)))
        else:
            stop, kind = end + pd.Timedelta(hours=3), "changeover"
        rows.append((m, end, stop, kind, np.nan))
        t = stop
    return pd.DataFrame(rows, columns=["machine_id", "start", "end", "kind", "load"])


def make_failures(m, sched):
    """Random unplanned failures of one machine, during runs and at least 20 days apart.

    Returns [(time, mode, ramp_days, amp)]:
      mode       "wear" (vibration + temperature rise first), "sudden" (no warning) or
                 "pressure_leak" (pressure drops; the new fault type, only in the late drop)
      ramp_days  how many days before the failure the warning signs start
      amp        how strong the warning signs are
    """
    if m in DRIFT or m == SHORT_HISTORY[0]:
        return []
    runs = sched[sched.kind == "run"]
    in_run = lambda t: ((runs.start <= t) & (runs.end > t)).any()   # failures only happen while running
    out = []
    for _ in range(rng.poisson(2.0)):
        t = START + pd.Timedelta(minutes=int(rng.integers(15 * 1440, (PACK_END - START).days * 1440 - 1440)))
        if in_run(t) and all(abs((t - f[0]).days) > 20 for f in out):
            mode = "wear" if rng.random() < 0.75 else "sudden"
            out.append((t, mode, float(rng.uniform(5, 12)), float(rng.uniform(0.6, 1.4))))
    if m in LATE_FAILURES:
        t = pd.Timestamp("2026-10-12", tz="UTC") + pd.Timedelta(minutes=int(rng.integers(0, 4 * 1440)))
        while not in_run(t):
            t += pd.Timedelta(hours=4)
        out = [f for f in out if (t - f[0]).days > 20]
        out.append((t, LATE_FAILURES[m], float(rng.uniform(7, 12)), float(rng.uniform(0.8, 1.2))))
    return sorted(out)


def make_readings(m, sched, failures):
    """10-minute sensor readings of one machine, including wear, drift, stuck sensor and gaps."""
    ts = pd.date_range(sched.start.min().ceil("10min"), LATE_END, freq="10min", inclusive="left")
    n = len(ts)
    t_days = ((ts - START) / pd.Timedelta("1D")).to_numpy()
    runs = sched[sched.kind == "run"].reset_index(drop=True)
    # Find the production run of every timestamp -> is the machine running, and at what load?
    idx = np.searchsorted(runs.start.to_numpy(), ts.to_numpy(), side="right") - 1
    running = (idx >= 0) & (ts.to_numpy() < runs.end.to_numpy()[np.clip(idx, 0, None)])
    load = np.where(idx >= 0, runs.load.to_numpy()[np.clip(idx, 0, None)], 1.0)

    # Normal behaviour: each machine type has its own level; load and noise move it around.
    base_t = 58 + {"A": 0, "B": 4, "C": 8}[mtype[m]] + rng.normal(0, 2)
    base_v = {"A": 1.8, "B": 2.2, "C": 2.6}[mtype[m]] * rng.uniform(0.9, 1.1)
    base_p = 6 + rng.normal(0, 0.4)
    temp = base_t + 10 * (load - 1) + 1.5 * np.sin(2 * np.pi * t_days) + rng.normal(0, 0.5, n)
    vib = base_v * load + rng.normal(0, 0.15, n)
    pres = base_p + rng.normal(0, 0.1, n)
    rpm = 1500 * load + rng.normal(0, 10, n)

    # Failures: warning signs ramp up (p goes 0 -> 1) before the failure, then the machine is
    # down (no readings) until the repair is finished.
    down = np.zeros(n, bool)
    for f_ts, mode, ramp, amp in failures:
        f_day = (f_ts - START) / pd.Timedelta("1D")
        p = np.clip((t_days - (f_day - ramp)) / ramp, 0, None) ** 1.5 * (t_days < f_day)
        if mode == "wear":
            vib += 2.5 * amp * p
            temp += 8 * amp * p
        elif mode == "pressure_leak":
            pres -= 1.8 * amp * p
        repair_end = f_ts + pd.Timedelta(hours=float(rng.uniform(4, 30)))
        down |= (ts >= f_ts) & (ts < repair_end)
    # Drifting vibration sensor: the reading creeps up although the machine is fine.
    for d_m, (d_start, rate) in DRIFT.items():
        if m == d_m:
            vib += rate * np.clip(t_days - (pd.Timestamp(d_start, tz="UTC") - START).days, 0, None)
    # Stuck pressure sensor: the same value for a whole week.
    if m == STUCK[0]:
        stuck = (ts >= pd.Timestamp(STUCK[1], tz="UTC")) & (ts < pd.Timestamp(STUCK[2], tz="UTC"))
        pres[stuck] = 6.12

    keep = running & ~down & (rng.random(n) > 0.01)            # 1% random missing readings
    df = pd.DataFrame({"machine_id": m, "ts": ts, "temperature_c": temp.round(2),
                       "vibration_mm_s": np.abs(vib).round(3), "pressure_bar": pres.round(3),
                       "rpm": rpm.round(1)})[keep]
    return df


def failure_wos(m, failures, start_id):
    """Corrective work orders for the failures inside the pack period (notes name the cause)."""
    rows = []
    for i, (f_ts, mode, _, _) in enumerate(f for f in failures if f[0] < PACK_END):
        closed = f_ts + pd.Timedelta(hours=float(rng.uniform(4, 30)))
        note = "" if mode == "sudden" and rng.random() < 0.3 else str(rng.choice(CAUSES))
        if mode == "sudden" and note:
            note = "Sudden trip, no warning. " + note
        rows.append((f"WO-{start_id + i:05d}", m, f_ts, closed, "corrective", note))
    return rows


def main():
    """Simulate every machine, add the planted traps, write all files and the data card."""
    for d in (RAW_DIR, LATE_DIR, HIDDEN_DIR):
        d.mkdir(parents=True, exist_ok=True)

    scheds, readings, truth, wos = [], [], [], []
    for m in machines:
        s = make_schedule(m)
        f = make_failures(m, s)
        scheds.append(s)
        readings.append(make_readings(m, s, f))
        truth += [(m, t, mode) for t, mode, _, _ in f]
        wos += failure_wos(m, f, start_id=len(wos) + 1)
        # Planned maintenance -> preventive work orders (a few with notes).
        pm = s[(s.kind == "planned_maintenance") & (s.start < PACK_END)]
        for _, r in pm.iterrows():
            note = str(rng.choice(["Routine PM: lubrication and filter change", "PM: belt tension checked",
                                   "PM: sensors cleaned"])) if rng.random() < 0.3 else ""
            wos.append((f"WO-{len(wos) + 1:05d}", m, r.start, r.end, "preventive", note))

    sched = pd.concat(scheds, ignore_index=True)
    # Trap: one planned changeover logged by the shift as a corrective "failure".
    co = sched[(sched.machine_id == TRAP_CHANGEOVER_MACHINE) & (sched.kind == "changeover")
               & (sched.start > pd.Timestamp("2026-06-01", tz="UTC"))].iloc[0]
    wos.append((f"WO-{len(wos) + 1:05d}", TRAP_CHANGEOVER_MACHINE, co.start + pd.Timedelta(minutes=20),
                co.end, "corrective", "Line stopped - breakdown?? (product changeover)"))

    sched["criticality"] = sched.machine_id.map(crit)
    sched["machine_type"] = sched.machine_id.map(mtype)
    sched = sched.drop(columns="load").sort_values(["machine_id", "start"])
    wo = pd.DataFrame(wos, columns=["wo_id", "machine_id", "opened_at", "closed_at", "type", "note"])
    wo = wo.sort_values("opened_at")
    r = pd.concat(readings, ignore_index=True)

    # Split readings into the data pack and the late drop, then add duplicates and invalid rows.
    pack = r[r.ts < PACK_END]
    late = r[(r.ts >= PACK_END) & (r.ts < LATE_END)]
    clean_rows = len(pack)
    dups = pack.sample(frac=0.003, random_state=SEED)                 # duplicate timestamps
    bad = pack.sample(n=25, random_state=SEED + 1).assign(rpm=-1.0)   # physically impossible
    pack = pd.concat([pack, dups, bad]).sort_values(["machine_id", "ts"], kind="stable")

    # ---- write the 4 pack files (sensors in UTC, work orders in plant local time IST)
    iso = "%Y-%m-%dT%H:%M:%SZ"
    pack.assign(ts=pack.ts.dt.strftime(iso)).to_csv(RAW_DIR / "sensor_readings.csv", index=False)
    ist = lambda s: s.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%dT%H:%M:%S%z")   # plant local time
    wo.assign(opened_at=ist(wo.opened_at), closed_at=ist(wo.closed_at)).to_csv(RAW_DIR / "work_orders.csv", index=False)
    sched.assign(start=sched.start.dt.strftime(iso), end=sched.end.dt.strftime(iso)).to_csv(
        RAW_DIR / "production_schedule.csv", index=False)
    parts = pd.DataFrame([
        ("P-BRG-6205", "part", "14", 3, ""), ("P-BRG-6310", "part", "0", 45, ""),   # zero stock, long lead
        ("P-SEAL-22", "part", "30", 5, ""), ("P-FAN-110", "part", "4", 10, ""),
        ("P-COUP-08", "part", "6", 7, ""),
        ("T-01", "technician", "senior_mechanical", None, "A"), ("T-02", "technician", "mechanical", None, "B"),
        ("T-03", "technician", "electrical", None, "C"), ("T-04", "technician", "senior_electrical", None, "A"),
    ], columns=["item", "kind", "qty_or_skill", "lead_time_days", "shift"])
    parts.to_csv(RAW_DIR / "parts_and_roster.csv", index=False)

    # Late drop: renamed column + new fault type (pressure_leak).
    late.assign(ts=late.ts.dt.strftime(iso)).rename(columns={"vibration_mm_s": "vibration_mms"}).to_csv(
        LATE_DIR / "sensor_readings_2026-10.csv", index=False)
    pd.DataFrame(truth, columns=["machine_id", "failure_ts", "mode"]).sort_values("failure_ts").to_csv(
        HIDDEN_DIR / "failures_truth.csv", index=False)

    # Data card (guide Annex D); validation compares the loaded row counts with it.
    card = {
        "name": "pm-synthetic-pack", "version": "1.0", "created": "2026-10-11",
        "how_made": "generate_data.py (numpy random generator, seed 42). No LLM, no real data.",
        "date_range": [str(START.date()), str((PACK_END - pd.Timedelta('1D')).date())],
        "machines": N_MACHINES,
        "expected_rows_after_cleaning": {"sensor_readings": clean_rows, "work_orders": len(wo),
                                         "production_schedule": len(sched), "parts_and_roster": len(parts)},
        "raw_rows": {"sensor_readings": len(pack), "duplicates_added": len(dups), "invalid_rows_added": len(bad)},
        "unplanned_failures_in_pack": int(sum(t < PACK_END for _, t, _ in truth)),
        "traps": {"drifting_sensor": list(DRIFT), "stuck_sensor": STUCK[0], "short_history": SHORT_HISTORY[0],
                  "changeover_logged_as_failure": TRAP_CHANGEOVER_MACHINE,
                  "leaky_notes": "corrective notes name the cause and are written after the failure",
                  "zero_stock_part": "P-BRG-6310", "timezones": "work orders in +05:30, sensors in UTC"},
        "late_drop": {"range": [str(PACK_END.date()), str((LATE_END - pd.Timedelta('1D')).date())],
                      "renamed_column": "vibration_mm_s -> vibration_mms", "new_fault_type": "pressure_leak",
                      "rows": len(late)},
        "privacy": "Fully synthetic. No real company or personal data.",
    }
    (RAW_DIR / "data_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
