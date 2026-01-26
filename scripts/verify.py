from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\Felix Petre\PycharmProjects\popsimprep")
control_col = "Insgesamt_Haushalte_Groesse_des_privaten_Haushalts_100m_Gitter_adj_ZENSUS100m"

ABS_TOL = 0
REL_TOL = 0.002  # 0.2 %

def as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

# ---- summary accumulators ----
n_total = 0
n_ok = 0
n_check = 0

max_abs_delta = 0.0
max_rel_delta = 0.0
worst_folder = None

any_geo_control_issue = False
any_nonzero_missing = False
any_mismatch = False
any_extra = False

for folder in sorted(root.glob("popsim_regiostar_*")):
    output_file = folder / "output" / "final_expanded_household_ids.csv"
    geo_file = folder / "data" / "geo_cross_walk.csv"
    control_file = folder / "data" / "control_totals_ZENSUS100m.csv"
    if not output_file.exists():
        continue

    n_total += 1

    geo = pd.read_csv(geo_file, dtype=str)
    controls = pd.read_csv(control_file, dtype={"ZENSUS100m": str})
    out = pd.read_csv(output_file, dtype=str, usecols=["ZENSUS100m", "ZENSUS1km", "STAAT"])

    geo_cells = set(geo["ZENSUS100m"])
    out_cells = set(out["ZENSUS100m"])
    control_cells = set(controls["ZENSUS100m"])

    # geo vs controls
    geo_missing_in_controls = geo_cells - control_cells
    controls_extra_vs_geo = control_cells - geo_cells
    geo_control_ok = not geo_missing_in_controls and not controls_extra_vs_geo
    if not geo_control_ok:
        any_geo_control_issue = True

    # geo vs out
    missing = sorted(geo_cells - out_cells)
    extra = out_cells - geo_cells
    if extra:
        any_extra = True

    # controls
    controls_num = pd.to_numeric(controls[control_col], errors="coerce").fillna(0)
    control_sum = float(controls_num.sum())
    out_len = int(len(out))
    delta = out_len - control_sum
    rel_delta = abs(delta) / control_sum if control_sum > 0 else 0.0

    max_abs_delta = max(max_abs_delta, abs(delta))
    if rel_delta > max_rel_delta:
        max_rel_delta = rel_delta
        worst_folder = folder.name

    control_map = controls.set_index("ZENSUS100m")[control_col]
    missing_nonzero = [c for c in missing if as_float(control_map.get(c, 0) or 0) > 0]
    if missing_nonzero:
        any_nonzero_missing = True

    # mismatches
    geo_map = geo.set_index("ZENSUS100m")[["ZENSUS1km", "STAAT"]]
    out_map = out.groupby("ZENSUS100m").agg({"ZENSUS1km": "first", "STAAT": "first"})
    idx = geo_map.index.intersection(out_map.index)
    mismatches_n = int((out_map.loc[idx] != geo_map.loc[idx]).any(axis=1).sum())
    if mismatches_n > 0:
        any_mismatch = True

    drift_ok = abs(delta) <= max(ABS_TOL, REL_TOL * control_sum)

    ok = (
        geo_control_ok
        and not extra
        and mismatches_n == 0
        and not missing_nonzero
        and drift_ok
    )

    if ok:
        n_ok += 1
        status = "OK ✅"
    else:
        n_check += 1
        status = "CHECK ⚠️"

    # ---- per-folder output ----
    print(f"\n{folder.name}  {status}")
    print(f"  geo↔controls: missing={len(geo_missing_in_controls)}, extra={len(controls_extra_vs_geo)}")
    print(f"  geo↔out     : missing={len(missing)} (nonzero={len(missing_nonzero)}), "
          f"extra={len(extra)}, mismatches={mismatches_n}")
    print(f"  totals      : control_sum={control_sum:.0f}, out_len={out_len}, "
          f"delta={delta:.0f} "
          f"(tol=max({ABS_TOL}, {REL_TOL:.3%}·control))")

# ---- FINAL SUMMARY ----
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"folders checked : {n_total}")
print(f"OK              : {n_ok}")
print(f"CHECK           : {n_check}")
print()
print(f"max |delta|     : {max_abs_delta:.0f}")
print(f"max rel delta   : {max_rel_delta:.4%}"
      + (f"  ({worst_folder})" if worst_folder else ""))
print()
print("structural issues observed:")
print(f"  geo↔controls mismatch : {'YES' if any_geo_control_issue else 'no'}")
print(f"  nonzero missing cells : {'YES' if any_nonzero_missing else 'no'}")
print(f"  attribute mismatches  : {'YES' if any_mismatch else 'no'}")
print(f"  extra output cells    : {'YES' if any_extra else 'no'}")

if n_check == 0:
    print("\n✔ All regions consistent — only negligible rounding drift.")
else:
    print("\n⚠ Some regions need inspection (see above).")
