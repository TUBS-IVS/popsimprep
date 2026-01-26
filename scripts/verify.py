from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\Felix Petre\PycharmProjects\popsimprep")
control_col = "Insgesamt_Haushalte_Groesse_des_privaten_Haushalts_100m_Gitter_adj_ZENSUS100m"

for folder in sorted(root.glob("popsim_regiostar_*")):
    output_file = folder / "output" / "final_expanded_household_ids.csv"
    geo_file = folder / "data" / "geo_cross_walk.csv"
    control_file = folder / "data" / "control_totals_ZENSUS100m.csv"
    if not output_file.exists():
        continue

    geo = pd.read_csv(geo_file, dtype=str)
    controls = pd.read_csv(control_file, dtype={"ZENSUS100m": str})
    out = pd.read_csv(output_file, dtype=str, usecols=["ZENSUS100m", "ZENSUS1km", "STAAT"])

    geo_cells = set(geo["ZENSUS100m"])
    out_cells = set(out["ZENSUS100m"])
    missing = sorted(geo_cells - out_cells)
    extra = out_cells - geo_cells

    control_map = controls.set_index("ZENSUS100m")[control_col]
    missing_nonzero = [c for c in missing if float(control_map.get(c, 0) or 0) > 0]

    geo_map = geo.set_index("ZENSUS100m")[["ZENSUS1km", "STAAT"]]
    out_map = out.groupby("ZENSUS100m").agg({"ZENSUS1km": "first", "STAAT": "first"})
    mismatches = (out_map.loc[geo_map.index.intersection(out_map.index)] != geo_map.loc[geo_map.index.intersection(out_map.index)]).any(axis=1)

    print(f"{folder.name}: missing={len(missing)} (nonzero={len(missing_nonzero)}), "
          f"extra={len(extra)}, mismatches={mismatches.sum()}")