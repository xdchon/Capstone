from __future__ import annotations

import csv
import html
import json
import math
import shutil
import sqlite3
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import setup_db


def _norm(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def _to_float(value: object) -> float | None:
    return setup_db._to_number(value)


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stdev(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else None


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    normalized = {_norm(col): col for col in columns}
    for candidate in candidates:
        exact = _norm(candidate)
        if exact in normalized:
            return normalized[exact]
    for col in columns:
        n = _norm(col)
        if any(_norm(candidate) in n for candidate in candidates):
            return col
    return None


def detect_trackmate_columns(columns: Sequence[str]) -> Dict[str, str | None]:
    non_intensity = [c for c in columns if "intensity" not in c.casefold()]
    return {
        "track_id": _find_column(non_intensity, ["TRACK_ID", "TRACKID", "TRACK ID"]),
        "x": _find_column(non_intensity, ["POSITION_X", "POSITION X", "X"]),
        "y": _find_column(non_intensity, ["POSITION_Y", "POSITION Y", "Y"]),
        "z": _find_column(non_intensity, ["POSITION_Z", "POSITION Z", "Z"]),
        "frame": _find_column(non_intensity, ["FRAME", "POSITION_T", "POSITION T", "T"]),
        "radius": _find_column(non_intensity, ["RADIUS", "DIAMETER", "SIZE", "AREA", "SURF", "VOL"]),
    }


def can_build_trackmate_report(db_path: Path, table_name: str, source_csv: str | None = None) -> bool:
    schema = setup_db.table_schema(db_path, table_name)
    cols = [name for name, _ in schema]
    detected = detect_trackmate_columns(cols)
    if not (detected["track_id"] and detected["x"] and detected["y"] and detected["frame"] and detected["radius"]):
        return False
    if source_csv is None:
        return True
    _columns, rows = _fetch_rows(db_path, table_name, source_csv)
    return bool(rows)


def _fetch_rows(db_path: Path, table_name: str, source_csv: str | None = None) -> Tuple[List[str], List[Dict[str, object]]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if source_csv is None:
            cur = conn.execute(f'SELECT * FROM "{table_name}";')
        else:
            cur = conn.execute(
                f'SELECT * FROM "{table_name}" WHERE source_csv = ?;',
                (source_csv,),
            )
        rows = [dict(row) for row in cur.fetchall()]
        return list(rows[0].keys()) if rows else [name for name, _ in setup_db.table_schema(db_path, table_name)], rows
    finally:
        conn.close()


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _feature_scale(rows: List[Dict[str, object]], feature_cols: Sequence[str]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for row in rows:
        matrix.append([float(row.get(col) or 0.0) for col in feature_cols])
    if not matrix:
        return []
    scaled = [[0.0 for _ in feature_cols] for _ in matrix]
    for j in range(len(feature_cols)):
        values = [row[j] for row in matrix]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        sd = math.sqrt(var) or 1.0
        for i, row in enumerate(matrix):
            scaled[i][j] = (row[j] - mean) / sd
    return scaled


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans(x: List[List[float]], k: int, max_iter: int = 80) -> Tuple[List[int], float]:
    if not x or len(x) < k:
        return [], float("inf")
    centers = [x[int(i * len(x) / k)] for i in range(k)]
    labels = [0] * len(x)
    for _ in range(max_iter):
        changed = False
        for i, row in enumerate(x):
            label = min(range(k), key=lambda c: _dist(row, centers[c]))
            if label != labels[i]:
                labels[i] = label
                changed = True
        new_centers = []
        for c in range(k):
            members = [row for row, label in zip(x, labels) if label == c]
            if not members:
                new_centers.append(centers[c])
            else:
                new_centers.append([sum(row[j] for row in members) / len(members) for j in range(len(x[0]))])
        centers = new_centers
        if not changed:
            break
    inertia = sum(_dist(row, centers[label]) ** 2 for row, label in zip(x, labels))
    return labels, inertia


def _agglomerative(x: List[List[float]], k: int) -> List[int]:
    clusters = [[i] for i in range(len(x))]
    while len(clusters) > k:
        best = (float("inf"), 0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = sum(_dist(x[a], x[b]) for a in clusters[i] for b in clusters[j]) / (len(clusters[i]) * len(clusters[j]))
                if d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        clusters[i].extend(clusters[j])
        del clusters[j]
    labels = [0] * len(x)
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = label
    return labels


def _silhouette(x: List[List[float]], labels: List[int]) -> float | None:
    if len(set(labels)) < 2 or len(x) < 3:
        return None
    scores = []
    for i, row in enumerate(x):
        own = labels[i]
        own_d = [_dist(row, x[j]) for j in range(len(x)) if labels[j] == own and j != i]
        if not own_d:
            continue
        a = sum(own_d) / len(own_d)
        b = min(
            sum(_dist(row, x[j]) for j in range(len(x)) if labels[j] == other) / labels.count(other)
            for other in set(labels)
            if other != own and labels.count(other)
        )
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return sum(scores) / len(scores) if scores else None


def _gmm_diag(x: List[List[float]], k: int, max_iter: int = 40) -> Tuple[List[int], float, float]:
    if not x or len(x) < k:
        return [], float("inf"), float("inf")
    labels, _ = _kmeans(x, k)
    d = len(x[0])
    n = len(x)
    weights = [1.0 / k] * k
    means = [[0.0] * d for _ in range(k)]
    vars_ = [[1.0] * d for _ in range(k)]
    resp = [[0.0] * k for _ in range(n)]
    for _ in range(max_iter):
        for c in range(k):
            members = [x[i] for i, label in enumerate(labels) if label == c]
            if not members:
                continue
            weights[c] = len(members) / n
            means[c] = [sum(row[j] for row in members) / len(members) for j in range(d)]
            vars_[c] = [max(1e-6, sum((row[j] - means[c][j]) ** 2 for row in members) / len(members)) for j in range(d)]
        for i, row in enumerate(x):
            probs = []
            for c in range(k):
                logp = math.log(max(weights[c], 1e-9))
                for j in range(d):
                    logp += -0.5 * (math.log(2 * math.pi * vars_[c][j]) + ((row[j] - means[c][j]) ** 2 / vars_[c][j]))
                probs.append(logp)
            max_log = max(probs)
            exps = [math.exp(p - max_log) for p in probs]
            total = sum(exps) or 1.0
            resp[i] = [v / total for v in exps]
        labels = [max(range(k), key=lambda c: resp[i][c]) for i in range(n)]
    log_likelihood = 0.0
    for row in x:
        comps = []
        for c in range(k):
            logp = math.log(max(weights[c], 1e-9))
            for j in range(d):
                logp += -0.5 * (math.log(2 * math.pi * vars_[c][j]) + ((row[j] - means[c][j]) ** 2 / vars_[c][j]))
            comps.append(logp)
        m = max(comps)
        log_likelihood += m + math.log(sum(math.exp(v - m) for v in comps))
    params = k * (2 * d) + (k - 1)
    aic = 2 * params - 2 * log_likelihood
    bic = math.log(n) * params - 2 * log_likelihood
    return labels, aic, bic


def _pca_2d(x: List[List[float]]) -> List[Tuple[float, float]]:
    if not x:
        return []
    # Small deterministic fallback: project to first two standardized features.
    if len(x[0]) == 1:
        return [(row[0], 0.0) for row in x]
    return [(row[0], row[1]) for row in x]


def _descriptor(value: float | None, low: float, high: float, labels: Tuple[str, str, str]) -> str:
    if value is None:
        return "unknown"
    if value <= low:
        return labels[0]
    if value >= high:
        return labels[2]
    return labels[1]


def _quantiles(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    return ordered[int(0.33 * (len(ordered) - 1))], ordered[int(0.66 * (len(ordered) - 1))]


def _compute_track_metrics(rows: List[Dict[str, object]], cols: Dict[str, str | None]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        tid = str(row.get(cols["track_id"] or "", "")).strip()
        if tid:
            grouped[tid].append(row)

    metrics = []
    step_rows = []
    for tid, track_rows in grouped.items():
        parsed = []
        warnings = []
        for row in track_rows:
            frame = _to_float(row.get(cols["frame"] or ""))
            x = _to_float(row.get(cols["x"] or ""))
            y = _to_float(row.get(cols["y"] or ""))
            z = _to_float(row.get(cols["z"] or "")) if cols.get("z") else 0.0
            radius = _to_float(row.get(cols["radius"] or ""))
            parsed.append({"frame": frame, "x": x, "y": y, "z": z, "radius": radius})
        parsed = [p for p in parsed if p["frame"] is not None]
        parsed.sort(key=lambda p: float(p["frame"]))
        if len(parsed) < len(track_rows):
            warnings.append("missing_frame")
        radii = [p["radius"] for p in parsed if p["radius"] is not None]
        coords = [p for p in parsed if p["x"] is not None and p["y"] is not None and p["frame"] is not None]
        if len(coords) < len(parsed):
            warnings.append("missing_position")
        if not radii:
            warnings.append("missing_radius")
        steps = []
        for a, b in zip(coords, coords[1:]):
            dt = float(b["frame"]) - float(a["frame"])
            if dt <= 0:
                warnings.append("nonpositive_time_step")
                continue
            dx = float(b["x"]) - float(a["x"])
            dy = float(b["y"]) - float(a["y"])
            dz = float(b["z"] or 0.0) - float(a["z"] or 0.0)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            speed = distance / dt
            steps.append({"dt": dt, "dx": dx, "dy": dy, "dz": dz, "distance": distance, "speed": speed})
            step_rows.append({
                "TRACK_ID": tid,
                "from_frame": a["frame"],
                "to_frame": b["frame"],
                "dt": dt,
                "distance": distance,
                "speed": speed,
            })
        n = len(parsed)
        if n < 5:
            warnings.append("short_track")
        speeds = [s["speed"] for s in steps]
        distances = [s["distance"] for s in steps]
        duration = (float(parsed[-1]["frame"]) - float(parsed[0]["frame"])) if len(parsed) >= 2 else 0.0
        path_length = sum(distances)
        if len(coords) >= 2:
            net = math.sqrt(
                (float(coords[-1]["x"]) - float(coords[0]["x"])) ** 2
                + (float(coords[-1]["y"]) - float(coords[0]["y"])) ** 2
                + (float(coords[-1]["z"] or 0.0) - float(coords[0]["z"] or 0.0)) ** 2
            )
        else:
            net = None
        msd_lag1 = _mean([d * d for d in distances]) if distances else None
        row = {
            "TRACK_ID": tid,
            "n_detections": n,
            "n_steps": len(steps),
            "track_duration": duration,
            "mean_radius": _mean(radii),
            "median_radius": _median(radii),
            "min_radius": min(radii) if radii else None,
            "max_radius": max(radii) if radii else None,
            "radius_variability": _stdev(radii),
            "estimated_area_from_radius": math.pi * (_mean(radii) ** 2) if radii and _mean(radii) is not None else None,
            "path_length": path_length,
            "net_displacement": net,
            "mean_speed": _mean(speeds),
            "median_speed": _median(speeds),
            "max_speed": max(speeds) if speeds else None,
            "speed_variability": _stdev(speeds),
            "straightness_directionality_ratio": _safe_ratio(net, path_length),
            "x_span": (max(p["x"] for p in coords) - min(p["x"] for p in coords)) if coords else None,
            "y_span": (max(p["y"] for p in coords) - min(p["y"] for p in coords)) if coords else None,
            "z_span": (max(float(p["z"] or 0.0) for p in coords) - min(float(p["z"] or 0.0) for p in coords)) if coords else None,
            "step_length_variability": _stdev(distances),
            "msd_lag1": msd_lag1,
            "warning_flags": ";".join(sorted(set(warnings))) if warnings else "",
            "is_high_confidence": not warnings and len(steps) >= 4 and path_length > 0,
        }
        metrics.append(row)
    return metrics, step_rows


def _add_descriptors(rows: List[Dict[str, object]], labels: List[int]) -> None:
    radii = [float(r["mean_radius"]) for r in rows if r.get("mean_radius") is not None]
    speeds = [float(r["median_speed"]) for r in rows if r.get("median_speed") is not None]
    straight = [float(r["straightness_directionality_ratio"]) for r in rows if r.get("straightness_directionality_ratio") is not None]
    r1, r2 = _quantiles(radii)
    s1, s2 = _quantiles(speeds)
    d1, d2 = _quantiles(straight)
    for row, label in zip(rows, labels):
        size = _descriptor(row.get("mean_radius"), r1, r2, ("small", "medium", "large"))
        speed = _descriptor(row.get("median_speed"), s1, s2, ("slow", "moderate-speed", "fast"))
        direction = _descriptor(row.get("straightness_directionality_ratio"), d1, d2, ("confined/tortuous", "mixed", "directional"))
        row["candidate_class_number"] = label + 1
        row["candidate_movement_class"] = f"Candidate class {label + 1}: {size}/{speed}/{direction}"
        row["candidate_size_class"] = size
        row["candidate_speed_class"] = speed
        row["directionality_descriptor"] = direction
        row["msd_motion_descriptor"] = "larger step variance" if (row.get("step_length_variability") or 0) > (_median([r.get("step_length_variability") or 0 for r in rows]) or 0) else "lower step variance"
        row["track_quality_descriptor"] = "high confidence" if row.get("is_high_confidence") else "review warnings"
        row["interpretation_notes"] = "Statistical candidate class only; not a confirmed biological chloroplast type."


def _write_scatter_svg(path: Path, title: str, rows: List[Dict[str, object]], x_key: str, y_key: str, label_key: str, x_label: str, y_label: str) -> None:
    width, height = 920, 620
    ml, mt, mr, mb = 82, 54, 30, 76
    pts = [(r.get(x_key), r.get(y_key), int(r.get(label_key, 0) or 0)) for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if not pts:
        path.write_text(f"<svg width='{width}' height='{height}'><text x='30' y='60'>No plot data</text></svg>", encoding="utf-8")
        return
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmax += 1
    if ymin == ymax: ymax += 1
    colors = ["#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#984ea3"]
    circles = []
    for x, y, label in pts:
        px = ml + (float(x) - xmin) / (xmax - xmin) * (width - ml - mr)
        py = mt + (1 - (float(y) - ymin) / (ymax - ymin)) * (height - mt - mb)
        circles.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{colors[label % len(colors)]}' fill-opacity='0.76'/>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="28" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>
<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#333"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#333"/>
{''.join(circles)}
<text x="{width/2-80:.1f}" y="{height-24}" font-family="Arial" font-size="14">{html.escape(x_label)}</text>
<text x="18" y="{mt+18}" font-family="Arial" font-size="14">{html.escape(y_label)}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_group_bar_svg(path: Path, title: str, rows: List[Dict[str, object]], class_key: str, value_key: str) -> None:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None:
            grouped[str(row.get(class_key))].append(float(row[value_key]))
    data = [{"label": k, "value": _median(v) or 0.0} for k, v in sorted(grouped.items())]
    width, height = 900, 480
    maxv = max([d["value"] for d in data] + [1.0])
    bars = []
    for i, d in enumerate(data):
        x = 90 + i * 130
        h = d["value"] / maxv * 300
        bars.append(f"<rect x='{x}' y='{380-h:.1f}' width='80' height='{h:.1f}' fill='#5f8f63'/><text x='{x}' y='405' font-size='12'>{html.escape(d['label'])}</text><text x='{x}' y='{370-h:.1f}' font-size='12'>{d['value']:.3g}</text>")
    path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><rect width='100%' height='100%' fill='white'/><text x='28' y='34' font-size='22' font-family='Arial' font-weight='700'>{html.escape(title)}</text><line x1='70' y1='380' x2='850' y2='380' stroke='#333'/>{''.join(bars)}</svg>", encoding="utf-8")


def build_trackmate_report_package(db_path: Path, table_name: str, output_dir: Path, source_csv: str | None = None) -> Path:
    report_dir = output_dir / "chloroplast_tracking_analysis"
    plots_dir = report_dir / "plots"
    tables_dir = report_dir / "tables"
    code_dir = report_dir / "code"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    plots_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)

    columns, rows = _fetch_rows(db_path, table_name, source_csv)
    if source_csv is not None and not rows:
        raise RuntimeError(f"No rows found for dataset '{source_csv}'.")
    detected = detect_trackmate_columns(columns)
    metrics, step_rows = _compute_track_metrics(rows, detected)
    high = [row.copy() for row in metrics if row["is_high_confidence"]]
    feature_cols = ["mean_radius", "median_speed", "straightness_directionality_ratio", "path_length", "speed_variability", "step_length_variability"]
    high = [row for row in high if all(row.get(col) is not None for col in feature_cols)]
    x = _feature_scale(high, feature_cols)

    comparisons = []
    primary_labels = [0] * len(high)
    best_score = -999.0
    for method in ["kmeans", "agglomerative", "gmm"]:
        for k in range(2, min(5, len(high)) + 1):
            if method == "kmeans":
                labels, inertia = _kmeans(x, k)
                sil = _silhouette(x, labels)
                aic = bic = None
            elif method == "agglomerative":
                labels = _agglomerative(x, k)
                inertia = None
                sil = _silhouette(x, labels)
                aic = bic = None
            else:
                labels, aic, bic = _gmm_diag(x, k)
                inertia = None
                sil = _silhouette(x, labels)
            comparisons.append({"model": method, "k": k, "silhouette": sil, "inertia": inertia, "aic": aic, "bic": bic})
            if sil is not None and sil > best_score and method in {"kmeans", "agglomerative"}:
                best_score = sil
                primary_labels = labels
    if high:
        _add_descriptors(high, primary_labels)

    high_by_id = {row["TRACK_ID"]: row for row in high}
    for row in metrics:
        hc = high_by_id.get(row["TRACK_ID"])
        if hc:
            row.update({k: hc.get(k) for k in ["candidate_class_number", "candidate_movement_class", "candidate_size_class", "candidate_speed_class", "directionality_descriptor", "msd_motion_descriptor", "track_quality_descriptor", "interpretation_notes"]})
        else:
            row.update({
                "candidate_class_number": "",
                "candidate_movement_class": "not classified",
                "candidate_size_class": "not classified",
                "candidate_speed_class": "not classified",
                "directionality_descriptor": "not classified",
                "msd_motion_descriptor": "not classified",
                "track_quality_descriptor": "low confidence or incomplete",
                "interpretation_notes": "Excluded from movement classification because warning flags or insufficient steps were present.",
            })

    class_summary = []
    for label, members in defaultdict(list, {k: [r for r in high if r.get("candidate_class_number") == k] for k in sorted(set(r.get("candidate_class_number") for r in high))}).items():
        class_summary.append({
            "candidate_class_number": label,
            "n_tracks": len(members),
            "median_radius": _median([r["mean_radius"] for r in members if r.get("mean_radius") is not None]),
            "median_speed": _median([r["median_speed"] for r in members if r.get("median_speed") is not None]),
            "median_straightness": _median([r["straightness_directionality_ratio"] for r in members if r.get("straightness_directionality_ratio") is not None]),
            "descriptor": members[0].get("candidate_movement_class") if members else "",
        })

    quality_counts = Counter(row["track_quality_descriptor"] for row in metrics)
    data_quality = [{"quality_category": key, "n_tracks": value} for key, value in sorted(quality_counts.items())]
    overview = [{
        "input_database": str(db_path),
        "source_csv": source_csv or "all datasets",
        "table": table_name,
        "n_spot_rows": len(rows),
        "n_tracks": len(metrics),
        "n_high_confidence_tracks": len(high),
        "ignored_intensity_metrics": True,
        "detected_track_id_column": detected["track_id"],
        "detected_radius_column": detected["radius"],
        "detected_frame_column": detected["frame"],
        "detected_x_column": detected["x"],
        "detected_y_column": detected["y"],
        "detected_z_column": detected["z"],
    }]
    metric_dictionary = [
        {"metric": "mean_radius", "description": "Average radius/size value for the track."},
        {"metric": "median_speed", "description": "Median step speed from neighboring detections."},
        {"metric": "path_length", "description": "Total distance traveled across valid steps."},
        {"metric": "net_displacement", "description": "Straight-line distance from first to last valid position."},
        {"metric": "straightness_directionality_ratio", "description": "net displacement divided by path length."},
        {"metric": "msd_lag1", "description": "Mean squared step displacement for lag 1."},
    ]

    all_fields = list(metrics[0].keys()) if metrics else []
    high_fields = list(high[0].keys()) if high else all_fields
    _write_csv(tables_dir / "all_tracks_metrics_with_warnings.csv", metrics, all_fields)
    _write_csv(tables_dir / "high_confidence_tracks_only_with_candidate_classes.csv", high, high_fields)
    _write_csv(tables_dir / "candidate_chloroplast_class_summary.csv", class_summary, ["candidate_class_number", "n_tracks", "median_radius", "median_speed", "median_straightness", "descriptor"])
    _write_csv(tables_dir / "cluster_model_comparison.csv", comparisons, ["model", "k", "silhouette", "inertia", "aic", "bic"])
    _write_csv(tables_dir / "metric_dictionary.csv", metric_dictionary, ["metric", "description"])
    _write_csv(tables_dir / "data_quality_summary.csv", data_quality, ["quality_category", "n_tracks"])
    _write_csv(tables_dir / "dataset_overview.csv", overview, list(overview[0].keys()))
    _write_csv(tables_dir / "step_speeds.csv", step_rows, ["TRACK_ID", "from_frame", "to_frame", "dt", "distance", "speed"])

    _write_scatter_svg(plots_dir / "01_size_vs_speed_candidate_classes.svg", "Size vs speed candidate classes", high, "mean_radius", "median_speed", "candidate_class_number", "mean radius", "median speed")
    pca = _pca_2d(x)
    pca_rows = [{"pc1": a, "pc2": b, "candidate_class_number": row.get("candidate_class_number", 0)} for (a, b), row in zip(pca, high)]
    _write_scatter_svg(plots_dir / "02_pca_size_movement_features.svg", "PCA-style feature projection", pca_rows, "pc1", "pc2", "candidate_class_number", "component 1", "component 2")
    _write_scatter_svg(plots_dir / "03_speed_vs_straightness.svg", "Speed vs straightness", high, "median_speed", "straightness_directionality_ratio", "candidate_class_number", "median speed", "straightness")
    _write_group_bar_svg(plots_dir / "04_radius_by_candidate_class.svg", "Median radius by candidate class", high, "candidate_class_number", "mean_radius")
    _write_group_bar_svg(plots_dir / "05_speed_by_candidate_class.svg", "Median speed by candidate class", high, "candidate_class_number", "median_speed")
    _write_group_bar_svg(plots_dir / "06_msd_by_candidate_class.svg", "MSD lag 1 by candidate class", high, "candidate_class_number", "msd_lag1")

    primary = max(comparisons, key=lambda r: (r["silhouette"] if r["silhouette"] is not None else -999))
    html_rows = "".join(f"<tr><td>{r['candidate_class_number']}</td><td>{r['n_tracks']}</td><td>{r['median_radius']}</td><td>{r['median_speed']}</td><td>{html.escape(str(r['descriptor']))}</td></tr>" for r in class_summary)
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Chloroplast Size-Movement Candidate Class Analysis</title>
<style>body{{font-family:Arial,sans-serif;max-width:1080px;margin:28px auto;line-height:1.45;color:#222}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #ccc;padding:8px;text-align:left}}th{{background:#f2f2f2}}img{{max-width:100%;border:1px solid #ddd;margin:12px 0 24px}}.note{{background:#fff8db;border:1px solid #ead48a;padding:12px;margin:16px 0}}.good{{background:#eef8ee;border:1px solid #b8d8b8;padding:12px;margin:16px 0}}</style>
</head><body>
<h1>Chloroplast Size-Movement Candidate Class Analysis</h1>
<p><b>Main question:</b> Are there candidate chloroplast classes based only on size and movement in one uploaded CSV file?</p>
<p><b>Dataset analysed:</b> <code>{html.escape(source_csv or "all datasets")}</code></p>
<div class="note"><p><b>Interpretation rule:</b> These are statistical candidate classes, not confirmed biological chloroplast types. Intensity metrics were ignored.</p></div>
<h2>What was analysed</h2>
<ul><li>Spot rows: <b>{len(rows)}</b></li><li>Tracks: <b>{len(metrics)}</b></li><li>High-confidence tracks used for clustering: <b>{len(high)}</b></li><li>Primary model selected by silhouette: <b>{primary['model']} k={primary['k']}</b></li></ul>
<h2>Detected columns</h2><pre>{html.escape(json.dumps(detected, indent=2))}</pre>
<h2>Candidate class summary</h2><table><tr><th>Class</th><th>Tracks</th><th>Median radius</th><th>Median speed</th><th>Descriptor</th></tr>{html_rows}</table>
<div class="good"><p><b>Main limitation:</b> separation can reflect track duration, missing data, or movement quality. Use the warning columns before interpreting candidate classes biologically.</p></div>
<h2>Graphs</h2>
<h3>Size vs speed clusters</h3><img src="plots/01_size_vs_speed_candidate_classes.svg">
<h3>PCA-style feature projection</h3><img src="plots/02_pca_size_movement_features.svg">
<h3>Speed vs straightness</h3><img src="plots/03_speed_vs_straightness.svg">
<h3>Radius distributions by candidate class</h3><img src="plots/04_radius_by_candidate_class.svg">
<h3>Movement metric distributions by candidate class</h3><img src="plots/05_speed_by_candidate_class.svg">
<h3>MSD curves / summary by candidate class</h3><img src="plots/06_msd_by_candidate_class.svg">
<h2>Files</h2><ul><li><code>tables/all_tracks_metrics_with_warnings.csv</code></li><li><code>tables/high_confidence_tracks_only_with_candidate_classes.csv</code></li><li><code>tables/candidate_chloroplast_class_summary.csv</code></li><li><code>tables/cluster_model_comparison.csv</code></li><li><code>tables/metric_dictionary.csv</code></li><li><code>tables/data_quality_summary.csv</code></li><li><code>tables/dataset_overview.csv</code></li><li><code>code/reproduce_analysis.py</code></li></ul>
</body></html>"""
    (report_dir / "chloroplast_tracking_analysis_report.html").write_text(report, encoding="utf-8")
    (report_dir / "README_chloroplast_tracking_analysis.md").write_text("# Chloroplast tracking analysis\n\nRun `python code/reproduce_analysis.py project.sqlite --out output_folder` to regenerate this package.\n", encoding="utf-8")
    shutil.copyfile(Path(__file__), code_dir / "trackmate_report.py")
    shutil.copyfile(Path(setup_db.__file__), code_dir / "setup_db.py")
    (code_dir / "reproduce_analysis.py").write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trackmate_report
parser = argparse.ArgumentParser()
parser.add_argument("sqlite_path")
parser.add_argument("--out", default="analysis_output")
parser.add_argument("--source-csv", default=None)
args = parser.parse_args()
print(trackmate_report.build_trackmate_report_package(Path(args.sqlite_path), "masks", Path(args.out), args.source_csv))
""",
        encoding="utf-8",
    )
    zip_path = output_dir / "chloroplast_tracking_analysis.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in report_dir.rglob("*"):
            z.write(path, path.relative_to(output_dir))
    return report_dir / "chloroplast_tracking_analysis_report.html"
