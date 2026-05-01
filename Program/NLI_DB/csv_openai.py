from __future__ import annotations

import csv
import html
import json
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPORT_ANALYSIS_PROMPT = textwrap.dedent(
    """
    Please analyse this chloroplast/object tracking dataset and create a full reproducible size-speed class analysis package.

    Main goal: determine whether the tracked objects separate into different exploratory classes based on size, speed, 3D position, and movement direction. Ignore intensity metrics unless they are needed only for file parsing.

    Please use only relevant non-intensity columns such as:

    TRACK_ID
    FRAME or POSITION_T
    POSITION_X
    POSITION_Y
    POSITION_Z if present
    RADIUS or equivalent size columns
    3D object size/shape columns if present

    Please produce a complete zipped output package with:

    A clear README explaining:
    what was in the file
    how many rows/spots/tracks/frames were analysed
    which rows were tracked vs untracked
    what columns were used
    which intensity columns were ignored
    what units are used
    limitations about pixels/frame and Z-scaling
    Per-track size and speed analysis:
    calculate track-level mean radius
    calculate median track speed from frame-to-frame movement
    calculate 2D and 3D displacement where possible
    calculate total distance travelled
    calculate net displacement
    calculate straightness as net displacement / total distance
    calculate signed movement components dx, dy, dz
    calculate movement direction angles in XY
    calculate Z elevation angle if Z is present
    Step-level movement table:
    one row per step between consecutive detections in the same track
    dx, dy, dz
    frame difference
    step distance
    step speed
    XY direction angle
    Z elevation angle
    Filtering:
    keep untracked spots for size histograms
    use only tracks with enough steps for speed/class analysis
    explain the filtering threshold clearly
    K-means clustering:
    cluster tracks using log-transformed and standardized size/speed features
    test k = 2 to 8
    create elbow/inertia and silhouette-score plots
    choose a practical k and explain why
    create a k-means class summary table
    label classes with readable names such as Large/fast, Small/fast, Medium/slow, etc.
    DBSCAN clustering:
    run DBSCAN on the same standardized size/speed features
    include a k-distance plot to justify eps
    include a DBSCAN parameter sweep table
    include DBSCAN cluster/noise summary
    compare DBSCAN labels to k-means classes with a crosstab
    explain whether DBSCAN supports true dense groups or mostly one continuous population
    Movement-direction analysis:
    make a rose/polar plot of step directions by k-means class
    make a net XY movement vector plot by k-means class
    make a Z elevation-angle distribution plot
    make a straightness-by-class plot
    make signed dx/dy/dz movement component plots
    create a movement_direction_by_kmeans_class.csv summary table
    3D size/shape analysis if 3D object data is present:
    calculate equivalent 3D radius if possible
    summarize object radius, volume, surface/shape metrics if available
    make 3D XYZ scatter plots
    make XY projection plots
    plot radius vs Z
    plot sphericity or shape descriptors vs radius/Z if present
    calculate nearest-neighbor distance/local density if possible
    run k-means and DBSCAN on 3D size/shape features
    include 3D class summary tables and crosstabs
    Plots to include:
    spot radius histogram
    step speed histogram
    size-speed scatter plot
    size-speed density/hexbin plot
    speed by size quartile boxplot
    k-means elbow plot
    k-means silhouette plot
    k-means cluster scatter plot
    class median radius plot
    class median speed plot
    spatial start map by class
    DBSCAN k-distance eps plot
    DBSCAN scatter plot
    DBSCAN group counts
    k-means vs DBSCAN crosstab plot
    3D size/shape plots if possible
    movement direction rose plot
    net XY movement vectors by class
    Z elevation angle by class
    straightness by class
    signed movement components by class
    Tables to include:
    track_summary_with_classes.csv
    step_speeds.csv
    cluster_summary.csv
    k_selection_summary.csv
    dbscan_summary.csv
    dbscan_parameter_sweep.csv
    kmeans_dbscan_crosstab.csv
    movement_direction_by_kmeans_class.csv
    overall_summary.json
    3d_time1_object_summary_with_classes.csv if possible
    3d_time1_kmeans_selection_summary.csv if possible
    3d_time1_kmeans_size_shape_summary.csv if possible
    3d_time1_dbscan_size_shape_summary.csv if possible
    3d_time1_dbscan_parameter_sweep.csv if possible
    3d_time1_kmeans_dbscan_crosstab.csv if possible
    3d_time1_overall_summary.json if possible
    Reports:
    create a Markdown README/report
    create an HTML report
    explain each graph in plain English
    explain what each class means biologically and statistically
    state clearly that these are exploratory statistical classes, not confirmed biological chloroplast types
    Code:
    include a full standalone Python script called reproduce_analysis.py
    include a second script called reproduce_3d_timepoint_analysis.py if the file supports 3D object-level analysis
    the scripts should regenerate all tables, plots, reports, and the zipped package from the original input file

    Important interpretation rules:

    Do not overclaim confirmed biological classes.
    Say whether the data look like discrete classes or a continuous population.
    Compare k-means and DBSCAN interpretations.
    Make clear that k-means forces all tracks into classes, while DBSCAN identifies dense groups and outliers.
    Explain that speed is in pixels/frame unless pixel size and frame interval are provided.
    Warn that Z movement should be interpreted cautiously if Z spacing differs from XY pixel size.
    Use readable descriptors and good labels throughout.
    """
).strip()


def csv_profile(csv_path: Path, preview_rows: int = 8) -> Dict[str, object]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        preview = []
        row_count = 0
        for row in reader:
            row_count += 1
            if len(preview) < preview_rows:
                preview.append({key: row.get(key, "") for key in columns[:40]})
    intensity_columns = [c for c in columns if "intensity" in c.casefold()]
    relevant_terms = ("track", "frame", "position", "radius", "diameter", "size", "area", "volume", "surface", "shape", "spheric")
    candidate_columns = [c for c in columns if "intensity" not in c.casefold() and any(t in c.casefold() for t in relevant_terms)]
    return {
        "csv_path": str(csv_path),
        "source_csv": csv_path.name,
        "row_count": row_count,
        "columns": columns,
        "candidate_non_intensity_columns": candidate_columns,
        "intensity_columns": intensity_columns,
        "preview_rows": preview,
    }


def upload_csv_file(client: object, csv_path: Path) -> str | None:
    """Upload the CSV to OpenAI when the installed SDK supports the Files API."""
    for purpose in ("user_data", "assistants"):
        try:
            with csv_path.open("rb") as f:
                uploaded = client.files.create(file=f, purpose=purpose)  # type: ignore[attr-defined]
            return str(getattr(uploaded, "id", "") or uploaded["id"])  # type: ignore[index]
        except Exception:
            continue
    return None


def _responses_text(client: object, model: str, prompt: str, csv_path: Path | None = None, reasoning_effort: str | None = None) -> str:
    file_id = upload_csv_file(client, csv_path) if csv_path else None
    content: List[Dict[str, object]] = []
    if file_id:
        content.append({"type": "input_file", "file_id": file_id})
    content.append({"type": "input_text", "text": prompt})
    kwargs: Dict[str, object] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    try:
        response = client.responses.create(**kwargs)  # type: ignore[attr-defined]
        text = getattr(response, "output_text", "").strip()
        return text or str(response)
    except Exception:
        # Fallback for older SDKs: include compact CSV context instead of file attachment.
        fallback_prompt = prompt
        if csv_path:
            fallback_prompt += "\n\nCSV profile and preview:\n" + json.dumps(csv_profile(csv_path), default=str)
        completion = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "user", "content": fallback_prompt}],
        )
        return completion.choices[0].message.content.strip()  # type: ignore[attr-defined]


def answer_csv_question(client: object, csv_path: Path, question: str, model: str) -> str:
    profile = csv_profile(csv_path)
    prompt = textwrap.dedent(
        f"""
        You are answering a scientist's question using the attached CSV file directly.
        Do not use or refer to a SQLite database.

        CSV profile:
        {json.dumps(profile, default=str)}

        Question:
        {question}

        Answer clearly. Use relevant CSV column names and say when a requested value
        cannot be computed from the available columns.
        """
    ).strip()
    return _responses_text(client, model, prompt, csv_path=csv_path)


def build_openai_csv_report_package(
    client: object,
    csv_path: Path,
    output_dir: Path,
    model: str,
    reasoning_effort: str,
) -> Path:
    report_dir = output_dir / "openai_csv_tracking_analysis"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    (report_dir / "inputs").mkdir(parents=True)
    (report_dir / "openai").mkdir()
    (report_dir / "code").mkdir()
    shutil.copyfile(csv_path, report_dir / "inputs" / csv_path.name)

    profile = csv_profile(csv_path)
    prompt = textwrap.dedent(
        f"""
        Use the attached CSV file directly as the source data. Do not assume any
        SQLite database or precomputed report table exists.

        CSV profile:
        {json.dumps(profile, default=str)}

        {REPORT_ANALYSIS_PROMPT}

        Return a concise but complete report-generation specification and narrative
        interpretation. If you cannot produce binary plot files directly, provide
        exact Python code structure and clearly list every required output file.
        """
    ).strip()
    result = _responses_text(client, model, prompt, csv_path=csv_path, reasoning_effort=reasoning_effort)

    (report_dir / "openai" / "report_prompt.txt").write_text(prompt, encoding="utf-8")
    (report_dir / "openai" / "openai_response.md").write_text(result, encoding="utf-8")
    (report_dir / "openai" / "csv_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__), report_dir / "code" / "csv_openai.py")

    readme = f"""# OpenAI CSV tracking analysis

This package was created from the original CSV file, not from the SQLite project table.

- Input CSV: `inputs/{csv_path.name}`
- Full prompt: `openai/report_prompt.txt`
- OpenAI response: `openai/openai_response.md`
- CSV profile: `openai/csv_profile.json`
- Reproduction script: `code/reproduce_analysis.py`

The OpenAI prompt explicitly asks for exploratory size-speed, movement-direction,
k-means, DBSCAN, and optional 3D object-level analysis. Classes should be treated
as exploratory statistical classes, not confirmed biological chloroplast types.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    reproduce = f'''#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import csv_openai


def get_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    key_file = Path(__file__).resolve().parents[1] / "openai_api_key.txt"
    if not api_key and key_file.is_file():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                api_key = stripped
                break
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or place openai_api_key.txt beside the report folder.")
    return OpenAI(api_key=api_key)

parser = argparse.ArgumentParser(description="Regenerate the OpenAI CSV tracking report package.")
parser.add_argument("csv_path", nargs="?", default=str(Path("inputs") / csv_path.name))
parser.add_argument("--out", default="reproduced_openai_csv_report")
parser.add_argument("--model", default="{model}")
parser.add_argument("--reasoning-effort", default="{reasoning_effort}")
args = parser.parse_args()

client = get_openai_client()
report = csv_openai.build_openai_csv_report_package(
    client=client,
    csv_path=Path(args.csv_path),
    output_dir=Path(args.out),
    model=args.model,
    reasoning_effort=args.reasoning_effort,
)
print(f"Report written to: {{report}}")
'''
    (report_dir / "code" / "reproduce_analysis.py").write_text(reproduce, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>OpenAI CSV Tracking Analysis</title>
<style>body{{font-family:Arial,sans-serif;max-width:1080px;margin:28px auto;line-height:1.45}}pre{{white-space:pre-wrap}}code{{background:#f6f6f6;padding:2px 4px}}</style></head>
<body>
<h1>OpenAI CSV Tracking Analysis</h1>
<p><b>Input CSV:</b> <code>{html.escape(csv_path.name)}</code></p>
<p><b>Rows:</b> {profile["row_count"]}</p>
<p><b>Important caveat:</b> exploratory statistical classes are not confirmed biological chloroplast types. Speed is pixels/frame unless calibration is provided; Z movement needs caution if Z spacing differs from XY pixel size.</p>
<h2>OpenAI Response</h2>
<pre>{html.escape(result)}</pre>
</body></html>"""
    (report_dir / "openai_csv_tracking_analysis.html").write_text(html_doc, encoding="utf-8")

    zip_path = output_dir / "openai_csv_tracking_analysis.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in report_dir.rglob("*"):
            z.write(path, path.relative_to(output_dir))
    return report_dir / "openai_csv_tracking_analysis.html"
