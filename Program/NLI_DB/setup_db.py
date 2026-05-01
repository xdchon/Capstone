from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
# Default CSV containing the per-object measurements (Quant output)
CSV_PATH = (
    BASE_DIR
    / "CSV_data"
    / "NIben_19_TriPer_CAT_TS_ON_v2_Capt3_DCN_T0_C0_TEST_Quant.csv"
)
# SQLite database that will store the table built from CSV files
DB_PATH = BASE_DIR / "nli_db.sqlite"
PROJECTS_DIR = BASE_DIR / "projects"
# Main table that stores per-object measurements
TABLE_NAME = "masks"

# Extra metadata columns used to support multiple images / CSV files
IMAGE_ID_COLUMN = "image_id"
SOURCE_FILE_COLUMN = "source_csv"
SOURCE_ROW_COLUMN = "source_row_number"
CSV_SOURCE_TABLE = "csv_sources"

# CSV columns that should be ignored in the retained project table.
# Keep this empty so the program does not silently drop source data.
IGNORED_COLUMNS: set[str] = set()
RESERVED_COLUMNS = {"id", IMAGE_ID_COLUMN, SOURCE_FILE_COLUMN, SOURCE_ROW_COLUMN}
ANALYSIS_EXCLUDED_COLUMNS = {
    "id",
    IMAGE_ID_COLUMN.casefold(),
    SOURCE_FILE_COLUMN.casefold(),
    SOURCE_ROW_COLUMN.casefold(),
    "nb",
    "name",
    "label",
    "type",
}


def project_db_path(project_name: str) -> Path:
    """Return the SQLite path for a named NLI project."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", project_name.strip()).strip("_")
    if not safe_name:
        safe_name = "nli_project"
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR / f"{safe_name}.sqlite"


def _clean_header(raw: str, fallback_index: int, used: set[str]) -> str:
    """Create a SQLite-safe column name without changing any cell values."""
    name = re.sub(r"\s+", " ", (raw or "").strip())
    if not name:
        name = f"column_{fallback_index + 1}"
    name = name.replace('"', "'")
    if name.casefold() in RESERVED_COLUMNS:
        name = f"csv_{name}"
    base = name
    suffix = 2
    while name.casefold() in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name.casefold())
    return name


def _to_number(value: object) -> float | None:
    """Parse a value for analysis only. Stored database values are not changed."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _table_exists(cur: sqlite3.Cursor, table_name: str) -> bool:
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    )
    return cur.fetchone() is not None


def _existing_columns(cur: sqlite3.Cursor, table_name: str) -> List[str]:
    cur.execute(f'PRAGMA table_info("{table_name}");')
    return [row[1] for row in cur.fetchall()]


def _add_missing_columns(
    cur: sqlite3.Cursor,
    table_name: str,
    columns: Sequence[Tuple[str, str]],
) -> None:
    existing = {name.casefold() for name in _existing_columns(cur, table_name)}
    for name, sql_type in columns:
        if name.casefold() not in existing:
            safe_name = name.replace('"', '""')
            cur.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {sql_type};')
            existing.add(name.casefold())


def _insert_row(
    cur: sqlite3.Cursor,
    table_name: str,
    columns: Sequence[str],
    values: Sequence[object],
) -> None:
    quoted = [f'"{name.replace(chr(34), chr(34) * 2)}"' for name in columns]
    placeholders = ", ".join(["?"] * len(columns))
    cur.execute(
        f'INSERT INTO "{table_name}" ({", ".join(quoted)}) VALUES ({placeholders});',
        list(values),
    )


def _ensure_csv_source_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        f'''
        CREATE TABLE IF NOT EXISTS "{CSV_SOURCE_TABLE}" (
            "source_csv" TEXT PRIMARY KEY,
            "csv_path" TEXT NOT NULL,
            "imported_at" TEXT DEFAULT CURRENT_TIMESTAMP
        );
        '''
    )


def register_csv_source(db_path: Path, source_csv: str, csv_path: Path) -> None:
    """Store where an imported CSV lives so reports can reopen the original file."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        _ensure_csv_source_table(cur)
        cur.execute(
            f'''
            INSERT INTO "{CSV_SOURCE_TABLE}" ("source_csv", "csv_path")
            VALUES (?, ?)
            ON CONFLICT("source_csv") DO UPDATE SET
                "csv_path" = excluded."csv_path",
                "imported_at" = CURRENT_TIMESTAMP;
            ''',
            (source_csv, str(csv_path.resolve())),
        )
        conn.commit()
    finally:
        conn.close()


def csv_source_paths(db_path: Path = DB_PATH) -> Dict[str, Path]:
    """Return source_csv -> original/cached CSV path for the active project."""
    if not db_path.is_file():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        _ensure_csv_source_table(cur)
        rows = cur.execute(
            f'SELECT "source_csv", "csv_path" FROM "{CSV_SOURCE_TABLE}" ORDER BY "source_csv";'
        ).fetchall()
        return {str(source): Path(str(path)) for source, path in rows}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def csv_source_path(db_path: Path, source_csv: str) -> Path | None:
    return csv_source_paths(db_path).get(source_csv)


def build_database(
    csv_path: Path = CSV_PATH,
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
    reset: bool = False,
    image_id: str | None = None,
) -> None:
    """
    Read a CSV file and populate a SQLite database.

    - Preserves every original CSV cell as text in '<table_name>'.
    - Also writes the same original values to a raw audit table named '<table_name>_raw'.
    - Numeric interpretation is done later in memory for reports; imported values are not rewritten.
    - Supports multiple CSV imports by appending rows to the same table.
    - When reset=True, drops and recreates the target table.
    - Adds metadata columns:
        * image_id: identifier derived from the CSV filename (or provided)
        * source_csv: the CSV filename
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if image_id is None:
        image_id = csv_path.stem
    raw_table_name = f"{table_name}_raw"

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        try:
            headers = next(reader)
        except StopIteration:
            raise RuntimeError("CSV file is empty; no header row found.")

        # Drop a trailing empty column if present (due to a trailing comma)
        if headers and headers[-1] == "":
            headers = headers[:-1]

        # Keep a copy of all original headers for alignment, with readable
        # duplicate-safe names for SQLite. Cell values are not changed.
        used_headers: set[str] = set()
        all_headers = [
            _clean_header(header, idx, used_headers) for idx, header in enumerate(headers)
        ]
        raw_headers = [f"raw_{header}" for header in all_headers]

        try:
            first_row = next(reader)
        except StopIteration:
            raise RuntimeError("CSV file contains a header but no data rows.")

        remaining_rows = [row for row in reader if row and not all(cell.strip() == "" for cell in row)]

        # Align first row with original headers length
        first_row = list(first_row)
        if len(first_row) > len(all_headers):
            first_row = first_row[: len(all_headers)]
        elif len(first_row) < len(all_headers):
            first_row.extend([""] * (len(all_headers) - len(first_row)))
        first_row_raw = list(first_row)

        # Determine which columns to keep. This is intentionally all columns by
        # default; project data should not be silently dropped.
        keep_indices = [i for i, _ in enumerate(all_headers)]

        headers = [all_headers[i] for i in keep_indices]
        incoming_type_map = {header: "TEXT" for header in headers}

        # Open/create the SQLite database
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()

            # Drop and recreate the table if requested
            if reset:
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
                cur.execute(f'DROP TABLE IF EXISTS "{raw_table_name}";')

            # Check if tables already exist
            exists = _table_exists(cur, table_name)
            raw_exists = _table_exists(cur, raw_table_name)

            raw_columns = [
                (IMAGE_ID_COLUMN, "TEXT"),
                (SOURCE_FILE_COLUMN, "TEXT"),
                (SOURCE_ROW_COLUMN, "INTEGER"),
                *[(name, "TEXT") for name in raw_headers],
            ]

            if not raw_exists:
                raw_defs = [
                    '"raw_id" INTEGER PRIMARY KEY AUTOINCREMENT',
                    f'"{IMAGE_ID_COLUMN}" TEXT',
                    f'"{SOURCE_FILE_COLUMN}" TEXT',
                    f'"{SOURCE_ROW_COLUMN}" INTEGER',
                ]
                for name in raw_headers:
                    safe_name = name.replace('"', '""')
                    raw_defs.append(f'"{safe_name}" TEXT')
                cur.execute(f'CREATE TABLE "{raw_table_name}" ({", ".join(raw_defs)});')
            else:
                _add_missing_columns(cur, raw_table_name, raw_columns)

            # Create the table if it does not exist
            if not exists:
                column_defs = [
                    '"id" INTEGER PRIMARY KEY AUTOINCREMENT',
                    f'"{IMAGE_ID_COLUMN}" TEXT',
                    f'"{SOURCE_FILE_COLUMN}" TEXT',
                    f'"{SOURCE_ROW_COLUMN}" INTEGER',
                ]
                for name in headers:
                    safe_name = name.replace('"', '""')
                    column_defs.append(f'"{safe_name}" TEXT')

                create_sql = (
                    f'CREATE TABLE "{table_name}" ({", ".join(column_defs)});'
                )
                cur.execute(create_sql)
            else:
                # Sanity-check that the existing table has the expected metadata columns.
                existing_cols = _existing_columns(cur, table_name)
                existing_lookup = {col.casefold() for col in existing_cols}
                if (
                    IMAGE_ID_COLUMN.casefold() not in existing_lookup
                    or SOURCE_FILE_COLUMN.casefold() not in existing_lookup
                ):
                    raise RuntimeError(
                        f"Existing table '{table_name}' has an old schema.\n"
                        f"Please remove the database file '{db_path.name}' or call "
                        "build_database(..., reset=True) once to recreate it "
                        "for the new Quant CSV format."
                    )
                _add_missing_columns(
                    cur,
                    table_name,
                    [(SOURCE_ROW_COLUMN, "INTEGER")]
                    + list(incoming_type_map.items()),
                )

            # Prepare INSERT statement (metadata columns + kept CSV columns)
            db_columns: List[str] = [
                IMAGE_ID_COLUMN,
                SOURCE_FILE_COLUMN,
                SOURCE_ROW_COLUMN,
            ] + list(headers)
            raw_db_columns: List[str] = [
                IMAGE_ID_COLUMN,
                SOURCE_FILE_COLUMN,
                SOURCE_ROW_COLUMN,
            ] + list(raw_headers)

            def align_row(row: Sequence[str]) -> List[str]:
                # Align row with original headers length
                row_list = list(row)
                if len(row_list) > len(all_headers):
                    row_list = row_list[: len(all_headers)]
                elif len(row_list) < len(all_headers):
                    row_list.extend([""] * (len(all_headers) - len(row_list)))
                return row_list

            def to_row_values(row: Sequence[str], source_row_number: int):
                row_list = align_row(row)
                filtered_row = [row_list[i] for i in keep_indices]

                return [image_id, csv_path.name, source_row_number] + filtered_row

            def to_raw_row_values(row: Sequence[str], source_row_number: int) -> List[object]:
                row_list = align_row(row)
                return [image_id, csv_path.name, source_row_number] + row_list

            # Insert the first data row
            _insert_row(cur, raw_table_name, raw_db_columns, to_raw_row_values(first_row_raw, 2))
            _insert_row(cur, table_name, db_columns, to_row_values(first_row, 2))

            # Insert remaining rows
            for offset, row in enumerate(remaining_rows, start=3):
                _insert_row(cur, raw_table_name, raw_db_columns, to_raw_row_values(row, offset))
                _insert_row(cur, table_name, db_columns, to_row_values(row, offset))

            conn.commit()
        finally:
            conn.close()

    register_csv_source(db_path, csv_path.name, csv_path)


def table_schema(
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
) -> List[Tuple[str, str]]:
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(f'PRAGMA table_info("{table_name}");')
        return [(row[1], row[2] or "TEXT") for row in cur.fetchall()]
    finally:
        conn.close()


def numeric_columns(
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
) -> List[str]:
    schema = table_schema(db_path, table_name)
    if not db_path.is_file():
        return []

    conn = sqlite3.connect(db_path)
    try:
        cols: List[str] = []
        for name, _col_type in schema:
            if name.casefold() in ANALYSIS_EXCLUDED_COLUMNS:
                continue
            safe_col = name.replace('"', '""')
            try:
                cur = conn.execute(
                    f'SELECT "{safe_col}" FROM "{table_name}" '
                    f'WHERE "{safe_col}" IS NOT NULL AND "{safe_col}" != "" LIMIT 200;'
                )
            except sqlite3.Error:
                continue

            sampled = [row[0] for row in cur.fetchall()]
            if not sampled:
                continue
            parsed = [_to_number(value) for value in sampled]
            numeric_count = sum(value is not None for value in parsed)
            if numeric_count and numeric_count / len(sampled) >= 0.8:
                cols.append(name)
        return cols
    finally:
        conn.close()


def _numeric_series(
    conn: sqlite3.Connection,
    table_name: str,
    column: str,
) -> List[float]:
    safe_col = column.replace('"', '""')
    cur = conn.execute(
        f'SELECT "{safe_col}" FROM "{table_name}" '
        f'WHERE "{safe_col}" IS NOT NULL AND "{safe_col}" != "";'
    )
    values: List[float] = []
    for (raw_value,) in cur.fetchall():
        number = _to_number(raw_value)
        if number is not None:
            values.append(number)
    return values


def _summary_from_values(values: List[float]) -> Dict[str, object]:
    if not values:
        return {"n": 0, "min_value": None, "avg_value": None, "max_value": None}
    return {
        "n": len(values),
        "min_value": min(values),
        "avg_value": sum(values) / len(values),
        "max_value": max(values),
    }


def imported_files(
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
) -> List[Tuple[str, int]]:
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            f'SELECT "{SOURCE_FILE_COLUMN}", COUNT(*) '
            f'FROM "{table_name}" GROUP BY "{SOURCE_FILE_COLUMN}" '
            f'ORDER BY "{SOURCE_FILE_COLUMN}";'
        )
        return [(row[0], row[1]) for row in cur.fetchall()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def column_stats(
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
    limit: int = 12,
) -> List[Dict[str, object]]:
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows: List[Dict[str, object]] = []
        for col in numeric_columns(db_path, table_name)[:limit]:
            rows.append({"column": col, **_summary_from_values(_numeric_series(conn, table_name, col))})
        return rows
    finally:
        conn.close()


def _pearson(values_a: List[float], values_b: List[float]) -> float | None:
    if len(values_a) < 3 or len(values_a) != len(values_b):
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    den_a = sum((a - mean_a) ** 2 for a in values_a)
    den_b = sum((b - mean_b) ** 2 for b in values_b)
    if den_a <= 0 or den_b <= 0:
        return None
    return num / ((den_a * den_b) ** 0.5)


def strongest_correlations(
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
    max_columns: int = 12,
    limit: int = 10,
) -> List[Dict[str, object]]:
    if not db_path.is_file():
        return []
    cols = numeric_columns(db_path, table_name)[:max_columns]
    if len(cols) < 2:
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        quoted = ", ".join(f'"{col.replace(chr(34), chr(34) * 2)}"' for col in cols)
        cur = conn.execute(f'SELECT {quoted} FROM "{table_name}";')
        data = {col: [] for col in cols}
        for row in cur.fetchall():
            for col in cols:
                data[col].append(_to_number(row[col]))

        results: List[Dict[str, object]] = []
        for i, left in enumerate(cols):
            for right in cols[i + 1 :]:
                pairs = [
                    (a, b)
                    for a, b in zip(data[left], data[right])
                    if a is not None and b is not None
                ]
                corr = _pearson([a for a, _ in pairs], [b for _, b in pairs])
                if corr is not None:
                    results.append(
                        {
                            "left": left,
                            "right": right,
                            "correlation": corr,
                            "n": len(pairs),
                        }
                    )
        results.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
        return results[:limit]
    finally:
        conn.close()


def main() -> None:
    # By default, (re)build the database from the default CSV,
    # replacing any existing table.
    build_database(reset=True)


if __name__ == "__main__":
    main()
