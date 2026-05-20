"""Rebuild the five BASIC indicator datasets without look-ahead leakage.

The raw Wind Excel files in tmp_data contain malformed workbook styles that
can break openpyxl. This script reads only worksheet XML values, then computes
all indicators from current/past observations.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tmp_data"
DATASET_DIR = ROOT / "Dataset"

START_DATE = pd.Timestamp("2010-01-04")
END_DATE = pd.Timestamp("2023-10-16")

RAW_FILES = {
    "000016.SH": ("000016.SH.xlsx", "combined_dataframe_000016SH_basic_indicators.csv"),
    "000100.SZ": ("000100.SZ.xlsx", "combined_dataframe_000100SZ_basic_indicators.csv"),
    "000300.SH": ("000300.SH.xlsx", "combined_dataframe_000300SH_basic_indicators.csv"),
    "002129.SZ": ("002129.SZ.xlsx", "combined_dataframe_002129SZ_basic_indicators.csv"),
    "300015.SZ": ("300015.SZ.xlsx", "combined_dataframe_300015SZ_basic_indicators.csv"),
}

OUTPUT_COLUMNS = [
    "Date",
    "Price",
    "Vol.",
    "weekday",
    "mom",
    "mom1",
    "mom2",
    "mom3",
    "ROC_5",
    "ROC_10",
    "ROC_15",
    "ROC_20",
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "Name",
]


def _column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _read_xlsx_values(path: Path) -> list[list[Any]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                text = "".join(t.text or "" for t in item.findall(".//a:t", ns))
                shared_strings.append(text)

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[Any]] = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values: list[Any] = []
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                while len(values) <= _column_index(ref):
                    values.append(None)
                value_node = cell.find("a:v", ns)
                value = value_node.text if value_node is not None else None
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared_strings[int(value)]
                values[_column_index(ref)] = value
            rows.append(values)
        return rows


def _excel_serial_to_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        serial = float(value)
    except (TypeError, ValueError):
        return pd.NaT
    if not math.isfinite(serial):
        return pd.NaT
    return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=serial))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_raw_market_file(path: Path) -> pd.DataFrame:
    rows = _read_xlsx_values(path)
    parsed = []
    for row in rows[1:]:
        if len(row) < 10:
            continue
        date = _excel_serial_to_timestamp(row[2])
        if pd.isna(date):
            continue
        parsed.append(
            {
                "code": row[0],
                "date": date.normalize(),
                "open": _to_float(row[3]),
                "high": _to_float(row[4]),
                "low": _to_float(row[5]),
                "close": _to_float(row[6]),
                "pct_chg": _to_float(row[7]),
                "volume": _to_float(row[8]),
                "amount": _to_float(row[9]),
            }
        )

    df = pd.DataFrame(parsed)
    if df.empty:
        raise ValueError(f"No usable market rows found in {path}")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df.dropna(subset=["date", "close"])
    return df.reset_index(drop=True)


def build_no_leakage_features(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Date"] = df["date"]
    out["Price"] = df["close"]

    volume = df["volume"].replace(0, np.nan)
    out["Vol."] = volume.pct_change(fill_method=None)
    out["weekday"] = df["date"].dt.weekday

    returns = df["close"].pct_change(fill_method=None)
    out["mom"] = returns
    out["mom1"] = returns.shift(1)
    out["mom2"] = returns.shift(2)
    out["mom3"] = returns.shift(3)

    for window in (5, 10, 15, 20):
        out[f"ROC_{window}"] = df["close"].pct_change(periods=window, fill_method=None) * 100

    for span in (10, 20, 50, 200):
        out[f"EMA_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    out["Name"] = code
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[(out["Date"] >= START_DATE) & (out["Date"] <= END_DATE)]
    out = out.dropna().reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out[OUTPUT_COLUMNS]


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for code, (input_name, output_name) in RAW_FILES.items():
        raw_path = TMP_DIR / input_name
        output_path = DATASET_DIR / output_name
        raw = load_raw_market_file(raw_path)
        rebuilt = build_no_leakage_features(raw, code)
        rebuilt.to_csv(output_path, index=False)
        summary.append((output_name, len(rebuilt), rebuilt["Date"].iloc[0], rebuilt["Date"].iloc[-1]))

    print("Rebuilt no-leakage BASIC indicator datasets:")
    for output_name, rows, start, end in summary:
        print(f"- {output_name}: {rows} rows, {start} to {end}")


if __name__ == "__main__":
    main()
