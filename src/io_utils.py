import re
import zipfile
from pathlib import Path
from xml.etree.ElementTree import iterparse

import numpy as np
import pandas as pd

from src.config import DATA_RAW_CSV, DATA_RAW_XLSX, TARGET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col_to_idx(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _load_shared_strings(zf: zipfile.ZipFile):
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    strings = []
    with zf.open("xl/sharedStrings.xml") as f:
        for _, elem in iterparse(f, events=("end",)):
            if elem.tag == NS + "si":
                strings.append("".join(t.text or "" for t in elem.iter(NS + "t")))
                elem.clear()
    return strings


def read_xlsx_fast(path: Path, sheet_xml: str = "xl/worksheets/sheet1.xml") -> pd.DataFrame:
    """Lee el xlsx sin depender de que Excel interprete mal PurchDate.

    En este archivo, PurchDate viene como segundos Unix pero con formato de fecha de Excel,
    por eso algunos lectores lo devuelven como #VALUE!/NaN. Este lector conserva el valor crudo.
    """
    with zipfile.ZipFile(path) as zf:
        shared_strings = _load_shared_strings(zf)
        rows, max_col = [], 0
        with zf.open(sheet_xml) as f:
            for _, row in iterparse(f, events=("end",)):
                if row.tag == NS + "row":
                    values = []
                    for cell in row.findall(NS + "c"):
                        ref = cell.attrib.get("r", "")
                        idx = _col_to_idx(ref)
                        if idx >= len(values):
                            values.extend([None] * (idx - len(values) + 1))
                        cell_type = cell.attrib.get("t")
                        v = cell.find(NS + "v")
                        val = None if v is None else v.text
                        if val is not None:
                            if cell_type == "s":
                                val = shared_strings[int(float(val))]
                            else:
                                try:
                                    val = float(val)
                                    if val.is_integer():
                                        val = int(val)
                                except Exception:
                                    pass
                        values[idx] = val
                    max_col = max(max_col, len(values))
                    rows.append(values)
                    row.clear()
    rows = [r + [None] * (max_col - len(r)) for r in rows]
    df = pd.DataFrame(rows[1:], columns=rows[0]).dropna(how="all")
    return df


def _fix_purch_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "PurchDate" not in df.columns:
        return df

    raw_numeric = pd.to_numeric(df["PurchDate"], errors="coerce")
    parsed = pd.to_datetime(df["PurchDate"], errors="coerce")

    # Si la mayoría viene numérica en rango Unix, usar segundos Unix.
    if raw_numeric.notna().mean() > 0.80 and raw_numeric.median() > 10_000_000:
        parsed = pd.to_datetime(raw_numeric, unit="s", errors="coerce")

    df["PurchDate"] = parsed
    return df


def load_kick_data(csv_path: Path = DATA_RAW_CSV, xlsx_path: Path = DATA_RAW_XLSX) -> pd.DataFrame:
    """Carga el dataset desde CSV DVC; si no existe, intenta XLSX."""
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    elif Path(xlsx_path).exists():
        df = read_xlsx_fast(Path(xlsx_path))
    else:
        raise FileNotFoundError(
            f"No encontré el dataset. Esperado: {csv_path} o {xlsx_path}. Ejecuta dvc pull si falta el CSV."
        )

    df = _fix_purch_date(df)
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("Int64")
    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
