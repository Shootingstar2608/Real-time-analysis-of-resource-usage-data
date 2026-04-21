"""Utilities for loading and normalizing Azure VM traces and Alibaba 2018 machine traces."""

from __future__ import annotations

import csv
import gzip
import glob
import logging
import os
import tarfile
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NORMALIZED_COLUMNS = [
    "timestamp",
    "vm_id",
    "min_cpu",
    "max_cpu",
    "avg_cpu",
    "cpu_range",
    "vm_category",
    "vm_core_count",
    "vm_memory_gb",
    "avg_memory",
    "network_in_mbps",
    "network_out_mbps",
    "disk_io_percent",
    "mem_gps",
    "mkpi",
    "ingestion_timestamp",
    "source_file",
    "data_source",
]

AZURE_CPU_PATTERNS = [
    "vm_cpu_readings-file-*-of-195.csv.gz",
    "vm_cpu_readings-*.csv.gz",
]

ALIBABA_USAGE_CANDIDATES = [
    "machine_usage_sample.csv",
    "machine_usage.csv",
    "machine_usage_sample.csv.gz",
    "machine_usage.csv.gz",
]

ALIBABA_META_CANDIDATES = [
    "machine_meta_sample.csv",
    "machine_meta.csv",
    "machine_meta_sample.csv.gz",
    "machine_meta.csv.gz",
]

ALIBABA_USAGE_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
]

ALIBABA_META_COLUMNS = [
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status",
]


def _open_text_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _open_csv_from_tar(tar_path: str, member_hint: str):
    tar = tarfile.open(tar_path, "r:gz")
    members = [m for m in tar.getmembers() if m.isfile() and member_hint in os.path.basename(m.name)]
    if not members:
        tar.close()
        raise FileNotFoundError(f"No member containing '{member_hint}' in {tar_path}")
    f = tar.extractfile(members[0])
    if f is None:
        tar.close()
        raise FileNotFoundError(f"Failed to extract {members[0].name} from {tar_path}")
    import io
    wrapper = io.TextIOWrapper(f, encoding="utf-8")
    wrapper._tar_handle = tar  # type: ignore[attr-defined]
    return wrapper


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _candidate_paths(data_dir: str, candidates: List[str]) -> List[str]:
    paths: List[str] = []
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            paths.append(p)
    return paths


def detect_dataset_type(data_dir: str) -> str:
    for pattern in AZURE_CPU_PATTERNS:
        if glob.glob(os.path.join(data_dir, pattern)):
            return "azure"

    if os.path.exists(os.path.join(data_dir, "vmtable.csv.gz")):
        return "azure"

    alibaba_candidates = _candidate_paths(data_dir, ALIBABA_USAGE_CANDIDATES)
    if alibaba_candidates:
        return "alibaba"

    if os.path.exists(os.path.join(data_dir, "machine_usage.tar.gz")):
        return "alibaba"

    raise FileNotFoundError(
        f"Could not detect dataset type inside: {data_dir}. "
        f"Expected Azure vm_cpu_readings*.csv.gz or Alibaba machine_usage*.csv."
    )


def find_alibaba_usage_path(data_dir: str) -> str:
    candidates = _candidate_paths(data_dir, ALIBABA_USAGE_CANDIDATES)
    if candidates:
        return candidates[0]

    tar_path = os.path.join(data_dir, "machine_usage.tar.gz")
    if os.path.exists(tar_path):
        return tar_path

    raise FileNotFoundError(f"No Alibaba machine_usage file found in {data_dir}")


def find_alibaba_meta_path(data_dir: str) -> Optional[str]:
    candidates = _candidate_paths(data_dir, ALIBABA_META_CANDIDATES)
    if candidates:
        return candidates[0]

    tar_path = os.path.join(data_dir, "machine_meta.tar.gz")
    if os.path.exists(tar_path):
        return tar_path

    return None


def find_azure_cpu_files(data_dir: str) -> List[str]:
    cpu_files: List[str] = []
    for pattern in AZURE_CPU_PATTERNS:
        cpu_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    return sorted(set(cpu_files))


def load_azure_vm_table(data_dir: str) -> Dict[str, Dict[str, object]]:
    vmtable_path = os.path.join(data_dir, "vmtable.csv.gz")
    if not os.path.exists(vmtable_path):
        logger.warning("Azure vmtable.csv.gz not found. Continuing without metadata.")
        return {}

    lookup: Dict[str, Dict[str, object]] = {}
    with gzip.open(vmtable_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 11:
                continue
            vm_id = row[0].strip()
            lookup[vm_id] = {
                "vm_category": row[8].strip() or "Unknown",
                "vm_core_count": _safe_int(row[9]),
                "vm_memory_gb": _safe_int(row[10]),
            }
    return lookup


def load_alibaba_machine_meta(data_dir: str) -> Dict[str, Dict[str, object]]:
    meta_path = find_alibaba_meta_path(data_dir)
    if meta_path is None:
        logger.warning("Alibaba machine_meta not found. Continuing without metadata enrichment.")
        return {}

    if meta_path.endswith(".tar.gz"):
        handle = _open_csv_from_tar(meta_path, "machine_meta")
    else:
        handle = _open_text_file(meta_path)

    lookup: Dict[str, Dict[str, object]] = {}
    with handle as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < len(ALIBABA_META_COLUMNS):
                continue
            machine_id = row[0].strip()
            ts = _safe_float(row[1], default=-1)
            prev = lookup.get(machine_id)
            # keep the latest known machine status snapshot
            if prev is None or ts >= prev.get("time_stamp", -1):
                lookup[machine_id] = {
                    "time_stamp": ts,
                    "vm_category": row[6].strip() or "Unknown",
                    "vm_core_count": _safe_int(row[4]),
                    "vm_memory_gb": _safe_int(row[5]),
                    "failure_domain_1": _safe_int(row[2], default=-1),
                    "failure_domain_2": row[3].strip() if len(row) > 3 else "",
                }
    return lookup


def iter_azure_records(data_dir: str, max_records: Optional[int] = None) -> Iterator[Dict[str, object]]:
    vm_lookup = load_azure_vm_table(data_dir)
    cpu_files = find_azure_cpu_files(data_dir)
    if not cpu_files:
        raise FileNotFoundError(f"No Azure vm_cpu_readings files found in {data_dir}")

    sent = 0
    for cpu_file in cpu_files:
        with gzip.open(cpu_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue
                vm_id = row[1].strip()
                min_cpu = _safe_float(row[2])
                max_cpu = _safe_float(row[3])
                avg_cpu = _safe_float(row[4])
                vm_info = vm_lookup.get(vm_id, {})
                yield {
                    "timestamp": _safe_float(row[0]),
                    "vm_id": vm_id,
                    "min_cpu": min_cpu,
                    "max_cpu": max_cpu,
                    "avg_cpu": avg_cpu,
                    "cpu_range": max_cpu - min_cpu,
                    "vm_category": vm_info.get("vm_category", "Unknown"),
                    "vm_core_count": vm_info.get("vm_core_count", 0),
                    "vm_memory_gb": vm_info.get("vm_memory_gb", 0),
                    "avg_memory": 0.0,
                    "network_in_mbps": 0.0,
                    "network_out_mbps": 0.0,
                    "disk_io_percent": 0.0,
                    "mem_gps": 0.0,
                    "mkpi": 0.0,
                    "ingestion_timestamp": datetime.utcnow().isoformat(),
                    "source_file": os.path.basename(cpu_file),
                    "data_source": "azure",
                }
                sent += 1
                if max_records is not None and sent >= max_records:
                    return


def iter_alibaba_records(data_dir: str, max_records: Optional[int] = None) -> Iterator[Dict[str, object]]:
    usage_path = find_alibaba_usage_path(data_dir)
    meta_lookup = load_alibaba_machine_meta(data_dir)

    if usage_path.endswith(".tar.gz"):
        handle = _open_csv_from_tar(usage_path, "machine_usage")
        source_name = os.path.basename(usage_path) + ":machine_usage"
    else:
        handle = _open_text_file(usage_path)
        source_name = os.path.basename(usage_path)

    sent = 0
    with handle as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < len(ALIBABA_USAGE_COLUMNS):
                continue

            machine_id = row[0].strip()
            avg_cpu = _safe_float(row[2])
            avg_memory = _safe_float(row[3])
            mem_gps = _safe_float(row[4])
            mkpi = _safe_float(row[5])
            net_in = _safe_float(row[6])
            net_out = _safe_float(row[7])
            disk_io = _safe_float(row[8])
            machine_info = meta_lookup.get(machine_id, {})

            yield {
                "timestamp": _safe_float(row[1]),
                "vm_id": machine_id,
                "min_cpu": avg_cpu,
                "max_cpu": avg_cpu,
                "avg_cpu": avg_cpu,
                "cpu_range": 0.0,
                "vm_category": machine_info.get("vm_category", "Unknown"),
                "vm_core_count": machine_info.get("vm_core_count", 0),
                "vm_memory_gb": machine_info.get("vm_memory_gb", 0),
                "avg_memory": avg_memory,
                "network_in_mbps": net_in,
                "network_out_mbps": net_out,
                "disk_io_percent": disk_io,
                "mem_gps": mem_gps,
                "mkpi": mkpi,
                "ingestion_timestamp": datetime.utcnow().isoformat(),
                "source_file": source_name,
                "data_source": "alibaba",
            }
            sent += 1
            if max_records is not None and sent >= max_records:
                return


def iter_normalized_records(
    data_dir: str,
    dataset: str = "auto",
    max_records: Optional[int] = None,
) -> Iterator[Dict[str, object]]:
    dataset = detect_dataset_type(data_dir) if dataset == "auto" else dataset.lower()
    if dataset == "azure":
        yield from iter_azure_records(data_dir, max_records=max_records)
        return
    if dataset == "alibaba":
        yield from iter_alibaba_records(data_dir, max_records=max_records)
        return
    raise ValueError(f"Unsupported dataset: {dataset}")


def records_to_dataframe(
    data_dir: str,
    dataset: str = "auto",
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    records = list(iter_normalized_records(data_dir=data_dir, dataset=dataset, max_records=max_records))
    if not records:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    df = pd.DataFrame(records)
    for col in NORMALIZED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[NORMALIZED_COLUMNS].sort_values(["timestamp", "vm_id"]).reset_index(drop=True)


def load_aggregated_series_by_timestamp(
    data_dir: str,
    dataset: str = "auto",
    target_unique_timestamps: Optional[int] = None,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    if target_unique_timestamps is None:
        target_unique_timestamps = 64

    agg: Dict[float, Dict[str, float]] = {}
    unique_ts_count = 0

    for record in iter_normalized_records(data_dir=data_dir, dataset=dataset, max_records=max_records):
        ts = _safe_float(record["timestamp"])
        if ts not in agg:
            if unique_ts_count >= target_unique_timestamps:
                break
            agg[ts] = {
                "count": 0.0,
                "min_cpu": 0.0,
                "max_cpu": 0.0,
                "avg_cpu": 0.0,
                "cpu_range": 0.0,
                "avg_memory": 0.0,
                "network_in_mbps": 0.0,
                "network_out_mbps": 0.0,
                "disk_io_percent": 0.0,
                "mem_gps": 0.0,
                "mkpi": 0.0,
            }
            unique_ts_count += 1

        agg[ts]["count"] += 1.0
        for field in [
            "min_cpu",
            "max_cpu",
            "avg_cpu",
            "cpu_range",
            "avg_memory",
            "network_in_mbps",
            "network_out_mbps",
            "disk_io_percent",
            "mem_gps",
            "mkpi",
        ]:
            agg[ts][field] += _safe_float(record.get(field, 0.0))

    rows = []
    for ts, stats in agg.items():
        cnt = max(stats["count"], 1.0)
        rows.append({
            "timestamp": ts,
            "vm_id": "GLOBAL",
            "min_cpu": stats["min_cpu"] / cnt,
            "max_cpu": stats["max_cpu"] / cnt,
            "avg_cpu": stats["avg_cpu"] / cnt,
            "cpu_range": stats["cpu_range"] / cnt,
            "avg_memory": stats["avg_memory"] / cnt,
            "network_in_mbps": stats["network_in_mbps"] / cnt,
            "network_out_mbps": stats["network_out_mbps"] / cnt,
            "disk_io_percent": stats["disk_io_percent"] / cnt,
            "mem_gps": stats["mem_gps"] / cnt,
            "mkpi": stats["mkpi"] / cnt,
            "vm_category": "GLOBAL",
            "vm_core_count": 0,
            "vm_memory_gb": 0,
            "ingestion_timestamp": "",
            "source_file": "aggregated",
            "data_source": dataset if dataset != "auto" else detect_dataset_type(data_dir),
        })

    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    df = pd.DataFrame(rows)
    for col in NORMALIZED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[NORMALIZED_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def select_feature_columns(
    df: pd.DataFrame,
    candidate_columns: Optional[List[str]] = None,
    min_non_nan_ratio: float = 0.80,
    min_std: float = 1e-8,
) -> List[str]:
    candidates = candidate_columns or [
        "min_cpu",
        "max_cpu",
        "avg_cpu",
        "cpu_range",
        "avg_memory",
        "network_in_mbps",
        "network_out_mbps",
        "disk_io_percent",
        "mem_gps",
        "mkpi",
    ]

    selected: List[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() < min_non_nan_ratio:
            continue
        if series.fillna(0.0).std() <= min_std:
            continue
        selected.append(col)

    if "avg_cpu" not in selected and "avg_cpu" in df.columns:
        selected.insert(0, "avg_cpu")

    # keep some cpu context if available
    for col in ["min_cpu", "max_cpu", "cpu_range"]:
        if col in df.columns and col not in selected:
            # only add when at least one selected column exists and no more than 4 cpu features duplicated
            if pd.to_numeric(df[col], errors="coerce").fillna(0.0).std() > min_std:
                selected.append(col)

    if not selected:
        selected = [c for c in ["min_cpu", "max_cpu", "avg_cpu", "cpu_range"] if c in df.columns]

    return selected
