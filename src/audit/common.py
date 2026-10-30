"""Shared paths and discovery helpers for the PAAWS Phase 1 audit.

Raw data lives read-only on the external SSD. Nothing in this module writes
outside the project directory.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

PAAWS_ROOT = Path("/mnt/e/FYP Data/PAAWS")
DATA_SUMMARY = PAAWS_ROOT / "PAAWS_Data_Summary_R2_version_1.2.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"
OUTPUTS = PROJECT_ROOT / "outputs"

NIGHT_PATTERN = re.compile(r"^(?P<participant>DS_\d+)-Sleep-Night(?P<night>\d+)$")
STAGES = ("N1", "N2", "N3", "REM")
SLEEP_STAGES = frozenset(STAGES)
ALL_STAGES = frozenset(STAGES + ("Wake",))

RESPIRATORY_EVENTS = frozenset(
    {
        "Apnea",
        "A. Obstructive",
        "A. Central",
        "A. Mixed",
        "Hypopnea",
        "H. Obstructive",
        "H. Central",
    }
)

# Sleep-stage epoch length is fixed at 30 s by the scoring software.
EPOCH_SECONDS = 30


@dataclass(frozen=True)
class Night:
    root_name: str
    participant: str
    night: int
    edf: Path
    csv: Path

    @property
    def night_id(self) -> str:
        return f"{self.participant}-N{self.night}"


def discover_nights() -> list[Night]:
    """Return every participant-night with an EDF and/or scored-events CSV."""
    nights: list[Night] = []
    roots = sorted(p for p in PAAWS_ROOT.iterdir() if p.is_dir())
    for root in roots:
        sleep_dir = root / "PAAWS_Sleep"
        if not sleep_dir.is_dir():
            continue
        for pdir in sorted(sleep_dir.iterdir()):
            if not pdir.is_dir() or not pdir.name.startswith("DS_"):
                continue
            edfs = {p.stem: p for p in pdir.glob("*.edf")}
            csvs = {p.name.replace("_scored_events.csv", ""): p for p in pdir.glob("*_scored_events.csv")}
            for stem in sorted(set(edfs) | set(csvs)):
                m = NIGHT_PATTERN.match(stem)
                if not m:
                    logging.warning("Unrecognised night file stem: %s", stem)
                    continue
                nights.append(
                    Night(
                        root_name=root.name,
                        participant=m.group("participant"),
                        night=int(m.group("night")),
                        edf=edfs.get(stem, pdir / f"{stem}.edf"),
                        csv=csvs.get(stem, pdir / f"{stem}_scored_events.csv"),
                    )
                )
    return nights


def configure_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def edf_header(path: Path) -> dict:
    """Parse an EDF header directly (fast, authoritative for labels/units).

    Only the fixed-size EDF header is read (~10-20 KB), never the signal data.
    """
    with path.open("rb") as fh:
        header = fh.read(256)
        if len(header) < 256:
            raise ValueError(f"Truncated EDF header: {path}")
        n_signals = int(header[252:256])
        sig_header = fh.read(256 * n_signals)
        if len(sig_header) < 256 * n_signals:
            raise ValueError(f"Truncated EDF signal header: {path}")

    def _block(start: int, width: int) -> list[str]:
        # EDF stores each field-major block across all signals consecutively.
        base = start * n_signals
        return [
            sig_header[base + i * width : base + (i + 1) * width].decode("latin-1").strip()
            for i in range(n_signals)
        ]

    labels = _block(0, 16)
    transducer = _block(16, 80)
    dimensions = _block(96, 8)
    phys_min = [float(v) for v in _block(104, 8)]
    phys_max = [float(v) for v in _block(112, 8)]
    dig_min = [int(float(v)) for v in _block(120, 8)]
    dig_max = [int(float(v)) for v in _block(128, 8)]
    prefilter = _block(136, 80)
    n_samples = [int(float(v)) for v in _block(216, 8)]

    n_records = int(header[236:244])
    record_duration = float(header[244:252])

    date_txt = header[168:176].decode("latin-1").strip()
    time_txt = header[176:184].decode("latin-1").strip()
    start_datetime = None
    try:
        day, month, year = (int(x) for x in date_txt.split("."))
        year += 1900 if year >= 85 else 2000
        hh, mm, ss = (int(x) for x in time_txt.split("."))
        start_datetime = f"{year:04d}-{month:02d}-{day:02d}T{hh:02d}:{mm:02d}:{ss:02d}"
    except (ValueError, AttributeError):
        start_datetime = None

    duration_s = n_records * record_duration
    channels = []
    annotation_channels = []
    for i, label in enumerate(labels):
        fs = n_samples[i] / record_duration if record_duration else float("nan")
        if label.strip().lower() == "edf annotations":
            annotation_channels.append(label)
            continue
        channels.append(
            {
                "channel_index": i,
                "channel_label": label,
                "sample_rate_hz": fs,
                "physical_dimension": dimensions[i],
                "physical_min": phys_min[i],
                "physical_max": phys_max[i],
                "digital_min": dig_min[i],
                "digital_max": dig_max[i],
                "prefilter": prefilter[i],
                "transducer": transducer[i],
                "samples_per_record": n_samples[i],
                "channel_sample_count": n_samples[i] * n_records,
                "channel_duration_h": (n_samples[i] * n_records) / fs / 3600 if fs else float("nan"),
            }
        )

    return {
        "n_records": n_records,
        "record_duration_s": record_duration,
        "recording_duration_h": duration_s / 3600,
        "n_channels": len(channels),
        "n_annotation_channels": len(annotation_channels),
        "start_datetime": start_datetime,
        "patient_field": header[8:88].decode("latin-1").strip(),
        "recording_field": header[88:168].decode("latin-1").strip(),
        "channels": channels,
    }
