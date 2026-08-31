#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Glacial Flow Mechanics (Elmer/Ice) Concept Graph v6.2 + QDWA
===============================================================================
Multi-level reasoning concept graph for glacial flow mechanics with the Elmer/Ice
finite element model. Integrates Query Distillation & Weighted Allocation (QDWA).
Focus: Rheology, basal sliding, numerical solvers, ice dynamics & thermodynamics.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import math
import os
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
import functools
import time
import io
import base64
import copy
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Iterator
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score, r2_score, mean_absolute_error,
    mean_squared_error, davies_bouldin_score, pairwise_distances
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches as mpatches
import seaborn as sns

from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# --- LLM & Query Analysis imports ---
from abc import ABC, abstractmethod
from enum import Enum as PyEnum

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    pipeline = None


# ============================================================================
# PERFORMANCE MONITORING DECORATOR
# ============================================================================
class PerformanceMonitor:
    _timings: Dict[str, float] = {}
    _call_counts: Dict[str, int] = {}

    @classmethod
    def reset(cls) -> None:
        cls._timings.clear()
        cls._call_counts.clear()

    @classmethod
    def get_report(cls) -> str:
        report = []
        for func_name, total_time in sorted(
            cls._timings.items(), key=lambda x: x[1], reverse=True
        ):
            count = cls._call_counts.get(func_name, 1)
            avg_time = total_time / count
            report.append(
                f"  {func_name}: {total_time:.3f}s total "
                f"({count} calls, {avg_time:.4f}s avg)"
            )
        return "\n".join(report)


# ============================================================================
# RUNTIME COMPUTATIONAL METRICS LOGGER
# ============================================================================
def get_memory_usage_mb() -> float:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return 0.0

class RunMetricsLogger:
    """Central logger for runtime computational metrics (steps + resources)."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_wall = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []
        self.counters: Dict[str, Any] = {}
        self._step_open: Optional[Dict[str, Any]] = None

    def start_step(self, name: str) -> None:
        self._sample_resources(tag=name, phase="start")
        self._step_open = {
            "name": name, "start": time.perf_counter(),
            "mem_start": get_memory_usage_mb(),
            "wall_start": time.time(),
        }

    def end_step(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self._step_open:
            return
        s = self._step_open
        elapsed = time.perf_counter() - s["start"]
        mem = get_memory_usage_mb()
        entry = {
            "name": s["name"],
            "duration_s": elapsed,
            "mem_start_mb": s["mem_start"],
            "mem_end_mb": mem,
            "mem_delta_mb": mem - s["mem_start"],
            "wall_clock": datetime.now().strftime("%H:%M:%S"),
            "elapsed_total_s": time.time() - self.start_wall,
        }
        if extra:
            entry["extra"] = extra
        self.steps.append(entry)
        self._sample_resources(tag=s["name"], phase="end")
        self._step_open = None

    def _sample_resources(self, tag: str, phase: str = "") -> None:
        self.resources.append({
            "wall_clock": datetime.now().strftime("%H:%M:%S"),
            "elapsed_s": time.time() - self.start_wall,
            "rss_mb": get_memory_usage_mb(),
            "threads": getattr(torch, "get_num_threads", lambda: 1)(),
            "cuda": torch.cuda.is_available(),
            "tag": f"{tag}:{phase}" if phase else tag,
        })

    def set(self, key: str, value: Any) -> None:
        self.counters[key] = value

    def add(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

    def step_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def resource_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.resources)

    def summary(self) -> Dict[str, Any]:
        df = self.step_df()
        total = float(df["duration_s"].sum()) if not df.empty else 0.0
        peak = max((r["rss_mb"] for r in self.resources), default=get_memory_usage_mb())
        return {
            "run_id": self.run_id,
            "total_duration_s": total,
            "wall_clock_s": time.time() - self.start_wall,
            "n_steps": len(self.steps),
            "peak_memory_mb": peak,
            "final_memory_mb": get_memory_usage_mb(),
            "counters": dict(self.counters),
            "function_timings": PerformanceMonitor.get_report(),
        }

    def save_log(self, folder: Optional[str] = None) -> str:
        folder = folder or os.path.join(SCRIPT_DIR, "metrics_logs")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"run_metrics_{self.run_id}.json")
        payload = {
            "summary": self.summary(),
            "steps": self.steps,
            "resources": self.resources,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

def get_metrics_logger() -> RunMetricsLogger:
    if "metrics_logger" not in st.session_state:
        st.session_state.metrics_logger = RunMetricsLogger()
    return st.session_state.metrics_logger

def render_live_metrics_panel(metrics: RunMetricsLogger, container) -> None:
    with container.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps completed", len(metrics.steps))
        c2.metric("Total time", f"{time.time() - metrics.start_wall:.1f}s")
        c3.metric("Memory (RSS)", f"{get_memory_usage_mb():.0f} MB")
        df = metrics.step_df()
        if not df.empty:
            last = df.iloc[-1]
            c4.metric("Last step", last["name"],
                      delta=f"{last['duration_s']:.2f}s", delta_color="off")
        with st.expander("📋 Live step table", expanded=False):
            if not df.empty:
                st.dataframe(
                    df[["name", "duration_s", "mem_start_mb",
                        "mem_end_mb", "mem_delta_mb", "wall_clock"]],
                    use_container_width=True,
                )
        rdf = metrics.resource_df()
        if len(rdf) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rdf["elapsed_s"], y=rdf["rss_mb"],
                mode="lines+markers", name="RSS Memory",
                text=rdf["tag"], hovertemplate=(
                    "<b>%{text}</b><br>Elapsed: %{x:.1f}s<br>Memory: %{y:.0f} MB"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                height=220, margin=dict(l=10, r=10, t=30, b=10),
                title="Memory timeline", xaxis_title="Elapsed (s)",
                yaxis_title="MB",
            )
            st.plotly_chart(fig, use_container_width=True)

def render_metrics_dashboard(metrics: RunMetricsLogger) -> None:
    if not metrics.steps:
        st.info("No metrics recorded yet — run an analysis first.")
        return
    df = metrics.step_df()
    rdf = metrics.resource_df()
    total = df["duration_s"].sum()
    peak = rdf["rss_mb"].max() if not rdf.empty else get_memory_usage_mb()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total runtime", f"{total:.2f} s")
    k2.metric("Wall clock", f"{time.time() - metrics.start_wall:.2f} s")
    k3.metric("Peak memory", f"{peak:.1f} MB")
    k4.metric("Steps executed", len(df))
    k5.metric("Avg step time", f"{df['duration_s'].mean():.2f} s")

    st.markdown("### ⏱️ Per-step duration")
    step_fig = px.bar(
        df.sort_values("duration_s", ascending=False),
        x="duration_s", y="name", orientation="h",
        color="duration_s", color_continuous_scale="Turbo",
        text=df["duration_s"].map(lambda v: f"{v:.2f}s"),
    )
    step_fig.update_layout(height=max(300, 25 * len(df)), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(step_fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        pie = px.pie(df, names="name", values="duration_s", hole=0.45,
                     title="Share of total runtime")
        st.plotly_chart(pie, use_container_width=True)
    with col_b:
        mem_fig = go.Figure()
        mem_fig.add_trace(go.Bar(x=df["name"], y=df["mem_start_mb"],
                                 name="Mem start", marker_color="#3b82f6"))
        mem_fig.add_trace(go.Bar(x=df["name"], y=df["mem_end_mb"],
                                 name="Mem end", marker_color="#ef4444"))
        mem_fig.update_layout(title="Memory before/after each step (MB)",
                              barmode="group", xaxis_tickangle=-45)
        st.plotly_chart(mem_fig, use_container_width=True)

    if len(rdf) > 1:
        st.markdown("### 💾 Resource timeline")
        timeline = go.Figure()
        timeline.add_trace(go.Scatter(
            x=rdf["elapsed_s"], y=rdf["rss_mb"], mode="lines+markers",
            name="RSS", text=rdf["tag"],
            hovertemplate="<b>%{text}</b><br>%{x:.1f}s — %{y:.0f} MB<extra></extra>",
        ))
        timeline.update_layout(xaxis_title="Elapsed (s)", yaxis_title="MB", height=280)
        st.plotly_chart(timeline, use_container_width=True)

    st.markdown("### 🔢 Numeric step log")
    st.dataframe(df, use_container_width=True)
    with st.expander("Raw resource samples"):
        st.dataframe(rdf, use_container_width=True)

    st.markdown("### 🧮 Pipeline counters")
    if metrics.counters:
        counter_df = pd.DataFrame(
            list(metrics.counters.items()), columns=["Counter", "Value"]
        )
        st.dataframe(counter_df, use_container_width=True)
    else:
        st.info("No counters recorded.")

    with st.expander("⚙️ Function-level timings (PerformanceMonitor)"):
        st.code(PerformanceMonitor.get_report() or "No timing data.", language="text")

    st.markdown("### 📥 Save log")
    if st.button("💾 Write metrics log to disk"):
        path = metrics.save_log()
        st.success(f"Log saved to `{path}`")
    json_str = json.dumps(
        {"summary": metrics.summary(), "steps": metrics.steps,
         "resources": metrics.resources},
        indent=2, default=str,
    )
    st.download_button("⬇️ Download metrics JSON", json_str.encode("utf-8"),
                       file_name=f"run_metrics_{metrics.run_id}.json",
                       mime="application/json")

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        func_name = func.__qualname__
        PerformanceMonitor._timings[func_name] = (
            PerformanceMonitor._timings.get(func_name, 0) + elapsed
        )
        PerformanceMonitor._call_counts[func_name] = (
            PerformanceMonitor._call_counts.get(func_name, 0) + 1
        )
        return result
    return wrapper


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Glacial Flow Mechanics (Elmer/Ice) Concept Graph v6.2 + QDWA",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)


# ============================================================================
# COLORMAP REGISTRY (50+)
# ============================================================================
SUPPORTED_COLORMAPS = {
    "viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno", "magma": "Magma",
    "cividis": "Cividis", "turbo": "Turbo", "jet": "Jet", "rainbow": "Rainbow",
    "hsv": "Hsv", "nipy_spectral": "NipySpectral", "gist_rainbow": "GistRainbow",
    "coolwarm": "Coolwarm", "RdBu": "RdBu", "seismic": "Seismic", "Spectral": "Spectral",
    "tab10": "Set1", "tab20": "Set2", "tab20b": "Set3", "Accent": "Accent",
    "Dark2": "Dark2", "Paired": "Paired", "Pastel1": "Pastel1", "Pastel2": "Pastel2",
    "cubehelix": "Cubehelix", "bone": "Bone", "gray": "Gray", "pink": "Pink",
    "spring": "Spring", "summer": "Summer", "autumn": "Autumn", "winter": "Winter",
    "cool": "Cool", "hot": "Hot", "twilight": "Twilight", "copper": "Copper",
    "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu",
    "BuPu": "BuPu", "GnBu": "GnBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn",
    "BuGn": "BuGn", "YlGn": "YlGn", "Greys": "Greys", "afmhot": "Afmhot",
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean",
    "prism": "Prism", "flag": "Flag", "gnuplot": "Gnuplot", "gnuplot2": "Gnuplot2",
    "brg": "BRG", "bwr": "BWR", "cmrmap": "CMRmap",
    "gist_gray": "GistGray", "gist_heat": "GistHeat", "gist_stern": "GistStern",
    "gist_yarg": "GistYarg", "binary": "Binary"
}


def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        try:
            cmap = cm.get_cmap(cmap_name, n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
        except Exception:
            try:
                cmap = matplotlib.colormaps.get_cmap("viridis").resampled(n)
            except Exception:
                cmap = cm.get_cmap("viridis", n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


# ============================================================================
# ROBUST FILE LOADER (JSON / JSONL / CSV / BibTeX)
# ============================================================================
# ============================================================================
# MICROTRANSFORMER CHART STYLING HELPERS
# ============================================================================
def plotly_continuous_scale(cmap_key: str, n: int = 12) -> List[str]:
    """Return a list of n hex colors from a matplotlib colormap."""
    return get_colormap_colors(cmap_key, n)

def apply_mt_chart_style(fig, theme: Dict, is_axial: bool = True):
    """Unify font/background/colorbar for microtransformer charts."""
    fam = st.session_state.get("mt_font_family", "Inter, Segoe UI, Roboto, sans-serif")
    tsize = int(st.session_state.get("mt_font_size", 11))
    fig.update_layout(
        font=dict(family=fam, size=tsize, color=theme["font"]),
        title_font=dict(family=fam, size=int(st.session_state.get("mt_title_size", 15))),
        paper_bgcolor=theme["plotly_paper"], plot_bgcolor=theme["plotly_bg"],
    )
    if is_axial:
        try:
            for ax in (fig.update_xaxes, fig.update_yaxes):
                ax(
                    showgrid=st.session_state.get("mt_show_grid", False),
                    gridcolor=theme["grid_color"],
                    tickfont=dict(family=fam, size=tsize, color=theme["axis_color"]),
                    title_font=dict(family=fam, size=tsize + 1),
                )
        except Exception:
            pass
    try:
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title=dict(text=st.session_state.get("mt_cbar_title", "Weight"),
                               font=dict(family=fam, size=tsize + 1, color=theme["font"])),
                    tickfont=dict(family=fam, size=max(8, tsize - 1), color=theme["axis_color"]),
                    thickness=st.session_state.get("mt_cbar_thick", 14),
                    outlinewidth=0,
                    len=st.session_state.get("mt_cbar_len", 0.8),
                )
            )
        )
    except Exception:
        pass
    return fig

def robust_load_file(filepath: Path):
    suffix = filepath.suffix.lower()
    if suffix == '.bib':
        return parse_bibtex_file(filepath)

    text = filepath.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"File is empty (0 bytes or only whitespace).")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    sanitized = re.sub(r'NaN', 'null', text)
    sanitized = re.sub(r'Infinity', 'null', sanitized)
    sanitized = re.sub(r'-Infinity', 'null', sanitized)
    sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if records:
        return records

    try:
        df = pd.read_csv(filepath)
        return df.to_dict(orient="records")
    except Exception:
        pass

    preview = text[:300]
    raise ValueError(
        f"Could not parse {filepath.name}. First 200 chars: {preview[:200]}..."
    )


def parse_bibtex_file(filepath: Path) -> List[Dict]:
    try:
        import bibtexparser
        from bibtexparser.bparser import BibTexParser
        from bibtexparser.customization import convert_to_unicode
        with open(filepath, 'r', encoding='utf-8') as bibfile:
            parser = BibTexParser()
            parser.customization = convert_to_unicode
            bib_database = bibtexparser.load(bibfile, parser=parser)
            records = []
            for entry in bib_database.entries:
                record = {
                    'title': entry.get('title', ''),
                    'abstract': entry.get('abstract', ''),
                    'author': entry.get('author', ''),
                    'year': entry.get('year', ''),
                    'journal': entry.get('journal', entry.get('booktitle', '')),
                    'doi': entry.get('doi', ''),
                    'keywords': entry.get('keywords', ''),
                    'entry_type': entry.get('ENTRYTYPE', ''),
                    'id': entry.get('ID', ''),
                    '_source_file': filepath.name,
                }
                records.append(record)
            return records
    except ImportError:
        st.warning(
            "bibtexparser not installed. Install with: pip install bibtexparser"
        )
        return []
    except Exception as e:
        st.error(f"BibTeX parse error for {filepath.name}: {e}")
        return []


@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    files = (
        sorted(Path(directory).glob("*.json"))
        + sorted(Path(directory).glob("*.bib"))
        + sorted(Path(directory).glob("*.csv"))
    )
    if not files:
        return []
    loaded = []
    for fp in files:
        try:
            data = robust_load_file(fp)
            if isinstance(data, list):
                loaded.append((str(fp.name), data))
            elif isinstance(data, dict):
                loaded.append((str(fp.name), [data]))
            else:
                loaded.append((str(fp.name), []))
        except Exception as e:
            st.error(f"Error loading `{fp.name}`: {e}")
            try:
                raw_bytes = fp.read_bytes()[:300]
                hex_str = raw_bytes.hex()
                formatted = ' '.join(
                    hex_str[i:i + 2] for i in range(0, len(hex_str), 2)
                )
                st.code(
                    f"Hex preview (first {len(raw_bytes)} bytes):\n{formatted}",
                    language="text",
                )
            except Exception:
                pass
    return loaded


@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    rows = []
    for fname, records in file_records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec["_source_file"] = fname
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df = df.replace({
        float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA
    })
    year_cols = [c for c in df.columns if 'year' in c.lower()]
    if year_cols:
        df["Year"] = pd.to_numeric(df[year_cols[0]], errors="coerce")
    elif "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


# ============================================================================
# ENHANCED ONTOLOGY & NLP REASONING SYSTEM (GLACIAL MECHANICS)
# ============================================================================
class ConceptType(Enum):
    MATERIAL = "material"
    PROCESS = "process"
    PROPERTY = "property"
    PHENOMENON = "phenomenon"
    METHOD = "method"
    PARAMETER = "parameter"
    MICROSTRUCTURE = "microstructure"
    MODEL = "model"
    GENERAL = "general"


class RelationshipType(Enum):
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    CAUSES = "causes"
    RESULTS_IN = "results_in"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    CO_OCCURS = "co_occurs"
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    BRIDGE = "bridge"
    CONSTRAINS = "constrains"
    MODIFIES = "modifies"
    CORRECTS = "corrects"
    SELECTS = "selects"
    INITIATES = "initiates"
    DRIVES = "drives"
    TRANSITIONS_TO = "transitions_to"
    REPLACES = "replaces"
    TRAINS = "trains"
    OUTPUTS = "outputs"
    LEARNS = "learns"
    CAPTURES = "captures"
    PARALLELIZES = "parallelizes"
    POSITIONS = "positions"
    IDENTIFIES = "identifies"
    FORMS = "forms"
    PROCESSES = "processes"
    STABILIZES = "stabilizes"
    PRESERVES = "preserves"
    GENERATES = "generates"
    COMPOSES = "composes"
    QUALIFIES = "qualifies"
    ENABLES = "enables"
    DISCOVERS = "discovers"
    PRE_TRAINS = "pre_trains"
    GENERALIZES = "generalizes"
    QUERIES = "queries"
    OPTIMIZES = "optimizes"
    VALIDATES = "validates"
    BOUNDS = "bounds"
    QUANTIFIES = "quantifies"
    EVALUATES = "evaluates"
    COMPARES = "compares"
    COMPUTES = "computes"
    MODELS = "models"
    AVERAGES = "averages"
    MAPS = "maps"
    SIMULATES = "simulates"
    DETECTS = "detects"
    INTEGRATES = "integrates"
    COUPLES = "couples"
    UPSCALES = "upscales"
    RESOLVES = "resolves"
    SYNCHRONIZES = "synchronizes"
    CHARACTERIZES = "characterizes"
    DECOMPOSES = "decomposes"
    DESIGNS = "designs"
    APPROXIMATES = "approximates"
    STRENGTHENS = "strengthens"
    EXPLAINS = "explains"
    INTERPRETS = "interprets"
    GROUPS = "groups"
    VISUALIZES = "visualizes"
    CONSTRUCTS = "constructs"
    FRAMES = "frames"
    ACCELERATES = "accelerates"
    ENFORCES = "enforces"
    CORRELATES = "correlates"

# ============================================================================
# EDGE COLOR REGISTRY — one distinct color per RelationshipType category
# ============================================================================

# ============================================================================
# RELATIONSHIP MAPPING FOR MICROTRANSFORMER
# ============================================================================
RELATIONSHIP_TO_IDX = {rel.name: i for i, rel in enumerate(RelationshipType)}
NUM_EDGE_TYPES = len(RelationshipType)

# 32 Specialized Latent Experts for Glacial Mechanics (Elmer/Ice)
GLACIER_EXPERT_LABELS = [
    # Rheology (8)
    "Glen's Flow Law", "Arrhenius Factor", "Glen Enhancement", "Anisotropic GOLF",
    "CAFFE Model", "Firn Compaction", "Damage Mechanics", "Critical Shear Rate",
    # Basal Sliding & Hydrology (8)
    "Weertman Friction", "Coulomb Friction", "Budd Friction", "Effective Pressure",
    "GlaDS Hydrology", "Double Continuum", "Grounding Line", "Contact Problem",
    # Numerical Solvers & Meshing (8)
    "ParStokes Solver", "Block Preconditioning", "Free Surface Evolution",
    "GMSH Remeshing", "Mesh Update", "Inverse Adjoint",
    "Streamline Stabilization", "Adjoint Data Assimilation",
    # Dynamics & Thermodynamics (8)
    "Full-Stokes Flow", "SSA Flow", "Enthalpy/Temp Solver",
    "Pressure Melting", "3D Calving", "Crevasse Depth",
    "Surface Mass Balance", "Basal Melt Parameterization"
]

EXPERT_CATEGORY = (
    ["rheology"] * 8 +
    ["sliding_hydro"] * 8 +
    ["numerical"] * 8 +
    ["dynamics_thermo"] * 8
)

EDGE_COLOR_REGISTRY: Dict[RelationshipType, str] = {
    # --- Semantic / structural ---
    RelationshipType.SYNONYM:           "#AAAAAA",
    RelationshipType.HYPERNYM:          "#5B9BD5",
    RelationshipType.HYPONYM:           "#5B9BD5",
    RelationshipType.PART_OF:           "#70AD47",
    RelationshipType.HAS_PART:          "#70AD47",
    RelationshipType.CO_OCCURS:         "#BFBFBF",

    # --- Causal / directional ---
    RelationshipType.CAUSES:            "#FF4444",
    RelationshipType.RESULTS_IN:        "#E06040",
    RelationshipType.INFLUENCES:        "#FF8C00",
    RelationshipType.DEPENDS_ON:        "#DAA520",
    RelationshipType.CONSTRAINS:        "#CC5500",
    RelationshipType.MODIFIES:          "#FF6347",
    RelationshipType.CORRECTS:          "#CD5C5C",
    RelationshipType.DRIVES:            "#DC143C",
    RelationshipType.ENABLES:           "#FF7F50",

    # --- Phase / thermodynamic transitions (reused for state changes) ---
    RelationshipType.TRANSITIONS_TO:    "#8A2BE2",
    RelationshipType.REPLACES:          "#9932CC",
    RelationshipType.FORMS:             "#9370DB",
    RelationshipType.STABILIZES:        "#7B68EE",
    RelationshipType.PRESERVES:         "#6A5ACD",

    # --- Computation / modeling ---
    RelationshipType.TRAINS:            "#00CED1",
    RelationshipType.OUTPUTS:           "#20B2AA",
    RelationshipType.LEARNS:            "#48D1CC",
    RelationshipType.CAPTURES:          "#40E0D0",
    RelationshipType.COMPUTES:          "#008B8B",
    RelationshipType.SIMULATES:         "#5F9EA0",
    RelationshipType.MODELS:            "#4682B4",
    RelationshipType.APPROXIMATES:      "#87CEEB",
    RelationshipType.MAPS:              "#00BFFF",

    # --- Analysis / evaluation ---
    RelationshipType.QUANTIFIES:        "#32CD32",
    RelationshipType.EVALUATES:         "#228B22",
    RelationshipType.COMPARES:          "#3CB371",
    RelationshipType.VALIDATES:         "#2E8B57",
    RelationshipType.AVERAGES:          "#66CDAA",
    RelationshipType.CORRELATES:        "#00FA9A",

    # --- Structural / architectural ---
    RelationshipType.PARALLELIZES:      "#FFD700",
    RelationshipType.POSITIONS:         "#FFC125",
    RelationshipType.IDENTIFIES:        "#F0E68C",
    RelationshipType.PROCESSES:         "#EEE8AA",
    RelationshipType.GROUPS:            "#DAA520",
    RelationshipType.INTEGRATES:        "#B8860B",
    RelationshipType.COUPLES:           "#CD950C",

    # --- Discovery / optimization ---
    RelationshipType.DISCOVERS:         "#FF69B4",
    RelationshipType.PRE_TRAINS:        "#FF1493",
    RelationshipType.GENERALIZES:       "#DB7093",
    RelationshipType.QUERIES:           "#C71585",
    RelationshipType.OPTIMIZES:         "#FF00FF",
    RelationshipType.DESIGNS:           "#BA55D3",
    RelationshipType.CONSTRUCTS:        "#DA70D6",

    # --- Advanced modeling ---
    RelationshipType.UPSCALES:          "#8B4513",
    RelationshipType.RESOLVES:          "#A0522D",
    RelationshipType.SYNCHRONIZES:      "#D2691E",
    RelationshipType.CHARACTERIZES:     "#CD853F",
    RelationshipType.DECOMPOSES:        "#DEB887",
    RelationshipType.FRAMES:            "#D2B48C",
    RelationshipType.COMPOSES:          "#BC8F8F",
    RelationshipType.QUALIFIES:         "#F4A460",

    # --- Explanation / visualization ---
    RelationshipType.STRENGTHENS:       "#7FFF00",
    RelationshipType.EXPLAINS:          "#ADFF2F",
    RelationshipType.INTERPRETS:        "#7CFC00",
    RelationshipType.VISUALIZES:        "#00FF7F",
    RelationshipType.ACCELERATES:       "#98FB98",
    RelationshipType.ENFORCES:          "#90EE90",

    # --- Generic fallback ---
    RelationshipType.SEMANTIC:          "#808080",
    RelationshipType.INFERRED:          "#A9A9A9",
    RelationshipType.BRIDGE:            "#C0C0C0",
    RelationshipType.SELECTS:           "#D3D3D3",
    RelationshipType.INITIATES:         "#696969",
    RelationshipType.DETECTS:           "#556B2F",
    RelationshipType.GENERATES:         "#6B8E23",
}

EDGE_COLOR_FALLBACK = "#888888"


def get_edge_color(rel_type: RelationshipType) -> str:
    return EDGE_COLOR_REGISTRY.get(rel_type, EDGE_COLOR_FALLBACK)


def get_edge_width(rel_type: RelationshipType) -> float:
    STRONG = {RelationshipType.CAUSES, RelationshipType.DRIVES,
              RelationshipType.FORMS, RelationshipType.STABILIZES,
              RelationshipType.DEPENDS_ON, RelationshipType.CONSTRAINS}
    MEDIUM = {RelationshipType.INFLUENCES, RelationshipType.RESULTS_IN,
              RelationshipType.MODIFIES, RelationshipType.ENABLES,
              RelationshipType.TRANSITIONS_TO, RelationshipType.COMPUTES}
    if rel_type in STRONG:
        return 3.0
    elif rel_type in MEDIUM:
        return 2.0
    return 1.0


def get_edge_style(rel_type: RelationshipType) -> str:
    DASHED = {RelationshipType.INFERRED, RelationshipType.CO_OCCURS,
              RelationshipType.SEMANTIC, RelationshipType.BRIDGE}
    return "dashed" if rel_type in DASHED else "solid"


def lighten_hex_color(hex_color: str, factor: float) -> str:
    if not hex_color.startswith('#'):
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class ConceptNode:
    canonical_name: str
    concept_type: ConceptType
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    related_processes: Set[str] = field(default_factory=set)
    related_properties: Set[str] = field(default_factory=set)
    definition: str = ""
    embedding: Optional[np.ndarray] = None

    def add_synonym(self, synonym: str) -> None:
        self.synonyms.add(synonym.lower().strip())

    def is_match(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if text_lower == self.canonical_name.lower():
            return True
        return text_lower in self.synonyms


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: RelationshipType
    confidence: float = 1.0
    evidence: str = ""
    inferred: bool = False


class DomainOntology:
    """Comprehensive ontology for Glacial Flow Mechanics (Elmer/Ice)."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self._build_ontology()

    def _build_ontology(self) -> None:
        # === 1. Rheology & Material Physics ===
        self._add_concept("glens_law", ConceptType.MODEL,
            synonyms={"glen", "glen's flow law"},
            definition="Constitutive law for ice deformation relating stress and strain rate.")
        self._add_concept("arrhenius_factor", ConceptType.PARAMETER,
            synonyms={"arrhenius", "rate factor"},
            definition="Temperature-dependent pre-exponential factor in Glen's law.")
        self._add_concept("glen_enhancement_factor", ConceptType.PARAMETER,
            synonyms={"enhancement", "e_factor"},
            definition="Multiplier to account for fabric or damage in Glen's law.")
        self._add_concept("anisotropic_ice", ConceptType.MATERIAL,
            synonyms={"anisotropic", "golf", "caffe"},
            definition="Ice with directional fabric (e.g., GOLF, CAFFE models).")
        self._add_concept("golf_model", ConceptType.MODEL,
            synonyms={"golf"},
            definition="General Orthotropic Linear Flow (GOLF) model for anisotropic ice.")
        self._add_concept("caffe_model", ConceptType.MODEL,
            synonyms={"caffe"},
            definition="Crystal fabric based flow enhancement (CAFFE) model.")
        self._add_concept("firn_densification", ConceptType.PROCESS,
            synonyms={"firn compaction"},
            definition="Densification of snow to firn and ice under overburden pressure.")
        self._add_concept("damage_mechanics", ConceptType.PROCESS,
            synonyms={"damage"},
            definition="Continuum damage mechanics for ice fracturing and weakening.")
        self._add_concept("critical_shear_rate", ConceptType.PARAMETER,
            synonyms={"shear rate"},
            definition="Threshold shear rate for non-Newtonian ice response.")
        self._add_concept("viscosity_model", ConceptType.PROPERTY,
            synonyms={"effective viscosity"},
            definition="Effective viscosity of ice computed at Gauss points.")
        self._add_concept("porous_law", ConceptType.MODEL,
            synonyms={"porous flow"},
            definition="Flow law for porous ice (e.g., firn).")
        self._add_concept("fabric_evolution", ConceptType.PHENOMENON,
            synonyms={"fabric", "crystal orientation"},
            definition="Evolution of ice crystal fabric under deformation.")

        # === 2. Basal Sliding & Hydrology ===
        self._add_concept("weertman_friction", ConceptType.MODEL,
            synonyms={"weertman"},
            definition="Non-linear Weertman-type basal sliding law.")
        self._add_concept("coulomb_friction", ConceptType.MODEL,
            synonyms={"coulomb"},
            definition="Water-pressure dependent Coulomb friction law.")
        self._add_concept("budd_friction", ConceptType.MODEL,
            synonyms={"budd"},
            definition="Budd friction law for basal sliding.")
        self._add_concept("effective_pressure", ConceptType.PARAMETER,
            synonyms={"N_eff"},
            definition="Effective pressure (ice overburden minus water pressure).")
        self._add_concept("glads_hydrology", ConceptType.MODEL,
            synonyms={"glads"},
            definition="Subglacial hydrology model with distributed channels.")
        self._add_concept("double_continuum_hydro", ConceptType.MODEL,
            synonyms={"double continuum"},
            definition="Double-continuum subglacial hydrology model.")
        self._add_concept("water_pressure", ConceptType.PARAMETER,
            synonyms={"p_w"},
            definition="Pore water pressure at the bed.")
        self._add_concept("geothermal_flux", ConceptType.PARAMETER,
            synonyms={"geothermal"},
            definition="Geothermal heat flux from the Earth's interior.")
        self._add_concept("contact_problem", ConceptType.METHOD,
            synonyms={"contact mechanics"},
            definition="Contact mechanics defining the grounding line interface.")
        self._add_concept("subglacial_cavities", ConceptType.PHENOMENON,
            synonyms={"cavities"},
            definition="Formation of water-filled cavities below the ice.")
        self._add_concept("lateral_friction", ConceptType.PROPERTY,
            synonyms={"side friction"},
            definition="Friction at lateral boundaries of outlet glaciers.")
        self._add_concept("shape_factor", ConceptType.PARAMETER,
            synonyms={"shape"},
            definition="Cross-sectional shape factor for flowline models.")

        # === 3. Numerical Solvers & Meshing ===
        self._add_concept("parstokes", ConceptType.METHOD,
            synonyms={"parstokes solver"},
            definition="Iterative parallel Stokes solver for large 3D applications.")
        self._add_concept("block_preconditioning", ConceptType.METHOD,
            synonyms={"preconditioner"},
            definition="Block upper triangular preconditioner for Stokes equations.")
        self._add_concept("free_surface_evolution", ConceptType.PROCESS,
            synonyms={"free surface"},
            definition="Evolution of the ice surface and grounding line.")
        self._add_concept("streamline_stabilization", ConceptType.METHOD,
            synonyms={"stabilization"},
            definition="Streamline upwind stabilization for advection-dominated problems.")
        self._add_concept("gmsh_remeshing", ConceptType.PROCESS,
            synonyms={"gmsh", "remeshing"},
            definition="Mesh adaptation following calving events using GMSH.")
        self._add_concept("mesh_update", ConceptType.PROCESS,
            synonyms={"mesh update"},
            definition="Updating the computational mesh to follow free surface.")
        self._add_concept("inverse_adjoint", ConceptType.METHOD,
            synonyms={"adjoint", "inverse"},
            definition="Linear adjoint method for inverse modeling.")
        self._add_concept("adjoint_data_assimilation", ConceptType.METHOD,
            synonyms={"data assimilation"},
            definition="Adjoint-based data assimilation for parameter estimation.")
        self._add_concept("flowline_geometry", ConceptType.PARAMETER,
            synonyms={"flowline"},
            definition="1D flowline geometry for simplified models.")
        self._add_concept("griddata_reader", ConceptType.METHOD,
            synonyms={"grid data"},
            definition="Reader for gridded input data (e.g., topography, SMB).")
        self._add_concept("ssa_flow", ConceptType.MODEL,
            synonyms={"ssa"},
            definition="Shallow Shelf Approximation (SSA) for ice shelves.")
        self._add_concept("sia_flow", ConceptType.MODEL,
            synonyms={"sia"},
            definition="Shallow Ice Approximation (SIA) for ice sheets.")

        # === 4. Ice Dynamics & Thermodynamics ===
        self._add_concept("full_stokes", ConceptType.MODEL,
            synonyms={"stokes"},
            definition="Full Navier-Stokes momentum equation without inertia.")
        self._add_concept("enthalpy_solver", ConceptType.METHOD,
            synonyms={"enthalpy"},
            definition="Thermodynamic solver handling temperate ice and CTS.")
        self._add_concept("temperature_solver", ConceptType.METHOD,
            synonyms={"temperature"},
            definition="Solver for the ice temperature field.")
        self._add_concept("pressure_melting_point", ConceptType.PROPERTY,
            synonyms={"pm", "melting point"},
            definition="Pressure-dependent melting point of ice.")
        self._add_concept("calving_3d", ConceptType.PROCESS,
            synonyms={"calving"},
            definition="3D calving model based on crevasse depth penetration.")
        self._add_concept("crevasse_depth", ConceptType.PROPERTY,
            synonyms={"crevasses"},
            definition="Depth of surface and basal crevasses.")
        self._add_concept("surface_mass_balance", ConceptType.PARAMETER,
            synonyms={"smb", "mass balance"},
            definition="Net accumulation/ablation at the ice surface.")
        self._add_concept("basal_melt_parameterization", ConceptType.PARAMETER,
            synonyms={"basal melt"},
            definition="Sub-shelf ocean melt rates.")
        self._add_concept("hydrostatic_loads", ConceptType.PROPERTY,
            synonyms={"hydrostatic"},
            definition="Hydrostatic pressure loads from ocean water.")
        self._add_concept("ismip_hom", ConceptType.MODEL,
            synonyms={"ismip"},
            definition="ISMIP-HOM benchmark for ice sheet models.")
        self._add_concept("ice_shelf_flow", ConceptType.PROPERTY,
            synonyms={"shelf flow"},
            definition="Flow dynamics of ice shelves.")
        self._add_concept("ablation_model", ConceptType.MODEL,
            synonyms={"ablation"},
            definition="Model for surface ablation (melting).")

        self._build_synonym_index()
        self._build_causal_chains()

    def _add_concept(
        self,
        canonical_name: str,
        concept_type: ConceptType,
        synonyms: Set[str] = None,
        hypernyms: Set[str] = None,
        hyponyms: Set[str] = None,
        definition: str = "",
        related_processes: Set[str] = None,
        related_properties: Set[str] = None,
    ) -> None:
        node = ConceptNode(
            canonical_name=canonical_name,
            concept_type=concept_type,
            synonyms=synonyms or set(),
            hypernyms=hypernyms or set(),
            hyponyms=hyponyms or set(),
            related_processes=related_processes or set(),
            related_properties=related_properties or set(),
            definition=definition,
        )
        self.concepts[canonical_name] = node

    def _build_synonym_index(self) -> None:
        self.synonym_to_canonical: Dict[str, str] = {}
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def _build_causal_chains(self) -> None:
        causal_chains = [
            # Rheology -> Dynamics
            ("glens_law", RelationshipType.INFLUENCES, "viscosity_model", 0.95),
            ("arrhenius_factor", RelationshipType.INFLUENCES, "glens_law", 0.90),
            ("damage_mechanics", RelationshipType.INFLUENCES, "glen_enhancement_factor", 0.85),
            ("anisotropic_ice", RelationshipType.INFLUENCES, "glen_enhancement_factor", 0.80),

            # Sliding & Hydro -> Dynamics
            ("coulomb_friction", RelationshipType.DEPENDS_ON, "effective_pressure", 0.90),
            ("glads_hydrology", RelationshipType.INFLUENCES, "water_pressure", 0.85),
            ("contact_problem", RelationshipType.CONSTRAINS, "grounding_line_dynamics", 0.95),
            ("geothermal_flux", RelationshipType.INFLUENCES, "temperature_solver", 0.80),
            ("weertman_friction", RelationshipType.INFLUENCES, "full_stokes", 0.85),
            ("budd_friction", RelationshipType.INFLUENCES, "full_stokes", 0.80),

            # Numerical -> Solvers
            ("parstokes", RelationshipType.OPTIMIZES, "full_stokes", 0.95),
            ("block_preconditioning", RelationshipType.STABILIZES, "full_stokes", 0.85),
            ("free_surface_evolution", RelationshipType.DRIVES, "mesh_update", 0.85),
            ("inverse_adjoint", RelationshipType.QUANTIFIES, "coulomb_friction", 0.85),
            ("streamline_stabilization", RelationshipType.ENABLES, "parstokes", 0.80),

            # Dynamics & Thermo
            ("enthalpy_solver", RelationshipType.INFLUENCES, "pressure_melting_point", 0.90),
            ("crevasse_depth", RelationshipType.CAUSES, "calving_3d", 0.95),
            ("calving_3d", RelationshipType.CAUSES, "gmsh_remeshing", 0.90),
            ("surface_mass_balance", RelationshipType.INFLUENCES, "ablation_model", 0.80),
            ("basal_melt_parameterization", RelationshipType.INFLUENCES, "ice_shelf_flow", 0.85),
            ("temperature_solver", RelationshipType.INFLUENCES, "enthalpy_solver", 0.90),

            # Cross-Domain Bridges
            ("coulomb_friction", RelationshipType.CONSTRAINS, "full_stokes", 0.85),
            ("temperature_solver", RelationshipType.INFLUENCES, "glads_hydrology", 0.65),
            ("parstokes", RelationshipType.COMPUTES, "grounding_line_dynamics", 0.90),
            ("geothermal_flux", RelationshipType.INFLUENCES, "basal_melt_parameterization", 0.75),
            ("ice_shelf_flow", RelationshipType.INFLUENCES, "calving_3d", 0.70),
        ]
        for source, rel_type, target, confidence in causal_chains:
            self.relationships.append(
                Relationship(source, target, rel_type, abs(confidence))
            )

    def resolve_concept(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in self.synonym_to_canonical:
            return self.synonym_to_canonical[text_lower]
        normalized = self._normalize_text(text_lower)
        if normalized in self.synonym_to_canonical:
            return self.synonym_to_canonical[normalized]
        variants = [
            text_lower.replace("-", " "),
            text_lower.replace(" ", "-"),
            text_lower.replace(" of ", " "),
            text_lower.replace(" for ", " "),
            text_lower.replace(" in ", " "),
            re.sub(r'\bs\b', '', text_lower),
            re.sub(r'\bes\b', '', text_lower),
        ]
        for variant in variants:
            if variant in self.synonym_to_canonical:
                return self.synonym_to_canonical[variant]
        return None

    def _normalize_text(self, text: str) -> str:
        text = re.sub(
            r'\b(the|a|an|of|for|in|with|by|to|and|or|on|at)\b', ' ', text
        )
        text = ' '.join(text.split())
        return text.strip()

    def get_concept_type(self, canonical_name: str) -> ConceptType:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].concept_type
        return ConceptType.GENERAL

    def get_hypernyms(self, canonical_name: str) -> Set[str]:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].hypernyms
        return set()

    def get_hyponyms(self, canonical_name: str) -> Set[str]:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].hyponyms
        return set()

    def get_definition(self, canonical_name: str) -> str:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].definition
        return ""

    def infer_path(
        self, source: str, target: str, max_depth: int = 3
    ) -> List[List[str]]:
        paths: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(current: str, target: str, path: List[str], depth: int) -> None:
            if depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            if current in visited:
                return
            visited.add(current)
            for rel in self.relationships:
                if rel.source == current and rel.confidence > 0.5:
                    path.append(rel.target)
                    dfs(rel.target, target, path, depth + 1)
                    path.pop()
            if current in self.concepts:
                for hyp in self.concepts[current].hypernyms:
                    path.append(hyp)
                    dfs(hyp, target, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(source, target, [source], 0)
        return paths

    def get_related_concepts(
        self, canonical_name: str, rel_type: RelationshipType = None
    ) -> List[Tuple[str, RelationshipType, float]]:
        related: List[Tuple[str, RelationshipType, float]] = []
        for rel in self.relationships:
            if rel.source == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.target, rel.rel_type, rel.confidence))
            elif rel.target == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.source, rel.rel_type, rel.confidence))
        return related


def ensure_ontology_populated() -> "DomainOntology":
    """
    Returns a fully populated DomainOntology instance, re-initialising it if necessary.
    """
    if "ontology" not in st.session_state or not st.session_state.ontology.concepts:
        st.session_state.ontology = DomainOntology()

    ontology = st.session_state.ontology
    has_property = any(node.concept_type == ConceptType.PROPERTY for node in ontology.concepts.values())
    if not has_property:
        st.session_state.ontology = DomainOntology()
    return st.session_state.ontology


# ============================================================================
# ADVANCED CONCEPT RESOLVER (AgNPs Pattern — Eager Precomputation)
# ============================================================================
# ============================================================================
# HIERARCHY LABEL BUILDER — enriches flat concept names with ancestor path
# ============================================================================

_HIERARCHY_PARENTS: Dict[str, Tuple[str, int]] = {
    "glacial_flow_mechanics": (None, 0),

    # Rheology & Material Physics
    "glens_law": ("Rheology & Material Physics", 1),
    "arrhenius_factor": ("Rheology & Material Physics", 1),
    "glen_enhancement_factor": ("Rheology & Material Physics", 1),
    "anisotropic_ice": ("Rheology & Material Physics", 1),
    "golf_model": ("Rheology & Material Physics", 1),
    "caffe_model": ("Rheology & Material Physics", 1),
    "firn_densification": ("Rheology & Material Physics", 1),
    "damage_mechanics": ("Rheology & Material Physics", 1),
    "critical_shear_rate": ("Rheology & Material Physics", 1),
    "viscosity_model": ("Rheology & Material Physics", 1),
    "porous_law": ("Rheology & Material Physics", 1),
    "fabric_evolution": ("Rheology & Material Physics", 1),

    # Basal Sliding & Hydrology
    "weertman_friction": ("Basal Sliding & Hydrology", 1),
    "coulomb_friction": ("Basal Sliding & Hydrology", 1),
    "budd_friction": ("Basal Sliding & Hydrology", 1),
    "effective_pressure": ("Basal Sliding & Hydrology", 1),
    "glads_hydrology": ("Basal Sliding & Hydrology", 1),
    "double_continuum_hydro": ("Basal Sliding & Hydrology", 1),
    "water_pressure": ("Basal Sliding & Hydrology", 1),
    "geothermal_flux": ("Basal Sliding & Hydrology", 1),
    "contact_problem": ("Basal Sliding & Hydrology", 1),
    "subglacial_cavities": ("Basal Sliding & Hydrology", 1),
    "lateral_friction": ("Basal Sliding & Hydrology", 1),
    "shape_factor": ("Basal Sliding & Hydrology", 1),

    # Numerical Solvers & Meshing
    "parstokes": ("Numerical Solvers & Meshing", 1),
    "block_preconditioning": ("Numerical Solvers & Meshing", 1),
    "free_surface_evolution": ("Numerical Solvers & Meshing", 1),
    "streamline_stabilization": ("Numerical Solvers & Meshing", 1),
    "gmsh_remeshing": ("Numerical Solvers & Meshing", 1),
    "mesh_update": ("Numerical Solvers & Meshing", 1),
    "inverse_adjoint": ("Numerical Solvers & Meshing", 1),
    "adjoint_data_assimilation": ("Numerical Solvers & Meshing", 1),
    "flowline_geometry": ("Numerical Solvers & Meshing", 1),
    "griddata_reader": ("Numerical Solvers & Meshing", 1),
    "ssa_flow": ("Numerical Solvers & Meshing", 1),
    "sia_flow": ("Numerical Solvers & Meshing", 1),

    # Ice Dynamics & Thermodynamics
    "full_stokes": ("Ice Dynamics & Thermodynamics", 1),
    "enthalpy_solver": ("Ice Dynamics & Thermodynamics", 1),
    "temperature_solver": ("Ice Dynamics & Thermodynamics", 1),
    "pressure_melting_point": ("Ice Dynamics & Thermodynamics", 1),
    "calving_3d": ("Ice Dynamics & Thermodynamics", 1),
    "crevasse_depth": ("Ice Dynamics & Thermodynamics", 1),
    "surface_mass_balance": ("Ice Dynamics & Thermodynamics", 1),
    "basal_melt_parameterization": ("Ice Dynamics & Thermodynamics", 1),
    "hydrostatic_loads": ("Ice Dynamics & Thermodynamics", 1),
    "ismip_hom": ("Ice Dynamics & Thermodynamics", 1),
    "ice_shelf_flow": ("Ice Dynamics & Thermodynamics", 1),
    "ablation_model": ("Ice Dynamics & Thermodynamics", 1),
}

def get_hierarchy_label(concept_key: str,
                        style: str = "arrow") -> str:
    leaf = concept_key.replace("_", " ").title()
    entry = _HIERARCHY_PARENTS.get(concept_key)
    if entry is None or entry[0] is None or style == "leaf":
        return leaf
    parent_label = entry[0]
    sep = " → " if style == "arrow" else " [" if style == "bracket" else " · "
    if style == "bracket":
        return f"{parent_label}{sep}{leaf}]"
    return f"{parent_label}{sep}{leaf}"


# ============================================================================
# GLACIAL MECHANICS KEYWORDS & PATTERNS
# ============================================================================
GLACIER_KEYWORDS = [
    "elmer/ice", "stokes", "ssa", "sia", "glens law", "arrhenius", "enthalpy",
    "calving", "crevasse", "grounding line", "hydrology", "glads", "friction",
    "weertman", "coulomb", "budd", "mesh", "gmsh", "remeshing", "smb", "basal melt",
    "viscosity", "damage", "fabric", "anisotropic", "golf", "caffe", "firn",
    "effective pressure", "water pressure", "geothermal", "contact problem",
    "parstokes", "preconditioning", "free surface", "streamline", "adjoint",
    "data assimilation", "flowline", "griddata", "ismip", "ice shelf", "ablation"
]

GLACIER_DESCRIPTOR_MAPPING = {
    r'glen|arrhenius|viscosity|anisotrop|golf|caffe|damage|firn': 'rheology',
    r'friction|weertman|coulomb|budd|hydrolog|glads|water pressure|grounding|contact': 'sliding_hydro',
    r'stokes|parstokes|precondition|gmsh|mesh|adjoint|inverse|solver|freesurface': 'numerical',
    r'enthalpy|temperature|calving|crevasse|smb|mass balance|basal melt|shelf|ablation': 'dynamics_thermo',
}

GLACIER_QUANTITATIVE_PATTERNS = [
    r'\bglens?\s*law\b', r'\barrhenius\s*factor\b', r'\bfull\s*stokes\b',
    r'\bssa\s*flow\b', r'\bsia\s*flow\b', r'\benthalpy\s*solver\b',
    r'\bgrounding\s*line\b', r'\bcontact\s*problem\b', r'\bsurface\s*mass\s*balance\b',
    r'\bbasal\s*melt\b', r'\b3d\s*calving\b', r'\bcrevasse\s*depth\b',
    r'\bweertman\s*friction\b', r'\bcoulomb\s*friction\b', r'\bglads\b',
    r'\bparstokes\b', r'\bgmsh\s*remeshing\b', r'\binverse\s*adjoint\b',
    r'\beffective\s*pressure\b', r'\bwater\s*pressure\b', r'\bgeothermal\s*flux\b',
    r'\bflowline\s*geometry\b', r'\bismip\s*hom\b', r'\bice\s*shelf\s*flow\b',
    r'\bfabric\s*evolution\b', r'\bdamage\s*mechanics\b',
]

ALL_GLACIER_KEYWORDS = GLACIER_KEYWORDS


def extract_glacier_concepts_from_text(text: str) -> List[str]:
    concepts = set()
    text_lower = text.lower()
    for pattern in GLACIER_QUANTITATIVE_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    for keyword in GLACIER_KEYWORDS:
        if keyword in text_lower:
            concepts.add(keyword)
    return list(concepts)

def is_valid_glacier_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw in concept_lower for kw in GLACIER_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in GLACIER_QUANTITATIVE_PATTERNS)
    generic = {'study', 'analysis', 'effect', 'role', 'investigation', 'research',
               'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
               'new', 'recent', 'various', 'different', 'significant', 'important'}
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic

def normalize_glacier_concept(concept: str) -> str:
    concept = concept.lower().strip()
    mapping = {
        r'\bglens?\s*law\b': 'glens_law',
        r'\barrhenius\s*factor\b': 'arrhenius_factor',
        r'\bfull\s*stokes\b': 'full_stokes',
        r'\bssa\s*flow\b': 'ssa_flow',
        r'\bsia\s*flow\b': 'sia_flow',
        r'\benthalpy\s*solver\b': 'enthalpy_solver',
        r'\bgrounding\s*line\b': 'grounding_line_dynamics',
        r'\bcontact\s*problem\b': 'contact_problem',
        r'\bsurface\s*mass\s*balance\b': 'surface_mass_balance',
        r'\bbasal\s*melt\b': 'basal_melt_parameterization',
        r'\b3d\s*calving\b': 'calving_3d',
        r'\bcrevasse\s*depth\b': 'crevasse_depth',
        r'\bweertman\s*friction\b': 'weertman_friction',
        r'\bcoulomb\s*friction\b': 'coulomb_friction',
        r'\bglads\b': 'glads_hydrology',
        r'\bparstokes\b': 'parstokes',
        r'\bgmsh\s*remeshing\b': 'gmsh_remeshing',
        r'\binverse\s*adjoint\b': 'inverse_adjoint',
        r'\beffective\s*pressure\b': 'effective_pressure',
        r'\bwater\s*pressure\b': 'water_pressure',
        r'\bgeothermal\s*flux\b': 'geothermal_flux',
        r'\bflowline\s*geometry\b': 'flowline_geometry',
        r'\bismip\s*hom\b': 'ismip_hom',
        r'\bice\s*shelf\s*flow\b': 'ice_shelf_flow',
        r'\bfabric\s*evolution\b': 'fabric_evolution',
        r'\bdamage\s*mechanics\b': 'damage_mechanics',
    }
    for pattern, replacement in mapping.items():
        concept = re.sub(pattern, replacement, concept)
    return concept


def categorize_glacier_concept(concept: str) -> str:
    concept_lower = concept.lower()
    for pattern, category in GLACIER_DESCRIPTOR_MAPPING.items():
        if re.search(pattern, concept_lower):
            return category
    return "general"

def get_glacier_category_color(concept: str, cmap_colors=None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    cat = categorize_glacier_concept(concept)
    color_map = {
        'rheology': '#3B82F6',
        'sliding_hydro': '#10B981',
        'numerical': '#F59E0B',
        'dynamics_thermo': '#EF4444',
        'general': '#A8D8EA',
    }
    return color_map.get(cat, '#7f7f7f')


def ensure_ontology_populated() -> "DomainOntology":
    if "ontology" not in st.session_state or not st.session_state.ontology.concepts:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology
    has_property = any(node.concept_type == ConceptType.PROPERTY for node in ontology.concepts.values())
    if not has_property:
        st.session_state.ontology = DomainOntology()
    return st.session_state.ontology

# ============================================================================
# ADVANCED CONCEPT RESOLVER
# ============================================================================
class AdvancedConceptResolver:
    def __init__(
        self,
        ontology: DomainOntology,
        embed_model,
        cache_max: int = 2000,
    ) -> None:
        self.ontology = ontology
        self.embed_model = embed_model
        self.resolution_cache: Dict[str, str] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max = max(100, int(cache_max))
        self.similarity_threshold = 0.85
        self.ontology_concepts_list: Optional[List[str]] = None
        self.ontology_embedding_matrix: Optional[np.ndarray] = None
        self._precompute_ontology_embeddings()

    def _trim_embedding_cache(self) -> None:
        if len(self.embedding_cache) > self._cache_max:
            keys = list(self.embedding_cache.keys())
            for k in keys[:int(len(keys) * 0.3)]:
                del self.embedding_cache[k]
            gc.collect()

    def _trim_resolution_cache(self) -> None:
        if len(self.resolution_cache) > self._cache_max * 4:
            keys = list(self.resolution_cache.keys())
            for k in keys[:int(len(keys) * 0.3)]:
                del self.resolution_cache[k]

    def _precompute_ontology_embeddings(self) -> None:
        concepts: List[str] = []
        all_texts: List[str] = []
        text_counts: List[int] = []

        for canonical, node in self.ontology.concepts.items():
            concepts.append(canonical)
            texts = [canonical] + list(node.synonyms)
            all_texts.extend(texts)
            text_counts.append(len(texts))

        if not all_texts:
            self.ontology_concepts_list = []
            self.ontology_embedding_matrix = np.empty((0, 0))
            return

        with torch.no_grad():
            all_embeddings = self.embed_model.encode(
                all_texts,
                show_progress_bar=False,
                batch_size=64,
                convert_to_numpy=True,
            )

        embeddings: List[np.ndarray] = []
        idx = 0
        for count in text_counts:
            concept_embs = all_embeddings[idx:idx + count]
            embeddings.append(np.mean(concept_embs, axis=0))
            idx += count

        del all_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.ontology_concepts_list = concepts
        self.ontology_embedding_matrix = (
            np.array(embeddings) if embeddings else np.empty((0, 0))
        )

    @timed
    def resolve(
        self, text: str, context: str = "", use_embedding: bool = True
    ) -> Optional[str]:
        self._trim_resolution_cache()
        text_lower = text.lower().strip()
        if text_lower in self.resolution_cache:
            return self.resolution_cache[text_lower]

        canonical = self.ontology.resolve_concept(text)
        if canonical:
            self.resolution_cache[text_lower] = canonical
            return canonical

        canonical = self._substring_match(text_lower)
        if canonical:
            self.resolution_cache[text_lower] = canonical
            return canonical

        if use_embedding and self.ontology_embedding_matrix.size > 0:
            canonical = self._embedding_match(text, context)
            if canonical:
                self.resolution_cache[text_lower] = canonical
                return canonical

        if context:
            canonical = self._context_disambiguation(text_lower, context)
            if canonical:
                self.resolution_cache[text_lower] = canonical
                return canonical

        return None

    @timed
    def resolve_batch(
        self, phrases: List[str], context: str = ""
    ) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        need_embedding: List[str] = []

        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower in self.resolution_cache:
                results[phrase] = self.resolution_cache[phrase_lower]
                continue
            canonical = self.ontology.resolve_concept(phrase)
            if canonical:
                self.resolution_cache[phrase_lower] = canonical
                results[phrase] = canonical
                continue
            sub_match = self._substring_match(phrase_lower)
            if sub_match:
                self.resolution_cache[phrase_lower] = sub_match
                results[phrase] = sub_match
                continue
            need_embedding.append(phrase)

        if need_embedding and self.ontology_embedding_matrix.size > 0:
            query_texts = [
                p if not context else f"{p} in context of {context}"
                for p in need_embedding
            ]
            with torch.no_grad():
                query_embs = self.embed_model.encode(
                    query_texts,
                    show_progress_bar=False,
                    batch_size=64,
                    convert_to_numpy=True,
                )
            sims = cosine_similarity(query_embs, self.ontology_embedding_matrix)
            best_indices = np.argmax(sims, axis=1)
            best_scores = np.max(sims, axis=1)
            for idx, phrase in enumerate(need_embedding):
                if best_scores[idx] > self.similarity_threshold:
                    canonical = self.ontology_concepts_list[best_indices[idx]]
                    self.resolution_cache[phrase.lower().strip()] = canonical
                    results[phrase] = canonical
                else:
                    results[phrase] = None
            del query_embs, sims, best_indices, best_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            for phrase in need_embedding:
                results[phrase] = None

        self._trim_resolution_cache()
        return results

    def _substring_match(self, text: str) -> Optional[str]:
        for canonical, node in self.ontology.concepts.items():
            all_forms = {canonical.lower()} | node.synonyms
            for form in all_forms:
                if form in text or text in form:
                    if len(form) > 4 and len(text) > 4:
                        return canonical
        return None

    def _embedding_match(self, text: str, context: str = "") -> Optional[str]:
        try:
            query_text = (
                text if not context else f"{text} in context of {context}"
            )
            if query_text not in self.embedding_cache:
                with torch.no_grad():
                    self.embedding_cache[query_text] = self.embed_model.encode(
                        query_text,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
            query_emb = self.embedding_cache[query_text]
            sims = cosine_similarity(
                [query_emb], self.ontology_embedding_matrix
            )[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] > self.similarity_threshold:
                return self.ontology_concepts_list[best_idx]
            return None
        except Exception:
            return None
        finally:
            self._trim_embedding_cache()

    def _context_disambiguation(
        self, text: str, context: str
    ) -> Optional[str]:
        context_lower = context.lower()
        if 'stokes' in text:
            if 'full' in context_lower:
                return 'full_stokes'
            elif 'ssa' in context_lower:
                return 'ssa_flow'
            elif 'sia' in context_lower:
                return 'sia_flow'
        if 'friction' in text:
            if 'weertman' in context_lower:
                return 'weertman_friction'
            elif 'coulomb' in context_lower:
                return 'coulomb_friction'
            elif 'budd' in context_lower:
                return 'budd_friction'
        if 'calving' in text:
            return 'calving_3d'
        if 'enthalpy' in text:
            return 'enthalpy_solver'
        return None


# ============================================================================
# ENHANCED CONCEPT EXTRACTOR (Glacier domain)
# ============================================================================
class EnhancedConceptExtractor:
    def __init__(
        self,
        ontology: DomainOntology,
        resolver: AdvancedConceptResolver,
        store_contexts: bool = False,
        store_documents: bool = True,
    ) -> None:
        self.ontology = ontology
        self.resolver = resolver
        self.concept_frequencies: Dict[str, int] = defaultdict(int)
        self.store_contexts = store_contexts
        self.store_documents = store_documents
        self.concept_contexts: Dict[str, List[str]] = defaultdict(list)
        self.document_concepts: Dict[int, List[str]] = defaultdict(list)
        self._build_extraction_patterns()
        all_keywords = self._get_all_keywords()
        if all_keywords:
            sorted_keywords = sorted(all_keywords, key=len, reverse=True)[:500]
            pattern = r'\b(' + '|'.join(
                re.escape(k) for k in sorted_keywords
            ) + r')\b'
            self._keyword_regex = re.compile(pattern, re.IGNORECASE)
        else:
            self._keyword_regex = None

    def _build_extraction_patterns(self) -> None:
        self.glacier_patterns = [
            r'\bglens?\s*law\b', r'\barrhenius\s*factor\b', r'\bfull\s*stokes\b',
            r'\bssa\s*flow\b', r'\bsia\s*flow\b', r'\benthalpy\s*solver\b',
            r'\bgrounding\s*line\b', r'\bcontact\s*problem\b', r'\bsurface\s*mass\s*balance\b',
            r'\bbasal\s*melt\b', r'\b3d\s*calving\b', r'\bcrevasse\s*depth\b',
            r'\bweertman\s*friction\b', r'\bcoulomb\s*friction\b', r'\bglads\b',
            r'\bparstokes\b', r'\bgmsh\s*remeshing\b', r'\binverse\s*adjoint\b',
            r'\beffective\s*pressure\b', r'\bwater\s*pressure\b', r'\bgeothermal\s*flux\b',
            r'\bflowline\s*geometry\b', r'\bismip\s*hom\b', r'\bice\s*shelf\s*flow\b',
            r'\bfabric\s*evolution\b', r'\bdamage\s*mechanics\b',
        ]
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.glacier_patterns
        ]
        self.cause_effect_patterns = [
            r'\b(increase|decrease|enhance|reduce)\w*\s+(?:in|of)\s+([\w\s-]+?)\s+(?:lead[s]?|result[s]?|cause[s]?)\s+(?:to|in)?\s+([\w\s-]+?)\b',
        ]
        self.compiled_cause_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.cause_effect_patterns
        ]

    @timed
    def extract_from_text(self, text: str, doc_id: int = 0, allowed_concepts: Optional[Set[str]] = None) -> List[str]:
        concepts: Set[str] = set()
        text_lower = text.lower()

        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = (
                        match[0] if match[0]
                        else (match[1] if len(match) > 1 else match[0])
                    )
                concept = match.lower().strip()
                if len(concept) > 3:
                    canonical = self.resolver.resolve(concept, context=text[:200])
                    if canonical:
                        if allowed_concepts is not None and canonical not in allowed_concepts:
                            continue
                        concepts.add(canonical)
                    else:
                        if allowed_concepts is not None:
                            continue
                        concepts.add(concept)

        context_concepts = self._extract_from_context_windows(text)
        if allowed_concepts is not None:
            context_concepts = {c for c in context_concepts if c in allowed_concepts}
        concepts.update(context_concepts)

        raw_concepts = set()
        for c in concepts:
            if c not in self.ontology.concepts and not self.resolver.resolve(c):
                raw_concepts.add(c)
        if raw_concepts:
            raw_list = list(raw_concepts)[:50]
            resolved_map = self.resolver.resolve_batch(raw_list, context="")
            for raw, canonical in resolved_map.items():
                if canonical:
                    if allowed_concepts is not None and canonical not in allowed_concepts:
                        continue
                    concepts.add(canonical)
                else:
                    if allowed_concepts is not None:
                        continue
                    concepts.add(raw)

        for concept in concepts:
            self.concept_frequencies[concept] += 1
            if self.store_contexts:
                self.concept_contexts[concept].append(text[:200])
        if self.store_documents:
            self.document_concepts[doc_id] = list(concepts)
        return list(concepts)

    def _extract_from_context_windows(
        self, text: str, window_size: int = 100
    ) -> Set[str]:
        if not self._keyword_regex:
            return set()
        candidate_phrases: Set[str] = set()
        text_lower = text.lower()
        match_count = 0
        for match in self._keyword_regex.finditer(text_lower):
            if match_count > 20:
                break
            match_count += 1
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            local_context = text_lower[start:end]
            phrases = re.findall(
                r'\b([a-z]+(?:[-\s][a-z]+){1,3})\b', local_context
            )
            for phrase in phrases:
                if 5 <= len(phrase) <= 40:
                    canonical = self.resolver.resolve(phrase, context=local_context)
                    if canonical:
                        candidate_phrases.add(canonical)
        return candidate_phrases

    def _get_all_keywords(self) -> Set[str]:
        keywords: Set[str] = set()
        for canonical, node in self.ontology.concepts.items():
            keywords.add(canonical)
            keywords.update(node.synonyms)
        return keywords

    def extract_relationships(self, text: str) -> List[Relationship]:
        relationships: List[Relationship] = []
        for pattern in self.compiled_cause_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) >= 2:
                    source = (
                        match[0] if isinstance(match[0], str) else match[1]
                    )
                    target = (
                        match[-1] if isinstance(match[-1], str) else match[0]
                    )
                    source_canon = self.resolver.resolve(source, context=text[:200])
                    target_canon = self.resolver.resolve(target, context=text[:200])
                    if (
                        source_canon and target_canon
                        and source_canon != target_canon
                    ):
                        rel = Relationship(
                            source=source_canon,
                            target=target_canon,
                            rel_type=RelationshipType.CAUSES,
                            confidence=0.7,
                            evidence=text[:150],
                        )
                        relationships.append(rel)
        return relationships

    def get_concept_frequencies(self) -> Dict[str, int]:
        return dict(self.concept_frequencies)


# ============================================================================
# REASONING-ENHANCED GRAPH BUILDER
# ============================================================================
class ReasoningEnhancedGraphBuilder:
    def __init__(
        self, ontology: DomainOntology, extractor: EnhancedConceptExtractor
    ) -> None:
        self.ontology = ontology
        self.extractor = extractor
        self.reasoning_paths: List[List[str]] = []
        self.inferred_edges: Set[Tuple[str, str]] = set()

    @timed
    def build_graph(
        self,
        all_concepts: List[List[str]],
        valid_concepts: List[str],
        concept_to_id: Dict[str, int],
        embed_model=None,
        config: Dict = None,
    ) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(3000)
        nx_graph = nx.Graph()

        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            freq = self.extractor.concept_frequencies.get(c, 0)
            definition = self.ontology.get_definition(c)
            nx_graph.add_node(
                c,
                frequency=freq,
                concept_type=concept_type.value,
                definition=definition,
                degree=0,
            )

        cooccurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for concepts in all_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if u != v:
                        key = tuple(sorted([u, v]))
                        cooccurrence_map[key] += 1

        for (u, v), count in cooccurrence_map.items():
            nx_graph.add_edge(
                u, v,
                weight=count,
                cooccurrence=count,
                semantic=0,
                edge_type='cooccurrence',
                inferred=False,
            )

        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(nx_graph, valid_concepts, embed_model, config)

        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)
            self._add_cause_effect_edges(nx_graph)
            self._add_hierarchical_edges(nx_graph, valid_concepts)

        self._compute_final_weights(nx_graph, config)
        return nx_graph

    def _add_semantic_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str],
        embed_model, config: Dict,
    ) -> None:
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts,
                    show_progress_bar=False,
                    batch_size=64,
                    convert_to_numpy=True,
                )
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i + 1:], start=i + 1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh:
                        if nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3:
                            nx_graph.add_edge(
                                c1, c2,
                                weight=sim * 2,
                                cooccurrence=0,
                                semantic=sim,
                                edge_type='semantic',
                                inferred=False,
                            )
            del embeddings, sim_matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")

    def _add_inferred_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        for rel in self.ontology.relationships:
            if rel.source in valid_concepts and rel.target in valid_concepts:
                if not nx_graph.has_edge(rel.source, rel.target):
                    nx_graph.add_edge(
                        rel.source, rel.target,
                        weight=rel.confidence * 2,
                        cooccurrence=0,
                        semantic=rel.confidence,
                        edge_type=rel.rel_type.value,
                        inferred=True,
                        confidence=rel.confidence,
                    )
                    self.inferred_edges.add((rel.source, rel.target))
        self._infer_cross_domain_bridges(nx_graph, valid_concepts)

    def _infer_cross_domain_bridges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        method_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.METHOD
        ]
        property_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.PROPERTY
        ]
        for method in method_nodes:
            for prop in property_nodes:
                if not nx_graph.has_edge(method, prop):
                    paths = self.ontology.infer_path(method, prop, max_depth=2)
                    if paths:
                        avg_confidence = 0.6
                        nx_graph.add_edge(
                            method, prop,
                            weight=avg_confidence,
                            cooccurrence=0,
                            semantic=avg_confidence,
                            edge_type='bridge',
                            inferred=True,
                            path=" -> ".join(paths[0]),
                        )
                        self.inferred_edges.add((method, prop))
                        self.reasoning_paths.append(paths[0])

    def _add_cause_effect_edges(self, nx_graph: nx.Graph) -> None:
        pass

    def _add_hierarchical_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        for concept in valid_concepts:
            if concept not in self.ontology.concepts:
                continue
            node = self.ontology.concepts[concept]
            for hypernym in node.hypernyms:
                if (
                    hypernym in valid_concepts
                    and not nx_graph.has_edge(concept, hypernym)
                ):
                    nx_graph.add_edge(
                        concept, hypernym,
                        weight=1.0, cooccurrence=0, semantic=0.95,
                        edge_type='hypernym', inferred=True,
                    )
            for hyponym in node.hyponyms:
                if (
                    hyponym in valid_concepts
                    and not nx_graph.has_edge(concept, hyponym)
                ):
                    nx_graph.add_edge(
                        concept, hyponym,
                        weight=1.0, cooccurrence=0, semantic=0.95,
                        edge_type='hyponym', inferred=True,
                    )

    def _compute_final_weights(
        self, nx_graph: nx.Graph, config: Dict
    ) -> None:
        cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.7)
        sem_weight = config.get("SEMANTIC_WEIGHT", 0.2)
        inf_weight = config.get("INFERENCE_WEIGHT", 0.1)
        for u, v, data in nx_graph.edges(data=True):
            cooc = data.get('cooccurrence', 0)
            sem = data.get('semantic', 0)
            inf = 1.0 if data.get('inferred', False) else 0
            conf = data.get('confidence', 0.5)
            data['weight'] = (
                cooc_weight * cooc
                + sem_weight * sem
                + inf_weight * inf * conf
            )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    if num_abstracts <= 50:
        return {
            "MIN_CONCEPT_FREQ": 2, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 1, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.72, "COOCCURRENCE_WEIGHT": 0.5,
            "SEMANTIC_WEIGHT": 0.5, "CLUSTER_SIMILARITY": 0.75,
            "TOP_N_CONCEPTS": 200, "MAX_CONCEPT_LENGTH": 6,
            "INFERENCE_WEIGHT": 0.1,
        }
    elif num_abstracts <= 500:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.78, "COOCCURRENCE_WEIGHT": 0.6,
            "SEMANTIC_WEIGHT": 0.3, "CLUSTER_SIMILARITY": 0.72,
            "TOP_N_CONCEPTS": 500, "MAX_CONCEPT_LENGTH": 8,
            "INFERENCE_WEIGHT": 0.1,
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 5, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 3, "USE_SEMANTIC_CLUSTERING": False,
            "SIMILARITY_THRESHOLD": 0.85, "COOCCURRENCE_WEIGHT": 0.7,
            "SEMANTIC_WEIGHT": 0.2, "CLUSTER_SIMILARITY": 0.68,
            "TOP_N_CONCEPTS": 1000, "MAX_CONCEPT_LENGTH": 10,
            "INFERENCE_WEIGHT": 0.1,
        }


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )


# ============================================================================
# CONCEPT DISTILLATION (Memory-safe)
# ============================================================================
def compute_concept_distillation(
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    all_texts: Union[List[str], Dict[int, str]],
    max_docs_per_concept: int = 30,
) -> pd.DataFrame:
    distill_data: List[Dict[str, Any]] = []
    doc_corpus: List[str] = []

    texts_is_dict = isinstance(all_texts, dict)
    n_texts = len(all_texts)

    for c in valid_concepts:
        doc_indices = concept_abstract_map.get(c, [])
        if max_docs_per_concept and len(doc_indices) > max_docs_per_concept:
            doc_indices = doc_indices[:max_docs_per_concept]
        if texts_is_dict:
            doc_text = " ".join([
                all_texts[i] for i in doc_indices
                if i in all_texts
            ])
        else:
            doc_text = " ".join([
                all_texts[i] for i in doc_indices
                if isinstance(i, int) and 0 <= i < n_texts
            ])
        doc_corpus.append(doc_text)

    tfidf = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        stop_words='english', max_features=2000,
    )
    try:
        if any(doc_corpus) and any(t.strip() for t in doc_corpus):
            tfidf_matrix = tfidf.fit_transform(doc_corpus)
            tfidf_scores = tfidf_matrix.max(axis=1).A1
            del tfidf_matrix
        else:
            tfidf_scores = np.ones(len(valid_concepts))
    except Exception:
        tfidf_scores = np.ones(len(valid_concepts))
    gc.collect()

    embed_model = load_embedding_model()

    for i, c in enumerate(valid_concepts):
        freq = len(concept_abstract_map.get(c, []))
        semantic_density = float(tfidf_scores[i])
        coherence = 0.0
        if freq > 1 and doc_corpus[i].strip():
            try:
                words = doc_corpus[i].split()[:20]
                with torch.no_grad():
                    concept_embeddings = embed_model.encode(
                        words, show_progress_bar=False,
                        batch_size=16, convert_to_numpy=True,
                    )
                if len(concept_embeddings) > 1:
                    sim_matrix = cosine_similarity(concept_embeddings)
                    coherence = float(np.mean(
                        sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                    ))
                    del sim_matrix
                del concept_embeddings, words
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                coherence = 0.0
        distill_data.append({
            "concept": c,
            "frequency": freq,
            "tfidf_weight": semantic_density,
            "semantic_density": semantic_density,
            "coherence_score": float(coherence),
            "distillation_efficiency": float(
                semantic_density * np.log1p(freq) * (0.5 + 0.5 * coherence)
            ),
        })

    del doc_corpus
    gc.collect()
    return pd.DataFrame(distill_data).sort_values(
        "distillation_efficiency", ascending=False
    )


# ============================================================================
# LEGACY GRAPH CONSTRUCTION (FALLBACK)
# ============================================================================
def build_hybrid_graph(
    all_concepts: List[List[str]],
    valid_concepts: List[str],
    concept_to_id: Dict[str, int],
    embed_model=None,
    config: Dict = None,
    ontology: DomainOntology = None,
) -> nx.Graph:
    if config is None:
        config = get_adaptive_config(3000)
    nx_graph = nx.Graph()
    for c in valid_concepts:
        concept_type = ontology.get_concept_type(c).value if ontology else 'general'
        definition = ontology.get_definition(c) if ontology else ''
        nx_graph.add_node(
            c, frequency=0, concept_type=concept_type, definition=definition,
        )
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(
                        u, v, weight=1, cooccurrence=1, semantic=0,
                        edge_type='cooccurrence',
                    )
                nx_graph.nodes[u]['frequency'] = (
                    nx_graph.nodes[u].get('frequency', 0) + 1
                )
                nx_graph.nodes[v]['frequency'] = (
                    nx_graph.nodes[v].get('frequency', 0) + 1
                )
    if embed_model and len(valid_concepts) >= 10:
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts, show_progress_bar=False,
                    batch_size=64, convert_to_numpy=True,
                )
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i + 1:], start=i + 1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh and (
                        nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3
                    ):
                        nx_graph.add_edge(
                            c1, c2, weight=sim * 2, cooccurrence=0,
                            semantic=sim, edge_type='semantic',
                        )
            del embeddings, sim_matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.9)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.1)
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    return nx_graph


def sample_edges_for_training(
    nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_to_id: Dict[str, int],
    config: Dict = None,
    memory_safe: bool = False,
) -> Tuple[List[Tuple], List[Tuple]]:
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs: List[Tuple[int, int]] = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    if memory_safe:
        target_negs = min(len(pos_pairs) * 2 if pos_pairs else 30, 2000)
    else:
        target_negs = min(len(pos_pairs) * 3 if pos_pairs else 30, 5000)
    attempts = 0
    max_attempts = 50000
    if memory_safe:
        path_lengths = {}
    else:
        try:
            path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=3))
        except Exception:
            path_lengths = {}
    while len(neg_pairs) < target_negs and attempts < max_attempts:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            attempts += 1
            continue
        dist = path_lengths.get(u_c, {}).get(v_c, 999)
        if dist == 2 or dist == 3:
            neg_pairs.append((int(u_idx), int(v_idx)))
        elif dist == 999 and np.random.rand() < 0.1:
            neg_pairs.append((int(u_idx), int(v_idx)))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append((int(u_idx), int(v_idx)))
    return pos_pairs, neg_pairs


# ============================================================================
# GNN MODEL
# ============================================================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, adj_indices, adj_values, num_nodes, h,
        pos_u, pos_v, neg_u, neg_v,
    ):
        A = sparse.FloatTensor(
            adj_indices, adj_values, torch.Size([num_nodes, num_nodes])
        ).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(
            self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1))
        )
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(
            torch.cat([h2[pos_u], h2[pos_v]], dim=1)
        ).squeeze(1)
        neg_scores = self.decoder(
            torch.cat([h2[neg_u], h2[neg_v]], dim=1)
        ).squeeze(1)
        return pos_scores, neg_scores, h2


# ============================================================================
# MICROTRANSFORMER #2: LatentMoE KG‑RAG EXTRACTOR
# ============================================================================
class LatentMoEKGExtractor(nn.Module):
    def __init__(self, num_nodes, num_edge_types, d_model=96, latent_dim=24,
                 n_experts=32, top_k=4, num_heads=4, num_layers=2):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        self.edge_embedding = nn.Embedding(num_edge_types, d_model)

        self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.up_proj = nn.Linear(latent_dim, d_model, bias=False)

        self.router = nn.Linear(latent_dim, n_experts)
        self.experts = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(n_experts)])
        self.top_k = top_k
        self.n_experts = n_experts

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True,
            dim_feedforward=d_model * 2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, node_seq, edge_seq, router_prior=None):
        node_emb = self.node_embedding(node_seq)
        if edge_seq.size(1) > 0:
            edge_emb = self.edge_embedding(edge_seq)
            node_emb[:, 1:, :] = node_emb[:, 1:, :] + edge_emb

        batch_size, seq_len, d_model = node_emb.shape
        latent_repr = self.down_proj(node_emb)

        flat_latent = latent_repr.view(batch_size * seq_len, -1)
        router_logits = self.router(flat_latent)
        if router_prior is not None:
            router_logits = router_logits + router_prior.to(router_logits.device)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        expert_outputs = torch.stack([self.experts[i](flat_latent) for i in range(self.n_experts)], dim=0)
        token_indices = torch.arange(batch_size * seq_len, device=flat_latent.device).unsqueeze(1).expand(-1, self.top_k)
        selected = expert_outputs[topk_indices, token_indices, :]
        weighted = topk_weights.unsqueeze(-1) * selected
        moe_output_flat = weighted.sum(dim=1)

        moe_output = moe_output_flat.view(batch_size, seq_len, -1)
        node_emb = node_emb + self.up_proj(moe_output)
        out = self.transformer(node_emb)
        out = self.output_proj(out)
        return out, routing_weights.view(batch_size, seq_len, -1)


def train_gnn(
    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs,
    progress_callback=None, epochs: int = 50, lr: float = 1e-3,
):
    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384
    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2:
            pos_pairs = [(nodes[0], nodes[1])]
        else:
            raise ValueError("Cannot train GNN with fewer than 2 concepts")
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    target_device = (
        node_features.device if node_features.numel() > 0
        else torch.device('cpu')
    )
    pos_u = torch.tensor(
        [p[0] for p in pos_pairs], dtype=torch.long, device=target_device
    )
    pos_v = torch.tensor(
        [p[1] for p in pos_pairs], dtype=torch.long, device=target_device
    )
    neg_u = (
        torch.tensor(
            [n[0] for n in neg_pairs], dtype=torch.long, device=target_device
        )
        if neg_pairs
        else torch.tensor([], dtype=torch.long, device=target_device)
    )
    neg_v = (
        torch.tensor(
            [n[1] for n in neg_pairs], dtype=torch.long, device=target_device
        )
        if neg_pairs
        else torch.tensor([], dtype=torch.long, device=target_device)
    )
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=128).to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if len(neg_pairs) == 0:
            pos_out, _, _ = model(
                adj_indices, adj_values, num_nodes, node_features,
                pos_u, pos_v, pos_u[:1], pos_v[:1],
            )
            loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
        else:
            pos_out, neg_out, _ = model(
                adj_indices, adj_values, num_nodes, node_features,
                pos_u, pos_v, neg_u, neg_v,
            )
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            loss = 0.5 * (pos_loss + neg_loss)
        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(
            adj_indices, adj_values, num_nodes, node_features,
            pos_u[:1], pos_v[:1],
            neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1],
            neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1],
        )
    return model, final_embeddings.cpu(), adj_indices.cpu(), adj_values.cpu()


# ============================================================================
# RESEARCH DIRECTION SCORING
# ============================================================================
def compute_research_direction_scores(
    model, node_features, final_emb, nx_graph,
    valid_concepts, concept_properties, ridge,
    embed_model, n_samples: int = 5000,
) -> pd.DataFrame:
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(
        n_concepts, size=min(n_samples, n_concepts * 5)
    )
    v_ids = np.random.randint(
        n_concepts, size=min(n_samples, n_concepts * 5)
    )
    candidate_pairs: List[Tuple[int, int, str, str]] = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx:
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            continue
        candidate_pairs.append((int(u_idx), int(v_idx), u_c, v_c))
    if not candidate_pairs:
        return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pair_features = torch.cat(
            [final_emb[u_tensor], final_emb[v_tensor]], dim=1
        )
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).numpy()
    with torch.no_grad():
        emb_np = embed_model.encode(
            valid_concepts, show_progress_bar=False,
            batch_size=64, convert_to_numpy=True,
        )
    cos_sims = np.sum(
        emb_np[u_tensor.numpy()] * emb_np[v_tensor.numpy()], axis=1
    )
    results: List[Dict[str, Any]] = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(
                    ridge.predict([[p_u, p_v, 1.0]])[0]
                )
            except Exception:
                expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = (
            np.exp(-0.5 * semantic_novelty)
            * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        )
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = (
            np.clip((expected_improvement - 50) / 200, 0, 1)
            if expected_improvement > 0 else 0
        )
        D_uv = (
            alpha['gnn'] * gnn_scores[i]
            + alpha['novelty'] * semantic_novelty
            + alpha['gain'] * norm_gain
            + alpha['feas'] * (1.0 - feasibility)
        )
        results.append({
            'concept_u': u_c, 'concept_v': v_c,
            'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty),
            'expected_property_gain': expected_improvement,
            'feasibility_score': float(feasibility),
            'composite_score': float(D_uv),
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    del emb_np
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df.head(min(100, len(df)))


# ============================================================================
# MATHEMATICAL VALIDATION
# ============================================================================
def validate_graph_metrics(
    nx_graph: nx.Graph, valid_concepts: List[str]
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if nx_graph.number_of_nodes() < 3:
        return metrics
    try:
        from networkx.algorithms import community
        partition = list(community.greedy_modularity_communities(nx_graph))
        metrics["modularity"] = community.modularity(nx_graph, partition)
        metrics["n_communities"] = len(partition)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0
    try:
        embed_model = load_embedding_model()
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts, show_progress_bar=False,
                batch_size=64, convert_to_numpy=True,
            )
        if len(valid_concepts) >= 3:
            labels = np.zeros(len(valid_concepts))
            for i, c in enumerate(valid_concepts):
                for idx, comm in enumerate(
                    partition if 'partition' in locals() else [[]]
                ):
                    if c in comm:
                        labels[i] = idx
                        break
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        else:
            metrics["silhouette_score"] = 0.0
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        metrics["silhouette_score"] = 0.0
    weights = [d.get('weight', 1) for _, _, d in nx_graph.edges(data=True)]
    if len(weights) > 10:
        p_values = []
        for w in weights[:50]:
            permuted = np.random.permutation(weights)
            p_values.append(np.sum(permuted >= w) / len(weights))
        metrics["edge_significance_p_mean"] = float(np.mean(p_values))
        metrics["edge_significant_count"] = int(
            sum(1 for p in p_values if p < 0.05)
        )
    else:
        metrics["edge_significance_p_mean"] = 1.0
        metrics["edge_significant_count"] = 0
    try:
        metrics["avg_betweenness"] = np.mean(
            list(nx.betweenness_centrality(nx_graph).values())
        )
        metrics["avg_closeness"] = np.mean(
            list(nx.closeness_centrality(nx_graph).values())
        )
    except Exception:
        pass
    return metrics


@st.cache_data(ttl=3600)
def compute_bootstrap_ci(
    scores: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05
) -> Tuple[float, float, float]:
    if len(scores) < 2:
        return float(np.mean(scores)), 0.0, 0.0
    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(float(np.mean(sample)))
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), ci_low, ci_high


# ============================================================================
# ADVANCED ANALYTICS (CACHED)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def detect_keyword_bursts(
    df_filtered: pd.DataFrame,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    text_columns: List[str],
    burst_threshold: float = 2.0,
) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame()
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 3:
        return pd.DataFrame()
    year_range = sorted(years.unique())
    burst_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 5:
            continue
        concept_years: List[int] = []
        for idx in doc_indices:
            if (
                idx < len(df_filtered)
                and pd.notna(df_filtered.iloc[idx].get("Year"))
            ):
                concept_years.append(int(df_filtered.iloc[idx]["Year"]))
        if len(concept_years) < 3:
            continue
        year_counts = Counter(concept_years)
        counts = [year_counts.get(y, 0) for y in year_range]
        if len(counts) < 3:
            continue
        window = max(2, len(counts) // 5)
        moving_avg = pd.Series(counts).rolling(
            window=window, min_periods=1
        ).mean()
        burst_scores: List[float] = []
        for i in range(window, len(counts)):
            if moving_avg.iloc[i - 1] > 0:
                ratio = counts[i] / max(moving_avg.iloc[i - 1], 0.1)
                burst_scores.append(float(ratio))
        if burst_scores:
            max_burst = max(burst_scores)
            burst_year = year_range[window + burst_scores.index(max_burst)]
            if max_burst >= burst_threshold:
                burst_data.append({
                    "concept": concept,
                    "burst_score": round(max_burst, 2),
                    "burst_year": burst_year,
                    "total_mentions": len(concept_years),
                    "year_range": f"{min(concept_years)}-{max(concept_years)}",
                })
    return pd.DataFrame(burst_data).sort_values(
        "burst_score", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def detect_semantic_drift(
    df_filtered: pd.DataFrame,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    text_columns: List[str],
    early_fraction: float = 0.3,
    late_fraction: float = 0.3,
) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame()
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 4:
        return pd.DataFrame()
    embed_model = load_embedding_model()
    sorted_years = sorted(years.unique())
    n_years = len(sorted_years)
    early_cutoff = sorted_years[int(n_years * early_fraction)]
    late_cutoff = sorted_years[int(n_years * (1 - late_fraction))]
    drift_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 10:
            continue
        early_texts: List[str] = []
        late_texts: List[str] = []
        for idx in doc_indices:
            if idx >= len(df_filtered):
                continue
            row = df_filtered.iloc[idx]
            year = row.get("Year")
            if pd.isna(year):
                continue
            year = int(year)
            text = " ".join([
                str(row.get(col, ""))
                for col in text_columns if pd.notna(row.get(col))
            ])
            if year <= early_cutoff:
                early_texts.append(text)
            elif year >= late_cutoff:
                late_texts.append(text)
        if len(early_texts) < 3 or len(late_texts) < 3:
            continue
        try:
            with torch.no_grad():
                early_emb = embed_model.encode(
                    early_texts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
                late_emb = embed_model.encode(
                    late_texts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
            early_centroid = np.mean(early_emb, axis=0)
            late_centroid = np.mean(late_emb, axis=0)
            drift = 1.0 - cosine_similarity(
                [early_centroid], [late_centroid]
            )[0][0]
            drift_data.append({
                "concept": concept,
                "semantic_drift": round(float(drift), 4),
                "early_papers": len(early_texts),
                "late_papers": len(late_texts),
                "early_period": f"{sorted_years[0]}-{early_cutoff}",
                "late_period": f"{late_cutoff}-{sorted_years[-1]}",
            })
            del early_emb, late_emb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            continue
    return pd.DataFrame(drift_data).sort_values(
        "semantic_drift", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def build_concept_genealogy(
    _nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    if _nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    try:
        pagerank = nx.pagerank(_nx_graph, weight='weight')
    except Exception:
        pagerank = {n: 1.0 for n in _nx_graph.nodes()}
    try:
        betweenness = nx.betweenness_centrality(_nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in _nx_graph.nodes()}
    genealogy_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in _nx_graph:
            continue
        pr = pagerank.get(concept, 0)
        bc = betweenness.get(concept, 0)
        freq = len(concept_abstract_map.get(concept, []))
        degree = _nx_graph.degree(concept)
        if (
            pr > np.percentile(list(pagerank.values()), 75)
            and degree > np.percentile(
                [_nx_graph.degree(n) for n in _nx_graph.nodes()], 75
            )
        ):
            generation = "Foundational (Parent)"
        elif (
            pr < np.percentile(list(pagerank.values()), 25)
            and degree < np.percentile(
                [_nx_graph.degree(n) for n in _nx_graph.nodes()], 25
            )
        ):
            generation = "Emerging (Child)"
        else:
            generation = "Intermediate"
        genealogy_data.append({
            "concept": concept,
            "pagerank": round(pr, 5),
            "betweenness": round(bc, 5),
            "frequency": freq,
            "degree": degree,
            "generation": generation,
        })
    return pd.DataFrame(genealogy_data).sort_values(
        "pagerank", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def detect_cross_domain_bridges(
    _nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    _BRIDGE_COLUMNS = [
        "concept", "bridge_score", "betweenness",
        "connected_categories", "categories",
        "degree", "own_category",
    ]
    if _nx_graph.number_of_nodes() < 5:
        return pd.DataFrame(columns=_BRIDGE_COLUMNS)
    category_map = {c: categorize_glacier_concept(c) for c in valid_concepts}
    try:
        betweenness = nx.betweenness_centrality(_nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in _nx_graph.nodes()}
    bridge_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in _nx_graph:
            continue
        neighbors = list(_nx_graph.neighbors(concept))
        if len(neighbors) < 2:
            continue
        own_cat = category_map.get(concept, 'general')
        neighbor_cats = [category_map.get(n, 'general') for n in neighbors]
        unique_cats = set(neighbor_cats)
        if len(unique_cats) < 2:
            continue
        bridge_score = betweenness.get(concept, 0) * len(unique_cats)
        bridge_data.append({
            "concept": concept,
            "bridge_score": round(bridge_score, 4),
            "betweenness": round(betweenness.get(concept, 0), 4),
            "connected_categories": len(unique_cats),
            "categories": ", ".join(sorted(unique_cats)),
            "degree": len(neighbors),
            "own_category": own_cat,
        })
    if not bridge_data:
        return pd.DataFrame(columns=_BRIDGE_COLUMNS)
    df = pd.DataFrame(bridge_data)
    if "bridge_score" not in df.columns:
        df["bridge_score"] = 0.0
    return df.sort_values("bridge_score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_network_motifs(_nx_graph: nx.Graph) -> Dict[str, Any]:
    if _nx_graph.number_of_nodes() < 3:
        return {}
    motifs: Dict[str, Any] = {}
    try:
        triangles = nx.triangles(_nx_graph)
        motifs["total_triangles"] = sum(triangles.values()) // 3
        motifs["avg_triangles_per_node"] = round(
            np.mean(list(triangles.values())), 2
        )
        motifs["nodes_in_triangles"] = sum(
            1 for v in triangles.values() if v > 0
        )
    except Exception:
        motifs["total_triangles"] = 0
    try:
        cliques = list(nx.find_cliques(_nx_graph))
        clique_sizes = [len(c) for c in cliques]
        motifs["total_cliques"] = len(cliques)
        motifs["max_clique_size"] = max(clique_sizes) if clique_sizes else 0
        motifs["avg_clique_size"] = (
            round(np.mean(clique_sizes), 2) if clique_sizes else 0
        )
        motifs["4cliques"] = sum(1 for c in clique_sizes if c >= 4)
    except Exception:
        motifs["total_cliques"] = 0
    try:
        clustering = nx.clustering(_nx_graph)
        stars: List[Tuple[str, int, float]] = []
        for node in _nx_graph.nodes():
            deg = _nx_graph.degree(node)
            clust = clustering.get(node, 0)
            if deg >= 5 and clust < 0.2:
                stars.append((node, deg, clust))
        stars.sort(key=lambda x: x[1], reverse=True)
        motifs["star_motifs"] = len(stars)
        motifs["top_stars"] = stars[:10]
    except Exception:
        motifs["star_motifs"] = 0
    return motifs


# ============================================================================
# CENTRALITY & DEGREE DISTRIBUTION
# ============================================================================
def compute_centrality_comparison(
    nx_graph: nx.Graph, valid_concepts: List[str]
) -> pd.DataFrame:
    if nx_graph.number_of_nodes() < 3:
        return pd.DataFrame()
    centrality_data: List[Dict[str, Any]] = []
    try:
        degree_c = dict(nx_graph.degree())
        betweenness_c = nx.betweenness_centrality(nx_graph, weight='weight')
        closeness_c = nx.closeness_centrality(nx_graph)
        eigenvector_c = nx.eigenvector_centrality(
            nx_graph, weight='weight', max_iter=1000
        )
        pagerank_c = nx.pagerank(nx_graph, weight='weight')
        for concept in valid_concepts:
            if concept not in nx_graph:
                continue
            centrality_data.append({
                "concept": concept,
                "degree": degree_c.get(concept, 0),
                "betweenness": round(betweenness_c.get(concept, 0), 5),
                "closeness": round(closeness_c.get(concept, 0), 5),
                "eigenvector": round(eigenvector_c.get(concept, 0), 5),
                "pagerank": round(pagerank_c.get(concept, 0), 5),
            })
    except Exception as e:
        st.warning(f"Centrality computation error: {e}")
    return pd.DataFrame(centrality_data)


def plot_degree_distribution(
    nx_graph: nx.Graph, theme: Dict = None
) -> go.Figure:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    degrees = [d for n, d in nx_graph.degree()]
    if len(degrees) < 3:
        return go.Figure()
    degree_counts = Counter(degrees)
    x = sorted(degree_counts.keys())
    y = [degree_counts[k] for k in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='Degree Distribution',
        marker=dict(size=10, color=theme.get('highlight_bg', '#ff6b6b')),
    ))
    fig.update_layout(
        title="Degree Distribution (Log-Log)",
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Degree (k)", yaxis_title="Frequency P(k)",
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    return fig


# ============================================================================
# PUBLICATION-READY EXPORTS
# ============================================================================
def export_publication_figure(
    nx_graph, valid_concepts, concept_abstract_map,
    cmap_name="viridis", dpi=300, figsize=(14, 12),
    filename="glacier_graph_pub.png",
) -> bytes:
    try:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5, iterations=200)
        plt.figure(figsize=figsize, dpi=dpi)
        node_colors = [get_glacier_category_color(n) for n in nx_graph.nodes()]
        node_sizes = [
            max(100, min(800, len(concept_abstract_map.get(n, [])) * 20 + 50))
            for n in nx_graph.nodes()
        ]
        nx.draw(
            nx_graph, pos,
            with_labels=True,
            node_color=node_colors,
            edge_color='lightgray',
            node_size=node_sizes,
            font_size=6,
            font_weight='bold',
            edgecolors='white',
            linewidths=1.5,
            width=0.5,
            alpha=0.9,
        )
        plt.title(
            "Glacial Flow Mechanics (Elmer/Ice) Concept Graph",
            fontsize=14, fontweight='bold', pad=20,
        )
        buf = io.BytesIO()
        plt.savefig(
            buf, format='png', dpi=dpi, bbox_inches='tight',
            facecolor='white', edgecolor='none',
        )
        buf.seek(0)
        plt.close()
        return buf.read()
    except Exception as e:
        st.error(f"Publication figure export failed: {e}")
        return b''


def generate_analysis_report(
    nx_graph, valid_concepts, concept_abstract_map,
    top_scores, distill_df, burst_df, drift_df,
    genealogy_df, bridge_df, motifs, val_metrics,
    df_filtered,
) -> str:
    report: List[str] = []
    report.append("# Glacial Flow Mechanics (Elmer/Ice) Concept Graph Analysis Report")
    report.append(
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
    )
    report.append("## 1. Dataset Overview")
    report.append(f"- **Total Records**: {len(df_filtered)}")
    if 'Year' in df_filtered.columns:
        years = df_filtered['Year'].dropna()
        report.append(
            f"- **Year Range**: {int(years.min())} - {int(years.max())}"
        )
    report.append(f"- **Total Concepts**: {len(valid_concepts)}")
    report.append(f"- **Total Edges**: {nx_graph.number_of_edges()}")
    report.append(f"- **Graph Density**: {nx.density(nx_graph):.4f}")
    report.append("")
    report.append("## 2. Top Concepts by Frequency")
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:20]
    for i, c in enumerate(top_concepts, 1):
        freq = len(concept_abstract_map.get(c, []))
        deg = nx_graph.degree(c)
        report.append(f"{i}. **{c}** - Freq: {freq}, Degree: {deg}")
    report.append("")
    report.append("## 3. Concept Distillation Efficiency (Top 15)")
    if not distill_df.empty:
        for _, row in distill_df.head(15).iterrows():
            report.append(
                f"- **{row['concept']}**: Efficiency="
                f"{row['distillation_efficiency']:.3f}, "
                f"Freq={row['frequency']}, "
                f"Coherence={row['coherence_score']:.3f}"
            )
    report.append("")
    report.append("## 4. Research Direction Recommendations (Top 10)")
    if not top_scores.empty:
        for i, (_, row) in enumerate(top_scores.head(10).iterrows(), 1):
            report.append(
                f"{i}. **{row['concept_u']}** + **{row['concept_v']}** - "
                f"Composite Score: {row['composite_score']:.3f}"
            )
    report.append("")
    report.append("## 5. Keyword Burst Detection")
    if not burst_df.empty:
        for _, row in burst_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Burst Score={row['burst_score']:.2f} "
                f"(Year {row['burst_year']})"
            )
    else:
        report.append("No significant keyword bursts detected.")
    report.append("")
    report.append("## 6. Semantic Drift Detection")
    if not drift_df.empty:
        for _, row in drift_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Drift={row['semantic_drift']:.4f} "
                f"({row['early_period']} -> {row['late_period']})"
            )
    else:
        report.append("No significant semantic drift detected.")
    report.append("")
    report.append("## 7. Cross-Domain Bridge Concepts")
    if not bridge_df.empty:
        for _, row in bridge_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Bridge Score={row['bridge_score']:.4f}, "
                f"Connects {row['connected_categories']} categories"
            )
    else:
        report.append("No cross-domain bridges detected.")
    report.append("")
    report.append("## 8. Network Motif Analysis")
    report.append(f"- Total Triangles: {motifs.get('total_triangles', 0)}")
    report.append(f"- Total Cliques: {motifs.get('total_cliques', 0)}")
    report.append(f"- Max Clique Size: {motifs.get('max_clique_size', 0)}")
    report.append(f"- Star Motifs: {motifs.get('star_motifs', 0)}")
    report.append("")
    report.append("## 9. Graph Validation Metrics")
    report.append(f"- Modularity: {val_metrics.get('modularity', 0):.3f}")
    report.append(
        f"- Silhouette Score: {val_metrics.get('silhouette_score', 0):.3f}"
    )
    report.append(f"- Number of Communities: {val_metrics.get('n_communities', 0)}")
    report.append(f"- Avg Betweenness: {val_metrics.get('avg_betweenness', 0):.3f}")
    report.append("")
    report.append("---")
    report.append("*Report generated by Glacial Flow Mechanics Concept Graph v6.2*")
    return "\n".join(report)


# ============================================================================
# GRAPH EDIT HISTORY
# ============================================================================
class GraphEditHistory:
    def __init__(self, max_history: int = 20) -> None:
        self.history: deque = deque(maxlen=max_history)
        self.redo_stack: deque = deque(maxlen=max_history)
        self._snapshot_counter = 0

    def save_snapshot(
        self, nx_graph, valid_concepts, concept_to_id,
        id_to_concept, concept_abstract_map,
    ) -> int:
        snapshot = {
            'id': self._snapshot_counter,
            'nx_graph': copy.copy(nx_graph),
            'valid_concepts': list(valid_concepts),
            'concept_to_id': dict(concept_to_id),
            'id_to_concept': dict(id_to_concept),
            'concept_abstract_map': {
                k: list(v) for k, v in concept_abstract_map.items()
            },
            'timestamp': datetime.now().isoformat(),
        }
        self.history.append(snapshot)
        self._snapshot_counter += 1
        self.redo_stack.clear()
        return snapshot['id']

    def undo(self) -> Optional[Dict]:
        if len(self.history) < 2:
            return None
        current = self.history.pop()
        self.redo_stack.append(current)
        previous = self.history[-1]
        return previous

    def redo(self) -> Optional[Dict]:
        if not self.redo_stack:
            return None
        snapshot = self.redo_stack.pop()
        self.history.append(snapshot)
        return snapshot

    def can_undo(self) -> bool:
        return len(self.history) >= 2

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def get_history_summary(self) -> List[str]:
        return [
            f"Snapshot {s['id']} @ {s['timestamp']}" for s in self.history
        ]


# ============================================================================
# CENTRALITY & METRICS DASHBOARD
# ============================================================================
def compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {}
    metrics: Dict[str, Any] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "clustering": (
            nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
        ),
        "connected_components": nx.number_connected_components(G),
        "avg_clustering": (
            nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
        ),
    }
    try:
        bc = nx.betweenness_centrality(
            G, normalized=True, k=min(100, G.number_of_nodes())
        )
        top_bridges = sorted(
            bc.items(), key=lambda x: x[1], reverse=True
        )[:10]
        metrics["top_bridges"] = top_bridges
        metrics["avg_betweenness"] = np.mean(list(bc.values()))
    except Exception:
        metrics["top_bridges"] = []
    return metrics

def display_metric_dashboard(metrics: Dict, theme=None) -> None:
    if not metrics:
        st.warning("No graph metrics available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"])
    col2.metric("Edges", metrics["edges"])
    col3.metric("Density", f"{metrics['density']:.3f}")
    col4.metric("Avg Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6, col7 = st.columns(3)
    col5.metric("Clustering", f"{metrics['clustering']:.3f}")
    col6.metric("Components", metrics["connected_components"])
    col7.metric(
        "Avg Betweenness", f"{metrics.get('avg_betweenness', 0):.3f}"
    )
    if metrics.get("top_bridges"):
        st.markdown("**Top Bridge Concepts (High Betweenness)**")
        bridge_df = pd.DataFrame(
            metrics["top_bridges"], columns=["Concept", "Bridge Score"]
        )
        st.dataframe(bridge_df, use_container_width=True)

# ============================================================================
# THEME CONFIGURATION
# ============================================================================
THEME_PRESETS = {
    "Bright (Default)": {
        "bg": "#ffffff", "font": "#1e293b",
        "tooltip_bg": "rgba(255,255,255,0.95)",
        "tooltip_border": "#cbd5e1", "tooltip_text": "#1e293b",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.45)",
        "edge_semantic": "rgba(251, 146, 60, 0.40)",
        "edge_bridge": "rgba(250, 204, 21, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(34, 197, 94, 0.45)",
        "edge_unknown": "rgba(148, 163, 184, 0.30)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b",
        "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.15)",
        "plotly_bg": "#ffffff", "plotly_paper": "#ffffff",
        "grid_color": "#e2e8f0", "axis_color": "#64748b",
    },
    "Dark": {
        "bg": "#0f172a", "font": "#e2e8f0",
        "tooltip_bg": "rgba(15, 23, 42, 0.95)",
        "tooltip_border": "#334155", "tooltip_text": "#e2e8f0",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.55)",
        "edge_semantic": "rgba(251, 146, 60, 0.50)",
        "edge_bridge": "rgba(250, 204, 21, 0.65)",
        "edge_inferred": "rgba(139, 92, 246, 0.60)",
        "edge_cause": "rgba(239, 68, 68, 0.65)",
        "edge_hypernym": "rgba(34, 197, 94, 0.55)",
        "edge_unknown": "rgba(148, 163, 184, 0.40)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b",
        "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.6)",
        "plotly_bg": "#0f172a", "plotly_paper": "#0f172a",
        "grid_color": "#1e293b", "axis_color": "#94a3b8",
    },
    "Midnight": {
        "bg": "#020617", "font": "#f1f5f9",
        "tooltip_bg": "rgba(2, 6, 23, 0.97)",
        "tooltip_border": "#1e293b", "tooltip_text": "#f1f5f9",
        "edge_cooccurrence": "rgba(99, 102, 241, 0.55)",
        "edge_semantic": "rgba(236, 72, 153, 0.50)",
        "edge_bridge": "rgba(34, 211, 238, 0.65)",
        "edge_inferred": "rgba(168, 85, 247, 0.60)",
        "edge_cause": "rgba(244, 63, 94, 0.65)",
        "edge_hypernym": "rgba(52, 211, 153, 0.55)",
        "edge_unknown": "rgba(71, 85, 105, 0.40)",
        "node_border": "#e2e8f0", "highlight_bg": "#f43f5e",
        "hover_bg": "#22d3ee",
        "shadow_color": "rgba(0,0,0,0.7)",
        "plotly_bg": "#020617", "plotly_paper": "#020617",
        "grid_color": "#0f172a", "axis_color": "#64748b",
    },
    "Warm": {
        "bg": "#fff7ed", "font": "#431407",
        "tooltip_bg": "rgba(255, 247, 237, 0.97)",
        "tooltip_border": "#fdba74", "tooltip_text": "#431407",
        "edge_cooccurrence": "rgba(234, 88, 12, 0.45)",
        "edge_semantic": "rgba(180, 83, 9, 0.40)",
        "edge_bridge": "rgba(202, 138, 4, 0.55)",
        "edge_inferred": "rgba(147, 51, 234, 0.50)",
        "edge_cause": "rgba(220, 38, 38, 0.55)",
        "edge_hypernym": "rgba(22, 163, 74, 0.45)",
        "edge_unknown": "rgba(120, 53, 15, 0.25)",
        "node_border": "#fff7ed", "highlight_bg": "#dc2626",
        "hover_bg": "#f59e0b",
        "shadow_color": "rgba(124, 45, 18, 0.15)",
        "plotly_bg": "#fff7ed", "plotly_paper": "#fff7ed",
        "grid_color": "#fed7aa", "axis_color": "#9a3412",
    },
    "Forest": {
        "bg": "#f0fdf4", "font": "#052e16",
        "tooltip_bg": "rgba(240, 253, 244, 0.97)",
        "tooltip_border": "#86efac", "tooltip_text": "#052e16",
        "edge_cooccurrence": "rgba(22, 163, 74, 0.45)",
        "edge_semantic": "rgba(5, 150, 105, 0.40)",
        "edge_bridge": "rgba(234, 179, 8, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(21, 128, 61, 0.45)",
        "edge_unknown": "rgba(20, 83, 45, 0.25)",
        "node_border": "#f0fdf4", "highlight_bg": "#15803d",
        "hover_bg": "#84cc16",
        "shadow_color": "rgba(20, 83, 45, 0.15)",
        "plotly_bg": "#f0fdf4", "plotly_paper": "#f0fdf4",
        "grid_color": "#bbf7d0", "axis_color": "#166534",
    },
    "Ocean": {
        "bg": "#ecfeff", "font": "#083344",
        "tooltip_bg": "rgba(236, 254, 255, 0.97)",
        "tooltip_border": "#67e8f9", "tooltip_text": "#083344",
        "edge_cooccurrence": "rgba(6, 182, 212, 0.45)",
        "edge_semantic": "rgba(14, 165, 233, 0.40)",
        "edge_bridge": "rgba(99, 102, 241, 0.55)",
        "edge_inferred": "rgba(168, 85, 247, 0.50)",
        "edge_cause": "rgba(244, 63, 94, 0.55)",
        "edge_hypernym": "rgba(13, 148, 136, 0.45)",
        "edge_unknown": "rgba(21, 94, 117, 0.25)",
        "node_border": "#ecfeff", "highlight_bg": "#0ea5e9",
        "hover_bg": "#22d3ee",
        "shadow_color": "rgba(8, 51, 68, 0.15)",
        "plotly_bg": "#ecfeff", "plotly_paper": "#ecfeff",
        "grid_color": "#a5f3fc", "axis_color": "#0e7490",
    },
}

PHYSICS_PRESETS = {
    "Stable (Default)": {
        "damping": 0.55, "gravity": -2500, "spring_length": 140,
        "spring_strength": 0.05, "central_gravity": 0.25,
        "stabilization": 2500,
    },
    "Fluid": {
        "damping": 0.25, "gravity": -1800, "spring_length": 120,
        "spring_strength": 0.05, "central_gravity": 0.30,
        "stabilization": 1500,
    },
    "Tight": {
        "damping": 0.70, "gravity": -4000, "spring_length": 80,
        "spring_strength": 0.08, "central_gravity": 0.20,
        "stabilization": 3000,
    },
    "Off": {
        "damping": 0.99, "gravity": 0, "spring_length": 200,
        "spring_strength": 0.0, "central_gravity": 0.0,
        "stabilization": 0,
    },
}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def render_graph_pyvis(
    nx_graph, concept_abstract_map, physics_enabled=True,
    min_node_size=8, max_node_size=40, cmap_name="viridis",
    custom_labels=None, node_label_size=12, node_label_position="center",
    top_n_nodes=0, theme=None, physics_preset=None,
    show_edge_weights=False, edge_label_mode="hover",
    use_abbreviated_labels=False,
    max_label_length=15,
    node_font_face="Inter, Segoe UI, Roboto, sans-serif",
    edge_label_size=10, edge_label_color=None,
    edge_label_position="middle", enable_node_highlight=True,
    show_definitions=True,
    edge_lightness=0.6,
    edge_color_mode="theme",
    custom_edge_color="#AAAAAA",
    tooltip_font_size=13,
    node_legend_font_size=13,
) -> None:
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(
            degrees.keys(), key=lambda x: degrees[x], reverse=True
        )[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    cmap_colors = get_colormap_colors(
        cmap_name, max(1, len(nx_graph.nodes()))
    )
    net = Network(
        height="780px", width="100%",
        bgcolor=theme['bg'], font_color=theme['font'],
        select_menu=True, notebook=False, cdn_resources='remote',
    )

    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
var options = {{
    "physics": {{
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {{
            "gravitationalConstant": {physics_preset['gravity']},
            "centralGravity": {physics_preset['central_gravity']},
            "springLength": {physics_preset['spring_length']},
            "springConstant": {physics_preset['spring_strength']},
            "damping": {physics_preset['damping']},
            "overlap": 0.15
        }},
        "stabilization": {{
            "enabled": true,
            "iterations": 500,
            "updateInterval": 50,
            "onlyDynamicEdges": true,
            "fit": true
        }}
    }},
    "interaction": {{
        "hover": true,
        "tooltipDelay": 180,
        "hideEdgesOnDrag": false,
        "zoomView": true,
        "dragView": true
    }}
}}
""")
    else:
        net.set_options("""
var options = {
    "physics": { "enabled": false },
    "interaction": {
        "hover": true, "dragNodes": true,
        "dragView": true, "zoomView": true
    }
}
""")

    label_map = {}
    n_counter = 1
    used_rel_types = {}

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(
            min_node_size + freq * 1.2, min_node_size, max_node_size
        ))
        color = get_glacier_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        original_label = (
            custom_labels.get(node, node) if custom_labels else node
        )
        label = get_hierarchy_label(node, style="arrow")

        if (
            use_abbreviated_labels
            and len(original_label) > max_label_length
        ):
            short_label = f"N{n_counter}"
            label_map[short_label] = original_label
            n_counter += 1
            label = short_label
            node_shape = 'circle'
            max_allowed_by_node = int(size * 0.55)
            inside_font_size = max(6, min(int(node_label_size), max_allowed_by_node))
            font_dict = {
                'color': '#ffffff',
                'size': inside_font_size,
                'face': node_font_face,
                'bold': True,
            }
        else:
            label = label
            node_shape = 'dot'
            v_adjust = 0
            if node_label_position == "top":
                v_adjust = -int(node_label_size) - 4
            elif node_label_position == "bottom":
                v_adjust = int(node_label_size) + 4
            font_dict = {
                'color': theme['font'],
                'size': int(node_label_size),
                'face': node_font_face,
                'strokeWidth': 0,
                'vadjust': v_adjust,
            }

        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        definition = nx_graph.nodes[node].get('definition', '')
        tooltip_content = (
            f"<div style='font-family:{node_font_face};'>"
            f"<b style='font-size:14px;color:{theme['highlight_bg']};'>"
            f"{node}</b><br>"
            f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>"
            f"Type:</span> {concept_type}<br>"
            f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>"
            f"Degree:</span> {degree}<br>"
            f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>"
            f"Frequency:</span> {freq}"
        )
        if show_definitions and definition:
            tooltip_content += (
                f"<br><span style='color:{theme['tooltip_text']};opacity:0.7;'>"
                f"Definition:</span> <i>{definition}</i>"
            )
        if use_abbreviated_labels and label != original_label:
            tooltip_content += (
                f"<br><span style='color:{theme['tooltip_text']};opacity:0.7;'>"
                f"Full Label:</span> {original_label}"
            )
        tooltip_content += "</div>"

        net.add_node(
            node, label=label, size=size,
            color={
                'background': color,
                'border': theme['node_border'],
                'highlight': {
                    'background': theme['highlight_bg'], 'border': '#ffffff'
                },
                'hover': {
                    'background': theme['hover_bg'], 'border': '#ffffff'
                },
            },
            font=font_dict, title=tooltip_content,
            borderWidth=2, borderWidthSelected=3,
            shadow={
                'enabled': True,
                'color': theme['shadow_color'],
                'size': 12, 'x': 4, 'y': 4,
            },
            shape=node_shape,
        )

    all_weights = [
        nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()
    ]
    weight_threshold = (
        float(np.percentile(all_weights, 80)) if all_weights else 0.0
    )

    for u, v in nx_graph.edges():
        w = float(nx_graph[u][v].get('weight', 1))
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)

        rel_type = RelationshipType.SEMANTIC
        if edge_type != 'unknown':
            try:
                rel_type = RelationshipType(edge_type)
            except ValueError:
                pass

        if edge_color_mode == "theme":
            if edge_type == 'unknown':
                base_color = theme['edge_unknown']
            else:
                base_color = get_edge_color(rel_type)
            if edge_lightness > 0:
                base_color = lighten_hex_color(base_color, edge_lightness)
        elif edge_color_mode == "uniform_grey":
            base_color = lighten_hex_color("#808080", edge_lightness)
        else:
            base_color = lighten_hex_color(custom_edge_color, edge_lightness)

        if edge_type == 'unknown' and edge_color_mode == "theme":
            base_color = theme['edge_unknown']

        width = float(get_edge_width(rel_type) * (0.5 + 0.5 * w))
        style = get_edge_style(rel_type)
        dashes = True if style == "dashed" or is_inferred else False

        actual_edge_label_color = (
            edge_label_color if edge_label_color else theme['font']
        )
        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)),
            width=width,
            color={
                'color': base_color,
                'highlight': theme['highlight_bg'],
                'hover': theme['hover_bg'],
                'opacity': 0.85,
            },
            smooth={"type": "dynamic"},
            title=(
                f"<span style='font-family:{node_font_face};'>"
                f"Weight: <b>{w:.2f}</b><br>"
                f"Type: {edge_type}<br>"
                f"Inferred: {is_inferred}</span>"
            ),
            dashes=dashes,
        )

        if (
            edge_label_mode == "all"
            or (edge_label_mode == "threshold" and w >= weight_threshold)
        ):
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {
                'color': actual_edge_label_color,
                'size': int(edge_label_size),
                'background': theme['tooltip_bg'],
                'strokeWidth': 2,
                'strokeColor': theme['node_border'],
                'align': edge_label_position,
                'face': node_font_face,
            }
        net.add_edge(u, v, **edge_kwargs)

        if rel_type not in used_rel_types:
            used_rel_types[rel_type] = rel_type.value.replace("_", " ").title()

    if used_rel_types:
        legend_rows = []
        for rt, human in sorted(used_rel_types.items(), key=lambda x: x[1]):
            c = get_edge_color(rt)
            if edge_color_mode == "theme":
                c = lighten_hex_color(c, edge_lightness) if edge_lightness > 0 else c
            elif edge_color_mode == "uniform_grey":
                c = lighten_hex_color("#808080", edge_lightness)
            else:
                c = lighten_hex_color(custom_edge_color, edge_lightness)
            w = get_edge_width(rt)
            s = get_edge_style(rt)
            border = 'border: 1px dashed #888;' if s == "dashed" else 'border: 1px solid transparent;'
            legend_rows.append(
                f'<tr>'
                f'<td style="padding:2px 6px;">'
                f'<span style="display:inline-block;width:{int(20*w)}px;height:3px;'
                f'background:{c};vertical-align:middle;{border}"></span>'
                f'</td>'
                f'<td style="padding:2px 6px;color:#ccc;font-size:11px;">{human}</td>'
                f'</tr>'
            )
        legend_html = (
            '<div style="background:#0d0d1a;border-radius:8px;padding:12px 16px;'
            f'margin-top:8px;max-height:280px;overflow-y:auto;">'
            f'<div style="color:#fff;font-size:13px;font-weight:bold;margin-bottom:6px;">'
            f'Edge Colors ({len(used_rel_types)} relationship types)</div>'
            f'<table style="border-collapse:collapse;">{"".join(legend_rows)}</table>'
            f'</div>'
        )
        net.add_node(
            "__legend__",
            label="",
            shape="dot",
            size=0,
            color="rgba(0,0,0,0)",
            fixed=True,
            x=-500,
            y=-500,
            physics=False,
            title=legend_html,
        )

    try:
        tmp_html = tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False, encoding='utf-8'
        )
        tmp_path = tmp_html.name
        net.write_html(tmp_path, notebook=False)
        tmp_html.close()
        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        if use_abbreviated_labels and label_map:
            label_map_json = json.dumps(label_map)
            hidden_div = (
                f'<div id="glacier-label-map-data" style="display:none;">'
                f'{label_map_json}</div>'
            )
            html_content = html_content.replace('</body>', hidden_div + '</body>')
        os.unlink(tmp_path)
    except Exception as e:
        st.error(f"PyVis HTML generation failed: {e}")
        html_content = net.generate_html()

    custom_css = f"""
<style>
body {{
    background: {theme['bg']};
    margin: 0;
    padding: 0;
    font-family: '{node_font_face}', sans-serif;
}}
#mynetwork {{
    border-radius: 16px;
    box-shadow: 0 12px 48px {theme['shadow_color']};
    outline: none;
}}
div.vis-tooltip {{
    background: {theme['tooltip_bg']} !important;
    color: {theme['tooltip_text']} !important;
    border: 1px solid {theme['tooltip_border']} !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    font-family: '{node_font_face}', sans-serif !important;
    font-size: {tooltip_font_size}px !important;
    line-height: 1.5 !important;
    box-shadow: 0 8px 32px {theme['shadow_color']} !important;
    max-width: 320px !important;
    white-space: normal !important;
}}
div.vis-network div.vis-manipulation {{
    background: {theme['tooltip_bg']} !important;
    border-top: 1px solid {theme['tooltip_border']} !important;
    color: {theme['font']} !important;
}}
.glacier-legend {{
    font-size: {node_legend_font_size}px !important;
}}
</style>
"""
    html_content = html_content.replace('</head>', custom_css + '</head>')

    if enable_node_highlight:
        highlight_js = r"""
<script>
(function() {
    var checkExist = setInterval(function() {
        if (typeof network !== 'undefined' && network !== null && network.body && network.body.data) {
            clearInterval(checkExist);
            var nodesDS = network.body.data.nodes;
            var edgesDS = network.body.data.edges;
            var savedNodeColors = {};
            var activeNodeId = null;
            var labelMode = 'short';
            var labelMap = {};
            (function initLabelMap() {
                var hidden = document.getElementById('glacier-label-map-data');
                if (hidden && hidden.textContent) {
                    try { labelMap = JSON.parse(hidden.textContent); } catch(e) {}
                }
            })();
            function resetAll() {
                var nodeRestores = [];
                for (var nid in savedNodeColors) {
                    nodeRestores.push({id: nid, color: savedNodeColors[nid]});
                }
                if (nodeRestores.length > 0) nodesDS.update(nodeRestores);
                savedNodeColors = {};
                activeNodeId = null;
                var panel = document.getElementById('edge-info-panel');
                if (panel) panel.style.display = 'none';
            }
            function resolveFullName(shortOrId) {
                if (labelMap && labelMap[shortOrId]) return labelMap[shortOrId];
                var n = nodesDS.get(shortOrId);
                if (n && n.title) {
                    var htmlStr = n.title.replace(/<br\s*\/?>/gi, '\n');
                    var tmp = document.createElement('div');
                    tmp.innerHTML = htmlStr;
                    var txt = (tmp.textContent || tmp.innerText || '').trim();
                    var firstLine = txt.split('\n')[0];
                    if (firstLine) return firstLine.replace(/<[^>]*>/g,'').trim();
                }
                return shortOrId;
            }
            function formatEdgeRow(e, idx, mode) {
                var typeColor = e.inferred ? '#8b5cf6' : '#0ea5e9';
                var badge = e.inferred ? ' <span style="background:#8b5cf6;color:white;padding:1px 4px;border-radius:3px;font-size:9px;">INFERRED</span>' : '';
                var typeBadge = '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:600;">' + e.type + '</span>';
                var fromName = (mode === 'short') ? e.from : resolveFullName(e.from);
                var toName = (mode === 'short') ? e.to : resolveFullName(e.to);
                return '<div style="padding:8px 10px;margin:4px 0;background:rgba(248,250,252,0.9);border-left:4px solid ' + typeColor + ';border-radius:6px;font-size:12px;">' +
                    '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">' +
                    '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;">' + fromName + '</span>' +
                    '<span style="color:#94a3b8;font-size:13px;">↔</span>' +
                    '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;">' + toName + '</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;gap:8px;padding-left:10px;">' +
                    '<span style="background:#0ea5e9;color:white;font-size:10px;padding:2px 6px;border-radius:4px;font-weight:700;">W: ' + e.weight + '</span>' +
                    typeBadge + badge +
                    '</div></div>';
            }
            function showEdgeInfoPanel(nodeId, connectedEdges) {
                var panel = document.getElementById('edge-info-panel');
                if (!panel) {
                    panel = document.createElement('div');
                    panel.id = 'edge-info-panel';
                    document.body.appendChild(panel);
                }
                panel.style.cssText = 'position:fixed;top:90px;right:20px;width:400px;max-height:calc(100vh - 110px);overflow-y:auto;z-index:9999;' +
                    'background:rgba(255,255,255,0.95);border:1px solid rgba(255,215,0,0.6);border-radius:16px;padding:0;' +
                    'font-family:Inter,Segoe UI,Roboto,sans-serif;box-shadow:0 20px 60px rgba(0,0,0,0.15);backdrop-filter:blur(20px);';
                var nodeData = nodesDS.get(nodeId);
                var nodeName = nodeId;
                var nodeDefinition = ""; var nodeType = ""; var nodeFreq = ""; var nodeDegree = "";
                if (nodeData && nodeData.title) {
                    var htmlStr = nodeData.title.replace(/<br\s*\/?>/gi, '\n');
                    var tmpDiv = document.createElement("div");
                    tmpDiv.innerHTML = htmlStr;
                    var tooltipText = (tmpDiv.textContent || tmpDiv.innerText || "").trim();
                    var defMatch = tooltipText.match(/Definition:\s*([^\n]+)/i);
                    if (defMatch && defMatch[1]) { nodeDefinition = defMatch[1].trim(); }
                    var typeMatch = tooltipText.match(/Type:\s*([^\n]+)/i);
                    if (typeMatch && typeMatch[1]) { nodeType = typeMatch[1].trim(); }
                    var freqMatch = tooltipText.match(/Frequency:\s*([^\n]+)/i);
                    if (freqMatch && freqMatch[1]) { nodeFreq = freqMatch[1].trim(); }
                    var degMatch = tooltipText.match(/Degree:\s*([^\n]+)/i);
                    if (degMatch && degMatch[1]) { nodeDegree = degMatch[1].trim(); }
                    var nameMatch = tooltipText.match(/^([^\n]+)/);
                    if (nameMatch) nodeName = nameMatch[1].replace(/<[^>]*>/g,'').trim();
                    if (!nodeName) nodeName = nodeId;
                }
                var html = '<div style="padding:16px 20px;background:linear-gradient(135deg,rgba(255,215,0,0.15),rgba(255,183,77,0.1));border-radius:16px 16px 0 0;border-bottom:2px solid rgba(255,215,0,0.4);">';
                html += '<div style="font-size:18px;font-weight:800;color:#1e293b;margin-bottom:8px;">🧊 ' + nodeName + '</div>';
                html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
                if (nodeType) html += '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">' + nodeType + '</span>';
                if (nodeDegree) html += '<span style="background:rgba(168,85,247,0.1);color:#a855f7;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Deg: ' + nodeDegree + '</span>';
                if (nodeFreq) html += '<span style="background:rgba(34,197,94,0.1);color:#22c55e;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Freq: ' + nodeFreq + '</span>';
                html += '</div></div>';
                if (nodeDefinition) {
                    html += '<div style="padding:12px 20px;background:rgba(251,191,36,0.06);border-bottom:1px solid rgba(0,0,0,0.04);">';
                    html += '<div style="font-size:10px;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-bottom:4px;">📖 Definition</div>';
                    html += '<div style="font-size:12px;color:#475569;font-style:italic;line-height:1.4;">' + nodeDefinition + '</div></div>';
                }
                html += '<div style="padding:10px 20px;background:rgba(248,250,252,0.8);border-bottom:1px solid rgba(0,0,0,0.04);display:flex;align-items:center;gap:10px;">';
                html += '<span style="font-size:10px;color:#94a3b8;font-weight:600;">Label Mode</span>';
                html += '<button id="btn-short" onclick="window._glacierSetLabelMode(\'short\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:#D32F2F;color:white;">Short</button>';
                html += '<button id="btn-full" onclick="window._glacierSetLabelMode(\'full\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:transparent;color:#64748b;">Full</button>';
                html += '</div>';
                html += '<div id="edges-container" style="padding:12px 16px 16px;">';
                var edgeList = [];
                connectedEdges.forEach(function(eId) {
                    var e = edgesDS.get(eId);
                    if (!e) return;
                    var fromNode = nodesDS.get(e.from); var toNode = nodesDS.get(e.to);
                    var fromLabel = fromNode ? (fromNode.label || e.from) : e.from;
                    var toLabel = toNode ? (toNode.label || e.to) : e.to;
                    var w = (typeof e.value === 'number') ? e.value : (e.width || 1);
                    var edgeType = 'unknown', isInferred = false;
                    if (e.title) {
                        var edgeHtml = e.title.replace(/<br\s*\/?>/gi, '\n');
                        var tmpDiv = document.createElement('div');
                        tmpDiv.innerHTML = edgeHtml;
                        var _txt = (tmpDiv.textContent || tmpDiv.innerText || '').trim();
                        var m = _txt.match(/Type:\s*([^\n]+)/);
                        if (m) edgeType = m[1].trim();
                        if (_txt.indexOf('Inferred: true') !== -1) isInferred = true;
                    }
                    edgeList.push({from: fromLabel, to: toLabel, weight: (typeof w === 'number') ? w.toFixed(2) : String(w), type: edgeType, inferred: isInferred});
                });
                edgeList.sort(function(a,b){ return parseFloat(b.weight)-parseFloat(a.weight); });
                edgeList.forEach(function(e, idx){ html += formatEdgeRow(e, idx, labelMode); });
                html += '</div>';
                panel.innerHTML = html;
                panel.style.display = 'block';
                panel._edgeList = edgeList;
                window._glacierSetLabelMode = function(mode) {
                    labelMode = mode;
                    var p = document.getElementById('edge-info-panel');
                    if (!p || !p._edgeList) return;
                    var btnShort = document.getElementById('btn-short');
                    var btnFull = document.getElementById('btn-full');
                    if (mode === 'short') {
                        btnShort.style.background = '#D32F2F'; btnShort.style.color = 'white';
                        btnFull.style.background = 'transparent'; btnFull.style.color = '#64748b';
                    } else {
                        btnFull.style.background = '#D32F2F'; btnFull.style.color = 'white';
                        btnShort.style.background = 'transparent'; btnShort.style.color = '#64748b';
                    }
                    var container = document.getElementById('edges-container');
                    if (container) {
                        var newHtml = '';
                        p._edgeList.forEach(function(e, idx){ newHtml += formatEdgeRow(e, idx, mode); });
                        container.innerHTML = newHtml;
                    }
                };
            }
            network.on("selectNode", function(params) {
                var nodeId = params.nodes[0];
                if (nodeId === "__legend__") {
                    network.unselectAll();
                    return;
                }
                if (activeNodeId !== null && activeNodeId !== nodeId) resetAll();
                activeNodeId = nodeId;
                var connectedEdges = network.getConnectedEdges(nodeId);
                var connectedNodes = network.getConnectedNodes(nodeId);
                var nodeUpdates = [];
                connectedNodes.forEach(function(nId) {
                    var n = nodesDS.get(nId);
                    if (n && !savedNodeColors[nId]) {
                        savedNodeColors[nId] = JSON.parse(JSON.stringify(n.color));
                        var newColor = JSON.parse(JSON.stringify(n.color));
                        if (typeof newColor === 'string') newColor = {background: newColor, border: '#FFD700'};
                        else newColor.border = '#FFD700';
                        nodeUpdates.push({id: nId, color: newColor, shadow: {enabled: true, color: 'rgba(255,215,0,0.5)', size: 15, x: 0, y: 0}});
                    }
                });
                if (nodeUpdates.length > 0) nodesDS.update(nodeUpdates);
                showEdgeInfoPanel(nodeId, connectedEdges);
            });
            network.on("deselectNode", function(){ resetAll(); });
            network.on("click", function(params){
                if (params.nodes.length === 0 && activeNodeId !== null) resetAll();
            });
        }
    }, 250);
})();
</script>
"""
        html_content = html_content.replace('</body>', highlight_js + '</body>')

    st.components.v1.html(html_content, height=790, scrolling=True)

    if use_abbreviated_labels and label_map:
        st.markdown("---")
        st.markdown("### 🗺️ Node Label Legend")
        st.caption("Hover over nodes in the interactive graph to see their full names and definitions.")
        sorted_legend = sorted(label_map.items(), key=lambda x: int(x[0][1:]))
        cols = st.columns(4)
        for i, (short, full) in enumerate(sorted_legend):
            with cols[i % 4]:
                st.markdown(
                    f"""<div class='glacier-legend' style='padding:8px; border-radius:6px; background-color:{theme.get('tooltip_bg', '#f8fafc')};
border-left:4px solid {theme.get('highlight_bg', '#ff6b6b')}; margin-bottom:6px; font-size:{node_legend_font_size}px;'>
<b style='color:{theme.get('highlight_bg', '#ff6b6b')}; font-size:{node_legend_font_size+1}px;'>{short}</b>:
<span style='font-size:{node_legend_font_size}px; color:{theme.get('font', '#1e293b')};'>{full}</span>
</div>""",
                    unsafe_allow_html=True,
                )

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button(
            "Download Interactive Graph (HTML)",
            data=html_bytes,
            file_name="glacier_concept_graph.html",
            mime="text/html",
        )
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")


def build_category_hierarchy(
    concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    top_n_per_category: int = 0,
) -> Tuple[List[str], List[str], List[float]]:
    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []

    category_map = {c: categorize_glacier_concept(c) for c in concepts}
    all_cat_displays = set()
    for cat in category_map.values():
        if cat == 'rheology':
            all_cat_displays.add('Rheology')
        elif cat == 'sliding_hydro':
            all_cat_displays.add('Sliding/Hydro')
        elif cat == 'numerical':
            all_cat_displays.add('Numerical')
        elif cat == 'dynamics_thermo':
            all_cat_displays.add('Dynamics/Thermo')
        else:
            all_cat_displays.add(cat.title())

    category_children: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for concept in concepts:
        cat = category_map.get(concept, 'general')
        freq = len(concept_abstract_map.get(concept, []))
        concept_display = concept.replace('_', ' ').title()
        if concept_display in all_cat_displays:
            continue
        # Map internal cat to display name
        if cat == 'rheology':
            display_cat = 'Rheology & Material Physics'
        elif cat == 'sliding_hydro':
            display_cat = 'Basal Sliding & Hydrology'
        elif cat == 'numerical':
            display_cat = 'Numerical Solvers & Meshing'
        elif cat == 'dynamics_thermo':
            display_cat = 'Ice Dynamics & Thermodynamics'
        else:
            display_cat = cat.replace('_', ' ').title()
        category_children[display_cat].append((concept, float(freq)))

    total_value = sum(freq for children in category_children.values() for _, freq in children)
    root_label = "Glacial Flow Mechanics"
    labels.append(root_label)
    parents.append("")
    values.append(float(total_value))

    for cat, children in sorted(category_children.items()):
        labels.append(cat)
        parents.append(root_label)
        cat_total = sum(w for _, w in children)
        values.append(float(cat_total))

        children.sort(key=lambda x: x[1], reverse=True)
        if top_n_per_category > 0:
            children = children[:top_n_per_category]

        for concept, freq in children:
            concept_display = concept.replace('_', ' ').title()
            labels.append(concept_display)
            parents.append(cat)
            values.append(float(freq))

    return labels, parents, values


def render_sunburst_chart(
    labels, parents, values, cmap_name="viridis",
    label_size=20, width=900, height=700,
    theme=None, branchvalues="total",
    show_labels=True, show_values=False,
    hover_info="all", color_continuous_scale=None,
    font_family="Arial, sans-serif",
    legend_font_size=12,
) -> None:
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    if len(labels) != len(parents) or len(labels) != len(values):
        st.error(f"Sunburst data mismatch: labels={len(labels)}, parents={len(parents)}, values={len(values)}")
        return

    if theme is None:
        theme = THEME_PRESETS.get("Bright (Default)", THEME_PRESETS["Bright (Default)"])

    parent_map = {labels[i]: parents[i] for i in range(len(labels))}

    def get_depth(label, visited=None):
        if visited is None:
            visited = set()
        if label in visited:
            return 0
        visited.add(label)
        p = parent_map.get(label, "")
        if p == "":
            return 0
        return 1 + get_depth(p, visited)

    try:
        depths = [get_depth(l) for l in labels]
    except RecursionError:
        st.error("Circular hierarchy detected in sunburst data.")
        return

    SYMBOL_LIBRARY = ['✦', '★', '●', '■', '▲', '◆', '⬟', '⬢', '◉', '◈',
                      '◇', '○', '□', '△', '◊']

    node_symbols: Dict[str, str] = {}
    for i, lab in enumerate(labels):
        d = depths[i]
        p = parents[i]
        if d == 0:
            node_symbols[lab] = SYMBOL_LIBRARY[0]
        else:
            ancestors: List[str] = []
            current = lab
            visited: Set[str] = set()
            while current != "" and current not in visited:
                visited.add(current)
                parent = parent_map.get(current, "")
                if parent != "" and parent in node_symbols:
                    ancestors.insert(0, node_symbols[parent])
                current = parent
            siblings = [
                labels[j] for j in range(len(labels))
                if parents[j] == p and depths[j] == d
            ]
            sym_idx = siblings.index(lab) if lab in siblings else 0
            own_symbol = SYMBOL_LIBRARY[(d + sym_idx) % len(SYMBOL_LIBRARY)]
            node_symbols[lab] = own_symbol

    display_labels: List[str] = []
    for i, lab in enumerate(labels):
        d = depths[i]
        if show_labels:
            if d == 0:
                display_labels.append(node_symbols[lab])
            else:
                chain: List[str] = []
                current = lab
                visited = set()
                while current != "" and current not in visited:
                    visited.add(current)
                    if current in node_symbols:
                        chain.insert(0, node_symbols[current])
                    current = parent_map.get(current, "")
                combo = "".join(chain[-3:]) if len(chain) > 3 else "".join(chain)
                display_labels.append(combo)
        else:
            display_labels.append(lab)

    unique_ids: List[str] = []
    seen: Dict[str, int] = {}
    for i, lab in enumerate(labels):
        base = f"{lab}_d{depths[i]}"
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base)
            seen[base] = 1

    parent_ids: List[str] = []
    for p in parents:
        if p == "":
            parent_ids.append("")
        else:
            found = False
            for i, lab in enumerate(labels):
                if lab == p:
                    parent_ids.append(unique_ids[i])
                    found = True
                    break
            if not found:
                parent_ids.append("")

    n_nodes = len(labels)
    cmap_to_use = color_continuous_scale or cmap_name or "Spectral"
    plot_colors: List[str] = []

    color_success = False
    try:
        cmap_obj = plt.cm.get_cmap(cmap_to_use)
        t_vals = np.linspace(0.05, 0.95, n_nodes)
        rgbas = [cmap_obj(t) for t in t_vals]
        plot_colors = [matplotlib.colors.to_hex(rgba) for rgba in rgbas]
        color_success = True
    except Exception:
        pass

    if not color_success:
        try:
            if hasattr(px.colors.sequential, cmap_to_use):
                px_scale = getattr(px.colors.sequential, cmap_to_use)
                plot_colors = [
                    px_scale[int(i * len(px_scale) / n_nodes) % len(px_scale)]
                    for i in range(n_nodes)
                ]
                color_success = True
        except Exception:
            pass

    if not color_success:
        try:
            from plotly.express import colors as px_colors
            qual_palettes = [
                px_colors.qualitative.Bold,
                px_colors.qualitative.Vivid,
                px_colors.qualitative.Safe,
                px_colors.qualitative.Pastel,
                px_colors.qualitative.Dark24,
                px_colors.qualitative.Light24,
            ]
            long_palette: List[str] = []
            for pal in qual_palettes:
                long_palette.extend(pal)
            plot_colors = [
                long_palette[i % len(long_palette)] for i in range(n_nodes)
            ]
            color_success = True
        except Exception:
            pass

    if not color_success:
        cmap_obj = plt.cm.get_cmap("tab20")
        plot_colors = [
            matplotlib.colors.to_hex(cmap_obj(i % 20 / 20))
            for i in range(n_nodes)
        ]

    legend_entries: List[Dict[str, Any]] = []
    for i, lab in enumerate(labels):
        d = depths[i]
        sym = display_labels[i]
        color = plot_colors[i]
        legend_entries.append({
            'symbol': sym,
            'label': lab,
            'depth': d,
            'color': color,
            'value': values[i],
        })
    legend_entries.sort(key=lambda x: (x['depth'], -x['value']))

    bv = branchvalues if branchvalues in ["total", "remainder"] else "total"
    if show_labels and show_values:
        textinfo = 'label+value'
    elif show_labels:
        textinfo = 'label'
    elif show_values:
        textinfo = 'value'
    else:
        textinfo = 'none'

    sunburst_colors = plot_colors.copy()
    for i in range(len(labels)):
        if depths[i] == 0:
            sunburst_colors[i] = theme.get("plotly_paper", "#f8f9fa")

    try:
        fig = go.Figure(go.Sunburst(
            ids=unique_ids,
            labels=display_labels,
            parents=parent_ids,
            values=values,
            customdata=labels,
            branchvalues=bv,
            marker=dict(
                colors=sunburst_colors,
                line=dict(width=0.5, color="rgba(255,255,255,0.25)"),
            ),
            textinfo=textinfo,
            hovertemplate=(
                '<b>%{customdata}</b><br>Value: %{value}<br>'
                'Symbol: %{label}<extra></extra>'
                if hover_info == "all"
                else '<b>%{customdata}</b><extra></extra>'
                if hover_info == "minimal"
                else '<extra></extra>'
            ),
            insidetextorientation="radial",
            textfont=dict(
                size=int(label_size), family=font_family, color="white"
            ),
        ))
    except Exception as e:
        st.error(f"Failed to create sunburst chart: {e}")
        return

    try:
        fig.update_layout(
            margin=dict(l=0, r=0, t=80, b=0),
            paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
            font=dict(color=theme.get("font", "#000000"), family=font_family),
            width=int(width),
            height=int(height),
            title=dict(
                text=(
                    "<b>Hierarchical Concept Map</b><br>"
                    "<sup>★ Parent | ★□ Child | ★□◆ Grandchild — Hover for names</sup>"
                ),
                font=dict(size=16, family=font_family),
            ),
            modebar=dict(
                orientation='h',
                bgcolor='rgba(255,255,255,0.7)',
                color='#333333',
                activecolor='#D32F2F',
            ),
        )
    except Exception as e:
        st.warning(f"Layout configuration issue: {e}")

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "💡 **Export:** Click the 📷 Camera icon (top-right of chart) "
        "to download high-res PNG/SVG/PDF."
    )

    if st.session_state.get('sunburst_show_legend', True):
        st.markdown("### 📊 Symbol-to-Label Legend")
        depth_names = {
            0: "Root", 1: "Category (Parent)", 2: "Concept (Child)",
            3: "Sub-Concept", 4: "Detail",
        }
        for d in sorted(set([e['depth'] for e in legend_entries])):
            depth_label = depth_names.get(d, f"Level {d}")
            st.markdown(f"**{depth_label}**")
            entries_at_depth = [
                e for e in legend_entries if e['depth'] == d
            ]
            n_cols = min(4, max(1, len(entries_at_depth)))
            cols = st.columns(n_cols)
            for i, entry in enumerate(entries_at_depth):
                with cols[i % n_cols]:
                    text_color = theme.get("font", "#333333")
                    st.markdown(
                        f"""<div style='padding:8px; border-radius:6px; background-color:{entry['color']}22;
border-left:4px solid {entry['color']}; margin-bottom:6px; font-size:{legend_font_size}px;'>
<span style='font-size:{legend_font_size+4}px; color:{entry['color']}; margin-right:6px;'>{entry['symbol']}</span>
<span style='font-size:{legend_font_size}px; color:{text_color}; font-weight:500;'>{entry['label']}</span>
<span style='font-size:{legend_font_size-1}px; color:#666; float:right;'>({entry['value']:.0f})</span>
</div>""",
                        unsafe_allow_html=True,
                    )


def render_graph_plotly_2d(
    nx_graph, concept_abstract_map,
    cmap_name="viridis", top_n_nodes=0, theme=None,
    show_edge_weights=False, node_label_size=10,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if nx_graph.number_of_nodes() == 0:
        st.info("No nodes to display in 2D plot.")
        return

    try:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5, iterations=100)
    except Exception:
        pos = {n: (0, 0) for n in nx_graph.nodes()}

    edge_x, edge_y = [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Freq: {freq}")
        node_color.append(get_glacier_category_color(node, cmap_colors))
        node_size.append(max(15, min(50, freq + 10)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='rgba(150,150,150,0.4)', width=1),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        text=[n.replace('_', ' ').title() for n in nx_graph.nodes()],
        textposition='top center',
        textfont=dict(size=node_label_size, color=theme.get('font', '#000')),
        hovertext=node_text,
        hoverinfo='text'
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
        font_color=theme.get('font', '#000000'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title="2D Concept Graph (Plotly)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_graph_plotly_3d(
    nx_graph, concept_abstract_map,
    cmap_name="viridis", top_n_nodes=0, theme=None,
    show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if nx_graph.number_of_nodes() == 0:
        st.info("No nodes to display in 3D plot.")
        return

    try:
        pos = nx.spring_layout(nx_graph, seed=42, dim=3, k=2.5, iterations=100)
    except Exception:
        pos = {n: (0, 0, 0) for n in nx_graph.nodes()}

    edge_x, edge_y, edge_z = [], [], []
    for u, v in nx_graph.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    node_x, node_y, node_z, node_text, node_color, node_size = [], [], [], [], [], []
    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))
    for node in nx_graph.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Freq: {freq}")
        node_color.append(get_glacier_category_color(node, cmap_colors))
        node_size.append(max(8, min(25, freq // 2 + 5)))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='rgba(150,150,150,0.3)', width=1),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers+text',
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        text=[n.replace('_', ' ').title() for n in nx_graph.nodes()],
        textposition='top center',
        textfont=dict(size=8, color=theme.get('font', '#000')),
        hovertext=node_text,
        hoverinfo='text'
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
        ),
        title="3D Concept Graph (Plotly)",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_graph_fallback(
    nx_graph, concept_abstract_map,
    theme=None, show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    st.info("Text-based graph summary (fallback mode)")

    st.markdown(f"**Nodes:** {nx_graph.number_of_nodes()}")
    st.markdown(f"**Edges:** {nx_graph.number_of_edges()}")

    top_nodes = sorted(
        nx_graph.nodes(),
        key=lambda n: len(concept_abstract_map.get(n, [])),
        reverse=True,
    )[:20]

    node_data = []
    for node in top_nodes:
        freq = len(concept_abstract_map.get(node, []))
        degree = nx_graph.degree(node)
        node_data.append({
            'Concept': node.replace('_', ' ').title(),
            'Frequency': freq,
            'Degree': degree,
            'Category': categorize_glacier_concept(node),
        })

    st.dataframe(pd.DataFrame(node_data), use_container_width=True)


def render_radar_chart(
    distill_df, top_k=15, cmap_name="viridis", theme=None,
    label_size=12, font_size=11
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if distill_df.empty:
        st.info("No distillation data for radar chart.")
        return

    df = distill_df.head(top_k).copy()
    if len(df) < 3:
        st.info("Need at least 3 concepts for radar chart.")
        return

    categories = df['concept'].tolist()
    categories = [c.replace('_', ' ').title() for c in categories]
    cmap_colors = get_colormap_colors(cmap_name, 3)

    fig = go.Figure()
    metrics_to_plot = [
        ('frequency', 'Frequency'),
        ('semantic_density', 'Semantic Density'),
        ('coherence_score', 'Coherence')
    ]

    for i, (metric, name) in enumerate(metrics_to_plot):
        if metric in df.columns:
            values = df[metric].tolist()
            values += values[:1]
            cat_loop = categories + categories[:1]
            fig.add_trace(go.Scatterpolar(
                r=values, theta=cat_loop,
                fill='toself', name=name,
                line=dict(color=cmap_colors[i], width=2),
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1.1],
                tickfont=dict(size=int(font_size), color=theme.get('axis_color', '#000'))
            ),
            angularaxis=dict(
                tickfont=dict(size=int(label_size), color=theme.get('font', '#000'))
            )
        ),
        showlegend=True,
        title=f"Concept Distillation Radar (Top {min(top_k, 10)})",
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        font=dict(color=theme.get('font', '#000000'), size=int(font_size)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=int(font_size))),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tsne_projection(
    valid_concepts, concept_abstract_map,
    embed_model, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 5:
        st.info("Need at least 5 concepts for t-SNE projection.")
        return

    try:
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts, show_progress_bar=False,
                batch_size=64, convert_to_numpy=True,
            )
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_concepts)-1))
        coords = tsne.fit_transform(embeddings)

        freqs = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
        categories = [categorize_glacier_concept(c) for c in valid_concepts]

        fig = px.scatter(
            x=coords[:, 0], y=coords[:, 1],
            color=categories,
            size=freqs,
            hover_name=[c.replace('_', ' ').title() for c in valid_concepts],
            title="t-SNE Projection of Concept Embeddings",
            template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
            plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
            font_color=theme.get('font', '#000000'),
        )
        st.plotly_chart(fig, use_container_width=True)

        del embeddings, coords
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        st.warning(f"t-SNE projection failed: {e}")


def render_community_detection(
    nx_graph, valid_concepts,
    concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if nx_graph.number_of_nodes() < 5:
        st.info("Need at least 5 nodes for community detection.")
        return

    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(nx_graph))

        node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community[node] = f"Community {i+1}"

        comm_data = []
        for node in valid_concepts:
            if node in node_community:
                comm_data.append({
                    'Concept': node.replace('_', ' ').title(),
                    'Community': node_community[node],
                    'Frequency': len(concept_abstract_map.get(node, [])),
                    'Degree': nx_graph.degree(node),
                })

        if comm_data:
            comm_df = pd.DataFrame(comm_data)
            st.dataframe(comm_df, use_container_width=True)

            fig = px.scatter(
                comm_df, x='Degree', y='Frequency',
                color='Community', hover_data=['Concept'],
                title="Communities by Degree vs Frequency",
                template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
                plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
                font_color=theme.get('font', '#000000'),
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Community detection failed: {e}")


def render_concept_growth(
    df_filtered, valid_concepts,
    concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        st.info("No year data available for growth analysis.")
        return

    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 2:
        st.info("Need at least 2 years for growth analysis.")
        return

    year_range = sorted(years.unique())
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:10]

    growth_data = []
    for year in year_range:
        year_mask = df_filtered["Year"] == year
        year_df = df_filtered[year_mask]
        year_text = ""
        for idx, row in year_df.iterrows():
            for col in df_filtered.columns:
                if pd.notna(row[col]):
                    year_text += " " + str(row[col])
        for concept in top_concepts:
            count = len(re.findall(r'\b' + re.escape(concept) + r'\b', year_text, re.I))
            growth_data.append({"Year": year, "Concept": concept, "Count": count})

    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        fig = px.line(
            growth_df, x="Year", y="Count", color="Concept",
            title="Concept Growth Over Time",
            template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
            plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
            font_color=theme.get('font', '#000000'),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_bubble_chart(
    nx_graph, valid_concepts,
    concept_abstract_map, distill_df, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if nx_graph.number_of_nodes() == 0:
        st.info("No graph data for bubble chart.")
        return

    bubble_data = []
    for concept in valid_concepts:
        freq = len(concept_abstract_map.get(concept, []))
        degree = nx_graph.degree(concept)
        category = categorize_glacier_concept(concept)

        efficiency = 0.0
        if not distill_df.empty and 'concept' in distill_df.columns:
            match = distill_df[distill_df['concept'] == concept]
            if not match.empty and 'distillation_efficiency' in match.columns:
                efficiency = match['distillation_efficiency'].values[0]

        bubble_data.append({
            'Concept': concept.replace('_', ' ').title(),
            'Frequency': freq,
            'Degree': degree,
            'Category': category,
            'Efficiency': efficiency,
        })

    if bubble_data:
        bubble_df = pd.DataFrame(bubble_data)
        fig = px.scatter(
            bubble_df, x='Degree', y='Frequency',
            size='Efficiency', color='Category',
            hover_name='Concept',
            title="Concept Importance Bubble Chart",
            size_max=50,
            template="plotly_white" if theme == THEME_PRESETS["Bright (Default)"] else "plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
            plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
            font_color=theme.get('font', '#000000'),
        )
        st.plotly_chart(fig, use_container_width=True)


def apply_graph_edits(
    nx_graph, valid_concepts, concept_to_id, id_to_concept,
    concept_abstract_map,
    nodes_to_remove=None, nodes_to_merge=None, merge_name=None,
    new_edge=None, new_edge_weight=1.0,
    min_degree=0, min_freq=0,
) -> Tuple[nx.Graph, List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]], bool]:
    edited = False

    if nodes_to_remove:
        for node in nodes_to_remove:
            if node in nx_graph:
                nx_graph.remove_node(node)
                edited = True
        valid_concepts = [c for c in valid_concepts if c not in nodes_to_remove]

    if nodes_to_merge and merge_name and len(nodes_to_merge) >= 2:
        merged_freq = sum(len(concept_abstract_map.get(c, [])) for c in nodes_to_merge)
        nx_graph.add_node(merge_name, frequency=merged_freq, concept_type='general')

        for node in nodes_to_merge:
            if node in nx_graph:
                for neighbor in list(nx_graph.neighbors(node)):
                    if neighbor not in nodes_to_merge and neighbor != merge_name:
                        if not nx_graph.has_edge(merge_name, neighbor):
                            nx_graph.add_edge(merge_name, neighbor, weight=1.0)
                nx_graph.remove_node(node)

        valid_concepts = [c for c in valid_concepts if c not in nodes_to_merge]
        if merge_name not in valid_concepts:
            valid_concepts.append(merge_name)
        edited = True

    if new_edge and len(new_edge) == 2:
        u, v = new_edge
        if u in nx_graph and v in nx_graph and u != v:
            nx_graph.add_edge(u, v, weight=float(new_edge_weight), cooccurrence=0, semantic=0, edge_type='manual')
            edited = True

    if min_degree > 0:
        low_degree = [n for n in nx_graph.nodes() if nx_graph.degree(n) < min_degree]
        for node in low_degree:
            nx_graph.remove_node(node)
        valid_concepts = [c for c in valid_concepts if c in nx_graph]
        if low_degree:
            edited = True

    if min_freq > 0:
        low_freq = [n for n in nx_graph.nodes() if len(concept_abstract_map.get(n, [])) < min_freq]
        for node in low_freq:
            nx_graph.remove_node(node)
        valid_concepts = [c for c in valid_concepts if c in nx_graph]
        if low_freq:
            edited = True

    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}

    return nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, edited


def render_concept_timeline(
    df_filtered, valid_concepts, concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        st.info("No 'Year' data available for timeline visualization.")
        return
    years = df_filtered["Year"].dropna().astype(int)
    if len(years) == 0:
        st.info("No valid year data found.")
        return
    year_range = sorted(years.unique())
    if len(year_range) < 2:
        st.info("Need at least 2 different years for timeline.")
        return
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:10]
    timeline_data: List[Dict[str, Any]] = []
    for year in year_range:
        year_mask = df_filtered["Year"] == year
        year_df = df_filtered[year_mask]
        year_text = ""
        for idx, row in year_df.iterrows():
            for col in df_filtered.columns:
                if pd.notna(row[col]):
                    year_text += " " + str(row[col])
        for concept in top_concepts:
            count = len(re.findall(
                r'\b' + re.escape(concept) + r'\b', year_text, re.I
            ))
            timeline_data.append({
                "Year": year, "Concept": concept, "Count": count,
            })
    if not timeline_data:
        st.info("No timeline data to display.")
        return
    timeline_df = pd.DataFrame(timeline_data)
    fig = px.line(
        timeline_df, x="Year", y="Count", color="Concept",
        title="Concept Frequency Over Time",
        labels={"Count": "Mentions", "Year": "Publication Year"},
        template=(
            "plotly_white" if theme == THEME_PRESETS["Bright (Default)"]
            else "plotly_dark"
        ),
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cooccurrence_heatmap(
    nx_graph, valid_concepts, concept_abstract_map,
    top_n=30, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:top_n]
    if len(top_concepts) < 3:
        st.info("Need at least 3 concepts for heatmap.")
        return
    n = len(top_concepts)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(top_concepts):
        for j, c2 in enumerate(top_concepts):
            if i == j:
                matrix[i][j] = len(concept_abstract_map.get(c1, []))
            elif nx_graph.has_edge(c1, c2):
                matrix[i][j] = nx_graph[c1][c2].get('cooccurrence', 0)
    fig = px.imshow(
        matrix, x=top_concepts, y=top_concepts,
        labels=dict(x="Concept", y="Concept", color="Co-occurrence"),
        title=f"Co-occurrence Heatmap (Top {n} Concepts)",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================
def export_graph(
    nx_graph, concept_abstract_map, export_format: str,
    include_metadata: bool = True,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    if export_format == "GraphML":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '6.1'
                nx_graph.graph['tool'] = 'Glacier-ConceptGraph'
            try:
                nx.write_graphml_lxml(nx_graph, "glacier_graph.graphml")
            except Exception:
                nx.write_graphml(nx_graph, "glacier_graph.graphml")
            with open("glacier_graph.graphml", "rb") as f:
                return f.read(), "application/graphml+xml", "glacier_graph.graphml"
        except Exception as e:
            st.error(f"GraphML export failed: {e}")
            return None, None, None
    elif export_format == "JSON (Full Metadata)":
        data = nx.node_link_data(nx_graph)
        if include_metadata:
            data['metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '6.1',
                'tool': 'Glacier-ConceptGraph',
                'node_count': len(nx_graph.nodes()),
                'edge_count': len(nx_graph.edges()),
                'inferred_edges': sum(
                    1 for u, v, d in nx_graph.edges(data=True)
                    if d.get('inferred', False)
                ),
                'categories': list(set(
                    categorize_glacier_concept(c) for c in nx_graph.nodes()
                )),
            }
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "glacier_graph_full.json"
    elif export_format == "JSON (Compact)":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "glacier_graph.json"
    elif export_format == "CSV (Edges + Metadata)":
        edge_data: List[Dict[str, Any]] = []
        for u, v, data in nx_graph.edges(data=True):
            row = {
                "source": u, "target": v,
                "weight": data.get('weight', 1),
                "cooccurrence": data.get('cooccurrence', 0),
                "semantic_similarity": data.get('semantic', 0),
                "edge_type": data.get('edge_type', 'unknown'),
                "inferred": data.get('inferred', False),
                "confidence": data.get('confidence', 1.0),
                "path": data.get('path', ''),
            }
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "glacier_edges_enhanced.csv"
    elif export_format == "CSV (Nodes + Metadata)":
        node_data: List[Dict[str, Any]] = []
        for node in nx_graph.nodes():
            row = {
                "concept": node,
                "frequency": len(concept_abstract_map.get(node, [])),
                "degree": nx_graph.degree(node),
                "concept_type": nx_graph.nodes[node].get('concept_type', 'general'),
                "definition": nx_graph.nodes[node].get('definition', ''),
                "category": categorize_glacier_concept(node),
            }
            row.update({
                k: v for k, v in nx_graph.nodes[node].items()
                if isinstance(v, (str, int, float, bool))
            })
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "glacier_nodes_enhanced.csv"
    elif export_format == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [
                get_glacier_category_color(n) for n in nx_graph.nodes()
            ]
            nx.draw(
                nx_graph, pos, with_labels=True,
                node_color=node_colors, edge_color='gray',
                node_size=400, font_size=7, font_weight='bold',
                edgecolors='white', linewidths=1,
            )
            buf = io.BytesIO()
            plt.savefig(
                buf, format='png', dpi=300,
                bbox_inches='tight', facecolor='white',
            )
            buf.seek(0)
            plt.close()
            return buf.read(), "image/png", "glacier_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif export_format == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=150)
            node_colors = [
                get_glacier_category_color(n) for n in nx_graph.nodes()
            ]
            nx.draw(
                nx_graph, pos, with_labels=True,
                node_color=node_colors, edge_color='gray',
                node_size=400, font_size=7, font_weight='bold',
                edgecolors='white', linewidths=1,
            )
            buf = io.BytesIO()
            plt.savefig(
                buf, format='svg', bbox_inches='tight', facecolor='white',
            )
            buf.seek(0)
            plt.close()
            return buf.read(), "image/svg+xml", "glacier_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    elif export_format == "GEXF":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '6.1'
            nx.write_gexf(nx_graph, "glacier_graph.gexf")
            with open("glacier_graph.gexf", "rb") as f:
                return f.read(), "application/xml", "glacier_graph.gexf"
        except Exception as e:
            st.error(f"GEXF export failed: {e}")
            return None, None, None
    return None, None, None


# ============================================================================
# REASONING DASHBOARD
# ============================================================================
def render_reasoning_dashboard(
    nx_graph, valid_concepts, ontology, extractor,
) -> None:
    st.subheader("🔍 Ontology-Based Reasoning Insights")
    type_counts: Dict[str, int] = defaultdict(int)
    for c in valid_concepts:
        if c in ontology.concepts:
            type_counts[ontology.concepts[c].concept_type.value] += 1
        else:
            type_counts["unknown"] += 1
    fig = px.pie(
        values=list(type_counts.values()),
        names=list(type_counts.keys()),
        title="Concept Type Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
    inferred_edges = [
        (u, v) for u, v, d in nx_graph.edges(data=True)
        if d.get('inferred', False)
    ]
    observed_edges = [
        (u, v) for u, v, d in nx_graph.edges(data=True)
        if not d.get('inferred', False)
    ]
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed Edges", len(observed_edges))
    col2.metric("Inferred Edges", len(inferred_edges))
    col3.metric(
        "Inference Ratio",
        f"{len(inferred_edges) / max(len(observed_edges), 1):.2f}",
    )
    rel_types: Dict[str, int] = defaultdict(int)
    for u, v, d in nx_graph.edges(data=True):
        rel_types[d.get('edge_type', 'unknown')] += 1
    if rel_types:
        rel_df = pd.DataFrame(
            [(k, v) for k, v in rel_types.items()],
            columns=['Relationship Type', 'Count'],
        )
        rel_df = rel_df.sort_values('Count', ascending=False)
        st.dataframe(rel_df, use_container_width=True)
        fig = px.bar(
            rel_df, x='Relationship Type', y='Count',
            title="Edge Type Distribution",
            color='Relationship Type',
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("🔗 Inferred Method-Property Chains")
    method_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.METHOD
    ]
    property_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.PROPERTY
    ]
    chains_found: List[Dict[str, Any]] = []
    for method in method_nodes[:5]:
        for prop in property_nodes[:5]:
            paths = ontology.infer_path(method, prop, max_depth=3)
            if paths:
                chains_found.append({
                    "Method": method,
                    "Property": prop,
                    "Path Length": len(paths[0]),
                    "Path": " → ".join(paths[0]),
                })
    if chains_found:
        st.dataframe(pd.DataFrame(chains_found), use_container_width=True)
    else:
        st.info("No direct inference chains found. Build graph with more concepts.")
    st.subheader("📚 Synonym Resolution Examples")
    synonym_examples = [
        ("glen's law", "glens_law"),
        ("arrhenius", "arrhenius_factor"),
        ("full stokes", "full_stokes"),
        ("ssa", "ssa_flow"),
        ("smb", "surface_mass_balance"),
    ]
    syn_data: List[Dict[str, Any]] = []
    for original, expected in synonym_examples:
        resolved = ontology.resolve_concept(original)
        syn_data.append({
            "Original": original,
            "Expected": expected,
            "Resolved": resolved,
            "Match": (
                "✅" if resolved == expected
                else ("⚠️" if resolved else "❌")
            ),
        })
    st.dataframe(pd.DataFrame(syn_data), use_container_width=True)
    st.subheader("🏛️ Concept Hierarchy")
    hierarchy_data: List[Dict[str, str]] = []
    for concept in valid_concepts[:20]:
        if concept in ontology.concepts:
            node = ontology.concepts[concept]
            if node.hypernyms:
                for hyp in node.hypernyms:
                    hierarchy_data.append({
                        "Child": concept, "Parent": hyp,
                        "Relation": "is-a",
                    })
            if node.hyponyms:
                for hyp in node.hyponyms:
                    if hyp in valid_concepts:
                        hierarchy_data.append({
                            "Parent": concept, "Child": hyp,
                            "Relation": "has-subtype",
                        })
    if hierarchy_data:
        st.dataframe(
            pd.DataFrame(hierarchy_data), use_container_width=True,
        )
    else:
        st.info("No hierarchical relationships found in current concept set.")


# ============================================================================
# BATCH PROCESSING MODE v6.0 (Streamlit Cloud ≤ 1 GB RAM)
# ============================================================================
# ============================================================================
# MICROTRANSFORMER #2: UI RENDERING (with Glacier expert labels)
# ============================================================================
def render_microtransformer_postprocessing_panel() -> Dict[str, Any]:
    """Postprocessing customization panel for Microtransformer visualizations."""
    with st.expander("🎨 Microtransformer Visualization Customization", expanded=False):
        st.markdown("**Colormap & Theme**")
        col1, col2 = st.columns(2)
        with col1:
            mt_cmap = st.selectbox("Colormap (Postprocessing)", options=list(SUPPORTED_COLORMAPS.keys()),
                index=list(SUPPORTED_COLORMAPS.keys()).index(st.session_state.get("mt_cmap", "viridis")) if st.session_state.get("mt_cmap", "viridis") in SUPPORTED_COLORMAPS else 0, key="mt_post_cmap")
            st.session_state["mt_cmap"] = mt_cmap
        with col2:
            st.checkbox("Reverse Colormap", key="mt_post_cmap_reverse")

        st.markdown("**Typography**")
        col3, col4 = st.columns(2)
        with col3: st.slider("Font Size", 8, 24, 11, key="mt_post_font_size")
        with col4: st.slider("Title Size", 12, 32, 15, key="mt_post_title_size")

        st.markdown("**Sankey & Bar Settings**")
        col5, col6, col7 = st.columns(3)
        with col5: st.slider("Sankey Node Padding", 5, 50, 15, key="mt_post_sankey_pad")
        with col6: st.slider("Sankey Link Opacity", 0.1, 1.0, 0.4, key="mt_post_sankey_opacity")
        with col7: st.selectbox("Bar Text Position", ["outside", "inside", "auto", "none"], key="mt_post_bar_text_pos")

    return {
        "cmap": st.session_state.get("mt_cmap", "viridis"),
        "font_size": st.session_state.get("mt_post_font_size", 11),
        "title_size": st.session_state.get("mt_post_title_size", 15),
        "font_family": st.session_state.get("mt_font_family", "Inter, Segoe UI, Roboto, sans-serif"),
        "sankey_pad": st.session_state.get("mt_post_sankey_pad", 15),
        "sankey_opacity": st.session_state.get("mt_post_sankey_opacity", 0.4),
        "bar_text_position": st.session_state.get("mt_post_bar_text_pos", "outside"),
    }

def render_chord_diagram(token_labels: List[str], routing_np: np.ndarray,
                         scale: List[str], theme: Dict) -> go.Figure:
    """
    Render a chord diagram showing token → expert flow.
    Falls back to a matrix heatmap if the chord library is unavailable.
    """
    try:
        from chord import Chord
        n_tokens, n_experts = routing_np.shape
        matrix = np.zeros((n_tokens + n_experts, n_tokens + n_experts))
        for i, token in enumerate(token_labels):
            top_indices = np.argsort(routing_np[i])[-3:][::-1]
            for j in top_indices:
                matrix[i, n_tokens + j] = routing_np[i][j]

        names = [f"T: {t}" for t in token_labels] + GLACIER_EXPERT_LABELS
        chord_fig = Chord(matrix, names=names, colors=scale)
        fig = chord_fig.plotly_chord()
        fig = apply_mt_chart_style(fig, theme, is_axial=False)
        fig.update_layout(
            title_text="Token → Expert Routing Chord Diagram",
            height=550,
            margin=dict(l=80, r=80, t=80, b=80),
        )
        return fig
    except ImportError:
        # Fallback: arc-style heatmap using Plotly
        n_tokens, n_experts = routing_np.shape
        matrix = np.zeros((n_tokens, n_experts))
        for i in range(n_tokens):
            top_indices = np.argsort(routing_np[i])[-3:][::-1]
            for j in top_indices:
                matrix[i, j] = routing_np[i][j]

        fig = px.imshow(
            matrix,
            x=GLACIER_EXPERT_LABELS,
            y=token_labels,
            labels=dict(x="Expert Domain", y="Path Token", color="Activation"),
            color_continuous_scale=scale,
            aspect="auto",
            height=450,
            title="Token → Expert Routing Matrix (Chord fallback)",
        )
        fig = apply_mt_chart_style(fig, theme, is_axial=True)
        return fig


def render_microtransformer_kg_rag_tab(analysis_data: Dict, ontology: DomainOntology):
    st.subheader("🧠 Microtransformer #2: LatentMoE KG‑RAG Extractor")
    st.markdown("""
    Encodes graph traversals (node‑edge‑node) using **Latent Mixture of Experts**.
    Pick a source and target concept – the shortest path is tokenised and fed through
    32 specialised latent domains for glacial mechanics (Elmer/Ice).
    """)

    if not analysis_data or "nx_graph" not in analysis_data:
        st.info("Please build the concept graph first.")
        return

    ontology = ensure_ontology_populated()
    nx_graph = analysis_data["nx_graph"]
    concept_to_id = analysis_data["concept_to_id"]
    num_nodes = len(concept_to_id)

    if num_nodes < 2:
        st.error("Graph has fewer than 2 nodes.")
        return

    type_order = [
        ConceptType.MATERIAL, ConceptType.PARAMETER, ConceptType.PHENOMENON,
        ConceptType.PROPERTY, ConceptType.PROCESS, ConceptType.METHOD,
        ConceptType.MODEL, ConceptType.MICROSTRUCTURE, ConceptType.GENERAL,
    ]
    type_labels = {
        ConceptType.MATERIAL: "📦 Materials",
        ConceptType.PARAMETER: "🎛️ Parameters",
        ConceptType.PHENOMENON: "⚡ Phenomena",
        ConceptType.PROPERTY: "📊 Properties",
        ConceptType.PROCESS: "🔥 Processes",
        ConceptType.METHOD: "🔬 Methods",
        ConceptType.MODEL: "🧠 Models",
        ConceptType.MICROSTRUCTURE: "🏗️ Microstructure",
        ConceptType.GENERAL: "📋 General",
    }

    graph_concepts = set(concept_to_id.keys())
    ontology_concepts = set(ontology.concepts.keys())
    all_available = sorted(graph_concepts | ontology_concepts)
    in_graph = {c for c in all_available if c in concept_to_id}

    grouped_options = []
    for ctype in type_order:
        members = [c for c in all_available if ontology.get_concept_type(c) == ctype]
        if members:
            grouped_options.append(f"--- {type_labels.get(ctype, ctype.value)} ---")
            for m in members:
                tag = " ✅" if m in in_graph else " ⚠️ not in graph"
                grouped_options.append(f"{m}{tag}")

    separator_set = {s for s in grouped_options if s.startswith("--- ")}

    def strip_tag(option_str: Optional[str]) -> str:
        if not option_str:
            return option_str
        for marker in [" ✅", " ⚠️ not in graph"]:
            if option_str.endswith(marker):
                return option_str[:-len(marker)]
        return option_str

    QUERY_CONCEPT_MAP = [
        (r"\bglen.*\barrhenius\b", "glens_law", "arrhenius_factor"),
        (r"\barrhenius.*\bglen\b", "arrhenius_factor", "glens_law"),
        (r"\bfriction.*\bpressure\b", "coulomb_friction", "effective_pressure"),
        (r"\bpressure.*\bfriction\b", "effective_pressure", "coulomb_friction"),
        (r"\bstokes.*\bparstokes\b", "full_stokes", "parstokes"),
        (r"\bparstokes.*\bstokes\b", "parstokes", "full_stokes"),
        (r"\bcalving.*\bcrevasse\b", "calving_3d", "crevasse_depth"),
        (r"\bcrevasse.*\bcalving\b", "crevasse_depth", "calving_3d"),
        (r"\benthalpy.*\btemperature\b", "enthalpy_solver", "temperature_solver"),
        (r"\btemperature.*\benthalpy\b", "temperature_solver", "enthalpy_solver"),
    ]

    quick_query = st.text_input(
        "Or type a natural‑language question:",
        value=st.session_state.get('synced_query', "How does Coulomb friction affect grounding line dynamics?"),
        key="mt_quick_query",
        placeholder="e.g., How does enthalpy solver couple with temperature?",
    )

    nlp_src, nlp_tgt = None, None
    if quick_query.strip():
        q_lower = quick_query.lower()
        for pattern, src_concept, tgt_concept in QUERY_CONCEPT_MAP:
            if re.search(pattern, q_lower):
                if src_concept in all_available:
                    nlp_src = src_concept
                if tgt_concept in all_available:
                    nlp_tgt = tgt_concept
                break

    def find_option_index(concept_name: str) -> int:
        for i, opt in enumerate(grouped_options):
            if strip_tag(opt) == concept_name:
                return i
        return 0

    if nlp_src and nlp_tgt:
        default_src_idx = find_option_index(nlp_src)
        default_tgt_idx = find_option_index(nlp_tgt)
    else:
        default_src_idx = find_option_index("coulomb_friction")
        default_tgt_idx = find_option_index("grounding_line_dynamics")

    col1, col2 = st.columns(2)
    with col1:
        src_option = st.selectbox(
            "Source Concept:",
            options=grouped_options,
            index=default_src_idx,
            key="mt_src_select",
        )
    with col2:
        tgt_option = st.selectbox(
            "Target Concept:",
            options=grouped_options,
            index=default_tgt_idx,
            key="mt_tgt_select",
        )

    selected_src = None if src_option in separator_set else strip_tag(src_option)
    selected_tgt = None if tgt_option in separator_set else strip_tag(tgt_option)

    if nlp_src and nlp_tgt:
        st.success(f"🧠 NLP parsed: **{nlp_src}** → **{nlp_tgt}** (dropdowns auto-set)")

    if selected_src and selected_tgt and selected_src == selected_tgt:
        st.error("Source and target must be different concepts.")
        selected_src = None

    torch.manual_seed(int(st.session_state.get("mt_seed", 42)))
    kg_model = LatentMoEKGExtractor(num_nodes, NUM_EDGE_TYPES)
    kg_model.eval()

    xi = st.sidebar.slider(
        "Router prior strength ξ (QDWA)",
        0.0, 2.0, 1.0, 0.25, key="mt_prior_xi"
    )
    router_prior = None
    la = st.session_state.get("last_query_analysis")
    if xi > 0 and la is not None and getattr(la, "category_weights", None):
        prior = np.array([
            math.log(max(la.category_weights.get(EXPERT_CATEGORY[i], 0.25), 1e-4))
            for i in range(32)
        ])
        router_prior = torch.tensor(xi * prior, dtype=torch.float32)

    st.markdown("---")
    st.markdown("#### 🔬 Expert Routing Activation Analysis")
    with st.expander("🎨 Chart Customization (colormap, fonts, colorbar)", expanded=False):
        _cmaps = list(SUPPORTED_COLORMAPS.keys())
        st.selectbox("Colormap:", options=_cmaps,
                     index=_cmaps.index(st.session_state.get("mt_cmap", "viridis"))
                     if st.session_state.get("mt_cmap", "viridis") in _cmaps else 0,
                     key="mt_cmap",
                     help="Sequential (viridis/inferno/turbo) best for heatmaps; jet/rainbow are popular but not colorblind-safe.")
        st.selectbox("Font family (labels, ticks, colorbar):",
                     ["Inter, Segoe UI, Roboto, sans-serif", "Arial, Helvetica, sans-serif",
                      "Georgia, serif", "Courier New, monospace", "Times New Roman, serif"],
                     key="mt_font_family")
        c1, c2, c3, c4 = st.columns(4)
        c1.slider("Tick font size", 8, 20, 11, key="mt_font_size")
        c2.slider("Title font size", 10, 26, 15, key="mt_title_size")
        c3.slider("Colorbar length", 0.3, 1.0, 0.8, 0.05, key="mt_cbar_len")
        c4.slider("Colorbar thickness (px)", 6, 40, 14, key="mt_cbar_thick")
        st.text_input("Colorbar title", value="Weight", key="mt_cbar_title")
        st.checkbox("Show gridlines", value=False, key="mt_show_grid")
        st.number_input("Torch seed (reproducible demo)", 0, 9999, 42, key="mt_seed")

    if st.button("⚡ Run LatentMoE Inference on Path", type="primary"):
        if not selected_src or not selected_tgt:
            st.warning("Please select both source and target.")
        elif selected_src == selected_tgt:
            st.warning("Source and target must be different.")
        else:
            path = None
            if selected_src in concept_to_id and selected_tgt in concept_to_id:
                if selected_src in nx_graph and selected_tgt in nx_graph:
                    try:
                        path = nx.shortest_path(nx_graph, source=selected_src, target=selected_tgt)
                        st.success(f"✅ Path found in graph: {' → '.join(path)}")
                    except nx.NetworkXNoPath:
                        pass

            if path is None:
                ontology_paths = ontology.infer_path(selected_src, selected_tgt, max_depth=3)
                if ontology_paths:
                    path = ontology_paths[0]
                    st.info(f"🔍 Ontology inference path: {' → '.join(path)}")
                else:
                    st.error(f"No path found between `{selected_src}` and `{selected_tgt}`.")
                    path = None

            if path is None:
                st.session_state.pop("mt_last_run", None)
            else:
                st.success(f"🚀 Processing Path: {' → '.join(path)}")
                node_indices = []
                ontology_only_tokens = []
                for n in path:
                    if n in concept_to_id:
                        node_indices.append(concept_to_id[n])
                    else:
                        node_indices.append(0)
                        ontology_only_tokens.append(n)
                node_seq = torch.tensor([node_indices], dtype=torch.long)

                edge_indices = []
                for i in range(len(path) - 1):
                    edge_data = nx_graph.get_edge_data(path[i], path[i+1])
                    if edge_data:
                        rel_str = edge_data.get('edge_type', 'semantic').upper()
                        edge_indices.append(RELATIONSHIP_TO_IDX.get(rel_str, 0))
                    else:
                        rel = next((r for r in ontology.relationships if r.source == path[i] and r.target == path[i+1]), None)
                        if rel:
                            edge_indices.append(RELATIONSHIP_TO_IDX.get(rel.rel_type.name, 0))
                        else:
                            edge_indices.append(RELATIONSHIP_TO_IDX.get("INFLUENCES", 0))
                edge_seq = torch.tensor([edge_indices], dtype=torch.long)

                if ontology_only_tokens:
                    st.caption(f"⚠️ Tokens {ontology_only_tokens} are ontology-only (node ID 0).")

                with torch.no_grad():
                    out, routing_weights = kg_model(node_seq, edge_seq, router_prior=router_prior)

                avg_weights = routing_weights.mean(dim=1).squeeze(0).numpy()
                st.session_state["mt_last_run"] = {
                    "path": path,
                    "routing": routing_weights.squeeze(0).numpy(),
                    "avg": avg_weights,
                }

    post_params = render_microtransformer_postprocessing_panel()

    run = st.session_state.get("mt_last_run")
    if run:
        path = run["path"]
        routing_np = run["routing"]
        avg_weights = run["avg"]
        token_labels = [n.replace("_", " ").title() for n in path]
        theme = THEME_PRESETS.get(st.session_state.get("theme", "Bright (Default)"), THEME_PRESETS["Bright (Default)"])
        scale = plotly_continuous_scale(post_params.get("cmap", "viridis"))

        st.markdown("#### 📊 Per‑Token Expert Routing Heatmap")
        per_token_df = pd.DataFrame(routing_np, index=token_labels, columns=GLACIER_EXPERT_LABELS)

        hover_text = []
        for i, token in enumerate(token_labels):
            token_row = []
            for j, expert in enumerate(GLACIER_EXPERT_LABELS):
                token_row.append(
                    f"<b>Token:</b> {token}<br>"
                    f"<b>Expert:</b> {expert}<br>"
                    f"<b>Weight:</b> {routing_np[i][j]:.4f}<br>"
                    f"<b>Rank:</b> {np.argsort(np.argsort(-routing_np[i]))[j] + 1}/32"
                )
            hover_text.append(token_row)

        fig_heat = px.imshow(
            per_token_df.T,
            labels=dict(x="Path Token", y="Expert Domain", color="Activation"),
            color_continuous_scale=scale,
            aspect="auto",
            height=450
        )
        fig_heat.update_traces(hoverinfo="text", text=hover_text, customdata=hover_text)
        fig_heat.update_layout(hovermode="closest")
        fig_heat.update_layout(
            font=dict(family=post_params.get("font_family"), size=post_params.get("font_size", 11), color=theme["font"]),
            title=dict(font=dict(size=post_params.get("title_size", 15)))
        )
        st.plotly_chart(apply_mt_chart_style(fig_heat, theme), use_container_width=True)

        st.markdown("#### 🌊 Token-to-Expert Routing Flow")
        chart_type = st.radio("Flow chart type:", ["Sankey", "Chord"], index=0, horizontal=True)

        if chart_type == "Sankey":
            sanky_tokens = [f"Token: {t}" for t in token_labels]
            sanky_experts = GLACIER_EXPERT_LABELS
            sankey_labels = sanky_tokens + sanky_experts

            sanky_sources = []
            sanky_targets = []
            sanky_values = []

            for i, token in enumerate(sanky_tokens):
                token_weights = routing_np[i]
                top_indices = np.argsort(token_weights)[-3:][::-1]
                for idx in top_indices:
                    val = float(token_weights[idx])
                    if val > 0:
                        sanky_sources.append(sankey_labels.index(token))
                        sanky_targets.append(sankey_labels.index(sanky_experts[idx]))
                        sanky_values.append(val)

            token_colors = px.colors.qualitative.Pastel[:len(sanky_tokens)]
            expert_colors = scale[:len(sanky_experts)]
            node_colors = token_colors + expert_colors

            link_colors = []
            for s in sanky_sources:
                hex_color = node_colors[s]
                if isinstance(hex_color, str) and hex_color.startswith('rgb('):
                    rgb_vals = hex_color[4:-1].split(',')
                    link_colors.append(f"rgba({rgb_vals[0].strip()},{rgb_vals[1].strip()},{rgb_vals[2].strip()},0.4)")
                elif isinstance(hex_color, str) and hex_color.startswith('#') and len(hex_color) == 7:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    link_colors.append(f"rgba({r},{g},{b},0.4)")
                else:
                    link_colors.append("rgba(100,100,100,0.4)")

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=post_params.get("sankey_pad", 15),
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=sankey_labels,
                    color=node_colors
                ),
                link=dict(
                    source=sanky_sources,
                    target=sanky_targets,
                    value=sanky_values,
                    color=link_colors
                )
            )])

            fig_sankey.update_layout(
                title_text="LatentMoE Routing Flow (Top 3 Experts per Token)",
                height=500,
                font=dict(
                    family=post_params.get("font_family", "Inter, Segoe UI, Roboto, sans-serif"),
                    size=max(11, post_params.get("font_size", 11) - 1),
                    color=theme.get("font", "#000000")
                )
            )
            st.plotly_chart(apply_mt_chart_style(fig_sankey, theme, is_axial=False), use_container_width=True)
        else:
            fig_chord = render_chord_diagram(token_labels, routing_np, scale, theme)
            st.plotly_chart(apply_mt_chart_style(fig_chord, theme, is_axial=False), use_container_width=True)

        st.markdown("#### 📊 Averaged Expert Activation")
        df_experts = pd.DataFrame({
            "Expert Domain": GLACIER_EXPERT_LABELS,
            "Activation Weight": avg_weights,
        }).sort_values("Activation Weight", ascending=False)

        df_active = df_experts[df_experts["Activation Weight"] > 0.05].copy()
        if df_active.empty:
            st.warning("No experts exceed the 0.05 activation threshold — showing all experts instead.")
            df_active = df_experts.copy()

        fig = px.bar(
            df_active,
            x="Expert Domain",
            y="Activation Weight",
            title=f"Top Activated Experts: {' → '.join(token_labels)}",
            color="Activation Weight",
            color_continuous_scale=scale,
            text=df_active["Activation Weight"].apply(lambda x: f"{x:.3f}")
        )
        fig.update_traces(
            textposition=post_params.get("bar_text_position", "outside"),
            textfont_size=post_params.get("bar_text_size", 12)
        )
        y_max = float(df_active["Activation Weight"].max()) if not df_active.empty else 1.0
        fig.update_layout(
            xaxis_tickangle=-45,
            height=450,
            yaxis=dict(range=[0, y_max * 1.15]),
            font=dict(family=post_params.get("font_family"), size=post_params.get("font_size", 11), color=theme["font"])
        )
        st.plotly_chart(apply_mt_chart_style(fig, theme), use_container_width=True)

        st.markdown("**Top Activated Experts:**")
        top_experts = df_experts.head(4)
        cols = st.columns(4)
        for i, (_, row) in enumerate(top_experts.iterrows()):
            with cols[i]:
                st.metric(row["Expert Domain"], f"{row['Activation Weight']:.3f}")

        st.markdown("#### 🔬 Scientific Interpretation")
        interpretation_map = {
            "Glen's Flow Law": "Non-linear rheology for ice deformation; key for accurate velocity fields.",
            "Coulomb Friction": "Water-pressure dependent basal sliding; critical for grounding line dynamics.",
            "ParStokes Solver": "Parallel Stokes solver enabling 3D full-Stokes simulations of ice flow.",
            "Enthalpy/Temp Solver": "Thermodynamic solver handling temperate ice and cold-temperate transition.",
            "Calving": "Crevasse-depth based calving models; drives front retreat and ice shelf evolution.",
            "Surface Mass Balance": "Accumulation/ablation forcing; dictates ice sheet mass budget.",
        }
        top3_names = df_experts.head(3)["Expert Domain"].tolist()
        for expert_name in top3_names:
            interp = interpretation_map.get(expert_name, "Routing capacity allocated to decode this ice dynamics domain.")
            st.info(f"**{expert_name}**: {interp}")

        st.markdown("#### 🔗 Reasoning Chain Along Path")
        for i in range(len(path) - 1):
            src_name = path[i].replace("_", " ").title()
            tgt_name = path[i + 1].replace("_", " ").title()
            rel_desc = "unknown"
            for r in ontology.relationships:
                if r.source == path[i] and r.target == path[i + 1]:
                    rel_desc = f"{r.rel_type.value} (confidence: {r.confidence:.2f})"
                    break
            token_experts = pd.DataFrame({"Expert": GLACIER_EXPERT_LABELS, "Weight": routing_np[i]}).sort_values("Weight", ascending=False)
            top2 = ", ".join(f"{row['Expert']} ({row['Weight']:.3f})" for _, row in token_experts.head(2).iterrows())
            st.markdown(f"**Step {i+1}**: `{src_name}` --[{rel_desc}]--> `{tgt_name}`  *(top experts: {top2})*")


def get_memory_usage_mb() -> float:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return 0.0


def split_into_batches(
    df: pd.DataFrame, batch_size: int
) -> Iterator[Tuple[int, pd.DataFrame]]:
    total_batches = math.ceil(len(df) / batch_size)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        yield start_idx, df.iloc[start_idx:end_idx]


def merge_graphs(existing_graph: nx.Graph, new_graph: nx.Graph) -> nx.Graph:
    merged = existing_graph
    for node, data in new_graph.nodes(data=True):
        if node in merged:
            merged.nodes[node]["frequency"] = (
                merged.nodes[node].get("frequency", 0)
                + data.get("frequency", 0)
            )
            for attr in ("concept_type", "definition"):
                if not merged.nodes[node].get(attr) and data.get(attr):
                    merged.nodes[node][attr] = data[attr]
        else:
            merged.add_node(node, **data)
    for u, v, data in new_graph.edges(data=True):
        if merged.has_edge(u, v):
            ed = merged[u][v]
            ed["cooccurrence"] = (
                ed.get("cooccurrence", 0) + data.get("cooccurrence", 0)
            )
            ed["semantic"] = max(
                ed.get("semantic", 0) or 0, data.get("semantic", 0) or 0
            )
            ed["inferred"] = bool(ed.get("inferred", False)) or bool(
                data.get("inferred", False)
            )
            if data.get("confidence") is not None:
                ed["confidence"] = max(
                    ed.get("confidence", 0), data["confidence"]
                )
            if data.get("path") and not ed.get("path"):
                ed["path"] = data["path"]
            if (
                ed.get("edge_type", "cooccurrence") == "cooccurrence"
                and data.get("edge_type") not in (None, "cooccurrence")
            ):
                ed["edge_type"] = data["edge_type"]
        else:
            merged.add_edge(u, v, **data)
    return merged


def recompute_edge_weights(nx_graph: nx.Graph, config: Dict) -> None:
    cooc_w = config.get("COOCCURRENCE_WEIGHT", 0.7)
    sem_w = config.get("SEMANTIC_WEIGHT", 0.2)
    inf_w = config.get("INFERENCE_WEIGHT", 0.1)
    for _, _, data in nx_graph.edges(data=True):
        cooc = data.get("cooccurrence", 0)
        sem = data.get("semantic", 0) or 0
        inf = 1.0 if data.get("inferred", False) else 0.0
        conf = data.get("confidence", 0.5)
        data["weight"] = cooc_w * cooc + sem_w * sem + inf_w * inf * conf


def extract_doc_metrics(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    # Extract potential glaciology-relevant numerical values
    vel_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:m/a|m yr-1|m yr\^-1)', text, re.I)
    if vel_matches:
        metrics['velocity_m_per_yr'] = [float(m) for m in vel_matches]
    temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', text, re.I)
    if temp_matches:
        metrics['temperature'] = [float(m) for m in temp_matches]
    press_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mpa|kpa|pa)', text, re.I)
    if press_matches:
        metrics['pressure'] = [float(m) for m in press_matches]
    return metrics


class IncrementalGraphBuilder(ReasoningEnhancedGraphBuilder):
    @timed
    def build_batch_graph(
        self,
        batch_concepts: List[List[str]],
        valid_concepts: List[str],
        concept_to_id: Dict[str, int],
        batch_doc_freq: Dict[str, int],
        embed_model=None,
        config: Dict = None,
    ) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(1000)
        nx_graph = nx.Graph()
        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            definition = self.ontology.get_definition(c)
            nx_graph.add_node(
                c,
                frequency=batch_doc_freq.get(c, 0),
                concept_type=concept_type.value,
                definition=definition,
                degree=0,
            )
        cooccurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for concepts in batch_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if u != v:
                        key = tuple(sorted([u, v]))
                        cooccurrence_map[key] += 1
        for (u, v), count in cooccurrence_map.items():
            nx_graph.add_edge(
                u, v,
                weight=float(count),
                cooccurrence=count,
                semantic=0.0,
                edge_type='cooccurrence',
                inferred=False,
            )
        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(
                nx_graph, valid_concepts, embed_model, config
            )
        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)
        self._add_hierarchical_edges(nx_graph, valid_concepts)
        self._compute_final_weights(nx_graph, config)
        return nx_graph


def reset_batch_state(clear_analysis: bool = False) -> None:
    st.session_state.batch_state = None
    st.session_state.pop("batch_trigger", None)
    if clear_analysis:
        st.session_state.analysis_data = None
        st.session_state.burst_df = None
        st.session_state.drift_df = None
        st.session_state.genealogy_df = None
        st.session_state.bridge_df = None
        st.session_state.motifs = {}
        st.session_state.edit_history = GraphEditHistory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_batch_processing_controls() -> None:
    st.markdown("---")
    st.subheader("📦 Batch Processing (≤1 GB RAM)")
    st.toggle(
        "Enable batch processing",
        key="batch_mode",
        help=(
            "Process documents in small batches with incremental graph "
            "merging and memory cleanup after each batch. Recommended for "
            "Streamlit Cloud free tier (1 GB RAM)."
        ),
    )
    if not st.session_state.get("batch_mode", False):
        return
    st.slider(
        "Batch size (documents)", 100, 2000, 1000, 100,
        key="batch_size",
        help="Smaller batches = lower peak memory but more merge steps.",
    )
    st.slider(
        "GNN epochs (final training)", 10, 50, 40, 5,
        key="batch_gnn_epochs",
        help="GNN is trained ONCE on the final merged graph.",
    )
    bs = st.session_state.get("batch_state")
    if bs:
        total = max(bs.get("total_batches", 1), 1)
        done = bs.get("next_batch", 0)
        st.progress(done / total)
        st.caption(
            f"Batch {done}/{total} • "
            f"{bs.get('docs_processed', len(bs.get('all_texts', {})))} "
            f"docs processed • "
            f"{len(bs.get('all_texts', {}))} texts cached"
        )
    col_next, col_all = st.columns(2)
    with col_next:
        if st.button(
            "▶️ Next batch", use_container_width=True,
            disabled=bool(bs and bs.get("done")),
        ):
            st.session_state["batch_trigger"] = "next"
    with col_all:
        if st.button(
            "⏩ All remaining", use_container_width=True,
            disabled=bool(bs and bs.get("done")),
        ):
            st.session_state["batch_trigger"] = "all"
    if bs:
        if st.button("🗑️ Reset batch state", use_container_width=True):
            reset_batch_state(clear_analysis=True)
            st.success("Batch state cleared!")
            st.rerun()
    else:
        st.caption(
            "Click 🚀 Build Concept Graph (or ▶️ Next batch) to start."
        )


BATCH_TEXT_STORE_CAP = 4000


def build_query_whitelist(st_session):
    if not st_session.get('query_focused_build', False):
        return None
    analysis = st_session.get('last_query_analysis')
    if analysis is None:
        return None
    whitelist = set(analysis.explicitly_mentioned)
    whitelist.update(analysis.inferred_concepts)
    whitelist.update(st_session.get('last_query_dynamic_concepts', set()))
    whitelist.update(st_session.get('last_query_bridge_concepts', {}))

    pdef = GLACIER_PROBLEM_DEFINITIONS.get(analysis.primary_problem)
    if pdef is not None:
        whitelist.update(pdef.key_concepts)
        whitelist.update(pdef.relevant_descriptors)
        whitelist.update(pdef.relevant_phenomena)
        whitelist.update(pdef.relevant_properties)
        for src, _rel, tgt in pdef.key_relationships:
            whitelist.update([src, tgt])

    ontology = st_session.get('ontology')
    if ontology is not None:
        whitelist = {c for c in whitelist if c in ontology.concepts}

    if len(whitelist) < 8 and ontology is not None:
        extra = set()
        for c in list(whitelist):
            node = ontology.concepts.get(c)
            if node:
                extra.update(node.hypernyms)
                extra.update(node.hyponyms)
                extra.update(node.related_processes)
                extra.update(node.related_properties)
        for r in ontology.relationships:
            if r.source in whitelist:
                extra.add(r.target)
            elif r.target in whitelist:
                extra.add(r.source)
        whitelist.update({c for c in extra if c in ontology.concepts})

    W = getattr(analysis, "category_weights", {})
    if (
        W
        and getattr(analysis, "concentration", 0.0) >= 0.5
        and ontology is not None
    ):
        k_star = max(W, key=W.get)
        whitelist.update(
            c
            for c in ontology.concepts
            if categorize_glacier_concept(c) == k_star
        )

    return whitelist


def run_batch_analysis(
    df_filtered: pd.DataFrame,
    selected_text_cols: List[str],
    ontology: DomainOntology,
    run_mode: str = "all",
) -> None:
    overall_start = time.perf_counter()
    metrics = get_metrics_logger()
    metrics.reset()
    batch_size = int(st.session_state.get("batch_size", 1000))
    total_docs = len(df_filtered)
    if total_docs == 0:
        st.error("No documents to process.")
        return
    total_batches = math.ceil(total_docs / batch_size)
    metrics.set("mode", "batch")
    metrics.set("total_docs", total_docs)
    live_panel = st.empty()
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

    data_hash = hashlib.md5(
        (
            f"{total_docs}|{'|'.join(selected_text_cols)}|"
            f"{df_filtered.index.min()}|{df_filtered.index.max()}"
        ).encode("utf-8")
    ).hexdigest()

    bs = st.session_state.get("batch_state")
    if bs is not None and (
        bs.get("data_hash") != data_hash
        or bs.get("batch_size") != batch_size
    ):
        st.info("Dataset or batch size changed — resetting batch state.")
        reset_batch_state(clear_analysis=False)
        bs = None
    if bs is None:
        bs = {
            "data_hash": data_hash,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "next_batch": 0,
            "all_concepts": [],
            "all_metrics": [],
            "all_texts": {},
            "valid_doc_indices": set(),
            "docs_processed": 0,
            "concept_freq": defaultdict(int),
            "concept_abstract_map": defaultdict(list),
            "merged_graph": None,
            "extractor": None,
            "resolver": None,
            "builder": None,
            "done": False,
        }
        st.session_state.batch_state = bs

    if bs["done"]:
        st.success("✅ All batches already processed — see results below.")
        return

    config = get_adaptive_config(total_docs)
    config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)

    whitelist = build_query_whitelist(st.session_state)
    is_query_focused = st.session_state.get('query_focused_build', False) and whitelist is not None and len(whitelist) > 0
    if is_query_focused:
        st.info(f"🎯 Query-focused batch mode: {len(whitelist)} concepts whitelisted. MIN_CONCEPT_FREQ lowered.")
        config["MIN_CONCEPT_FREQ"] = 1 if len(whitelist) <= 15 else 2
    config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
    config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
    config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.7)
    config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.2)
    config["INFERENCE_WEIGHT"] = st.session_state.get('inf_weight', 0.1)

    use_ontology = st.session_state.get('use_ontology', True)
    embed_model = load_embedding_model()

    if use_ontology and bs["extractor"] is None:
        with st.spinner("Initializing ontology resolver (one-time)..."):
            resolver = AdvancedConceptResolver(
                ontology, embed_model, cache_max=2000,
            )
            extractor = EnhancedConceptExtractor(
                ontology, resolver,
                store_contexts=False, store_documents=False,
            )
            builder = IncrementalGraphBuilder(ontology, extractor)
            bs["resolver"] = resolver
            bs["extractor"] = extractor
            bs["builder"] = builder
            st.session_state.resolver = resolver
            st.session_state.extractor = extractor
        gc.collect()

    pending = list(range(bs["next_batch"], total_batches))
    if run_mode == "next":
        pending = pending[:1]
    if not pending:
        st.success("✅ Nothing left to process.")
        return

    progress_bar = st.progress(0.0)
    status = st.status("📦 Batch processing running...", expanded=True)

    def _process_one_batch(batch_num: int) -> None:
        start = batch_num * batch_size
        end = min(start + batch_size, total_docs)
        batch_df = df_filtered.iloc[start:end]
        n_this = len(batch_df)
        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        with status:
            st.write(
                f"📦 Batch {batch_num + 1}/{total_batches} — "
                f"docs {start}–{end - 1} ({n_this} docs)"
            )
        batch_concepts: List[List[str]] = []
        batch_metrics: List[Dict] = []
        batch_doc_freq: Dict[str, int] = defaultdict(int)
        extractor = bs["extractor"]

        for local_i, (_, row) in enumerate(batch_df.iterrows()):
            text = " ".join([
                str(row[col]) for col in selected_text_cols
                if col in row and pd.notna(row[col])
            ])
            if use_ontology and extractor is not None:
                concepts = extractor.extract_from_text(text, start + local_i, allowed_concepts=whitelist)
            else:
                concepts = extract_glacier_concepts_from_text(text)
            batch_concepts.append(concepts)
            batch_metrics.append(extract_doc_metrics(text))
            unique_concepts = set(concepts)
            for c in unique_concepts:
                batch_doc_freq[c] += 1
                bs["concept_freq"][c] += 1
                bs["concept_abstract_map"][c].append(start + local_i)
            has_valid = any(
                bs["concept_freq"].get(c, 0) >= min_freq
                for c in unique_concepts
            )
            if has_valid:
                bs["all_texts"][start + local_i] = (
                    text[:BATCH_TEXT_STORE_CAP]
                )
                bs["valid_doc_indices"].add(start + local_i)
            bs["docs_processed"] += 1
            del text
            if (local_i + 1) % 100 == 0 or (local_i + 1) == n_this:
                frac = (batch_num + (local_i + 1) / n_this) / total_batches
                progress_bar.progress(min(0.90 * frac, 0.90))
                with status:
                    st.write(f"  … {local_i + 1}/{n_this} docs extracted")

        bs["all_concepts"].extend(batch_concepts)
        bs["all_metrics"].extend(batch_metrics)

        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        top_n = config.get("TOP_N_CONCEPTS", 1000)
        batch_unique: Set[str] = set()
        for cs in batch_concepts:
            batch_unique.update(cs)
        batch_valid = [
            c for c in batch_unique
            if bs["concept_freq"].get(c, 0) >= min_freq
        ]
        batch_valid.sort(
            key=lambda c: bs["concept_freq"][c], reverse=True
        )
        batch_valid = batch_valid[:top_n]
        concept_to_id_batch = {c: i for i, c in enumerate(batch_valid)}

        if use_ontology and bs["builder"] is not None:
            batch_graph = bs["builder"].build_batch_graph(
                batch_concepts, batch_valid, concept_to_id_batch,
                batch_doc_freq, embed_model, config,
            )
        else:
            batch_graph = build_hybrid_graph(
                batch_concepts, batch_valid, concept_to_id_batch,
                embed_model, config, ontology,
            )

        if bs["merged_graph"] is None:
            bs["merged_graph"] = batch_graph
        else:
            bs["merged_graph"] = merge_graphs(bs["merged_graph"], batch_graph)
        recompute_edge_weights(bs["merged_graph"], config)
        bs["next_batch"] = batch_num + 1

        g = bs["merged_graph"]
        with status:
            st.write(
                f"✅ Batch {batch_num + 1} done — cumulative graph: "
                f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
                f"| peak RSS ≈ {get_memory_usage_mb():.0f} MB"
            )
        del batch_concepts, batch_metrics, batch_doc_freq
        del batch_graph, batch_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _finalize() -> None:
        merged = bs["merged_graph"]
        if merged is None or merged.number_of_nodes() == 0:
            st.error("No graph could be built from the processed batches.")
            return
        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        top_n = config.get("TOP_N_CONCEPTS", 1000)
        with status:
            st.write("🧩 Finalizing — selecting top concepts...")
        valid_concepts = [
            c for c, f in bs["concept_freq"].items() if f >= min_freq
        ]
        valid_concepts.sort(
            key=lambda c: len(bs["concept_abstract_map"].get(c, [])),
            reverse=True,
        )
        valid_concepts = valid_concepts[:top_n]
        if len(valid_concepts) < 5:
            st.error(
                "Too few concepts extracted. "
                "Try lowering frequency thresholds."
            )
            return
        valid_set = set(valid_concepts)
        drop_nodes = [n for n in merged.nodes() if n not in valid_set]
        merged.remove_nodes_from(drop_nodes)
        del drop_nodes
        concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
        id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
        concept_abstract_map = {
            c: bs["concept_abstract_map"][c] for c in valid_concepts
        }
        progress_bar.progress(0.90)

        with status:
            st.write("🔢 Generating node embeddings...")
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
            node_features = torch.tensor(embeddings, dtype=torch.float32)
            del embeddings
        except Exception:
            node_features = torch.randn(len(valid_concepts), 384)
        gc.collect()

        with status:
            st.write("🧠 Training GraphSAGE (final, once)...")
        pos_pairs, neg_pairs = sample_edges_for_training(
            merged, valid_concepts, concept_to_id, config, memory_safe=True,
        )
        epochs = int(st.session_state.get("batch_gnn_epochs", 40))

        def _gnn_progress(epoch, loss):
            frac = 0.90 + (epoch / max(epochs, 1)) * 0.05
            progress_bar.progress(min(frac, 0.95))
            if epoch % 10 == 0:
                with status:
                    st.write(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

        gnn_model, final_emb, adj_indices, adj_values = train_gnn(
            node_features, merged, concept_to_id,
            pos_pairs, neg_pairs, _gnn_progress, epochs=epochs,
        )
        del pos_pairs, neg_pairs, adj_indices, adj_values
        gc.collect()

        with status:
            st.write("🎯 Scoring research directions...")
        concept_properties: Dict[str, float] = {}
        all_metrics = bs["all_metrics"]
        for concept in valid_concepts:
            values: List[float] = []
            for idx in concept_abstract_map.get(concept, []):
                if idx < len(all_metrics):
                    for metric_values in all_metrics[idx].values():
                        values.extend(metric_values)
            concept_properties[concept] = (
                float(np.median(values)) if values else 0.0
            )
        X_feat: List[List[float]] = []
        y_target: List[float] = []
        for u, v in merged.edges():
            pu = concept_properties.get(u, 0)
            pv = concept_properties.get(v, 0)
            w = merged[u][v].get('weight', 1)
            X_feat.append([pu, pv, w])
            y_target.append(
                max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0
            )
        ridge = None
        if len(X_feat) > 5:
            ridge = Ridge(alpha=1.0).fit(
                np.array(X_feat), np.array(y_target)
            )
        top_scores = compute_research_direction_scores(
            gnn_model, node_features, final_emb, merged,
            valid_concepts, concept_properties, ridge, embed_model,
        )
        del X_feat, y_target, node_features
        gc.collect()

        with status:
            st.write("🧪 Distillation + advanced analytics...")
        distill_df = compute_concept_distillation(
            valid_concepts, concept_abstract_map, bs["all_texts"],
            max_docs_per_concept=30,
        )
        burst_df = None
        drift_df = None
        genealogy_df = None
        bridge_df = None
        motifs: Dict[str, Any] = {}
        try:
            burst_df = detect_keyword_bursts(
                df_filtered, valid_concepts,
                concept_abstract_map, selected_text_cols,
            )
            drift_df = detect_semantic_drift(
                df_filtered, valid_concepts,
                concept_abstract_map, selected_text_cols,
            )
            genealogy_df = build_concept_genealogy(
                merged, valid_concepts, concept_abstract_map,
            )
            bridge_df = detect_cross_domain_bridges(
                merged, valid_concepts, concept_abstract_map,
            )
            motifs = analyze_network_motifs(merged)
        except Exception as e:
            st.warning(f"Some analytics skipped: {e}")
        st.session_state.burst_df = burst_df
        st.session_state.drift_df = drift_df
        st.session_state.genealogy_df = genealogy_df
        st.session_state.bridge_df = bridge_df
        st.session_state.motifs = motifs
        gc.collect()

        analysis_data = {
            "valid_concepts": valid_concepts,
            "concept_to_id": concept_to_id,
            "id_to_concept": id_to_concept,
            "concept_abstract_map": concept_abstract_map,
            "nx_graph": merged,
            "concept_properties": concept_properties,
            "ridge": ridge,
            "top_scores": top_scores,
            "distill_df": distill_df,
            "gnn_model": gnn_model,
            "final_emb": final_emb,
            "embed_model": embed_model,
            "all_metrics": bs["all_metrics"],
            "all_texts": bs["all_texts"],
            "config": config,
            "df_filtered": df_filtered,
            "selected_text_cols": selected_text_cols,
            "batch_info": {
                "mode": "batch",
                "batch_size": batch_size,
                "total_batches": total_batches,
                "total_docs": total_docs,
            },
        }
        if use_ontology:
            analysis_data.update({
                "ontology": ontology,
                "resolver": bs["resolver"],
                "extractor": bs["extractor"],
                "graph_builder": bs["builder"],
                "reasoning_paths": (
                    bs["builder"].reasoning_paths if bs["builder"] else []
                ),
            })
        st.session_state.analysis_data = analysis_data
        st.session_state.edit_history = GraphEditHistory()
        st.session_state.edit_history.save_snapshot(
            merged, valid_concepts, concept_to_id,
            id_to_concept, concept_abstract_map,
        )
        bs["all_concepts"] = []
        bs["valid_doc_indices"] = set()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bs["done"] = True

    try:
        for b in pending:
            _process_one_batch(b)
        if bs["next_batch"] >= total_batches:
            with status:
                st.write("🏁 All batches processed — finalizing...")
            _finalize()
            total_time = time.perf_counter() - overall_start
            progress_bar.progress(1.0)
            status.update(
                label=(
                    f"Batch analysis complete! ({total_time:.1f}s, "
                    f"peak RSS ≈ {get_memory_usage_mb():.0f} MB)"
                ),
                state="complete", expanded=False,
            )
            st.success(
                f"✅ All {total_batches} batches processed in "
                f"{total_time:.1f}s — peak memory ≈ "
                f"{get_memory_usage_mb():.0f} MB"
            )
        else:
            status.update(
                label=(
                    f"Batch {bs['next_batch']}/{total_batches} complete"
                ),
                state="complete", expanded=False,
            )
            st.info(
                f"📦 {total_batches - bs['next_batch']} batch(es) remaining "
                f"— click ▶️ Next batch or ⏩ All remaining in the sidebar."
            )
    except Exception as e:
        st.error(f"Batch pipeline error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# LLM QUERY PANEL (Sidebar) and Q&A TAB (Main)
# ============================================================================
class GlacierProblem(Enum):
    RHEOLOGY_TUNING = "rheology_tuning"
    BASAL_SLIDING_INVERSION = "basal_sliding_inversion"
    GROUNDING_LINE_DYNAMICS = "grounding_line_dynamics"
    CALVING_REMESHING = "calving_remeshing"
    THERMODYNAMIC_COUPLING = "thermodynamic_coupling"
    GENERAL = "general"
    MULTI_PROBLEM = "multi_problem"

@dataclass
class GlacierProblemDefinition:
    problem_id: GlacierProblem
    title: str
    scientific_description: str
    root_cause: str
    key_concepts: List[str]
    key_relationships: List[Tuple[str, str, str]]
    solution_directions: List[str]
    relevant_descriptors: List[str]
    relevant_phenomena: List[str]
    relevant_properties: List[str]
    example_queries: List[str]
    visualization_focus: List[str]

    def get_ontology_concepts(self) -> Set[str]:
        concepts = set(self.key_concepts + self.relevant_descriptors +
                       self.relevant_phenomena + self.relevant_properties)
        for src, _, tgt in self.key_relationships:
            concepts.update([src, tgt])
        return concepts

GLACIER_PROBLEM_DEFINITIONS: Dict[GlacierProblem, GlacierProblemDefinition] = {
    GlacierProblem.RHEOLOGY_TUNING: GlacierProblemDefinition(
        problem_id=GlacierProblem.RHEOLOGY_TUNING,
        title="Ice Rheology and Damage Tuning",
        scientific_description="Optimizing Glen's flow law parameters, Arrhenius factor, and damage mechanics for accurate ice viscosity.",
        root_cause="Uncertainty in rate factor and enhancement due to temperature and fabric.",
        key_concepts=["glens_law", "arrhenius_factor", "damage_mechanics", "viscosity_model", "glen_enhancement_factor"],
        key_relationships=[("arrhenius_factor", "INFLUENCES", "glens_law"), ("damage_mechanics", "INFLUENCES", "glen_enhancement_factor")],
        solution_directions=["Parameter calibration from borehole data", "Enthalpy coupling", "Anisotropic GOLF integration"],
        relevant_descriptors=["critical_shear_rate"],
        relevant_phenomena=["fabric_evolution"],
        relevant_properties=["full_stokes"],
        example_queries=["How does temperature affect Glen's law rate factor?", "Impact of damage on ice viscosity"],
        visualization_focus=["rheology_comparison", "damage_maps"]
    ),
    GlacierProblem.GROUNDING_LINE_DYNAMICS: GlacierProblemDefinition(
        problem_id=GlacierProblem.GROUNDING_LINE_DYNAMICS,
        title="Grounding Line Migration and Contact Mechanics",
        scientific_description="Solving the contact problem between bedrock and ice to track grounding line evolution.",
        root_cause="Hydrostatic pressure and friction laws control the position of the grounding line.",
        key_concepts=["contact_problem", "grounding_line_dynamics", "full_stokes", "hydrostatic_loads"],
        key_relationships=[("contact_problem", "CONSTRAINS", "grounding_line_dynamics"), ("hydrostatic_loads", "INFLUENCES", "grounding_line_dynamics")],
        solution_directions=["Mesh adaptation", "Sub-element parameterization", "Friction law coupling"],
        relevant_descriptors=["water_pressure", "coulomb_friction"],
        relevant_phenomena=["basal_melt_parameterization"],
        relevant_properties=["ice_shelf_flow"],
        example_queries=["How does contact problem define grounding line migration?", "Effect of hydrostatic pressure on grounding line"],
        visualization_focus=["gl_migration", "contact_forces"]
    ),
    GlacierProblem.BASAL_SLIDING_INVERSION: GlacierProblemDefinition(
        problem_id=GlacierProblem.BASAL_SLIDING_INVERSION,
        title="Basal Sliding Inversion with Inverse Adjoint",
        scientific_description="Using inverse methods to infer basal friction coefficients from observed surface velocities.",
        root_cause="Basal friction (Coulomb/Weertman) is poorly constrained; data assimilation improves accuracy.",
        key_concepts=["inverse_adjoint", "coulomb_friction", "weertman_friction", "effective_pressure"],
        key_relationships=[("inverse_adjoint", "QUANTIFIES", "coulomb_friction"), ("effective_pressure", "INFLUENCES", "coulomb_friction")],
        solution_directions=["Adjoint data assimilation", "Regularization techniques", "Uncertainty quantification"],
        relevant_descriptors=["water_pressure", "geothermal_flux"],
        relevant_phenomena=["subglacial_cavities"],
        relevant_properties=["full_stokes"],
        example_queries=["How to invert for basal friction using adjoint?", "What data is needed for Coulomb friction calibration?"],
        visualization_focus=["friction_maps", "velocity_misfit"]
    ),
    GlacierProblem.CALVING_REMESHING: GlacierProblemDefinition(
        problem_id=GlacierProblem.CALVING_REMESHING,
        title="Calving and Mesh Adaptation",
        scientific_description="3D calving based on crevasse depth and subsequent mesh remeshing with GMSH.",
        root_cause="Crevasse penetration leads to iceberg detachment; mesh must adapt.",
        key_concepts=["calving_3d", "crevasse_depth", "gmsh_remeshing", "mesh_update", "free_surface_evolution"],
        key_relationships=[("crevasse_depth", "CAUSES", "calving_3d"), ("calving_3d", "CAUSES", "gmsh_remeshing")],
        solution_directions=["Crevasse depth criterion", "Mesh refinement", "Parallel remeshing"],
        relevant_descriptors=["surface_mass_balance"],
        relevant_phenomena=["basal_melt_parameterization"],
        relevant_properties=["ice_shelf_flow"],
        example_queries=["How does crevasse depth determine calving?", "Best remeshing strategy after calving"],
        visualization_focus=["calving_front", "mesh_quality"]
    ),
    GlacierProblem.THERMODYNAMIC_COUPLING: GlacierProblemDefinition(
        problem_id=GlacierProblem.THERMODYNAMIC_COUPLING,
        title="Thermodynamic Coupling: Enthalpy and Temperature",
        scientific_description="Coupling enthalpy solver with temperature and pressure melting point to capture cold-temperate ice.",
        root_cause="Temperature and enthalpy fields affect ice viscosity and phase change.",
        key_concepts=["enthalpy_solver", "temperature_solver", "pressure_melting_point", "full_stokes"],
        key_relationships=[("enthalpy_solver", "INFLUENCES", "pressure_melting_point"), ("temperature_solver", "INFLUENCES", "enthalpy_solver")],
        solution_directions=["Enthalpy method for temperate ice", "Cold-temperate transition", "Coupling with hydrology"],
        relevant_descriptors=["geothermal_flux"],
        relevant_phenomena=["basal_melt_parameterization"],
        relevant_properties=["viscosity_model"],
        example_queries=["How does enthalpy solver handle temperate ice?", "Coupling temperature with full-Stokes"],
        visualization_focus=["temperature_profiles", "enthalpy_maps"]
    ),
    GlacierProblem.GENERAL: GlacierProblemDefinition(
        problem_id=GlacierProblem.GENERAL,
        title="General Glacial Flow Inquiry",
        scientific_description="General question about ice dynamics and Elmer/Ice modeling.",
        root_cause="N/A",
        key_concepts=["full_stokes", "glens_law", "coulomb_friction", "enthalpy_solver"],
        key_relationships=[],
        solution_directions=[],
        relevant_descriptors=[],
        relevant_phenomena=[],
        relevant_properties=[],
        example_queries=["What are the main components of Elmer/Ice for glacier modeling?"],
        visualization_focus=["general_overview"]
    ),
    GlacierProblem.MULTI_PROBLEM: GlacierProblemDefinition(
        problem_id=GlacierProblem.MULTI_PROBLEM,
        title="Multi-Problem Glacier Inquiry",
        scientific_description="Inquiry spanning multiple core problems in glacial mechanics.",
        root_cause="N/A",
        key_concepts=[],
        key_relationships=[],
        solution_directions=[],
        relevant_descriptors=[],
        relevant_phenomena=[],
        relevant_properties=[],
        example_queries=[],
        visualization_focus=["multi_problem_comparison"]
    ),
}

# ============================================================================
# QDWA: QUERY DISTILLATION & WEIGHTED ALLOCATION (4-ROTOR STRATEGY)
# ============================================================================
FOUR_CATEGORIES = ["rheology", "sliding_hydro", "numerical", "dynamics_thermo"]
CATEGORY_DISPLAY = {
    "rheology": "Rheology & Material Physics",
    "sliding_hydro": "Basal Sliding & Hydrology",
    "numerical": "Numerical Solvers & Meshing",
    "dynamics_thermo": "Ice Dynamics & Thermodynamics",
}
_LABEL2KEY = {
    "Rheology & Material Physics": "rheology",
    "Basal Sliding & Hydrology": "sliding_hydro",
    "Numerical Solvers & Meshing": "numerical",
    "Ice Dynamics & Thermodynamics": "dynamics_thermo",
}

CATEGORY_SEEDS: Dict[str, List[str]] = {k: [] for k in FOUR_CATEGORIES}
for _c, (_parent, _) in _HIERARCHY_PARENTS.items():
    if _parent in _LABEL2KEY:
        CATEGORY_SEEDS[_LABEL2KEY[_parent]].append(_c)

PROBLEM_TO_CATEGORY = {
    GlacierProblem.RHEOLOGY_TUNING: "rheology",
    GlacierProblem.GROUNDING_LINE_DYNAMICS: "sliding_hydro",
    GlacierProblem.BASAL_SLIDING_INVERSION: "sliding_hydro",
    GlacierProblem.CALVING_REMESHING: "numerical",
    GlacierProblem.THERMODYNAMIC_COUPLING: "dynamics_thermo",
    GlacierProblem.GENERAL: None,
    GlacierProblem.MULTI_PROBLEM: None,
}

# Blocks b(i) for the 32 LatentMoE experts (order = GLACIER_EXPERT_LABELS)
EXPERT_CATEGORY = (
    ["rheology"] * 8 +
    ["sliding_hydro"] * 8 +
    ["numerical"] * 8 +
    ["dynamics_thermo"] * 8
)

@st.cache_resource(show_spinner=False)
def get_category_anchors() -> Tuple[List[str], np.ndarray]:
    model = load_embedding_model()
    rows = []
    for k in FOUR_CATEGORIES:
        texts = [c.replace("_", " ") for c in CATEGORY_SEEDS[k]] or [CATEGORY_DISPLAY[k]]
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        mu = emb.mean(axis=0)
        rows.append(mu / (np.linalg.norm(mu) + 1e-9))
    return FOUR_CATEGORIES, np.stack(rows)

def soft_category_membership(term: str, beta: float = 8.0) -> Dict[str, float]:
    try:
        keys, A = get_category_anchors()
        t = load_embedding_model().encode(
            term.replace("_", " "), show_progress_bar=False, convert_to_numpy=True
        )
        t = t / (np.linalg.norm(t) + 1e-9)
        z = beta * (A @ t)
        e = np.exp(z - z.max())
        return dict(zip(keys, e / e.sum()))
    except Exception:
        hard = categorize_glacier_concept(term)
        return {k: (0.9 if k == hard else 0.1 / 3) for k in FOUR_CATEGORIES}

def allocate_category_weights(
    raw: Dict[str, float], alpha: float = 0.25
) -> Dict[str, float]:
    tot = 4 * alpha + sum(raw.get(k, 0.0) for k in FOUR_CATEGORIES)
    return {k: (alpha + raw.get(k, 0.0)) / tot for k in FOUR_CATEGORIES}

def concentration_of(W: Dict[str, float]) -> float:
    H = -sum(w * math.log(w) for w in W.values() if w > 1e-12)
    return float(np.clip(1.0 - H / math.log(4), 0.0, 1.0))

def compute_query_category_weights(
    query: str,
    primary_problem: Optional[GlacierProblem] = None,
) -> Tuple[Dict[str, float], float, int]:
    q = query.lower()
    raw = {k: 0.0 for k in FOUR_CATEGORIES}

    for p, kws in FallbackAnalyzer.PROBLEM_KEYWORDS.items():
        if p == GlacierProblem.GENERAL:
            continue
        hits = sum(1 for kw in kws if kw in q)
        if hits:
            cat = PROBLEM_TO_CATEGORY.get(p)
            if cat:
                raw[cat] += hits

    if primary_problem is not None:
        pdef = GLACIER_PROBLEM_DEFINITIONS.get(primary_problem)
        if pdef is not None:
            for concept in pdef.key_concepts:
                cat = categorize_glacier_concept(concept)
                if cat in raw:
                    raw[cat] += 1.0
            for concept in (
                pdef.relevant_descriptors
                + pdef.relevant_phenomena
                + pdef.relevant_properties
            ):
                cat = categorize_glacier_concept(concept)
                if cat in raw:
                    raw[cat] += 0.5

    for term in extract_glacier_concepts_from_text(q):
        for k, m in soft_category_membership(term).items():
            raw[k] += m

    W = allocate_category_weights(raw)
    c = concentration_of(W)
    K = 2 + int(round(2 * c))
    return W, c, K

def render_category_weight_gauge(
    W: Dict[str, float], c: float, container=None
) -> None:
    if container is None:
        container = st
    cols = container.columns(4)
    for col, k in zip(cols, FOUR_CATEGORIES):
        col.metric(CATEGORY_DISPLAY[k], f"{W.get(k, 0.25):.2f}")
    container.caption(
        f"Rotor concentration c={c:.2f} -> subgraph depth K={2 + int(round(2*c))}"
    )

# ============================================================================
# QDWA CHART CUSTOMIZATION HELPERS (Publication-Quality)
# ============================================================================

def render_qdwa_customization_panel() -> Dict[str, Any]:
    """Comprehensive customization panel for QDWA visualizations."""
    with st.expander("🎨 QDWA Visualization Customization", expanded=False):
        st.markdown("**Colormap & Theme**")
        col1, col2 = st.columns(2)
        with col1:
            qdwa_cmap = st.selectbox(
                "Colormap (QDWA)",
                options=list(SUPPORTED_COLORMAPS.keys()),
                index=list(SUPPORTED_COLORMAPS.keys()).index(st.session_state.get("qdwa_cmap", "viridis")) if st.session_state.get("qdwa_cmap", "viridis") in SUPPORTED_COLORMAPS else 0,
                key="qdwa_post_cmap"
            )
            st.session_state["qdwa_cmap"] = qdwa_cmap
        with col2:
            st.checkbox("Reverse Colormap", key="qdwa_cmap_reverse")

        st.markdown("**Typography**")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.slider("Font Size", 8, 24, 11, key="qdwa_post_font_size")
        with col4:
            st.slider("Title Size", 12, 32, 15, key="qdwa_post_title_size")
        with col5:
            st.selectbox("Font Family", [
                "Inter, Segoe UI, Roboto, sans-serif", "Arial, Helvetica, sans-serif",
                "Georgia, serif", "Courier New, monospace"
            ], key="qdwa_post_font_family")

        st.markdown("**Sankey Settings**")
        col6, col7, col8 = st.columns(3)
        with col6: st.slider("Node Padding", 5, 50, 20, key="qdwa_post_sankey_pad")
        with col7: st.slider("Node Thickness", 10, 40, 25, key="qdwa_post_sankey_thick")
        with col8: st.slider("Link Opacity", 0.1, 1.0, 0.4, key="qdwa_post_sankey_opacity")

        st.markdown("**Bar Chart Settings**")
        col9, col10 = st.columns(2)
        with col9: st.selectbox("Text Position", ["outside", "inside", "auto", "none"], key="qdwa_post_bar_text_pos")
        with col10: st.slider("Text Size", 6, 16, 10, key="qdwa_post_bar_text_size")

        preview_cmap = st.session_state.get("qdwa_cmap", "viridis")
        colors = get_colormap_colors(preview_cmap.replace("_r", ""), 10)
        st.markdown(
            f"<div style='display:flex; height:15px; border-radius:3px; overflow:hidden;'>"
            + "".join(f"<div style='flex:1; background:{c};'></div>" for c in colors)
            + "</div>", unsafe_allow_html=True
        )

        if st.button("🔄 Reset to Defaults", key="qdwa_post_reset"):
            st.session_state["qdwa_cmap"] = "viridis"
            st.rerun()

    return {
        "cmap": st.session_state.get("qdwa_cmap", "viridis"),
        "font_size": st.session_state.get("qdwa_post_font_size", 11),
        "title_size": st.session_state.get("qdwa_post_title_size", 15),
        "font_family": st.session_state.get("qdwa_post_font_family", "Inter, Segoe UI, Roboto, sans-serif"),
        "sankey_node_pad": st.session_state.get("qdwa_post_sankey_pad", 20),
        "sankey_node_thickness": st.session_state.get("qdwa_post_sankey_thick", 25),
        "sankey_link_opacity": st.session_state.get("qdwa_post_sankey_opacity", 0.4),
        "bar_text_position": st.session_state.get("qdwa_post_bar_text_pos", "outside"),
        "bar_text_size": st.session_state.get("qdwa_post_bar_text_size", 10),
    }

def apply_qdwa_pub_style(fig, is_axial=True):
    font = st.session_state.get("qdwa_font", "Inter, sans-serif")
    title_size = st.session_state.get("qdwa_title_size", 18)
    tick_size = st.session_state.get("qdwa_tick_size", 12)
    mt = st.session_state.get("qdwa_margin_t", 50)
    mb = st.session_state.get("qdwa_margin_b", 40)
    ml = st.session_state.get("qdwa_margin_l", 60)
    mr = st.session_state.get("qdwa_margin_r", 40)
    tl = st.session_state.get("qdwa_tick_len", 5)
    
    fig.update_layout(
        font=dict(family=font, size=tick_size, color="#1e293b"),
        title_font=dict(family=font, size=title_size, color="#0f172a"),
        margin=dict(l=ml, r=mr, t=mt, b=mb),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if is_axial:
        fig.update_xaxes(tickfont=dict(size=tick_size, family=font, color="#475569"), ticklen=tl, showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
        fig.update_yaxes(tickfont=dict(size=tick_size, family=font, color="#475569"), ticklen=tl, showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig

def render_qdwa_bar(df: pd.DataFrame, scale: List[str]):
    sorted_df = df.sort_values("Weight", ascending=True)
    colors = []
    border_colors = []
    border_widths = []
    lw = st.session_state.get("qdwa_line_width", 1.5)
    
    for _, row in sorted_df.iterrows():
        colors.append(row["Color"])
        if row["Is Major"]:
            border_colors.append("#FFD700")
            border_widths.append(lw * 2)
        else:
            border_colors.append(row["Color"])
            border_widths.append(lw)
            
    fig = go.Figure(go.Bar(
        x=sorted_df["Weight"], y=sorted_df["Short Name"], orientation="h",
        marker_color=colors, marker_line_color=border_colors, marker_line_width=border_widths,
        text=sorted_df["Weight"].map("{:.1%}".format), textposition="outside",
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>",
    ))
    
    major_row = sorted_df[sorted_df["Is Major"]].iloc[0]
    fig.add_annotation(x=major_row["Weight"], y=major_row["Short Name"], text="⭐ MAJOR",
                       showarrow=True, arrowhead=2, ax=40, ay=0,
                       font=dict(size=13, color="#FFD700", family="sans-serif"))
    
    fig.update_layout(title="QDWA Category Weights (Major Highlighted ⭐)",
                      xaxis_title="Weight", xaxis_range=[0, max(sorted_df["Weight"]) * 1.35],
                      height=300, showlegend=False)
    
    st.plotly_chart(apply_qdwa_pub_style(fig), use_container_width=True)

def render_qdwa_donut(df: pd.DataFrame, scale: List[str]):
    cats = list(df["Short Name"])
    weights = list(df["Weight"])
    colors = list(df["Color"])
    major_cat = df.loc[df["Is Major"], "Short Name"].iloc[0]
    
    pull = [0.12 if c == major_cat else 0 for c in cats]
    
    fig = go.Figure(go.Pie(
        labels=[f"{c}\n({w:.1%})" for c, w in zip(cats, weights)],
        values=weights,
        marker_colors=colors,
        hole=0.50,
        pull=pull,
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.4f}<extra></extra>",
    ))
    
    fig.add_annotation(
        text=f"⭐ {major_cat}",
        x=0.5, y=0.5,
        font=dict(size=16, color="#FFD700", family="sans-serif"),
        showarrow=False,
    )
    
    fig.update_layout(
        title="QDWA Weight Distribution (Donut)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(font=dict(size=12)),
    )
    
    st.plotly_chart(apply_qdwa_pub_style(fig, is_axial=False), use_container_width=True)

def render_qdwa_radar(df: pd.DataFrame, scale: List[str]):
    cats = list(df["Short Name"])
    weights = list(df["Weight"])
    colors = list(df["Color"])
    n = len(cats)
    
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    weights_closed = weights + [weights[0]]
    theta_closed = theta + [theta[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=weights_closed,
        theta=theta_closed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.25)",
        line=dict(color="#6366F1", width=2),
        name="Weight profile",
        hovertemplate="<b>%{theta}</b><br>Weight: %{r:.4f}<extra></extra>",
    ))
    
    for i in range(n):
        fig.add_trace(go.Scatterpolar(
            r=[weights[i]],
            theta=[np.degrees(theta[i])],
            mode="markers+text",
            marker=dict(size=14, color=colors[i], line=dict(width=2, color="white")),
            text=[f"{weights[i]:.1%}"],
            textposition="top center",
            textfont=dict(size=12, color=colors[i]),
            name=cats[i],
            showlegend=False,
            hoverinfo="skip",
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(weights) * 1.4]),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        title="QDWA Weight Radar",
        height=420,
        margin=dict(l=60, r=60, t=50, b=40),
    )
    
    st.plotly_chart(apply_qdwa_pub_style(fig, is_axial=False), use_container_width=True)

def render_qdwa_sankey(df: pd.DataFrame, query: str, scale: List[str]):
    labels = [f"Query:\n{query[:40]}…"] + list(df["Short Name"])
    sources = [0, 0, 0, 0]
    targets = [1, 2, 3, 4]
    values = list(df["Weight"])
    colors = list(df["Color"])
    
    node_colors = ["#6366F1"] + colors
    
    link_colors = []
    for c in colors:
        if isinstance(c, str) and c.startswith('#') and len(c) == 7:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            link_colors.append(f"rgba({r},{g},{b},0.5)")
        else:
            link_colors.append(c)
    
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=30,
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=link_colors,
        ),
    )])
    
    fig.update_layout(
        title="QDWA Weight Flow (Query → Categories)",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=13),
    )
    
    st.plotly_chart(apply_qdwa_pub_style(fig, is_axial=False), use_container_width=True)


def render_qdwa_sankey_twoway(query: str, df: pd.DataFrame, scale: List[str]):
    terms = extract_glacier_concepts_from_text(query.lower())
    if not terms:
        terms = ["general_glacial_flow"]

    term_memberships = {}
    for t in terms:
        term_memberships[t] = soft_category_membership(t)

    labels = [f"Query:\n{query[:35]}..."]
    term_labels = [t.replace("_", " ").title() for t in terms]
    labels.extend(term_labels)
    cat_labels = list(df["Short Name"])
    labels.extend(cat_labels)
    labels.append("Final\nWeights")

    sources, targets, values, link_colors = [], [], [], []

    q_idx = 0
    for i, t in enumerate(terms):
        sources.append(q_idx)
        targets.append(1 + i)
        values.append(1.0 / len(terms))
        link_colors.append("rgba(99, 102, 241, 0.3)")

    for i, t in enumerate(terms):
        mems = term_memberships[t]
        for j, cat_key in enumerate(FOUR_CATEGORIES):
            val = mems.get(cat_key, 0)
            if val > 0.05:
                sources.append(1 + i)
                targets.append(1 + len(terms) + j)
                values.append(val)
                c = CATEGORY_COLORS.get(CATEGORY_DISPLAY[cat_key], "#64748b")
                if isinstance(c, str) and c.startswith('#') and len(c) == 7:
                    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                    link_colors.append(f"rgba({r},{g},{b},0.4)")
                else:
                    link_colors.append("rgba(100,100,100,0.4)")

    for j, (_, row) in enumerate(df.iterrows()):
        sources.append(1 + len(terms) + j)
        targets.append(1 + len(terms) + len(FOUR_CATEGORIES))
        values.append(row["Weight"])
        c = row["Color"]
        if isinstance(c, str) and c.startswith('#') and len(c) == 7:
            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            link_colors.append(f"rgba({r},{g},{b},0.7)")
        else:
            link_colors.append("rgba(100,100,100,0.7)")

    node_colors = ["#6366F1"] + ["#94A3B8"]*len(terms) + list(df["Color"]) + ["#10B981"]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=25, label=labels, color=node_colors,
                  line=dict(color="#fff", width=1)),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    )])

    fig.update_layout(
        title_text="QDWA Two-Way Flow: Query → Terms → Categories → Weights",
        height=500,
        font=dict(size=13, color="#1e293b")
    )
    st.plotly_chart(apply_qdwa_pub_style(fig, is_axial=False), use_container_width=True)

def render_qdwa_chord(df: pd.DataFrame, scale: List[str]):
    cats = list(df["Short Name"])
    weights = list(df["Weight"])
    colors = list(df["Color"])
    n = len(cats)
    
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.75
    
    fig = go.Figure()
    for i in range(n):
        fig.add_trace(go.Barpolar(
            r=[weights[i]],
            theta=[np.degrees(theta[i])],
            width=[np.degrees(width)],
            marker_color=colors[i],
            marker_line_color="white",
            marker_line_width=2,
            name=cats[i],
            hoverinfo="text",
            text=f"{cats[i]}: {weights[i]:.1%}",
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(weights) * 1.3]),
            angularaxis=dict(visible=False),
        ),
        showlegend=True,
        title="QDWA Category Chord (Radial)",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(apply_qdwa_pub_style(fig, is_axial=False), use_container_width=True)
    
    st.markdown("#### Category Interaction Heatmap")
    render_qdwa_heatmap(df, scale)

def render_qdwa_heatmap(df: pd.DataFrame, scale: List[str]):
    cats = list(df["Short Name"])
    matrix = np.array([
        list(df["Weight"]),
        list(df["Raw Similarity"]),
        list(df["Keyword Hits"] / (df["Keyword Hits"].max() + 1)),
    ])
    
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=cats,
        y=["Final Weight", "Raw Cosine Sim", "Keyword Hits (norm)"],
        colorscale=scale,
        texttemplate="%{z:.3f}",
        textfont=dict(size=14, color="white"),
        hovertemplate=(
            "<b>%{y}</b> × <b>%{x}</b><br>Value: %{z:.4f}<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(title="Value"),
    ))
    
    major_idx = df.index[df["Is Major"]][0]
    fig.add_vrect(
        x0=major_idx - 0.45, x1=major_idx + 0.45,
        line_width=3, line_color="#FFD700",
        fillcolor="#FFD700", opacity=0.08,
    )
    
    fig.update_layout(
        title="QDWA Weight Decomposition Heatmap (⭐ = major column)",
        height=260,
        margin=dict(l=140, r=40, t=50, b=30),
        xaxis_side="bottom",
    )
    
    st.plotly_chart(apply_qdwa_pub_style(fig), use_container_width=True)

# ============================================================================
# QUERY ANALYSIS DATA STRUCTURES
# ============================================================================
@dataclass
class ConceptPriority:
    concept_name: str
    concept_type: str
    composite_score: float
    direct_score: float
    problem_affinity_score: float
    causal_path_score: float
    is_explicitly_mentioned: bool
    is_inferred: bool
    inference_reason: str = ""
    ppr_score: float = 0.0
    qc_pmi: float = 0.0
    semantic_resonance: float = 0.0
    cde: float = 0.0
    causal_proximity: float = 0.0

    def to_dict(self) -> Dict:
        return {**self.__dict__, "score": round(self.composite_score, 3)}

@dataclass
class QueryAnalysisResult:
    original_query: str
    normalized_query: str
    primary_problem: GlacierProblem
    secondary_problems: List[GlacierProblem]
    problem_confidences: Dict[str, float]
    explicitly_mentioned: List[str]
    inferred_concepts: List[str]
    all_relevant_concepts: List[str]
    concept_priorities: Dict[str, ConceptPriority] = field(default_factory=dict)
    query_type: str = "general"
    emphasis_direction: str = "cause"
    comparison_pairs: List[Tuple[str, str]] = field(default_factory=list)
    subgraph_depth: int = 2
    priority_threshold: float = 0.3
    focus_nodes: List[str] = field(default_factory=list)
    bridge_nodes: List[str] = field(default_factory=list)
    suggested_layout: str = "force"
    highlight_paths: List[List[str]] = field(default_factory=list)
    visualization_focus: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    category_weights: Dict[str, float] = field(default_factory=dict)
    concentration: float = 0.0

    def get_top_concepts(self, n: int = 10) -> List[ConceptPriority]:
        return sorted(self.concept_priorities.values(), key=lambda x: x.composite_score, reverse=True)[:n]

    def get_concepts_above_threshold(self, threshold: float = None) -> List[str]:
        thresh = threshold or self.priority_threshold
        return [name for name, cp in self.concept_priorities.items() if cp.composite_score >= thresh]

# ============================================================================
# LLM QUERY ANALYZERS (Fallback, OpenAI, Local)
# ============================================================================
class LLMQueryAnalyzer(ABC):
    @abstractmethod
    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        pass
    @abstractmethod
    def is_available(self) -> bool:
        pass

class FallbackAnalyzer(LLMQueryAnalyzer):
    PROBLEM_KEYWORDS = {
        GlacierProblem.RHEOLOGY_TUNING: {"glen", "arrhenius", "viscosity", "damage", "enhancement", "anisotropic"},
        GlacierProblem.GROUNDING_LINE_DYNAMICS: {"grounding", "contact", "hydrostatic", "water pressure"},
        GlacierProblem.BASAL_SLIDING_INVERSION: {"friction", "weertman", "coulomb", "budd", "adjoint", "inverse"},
        GlacierProblem.CALVING_REMESHING: {"calving", "crevasse", "gmsh", "remesh", "free surface"},
        GlacierProblem.THERMODYNAMIC_COUPLING: {"enthalpy", "temperature", "melting", "thermo", "cold-temperate"},
        GlacierProblem.GENERAL: {"elmer/ice", "glacier", "ice sheet", "ice shelf", "stokes"},
    }
    def is_available(self) -> bool:
        return True

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        q = query.lower().strip()
        problem_scores = {p: sum(1 for kw in kws if kw in q) for p, kws in self.PROBLEM_KEYWORDS.items()}
        primary = max(problem_scores, key=problem_scores.get) if sum(problem_scores.values()) > 0 else GlacierProblem.GENERAL
        secondary = [p for p, s in sorted(problem_scores.items(), key=lambda x: -x[1]) if s > 0 and p != primary][:2]

        W, c, K = compute_query_category_weights(query, primary_problem=primary)

        explicitly_mentioned = []
        for canonical, node in ontology.concepts.items():
            if canonical.replace("_", " ") in q or any(syn.replace("_", " ") in q for syn in node.synonyms):
                explicitly_mentioned.append(canonical)

        inferred = []
        if primary != GlacierProblem.GENERAL:
            pdef = GLACIER_PROBLEM_DEFINITIONS[primary]
            for concept in pdef.get_ontology_concepts():
                if concept not in explicitly_mentioned and concept in ontology.concepts:
                    inferred.append(concept)

        all_relevant = list(dict.fromkeys(explicitly_mentioned + inferred))
        priorities = {}
        pdef = GLACIER_PROBLEM_DEFINITIONS.get(primary, GLACIER_PROBLEM_DEFINITIONS[GlacierProblem.GENERAL])
        problem_concept_set = pdef.get_ontology_concepts()

        for concept in all_relevant:
            is_explicit = concept in explicitly_mentioned
            priorities[concept] = ConceptPriority(
                concept_name=concept, concept_type=ontology.get_concept_type(concept).value,
                composite_score=(1.0 if is_explicit else 0.6) * 0.5 + (1.0 if concept in problem_concept_set else 0.4) * 0.5,
                direct_score=1.0 if is_explicit else 0.6, problem_affinity_score=1.0 if concept in problem_concept_set else 0.4,
                causal_path_score=0.5, is_explicitly_mentioned=is_explicit, is_inferred=not is_explicit,
                inference_reason="problem_affinity" if not is_explicit else "explicit_mention"
            )

        query_type = "general"
        if any(w in q for w in ["compare", "vs", "versus", "difference"]):
            query_type = "comparison"
        elif any(w in q for w in ["why", "cause", "reason", "lead to"]):
            query_type = "causal"
        elif any(w in q for w in ["how", "improve", "enhance", "optimize", "strategy"]):
            query_type = "solution"

        highlight_paths = [[src, tgt] for src, rel, tgt in pdef.key_relationships if src in ontology.concepts and tgt in ontology.concepts]
        total = max(sum(problem_scores.values()), 1)

        return QueryAnalysisResult(
            original_query=query, normalized_query=q, primary_problem=primary, secondary_problems=secondary,
            problem_confidences={p.value: s / total for p, s in problem_scores.items()},
            explicitly_mentioned=explicitly_mentioned, inferred_concepts=inferred, all_relevant_concepts=all_relevant,
            concept_priorities=priorities, query_type=query_type, emphasis_direction="cause" if query_type == "causal" else "neutral",
            subgraph_depth=K, priority_threshold=0.3, focus_nodes=explicitly_mentioned[:5], bridge_nodes=inferred[:3],
            suggested_layout="force" if query_type != "comparison" else "bisected", highlight_paths=highlight_paths,
            visualization_focus=pdef.visualization_focus, reasoning_chain=[f"Query normalized: '{q}'", f"Primary problem: {primary.value}"],
            confidence=min(sum(problem_scores.values()) / 3.0, 1.0),
            category_weights=W, concentration=c,
        )

class OpenAIQueryAnalyzer(LLMQueryAnalyzer):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self._client = None
        self._pending_new_concepts = []
        self._pending_new_relationships = []

    def _get_client(self):
        if self._client is None and self.api_key and OpenAI is not None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return self._get_client() is not None

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        client = self._get_client()
        if client is None:
            return FallbackAnalyzer().analyze_query(query, ontology)

        concept_list = list(ontology.concepts.keys())[:50]
        system_prompt = """You are an expert in glacial flow mechanics and Elmer/Ice modeling. Analyze the user's query and return ONLY valid JSON with:
        1. "primary_problem": One of: rheology_tuning, basal_sliding_inversion, grounding_line_dynamics, calving_remeshing, thermodynamic_coupling, general, multi_problem
        2. "explicitly_mentioned": List of canonical concept names from the query (use snake_case)
        3. "inferred_concepts": List of additional relevant concepts the query implies
        4. "query_type": One of: causal, comparison, solution, definition, general
        5. "highlight_paths": List of [source, target] concept pairs to highlight
        6. "reasoning_chain": List of strings explaining analysis steps
        7. "new_concepts": List of objects with "name" (snake_case), "type" (material/property/phenomenon/process/method/parameter/model), "definition", "synonyms" (list)
        8. "new_relationships": List of [source, relationship_type, target, confidence] for NEW relationships between EXISTING concepts."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Analyze: '{query}'. Available concepts: {', '.join(concept_list)}"}],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(response.choices[0].message.content)
            self._pending_new_concepts = parsed.get("new_concepts", [])
            self._pending_new_relationships = parsed.get("new_relationships", [])
            problem_map = {p.value: p for p in GlacierProblem}
            primary = problem_map.get(parsed.get("primary_problem", "general"), GlacierProblem.GENERAL)
            W, c, K = compute_query_category_weights(query, primary_problem=primary)
            explicitly_mentioned = [c for c in parsed.get("explicitly_mentioned", []) if c in ontology.concepts]
            inferred = [c for c in parsed.get("inferred_concepts", []) if c in ontology.concepts and c not in explicitly_mentioned]
            priorities = {c: ConceptPriority(c, ontology.get_concept_type(c).value, 0.9 if c in explicitly_mentioned else 0.6, 1.0 if c in explicitly_mentioned else 0.5, 0.8, 0.5, c in explicitly_mentioned, c not in explicitly_mentioned, "llm_inferred") for c in list(dict.fromkeys(explicitly_mentioned + inferred))}
            return QueryAnalysisResult(
                original_query=query, normalized_query=query.lower().strip(), primary_problem=primary, secondary_problems=[],
                problem_confidences={}, explicitly_mentioned=explicitly_mentioned, inferred_concepts=inferred, all_relevant_concepts=list(dict.fromkeys(explicitly_mentioned + inferred)),
                concept_priorities=priorities, query_type=parsed.get("query_type", "general"), emphasis_direction="cause",
                subgraph_depth=K, priority_threshold=0.3, focus_nodes=explicitly_mentioned[:5], bridge_nodes=inferred[:3],
                suggested_layout="bisected" if parsed.get("query_type") == "comparison" else "force",
                highlight_paths=[[p[0], p[1]] for p in parsed.get("highlight_paths", []) if len(p) >= 2],
                visualization_focus=GLACIER_PROBLEM_DEFINITIONS[primary].visualization_focus,
                reasoning_chain=parsed.get("reasoning_chain", ["LLM analysis completed"]), confidence=0.85,
                category_weights=W, concentration=c,
            )
        except Exception as e:
            st.warning(f"OpenAI analysis failed ({e}), falling back to rule-based.")
            return FallbackAnalyzer().analyze_query(query, ontology)

class LocalLLMQueryAnalyzer(LLMQueryAnalyzer):
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False
        self._pending_new_concepts = []
        self._pending_new_relationships = []

    def _load_model(self):
        if self._loaded:
            return
        try:
            if pipeline is None:
                st.warning("transformers not installed; cannot use local LLM.")
                return
            st.info(f"⏳ Loading local model: `{self.model_name}`… (first run may take 1–2 min)")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            load_kwargs: Dict[str, Any] = {}
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                try:
                    load_kwargs["load_in_8bit"] = True
                except Exception:
                    pass
            else:
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = None
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            self._loaded = True
            st.success(f"✅ Model `{self.model_name}` loaded!")
        except Exception as e:
            st.warning(f"⚠️ Failed to load local model `{self.model_name}`: {e}")
            self._loaded = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_available(self) -> bool:
        self._load_model()
        return self._loaded

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        if not self.is_available():
            return FallbackAnalyzer().analyze_query(query, ontology)
        prompt = f"[INST] You are an expert in glacial flow mechanics and Elmer/Ice. Analyze: '{query}'. Return ONLY valid JSON with: primary_problem, explicitly_mentioned (snake_case list), inferred_concepts (list), query_type, highlight_paths (list of [src, tgt]), reasoning_chain (list). [/INST]"
        try:
            result = self._pipeline(prompt)[0]["generated_text"]
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                fake_openai = OpenAIQueryAnalyzer()
                fake_openai._pending_new_concepts = parsed.get("new_concepts", [])
                fake_openai._pending_new_relationships = parsed.get("new_relationships", [])
                return fake_openai.analyze_query(query, ontology)
        except Exception as e:
            st.warning(f"Local LLM parsing failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return FallbackAnalyzer().analyze_query(query, ontology)

    def unload_model(self) -> None:
        if self._pipeline is not None:
            if hasattr(self._pipeline, 'tokenizer'):
                del self._pipeline.tokenizer
            if hasattr(self._pipeline, 'model'):
                del self._pipeline.model
            del self._pipeline
            self._pipeline = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class LLMQueryAnalyzerFactory:
    def __init__(self):
        self._openai_cache: Optional[OpenAIQueryAnalyzer] = None
        self._local_cache: Dict[str, LocalLLMQueryAnalyzer] = {}
        self._fallback = FallbackAnalyzer()

    def get_analyzer(self, mode: str = "auto", api_key: str = None, local_model: str = None) -> LLMQueryAnalyzer:
        if mode == "openai":
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            return self._openai_cache
        elif mode == "local":
            model = local_model
            if model is None:
                return self._fallback
            if model not in self._local_cache:
                self._local_cache[model] = LocalLLMQueryAnalyzer(model)
            return self._local_cache[model]
        elif mode == "fallback":
            return self._fallback
        else:  # auto
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            if self._openai_cache.is_available():
                return self._openai_cache
            model = local_model
            if model is None:
                return self._fallback
            if model not in self._local_cache:
                self._local_cache[model] = LocalLLMQueryAnalyzer(model)
            if self._local_cache[model].is_available():
                return self._local_cache[model]
            return self._fallback

# -----------------------------------------------------------------------------
# DYNAMIC ONTOLOGY EXPANDER
# -----------------------------------------------------------------------------
class DynamicOntologyExpander:
    REL_STR_TO_ENUM = {r.value: r for r in RelationshipType}
    for _k, _v in list(REL_STR_TO_ENUM.items()):
        REL_STR_TO_ENUM[_k.upper()] = _v
    TYPE_STR_TO_ENUM = {t.value: t for t in ConceptType}

    def __init__(self, ontology: Any):
        self.ontology = ontology
        self.mutation_log: List[Dict[str, Any]] = []
        self.session_concepts_added: Set[str] = set()
        self.session_relationships_added: List[Tuple[str, str, RelationshipType, float]] = []
        self.query_bridge_concepts: Dict[str, str] = {}
        self.priority_overrides: Dict[str, float] = {}
        self._base_concept_count = len(ontology.concepts)
        self._base_rel_count = len(ontology.relationships)

    @property
    def stats(self) -> Dict[str, int]:
        return {"base_concepts": self._base_concept_count, "base_relationships": self._base_rel_count,
                "concepts_added": len(self.session_concepts_added), "relationships_added": len(self.session_relationships_added),
                "bridge_concepts": len(self.query_bridge_concepts), "total_mutations": len(self.mutation_log)}

    def apply_query_analysis(self, analysis: QueryAnalysisResult, analyzer: LLMQueryAnalyzer = None) -> Dict[str, Any]:
        changes = {"concepts_added": [], "relationships_added": [], "bridges_created": []}
        for concept_name, priority in analysis.concept_priorities.items():
            if concept_name in self.ontology.concepts:
                self.priority_overrides[concept_name] = priority.composite_score

        new_concepts_raw = getattr(analyzer, '_pending_new_concepts', []) if hasattr(analyzer, '_pending_new_concepts') else []
        new_rels_raw = getattr(analyzer, '_pending_new_relationships', []) if hasattr(analyzer, '_pending_new_relationships') else []

        for concept_data in new_concepts_raw:
            result = self._add_concept_from_llm(concept_data, analysis.original_query)
            if result:
                changes["concepts_added"].append(result)
        for rel_data in new_rels_raw:
            result = self._add_relationship_from_llm(rel_data, analysis.original_query)
            if result:
                changes["relationships_added"].append(result)

        for concept in analysis.inferred_concepts:
            if concept not in self.ontology.concepts:
                bridge_result = self._create_bridge_concept(concept, analysis.original_query, analysis.primary_problem)
                if bridge_result:
                    changes["bridges_created"].append(bridge_result)

        self.ontology._build_synonym_index()
        return changes

    def _add_concept_from_llm(self, concept_data: Dict, source_query: str) -> Optional[Dict]:
        name = concept_data.get("name", "").strip().lower().replace(" ", "_")
        if not name or name in self.ontology.concepts or name in self.session_concepts_added:
            return None
        concept_type = self.TYPE_STR_TO_ENUM.get(concept_data.get("type", "general"), ConceptType.GENERAL)
        synonyms = set(s.lower().strip() for s in concept_data.get("synonyms", []) if isinstance(s, str))
        definition = concept_data.get("definition", f"LLM-inferred concept from query: {source_query}")

        self.ontology._add_concept(name, concept_type, synonyms=synonyms, definition=definition)
        self.ontology.synonym_to_canonical[name.lower()] = name
        for syn in synonyms:
            self.ontology.synonym_to_canonical[syn] = name
        self.session_concepts_added.add(name)

        for rel_tuple in concept_data.get("relate_to", []):
            if len(rel_tuple) >= 2:
                target, rel_type_str = rel_tuple[0], rel_tuple[1] if len(rel_tuple) > 1 else "influences"
                conf = float(rel_tuple[2]) if len(rel_tuple) > 2 else 0.7
                rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
                if target in self.ontology.concepts:
                    self.ontology.relationships.append(Relationship(name, target, rel_enum, conf))
                    self.session_relationships_added.append((name, target, rel_enum, conf))

        self.mutation_log.append({"type": "add_concept", "concept": name, "concept_type": concept_type.value, "source_query": source_query})
        return {"name": name, "type": concept_type.value, "synonyms": list(synonyms)}

    def _add_relationship_from_llm(self, rel_data: List, source_query: str) -> Optional[Dict]:
        if len(rel_data) < 3:
            return None
        source, rel_type_str, target = str(rel_data[0]).strip().lower().replace(" ", "_"), str(rel_data[1]).upper(), str(rel_data[2]).strip().lower().replace(" ", "_")
        confidence = float(rel_data[3]) if len(rel_data) > 3 else 0.7
        if source not in self.ontology.concepts or target not in self.ontology.concepts:
            return None
        rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
        self.ontology.relationships.append(Relationship(source, target, rel_enum, confidence))
        self.session_relationships_added.append((source, target, rel_enum, confidence))
        self.mutation_log.append({"type": "add_relationship", "source": source, "target": target, "rel_type": rel_enum.value, "source_query": source_query})
        return {"source": source, "target": target, "rel_type": rel_enum.value, "confidence": confidence}

    def _create_bridge_concept(self, missing_concept: str, source_query: str, problem: GlacierProblem) -> Optional[Dict]:
        bridge_name = f"query_bridge_{missing_concept.replace(' ', '_').lower()}"
        if bridge_name in self.ontology.concepts:
            return None
        pdef = GLACIER_PROBLEM_DEFINITIONS.get(problem, GLACIER_PROBLEM_DEFINITIONS[GlacierProblem.GENERAL])
        self.ontology._add_concept(bridge_name, ConceptType.GENERAL, synonyms={missing_concept.lower()}, definition=f"Query-inferred bridge: '{missing_concept}'")
        self.ontology.synonym_to_canonical[bridge_name] = bridge_name
        self.ontology.synonym_to_canonical[missing_concept.lower()] = bridge_name

        connected = []
        for key_concept in pdef.key_concepts[:3]:
            if key_concept in self.ontology.concepts:
                self.ontology.relationships.append(Relationship(bridge_name, key_concept, RelationshipType.BRIDGE, 0.5))
                self.session_relationships_added.append((bridge_name, key_concept, RelationshipType.BRIDGE, 0.5))
                connected.append(key_concept)
        self.session_concepts_added.add(bridge_name)
        self.query_bridge_concepts[bridge_name] = source_query
        self.mutation_log.append({"type": "create_bridge", "bridge_name": bridge_name, "original_term": missing_concept, "connected_to": connected})
        return {"bridge": bridge_name, "for": missing_concept, "connected_to": connected}

    def get_priority_boosted_scores(self, base_priorities: Dict[str, ConceptPriority]) -> Dict[str, ConceptPriority]:
        boosted = {}
        for name, priority in base_priorities.items():
            boost = self.priority_overrides.get(name, 0.0)
            if boost > 0:
                bp = copy.deepcopy(priority)
                bp.composite_score = min(bp.composite_score + boost * 0.2, 1.0)
                bp.causal_path_score = boost * 0.2
                boosted[name] = bp
            else:
                boosted[name] = priority
        return boosted

    def undo_last_mutation(self) -> Optional[Dict]:
        if not self.mutation_log:
            return None
        mutation = self.mutation_log.pop()
        if mutation["type"] == "add_concept":
            name = mutation["concept"]
            if name in self.ontology.concepts:
                del self.ontology.concepts[name]
                self.session_concepts_added.discard(name)
                self.ontology.relationships = [r for r in self.ontology.relationships if r.source != name and r.target != name]
        elif mutation["type"] == "add_relationship":
            self.ontology.relationships = [r for r in self.ontology.relationships if not (r.source == mutation["source"] and r.target == mutation["target"] and r.rel_type.value == mutation["rel_type"])]
        elif mutation["type"] == "create_bridge":
            bridge_name = mutation["bridge_name"]
            if bridge_name in self.ontology.concepts:
                del self.ontology.concepts[bridge_name]
                self.session_concepts_added.discard(bridge_name)
                self.query_bridge_concepts.pop(bridge_name, None)
        self.ontology._build_synonym_index()
        return mutation

    def reset_to_base(self) -> Dict[str, int]:
        for name in list(self.session_concepts_added):
            if name in self.ontology.concepts:
                del self.ontology.concepts[name]
        self.ontology.relationships = self.ontology.relationships[:self._base_rel_count]
        self.session_concepts_added.clear()
        self.session_relationships_added.clear()
        self.query_bridge_concepts.clear()
        self.priority_overrides.clear()
        self.mutation_log.clear()
        self.ontology._build_synonym_index()
        return {"concepts_removed": len(self.session_concepts_added), "relationships_removed": len(self.ontology.relationships) - self._base_rel_count}


# ============================================================================
# PRIORITY-GUIDED SUBGRAPH EXTRACTOR (QDWA Enhanced)
# ============================================================================
class PriorityGuidedSubgraphExtractor:
    def __init__(self, full_graph: nx.Graph, ontology: Any, expander: DynamicOntologyExpander):
        self.full_graph = full_graph
        self.ontology = ontology
        self.expander = expander

    def extract(self, analysis: QueryAnalysisResult, query_embedding: np.ndarray = None) -> nx.Graph:
        raw_seed_nodes = set(analysis.focus_nodes + analysis.get_concepts_above_threshold())
        seed_nodes = {n for n in raw_seed_nodes if n in self.full_graph}
        if not seed_nodes:
            seed_nodes = {n for n, d in self.full_graph.nodes(data=True) if d.get("priority_score", 0) >= 0.3}

        W = getattr(analysis, "category_weights", None) or {k: 0.25 for k in FOUR_CATEGORIES}
        k_star = max(W, key=W.get)
        rho, eta, kappa = 0.15, 0.5, 0.5

        G2 = self.full_graph.copy()
        for u, v, d in G2.edges(data=True):
            cu, cv = categorize_glacier_concept(u), categorize_glacier_concept(v)
            boost = 1.0 + eta * W.get(cu, 0.0) * (1 if cu == cv else 0)
            if d.get("edge_type", "") in ("causes", "depends_on", "influences"):
                boost += kappa * W.get(cv, 0.0)
            d["weight"] = d.get("weight", 1.0) * boost

        personalization = {
            n: (1.0 if n in seed_nodes else 0.0)
            + rho * W.get(categorize_glacier_concept(n), 0.0)
            for n in G2.nodes()
        }
        try:
            ppr_scores = nx.pagerank(G2, personalization=personalization, alpha=0.85)
        except Exception:
            ppr_scores = {n: 1.0 / len(G2) for n in G2}

        for node in self.full_graph.nodes():
            ppr = ppr_scores.get(node, 0.0)
            srs = self._compute_semantic_resonance(node, query_embedding) if query_embedding is not None else 0.5
            combined = 0.6 * ppr + 0.4 * srs
            self.full_graph.nodes[node]["priority_score"] = combined
            self.full_graph.nodes[node]["ppr_score"] = ppr
            self.full_graph.nodes[node]["semantic_resonance"] = srs

            if node in analysis.concept_priorities:
                cp = analysis.concept_priorities[node]
                self.full_graph.nodes[node]["is_explicit"] = cp.is_explicitly_mentioned
                self.full_graph.nodes[node]["is_inferred"] = cp.is_inferred
            elif node in self.expander.session_concepts_added:
                self.full_graph.nodes[node]["is_explicit"] = False
                self.full_graph.nodes[node]["is_inferred"] = True
                self.full_graph.nodes[node]["is_llm_added"] = True
            else:
                self.full_graph.nodes[node]["is_explicit"] = False
                self.full_graph.nodes[node]["is_inferred"] = False

        threshold = 0.1
        selected_nodes = {n for n, d in self.full_graph.nodes(data=True) if d.get("priority_score", 0) >= threshold}
        selected_nodes.update(seed_nodes)

        K = max(1, getattr(analysis, "subgraph_depth", 2))
        frontier = set(selected_nodes)
        for _ in range(K - 1):
            nxt = set()
            for n in frontier:
                for nb in self.full_graph.neighbors(n):
                    if nb in selected_nodes:
                        continue
                    ncat = {categorize_glacier_concept(x) for x in self.full_graph.neighbors(nb)}
                    if categorize_glacier_concept(nb) == k_star or (len(ncat) >= 2 and self.full_graph.degree(nb) > 2):
                        nxt.add(nb)
            selected_nodes |= nxt
            frontier = nxt

        subgraph = self.full_graph.subgraph(selected_nodes).copy()
        return subgraph

    def _compute_semantic_resonance(self, concept: str, query_emb: np.ndarray) -> float:
        embed_model = st.session_state.get('embed_model')
        if embed_model is None:
            return 0.5
        try:
            concept_emb = embed_model.encode(concept, convert_to_numpy=True)
            sim = np.dot(query_emb, concept_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(concept_emb) + 1e-8)
            return float(np.clip(sim, 0, 1))
        except Exception:
            return 0.5


# ============================================================================
# GRAPH‑RAG ANSWER GENERATOR
# ============================================================================
class GraphRAGAnswerGenerator:
    def __init__(self, analyzer: LLMQueryAnalyzer):
        self.analyzer = analyzer

    def generate_ground_response(self, query: str, analysis: QueryAnalysisResult, subgraph: nx.Graph,
                                 concept_abstract_map: Dict[str, List[int]], all_texts: Union[List[str], Dict[int, str]],
                                 max_docs_per_concept: int = 2) -> str:
        top_nodes = sorted(subgraph.nodes(data=True), key=lambda x: x[1].get("priority_score", 0.0), reverse=True)[:5]
        evidence_snippets = []
        for node, attrs in top_nodes:
            doc_indices = concept_abstract_map.get(node, [])[:max_docs_per_concept]
            for idx in doc_indices:
                if isinstance(all_texts, dict):
                    text = all_texts.get(idx, "")
                else:
                    text = all_texts[idx] if 0 <= idx < len(all_texts) else ""
                if text:
                    clean_text = re.sub(r'\s+', ' ', text).strip()[:400]
                    evidence_snippets.append("- **" + node + "**: " + clean_text + "...")
        nl = chr(10)
        prompt = "You are an expert in glacial flow mechanics and Elmer/Ice modeling. Answer the user's query based *strictly* on the provided graph context and evidence snippets." + nl
        prompt += "User Query: " + repr(query) + nl
        prompt += "Identified Core Problem: " + analysis.primary_problem.value.replace("_", " ").title() + nl
        if getattr(analysis, "category_weights", None):
            prompt += "Category focus weights (W): " + ", ".join(f"{k}={v:.2f}" for k, v in analysis.category_weights.items()) + nl
        prompt += "Key Graph Concepts: " + ", ".join([n for n, _ in top_nodes]) + nl
        prompt += "Evidence Snippets from Literature:" + nl
        if evidence_snippets:
            prompt += nl.join(evidence_snippets) + nl
        else:
            prompt += "No direct text snippets found. Rely on your general knowledge of glacial mechanics but note the lack of specific retrieved context." + nl
        prompt += "Instructions:" + nl
        prompt += "1. Provide a direct, scientifically accurate answer (2-3 paragraphs)." + nl
        prompt += "2. Explicitly mention how the key concepts interact (e.g., causal chains like 'Coulomb friction depends on effective pressure')." + nl
        prompt += "3. If the retrieved evidence is insufficient, state what specific data is missing."

        if isinstance(self.analyzer, OpenAIQueryAnalyzer) and self.analyzer.is_available():
            return self._call_llm_for_answer(prompt, self.analyzer, query, analysis, top_nodes, evidence_snippets)
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _call_llm_for_answer(self, prompt: str, analyzer: LLMQueryAnalyzer, query: str, analysis: QueryAnalysisResult, top_nodes, evidence_snippets) -> str:
        client = analyzer._get_client()
        if client:
            try:
                response = client.chat.completions.create(
                    model=analyzer.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800
                )
                return response.choices[0].message.content
            except Exception as e:
                fallback_text = self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)
                return "⚠️ LLM API Error: " + str(e) + chr(10) + chr(10) + fallback_text
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _generate_fallback_answer(self, query: str, analysis: Optional[QueryAnalysisResult], top_nodes, snippets: List[str]) -> str:
        nl = chr(10)
        fallback_text = "### Analysis of: '" + query + "'" + nl + nl
        if analysis is not None:
            primary = getattr(analysis, 'primary_problem', None)
            fallback_text += "**Core Problem Identified:** " + (primary.value.replace('_', ' ').title() if primary else 'Unknown') + nl + nl
        else:
            fallback_text += "**Core Problem Identified:** (analysis unavailable)" + nl + nl
        fallback_text += "**Key Concepts in Focus:**" + nl
        fallback_text += nl.join(["- **" + node + "** (" + attrs.get("concept_type", "general") + "): Priority Score " + str(round(attrs.get("priority_score", 0), 2)) for node, attrs in top_nodes])
        if snippets:
            fallback_text += nl + "**Retrieved Evidence Context:**" + nl + nl.join(snippets[:3]) + nl
        else:
            fallback_text += nl + "*Note: No direct text snippets were linked to these concepts in the current dataset.*" + nl
        fallback_text += nl + "**System Reasoning Chain:**" + nl
        if analysis is not None:
            reasoning_chain = getattr(analysis, 'reasoning_chain', [])
            fallback_text += nl.join(["- " + step for step in reasoning_chain])
        else:
            fallback_text += "- No reasoning chain available (analysis was None)." + nl
        return fallback_text


# ============================================================================
# QUERY SESSION MANAGER
# ============================================================================
class QuerySessionManager:
    SESSION_KEY = "glacier_query_session"
    @classmethod
    def init_session(cls) -> Dict[str, Any]:
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = {"query_history": [], "analysis_history": [], "mutation_history": [], "analyzer_mode": "auto", "total_concepts_added": 0, "total_relationships_added": 0}
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def record_query(cls, query: str, analysis: QueryAnalysisResult, mutations: Dict[str, Any]) -> None:
        session = cls.init_session()
        session["query_history"].append(query)
        session["analysis_history"].append({"query": query, "primary_problem": analysis.primary_problem.value, "query_type": analysis.query_type, "concepts_found": len(analysis.all_relevant_concepts), "explicit": len(analysis.explicitly_mentioned), "inferred": len(analysis.inferred_concepts), "confidence": analysis.confidence, "timestamp": datetime.now().isoformat()})
        session["mutation_history"].append({"query": query, "concepts_added": len(mutations.get("concepts_added", [])), "relationships_added": len(mutations.get("relationships_added", [])), "bridges_created": len(mutations.get("bridges_created", [])), "timestamp": datetime.now().isoformat()})
        session["total_concepts_added"] += len(mutations.get("concepts_added", []))
        session["total_relationships_added"] += len(mutations.get("relationships_added", []))

    @classmethod
    def get_session(cls) -> Dict[str, Any]:
        return cls.init_session()
    @classmethod
    def clear_session(cls) -> None:
        if cls.SESSION_KEY in st.session_state:
            del st.session_state[cls.SESSION_KEY]


# ============================================================================
# QDWA — Query Distillation & Weighted Allocation (FULL IMPLEMENTATION)
# ============================================================================
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import hashlib

# The four categories mapped to ontology concept groups
QDWA_CATEGORIES = {
    "Rheology & Material Physics": [
        "glens_law", "arrhenius_factor", "glen_enhancement_factor", "anisotropic_ice",
        "golf_model", "caffe_model", "firn_densification", "damage_mechanics",
        "critical_shear_rate", "viscosity_model", "porous_law", "fabric_evolution"
    ],
    "Basal Sliding & Hydrology": [
        "weertman_friction", "coulomb_friction", "budd_friction", "effective_pressure",
        "glads_hydrology", "double_continuum_hydro", "water_pressure", "geothermal_flux",
        "contact_problem", "subglacial_cavities", "lateral_friction", "shape_factor"
    ],
    "Numerical Solvers & Meshing": [
        "parstokes", "block_preconditioning", "free_surface_evolution", "streamline_stabilization",
        "gmsh_remeshing", "mesh_update", "inverse_adjoint", "adjoint_data_assimilation",
        "flowline_geometry", "griddata_reader", "ssa_flow", "sia_flow"
    ],
    "Ice Dynamics & Thermodynamics": [
        "full_stokes", "enthalpy_solver", "temperature_solver", "pressure_melting_point",
        "calving_3d", "crevasse_depth", "surface_mass_balance", "basal_melt_parameterization",
        "hydrostatic_loads", "ismip_hom", "ice_shelf_flow", "ablation_model"
    ]
}

CATEGORY_SHORT_NAMES = {
    "Rheology & Material Physics": "Rheology",
    "Basal Sliding & Hydrology": "Sliding/Hydro",
    "Numerical Solvers & Meshing": "Numerical",
    "Ice Dynamics & Thermodynamics": "Dynamics/Thermo",
}

CATEGORY_COLORS = {
    "Rheology & Material Physics":          "#3B82F6",  # Blue
    "Basal Sliding & Hydrology":            "#10B981",  # Emerald
    "Numerical Solvers & Meshing":          "#F59E0B",  # Amber
    "Ice Dynamics & Thermodynamics":        "#EF4444",  # Red
}

class QDWAWeightAllocator:
    def __init__(
        self,
        ontology: "DomainOntology",
        encoder: "SentenceTransformer",
        temperature: float = 0.15,
        keyword_boost: float = 0.30,
    ):
        self.ontology = ontology
        self.encoder = encoder
        self.temperature = temperature
        self.keyword_boost = keyword_boost
        self._centroids: Optional[Dict[str, np.ndarray]] = None
        self._last_query: str = ""
        self._last_weights: Dict[str, float] = {}
        self._last_raw_scores: Dict[str, float] = {}
        self._last_keyword_hits: Dict[str, int] = {}
        self._last_query_embedding: Optional[np.ndarray] = None

    def build_centroids(self) -> None:
        self._centroids = {}
        for cat_name, concept_keys in QDWA_CATEGORIES.items():
            embeddings = []
            for key in concept_keys:
                node = self.ontology.concepts.get(key)
                if node is not None and node.embedding is not None:
                    embeddings.append(node.embedding)
                else:
                    emb = self.encoder.encode(
                        key.replace("_", " "), normalize_embeddings=True
                    )
                    embeddings.append(emb)
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                self._centroids[cat_name] = centroid

    @property
    def centroids(self) -> Dict[str, np.ndarray]:
        if self._centroids is None:
            self.build_centroids()
        return self._centroids

    def compute_weights(self, query: str) -> Dict[str, float]:
        self._last_query = query
        query_emb = self.encoder.encode(query, normalize_embeddings=True)
        self._last_query_embedding = query_emb

        raw_scores = {}
        for cat_name, centroid in self.centroids.items():
            sim = float(np.dot(query_emb, centroid))
            raw_scores[cat_name] = sim
        self._last_raw_scores = raw_scores

        temp_scaled = {k: v / self.temperature for k, v in raw_scores.items()}
        max_val = max(temp_scaled.values())
        exp_vals = {k: np.exp(v - max_val) for k, v in temp_scaled.items()}
        total_exp = sum(exp_vals.values())
        weights = {k: v / total_exp for k, v in exp_vals.items()}

        query_lower = query.lower()
        keyword_hits = {}
        for cat_name, concept_keys in QDWA_CATEGORIES.items():
            hits = 0
            for key in concept_keys:
                node = self.ontology.concepts.get(key)
                if key.replace("_", " ") in query_lower:
                    hits += 1
                if node:
                    for syn in node.synonyms:
                        if syn in query_lower:
                            hits += 1
            keyword_hits[cat_name] = hits
        self._last_keyword_hits = keyword_hits

        beta = self.keyword_boost
        boosted = {
            k: weights[k] * (1.0 + beta * keyword_hits[k])
            for k in weights
        }
        total_boosted = sum(boosted.values())
        final_weights = {k: v / total_boosted for k, v in boosted.items()}
        self._last_weights = final_weights
        return final_weights

    @property
    def weights(self) -> Dict[str, float]:
        return self._last_weights

    @property
    def raw_scores(self) -> Dict[str, float]:
        return self._last_raw_scores

    @property
    def keyword_hits(self) -> Dict[str, int]:
        return self._last_keyword_hits

    @property
    def query_embedding(self) -> Optional[np.ndarray]:
        return self._last_query_embedding

    def get_major_category(self) -> Tuple[str, float]:
        if not self._last_weights:
            return ("N/A", 0.0)
        return max(self._last_weights.items(), key=lambda x: x[1])

    def get_ranked_categories(self) -> List[Tuple[str, float]]:
        return sorted(self._last_weights.items(), key=lambda x: x[1], reverse=True)

    def weights_dataframe(self) -> pd.DataFrame:
        rows = []
        for cat, w in self._last_weights.items():
            rows.append({
                "Category": cat,
                "Short Name": CATEGORY_SHORT_NAMES[cat],
                "Weight": w,
                "Raw Similarity": self._last_raw_scores.get(cat, 0.0),
                "Keyword Hits": self._last_keyword_hits.get(cat, 0),
                "Color": CATEGORY_COLORS[cat],
                "Is Major": cat == self.get_major_category()[0],
            })
        return pd.DataFrame(rows)

@st.cache_resource(show_spinner="Building QDWA weight allocator…")
def init_qdwa_allocator(
    _ontology: "DomainOntology",
    _encoder: "SentenceTransformer",
    temperature: float = 0.15,
    keyword_boost: float = 0.30,
) -> QDWAWeightAllocator:
    allocator = QDWAWeightAllocator(
        ontology=_ontology,
        encoder=_encoder,
        temperature=temperature,
        keyword_boost=keyword_boost,
    )
    allocator.build_centroids()
    return allocator

def run_qdwa_and_show_weights(qdwa: QDWAWeightAllocator, query: str):
    weights = qdwa.compute_weights(query)
    df = qdwa.weights_dataframe()
    major_cat, major_w = qdwa.get_major_category()

    st.markdown("### 🎯 QDWA Category Weights")
    cols = st.columns(4)
    for i, row in df.iterrows():
        with cols[i]:
            is_major = row["Is Major"]
            border = "3px solid #FFD700" if is_major else "1px solid #333"
            st.markdown(
                f"""
                <div style="border:{border}; border-radius:8px; padding:12px;
                            background:{row['Color']}22; text-align:center;">
                    <div style="font-size:12px; color:{row['Color']}; font-weight:600;">
                        {row['Short Name']}
                    </div>
                    <div style="font-size:28px; font-weight:700; color:{row['Color']};
                                margin:4px 0;">
                        {row['Weight']:.1%}
                    </div>
                    <div style="font-size:10px; color:#888;">
                        raw={row['Raw Similarity']:.3f} · hits={int(row['Keyword Hits'])}
                    </div>
                    {"⭐ MAJOR" if is_major else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("📊 Weight detail table", expanded=False):
        show_df = df.drop(columns=["Color"]).copy()
        show_df["Weight"] = show_df["Weight"].map("{:.4f}".format)
        show_df["Raw Similarity"] = show_df["Raw Similarity"].map("{:.4f}".format)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    return df, major_cat, major_w

def render_qdwa_tab(qdwa: QDWAWeightAllocator, query: str, metrics: RunMetricsLogger):
    metrics.start_step("QDWA weight computation")
    df, major_cat, major_w = run_qdwa_and_show_weights(qdwa, query)
    metrics.end_step(extra={"major": major_cat, "major_weight": major_w})
    
    render_qdwa_customization_panel()
    
    st.markdown("### 🧠 Methodology & Category Dominance")
    W = qdwa.weights
    H = -sum(w * math.log(w) for w in W.values() if w > 1e-12)
    c = float(np.clip(1.0 - H / math.log(4), 0.0, 1.0))
    
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        if c > 0.6:
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 8px; background: rgba(16, 185, 129, 0.1); border-left: 5px solid #10B981;">
                <h4 style="margin:0; color:#065F46;">🎯 Major Category Focus (Specialized)</h4>
                <p style="margin:5px 0 0 0; color:#047857;">
                    The query is heavily dominated by <b>{major_cat}</b> ({major_w:.1%}). 
                    The system will prioritize deep-dive experts and causal chains in this domain.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif c < 0.4:
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 8px; background: rgba(59, 130, 246, 0.1); border-left: 5px solid #3B82F6;">
                <h4 style="margin:0; color:#1E3A8A;">🌐 Plural / Multi-Disciplinary</h4>
                <p style="margin:5px 0 0 0; color:#1E40AF;">
                    The query spans multiple domains relatively equally. The system will activate cross-domain bridge concepts and broad routing.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 8px; background: rgba(245, 158, 11, 0.1); border-left: 5px solid #F59E0B;">
                <h4 style="margin:0; color:#78350F;">⚖️ Balanced Distribution</h4>
                <p style="margin:5px 0 0 0; color:#92400E;">
                    The query has a primary focus on <b>{major_cat}</b> but maintains strong relevance to secondary categories.
                </p>
            </div>
            """, unsafe_allow_html=True)
    with col_m2:
        st.metric("Concentration Index ($c$)", f"{c:.3f}")
        st.caption("1.0 = Single Category\n0.0 = Uniform Plural")
        
    st.markdown("---")
    
    chart_options = ["📊 Bar Chart", "🍩 Donut Chart", "🕸️ Radar / Spider", 
                     "🔄 Sankey Flow (Simple)", "🌊 Two-Way Sankey (Terms→Cats)",
                     "🎛️ Chord (Radial)", "🔥 Heatmap", "📈 All Charts (Gallery)"]
    chosen = st.selectbox("Select visualization", chart_options, index=0)
    metrics.start_step(f"QDWA chart: {chosen}")
    
    scale = plotly_continuous_scale(st.session_state.get("qdwa_cmap", "viridis"))
    
    if "Bar" in chosen: 
        render_qdwa_bar(df, scale)
    elif "Donut" in chosen: 
        render_qdwa_donut(df, scale)
    elif "Radar" in chosen: 
        render_qdwa_radar(df, scale)
    elif "Sankey" in chosen and "Two-Way" not in chosen: 
        render_qdwa_sankey(df, query, scale)
    elif "Two-Way" in chosen:
        render_qdwa_sankey_twoway(query, df, scale)
    elif "Chord" in chosen: 
        render_qdwa_chord(df, scale)
    elif "Heatmap" in chosen: 
        render_qdwa_heatmap(df, scale)
    elif "Gallery" in chosen:
        c1, c2 = st.columns(2)
        with c1: 
            render_qdwa_bar(df, scale)
        with c2: 
            render_qdwa_donut(df, scale)
        render_qdwa_radar(df, scale)
        c3, c4 = st.columns(2)
        with c3: 
            render_qdwa_sankey(df, query, scale)
        with c4: 
            render_qdwa_heatmap(df, scale)
    metrics.end_step()
    
    with st.expander("🧮 QDWA Mathematical Framework (Robust Explanation)", expanded=False):
        st.markdown(r"""
        #### 🧮 QDWA Mathematical Framework
        The Query Distillation & Weighted Allocation (QDWA) strategy maps a natural language query $\mathbf{q}$ to a 4-dimensional weight vector $\mathbf{W} = [\alpha_1, \alpha_2, \alpha_3, \alpha_4]$ representing relevance to Rheology, Sliding/Hydro, Numerical, and Dynamics/Thermo domains.
        
        **Stage 1: Query Embedding**
        $$ \mathbf{e}_q = f_{\text{enc}}(\mathbf{q}) \in \mathbb{R}^d $$
        
        **Stage 2: Category Centroids**
        $$ \mathbf{e}_{c_i} = \frac{1}{|S_i|} \sum_{j \in S_i} \mathbf{e}_j, \quad \hat{\mathbf{e}}_{c_i} = \frac{\mathbf{e}_{c_i}}{\|\mathbf{e}_{c_i}\|_2} $$
        
        **Stage 3: Cosine Similarity & Temperature Softmax**
        $$ s_i = \hat{\mathbf{e}}_q \cdot \hat{\mathbf{e}}_{c_i}, \quad \alpha_i = \frac{\exp(s_i / \tau)}{\sum_{j=1}^4 \exp(s_j / \tau)} $$
        
        **Stage 4: Keyword Evidence Boost**
        $$ \alpha_i^{\text{final}} = \frac{\alpha_i (1 + \beta \cdot k_i)}{\sum_j \alpha_j (1 + \beta \cdot k_j)} $$
        
        **Stage 5: Concentration & Routing Depth**
        $$ H(\mathbf{W}) = -\sum_{i=1}^4 \alpha_i^{\text{final}} \log(\alpha_i^{\text{final}}), \quad c = 1 - \frac{H(\mathbf{W})}{\log 4} $$
        """)
        st.markdown(f"""
        | Parameter | Current Value | Meaning |
        |-----------|-------|---------|
        | τ (temperature) | {qdwa.temperature} | Low → sharper distribution |
        | β (boost) | {qdwa.keyword_boost} | Per-keyword-hit multiplier |
        | $k_i$ (keyword hits) | {dict(qdwa.keyword_hits)} | Exact matches in category $i$ |
        | $c$ (concentration) | {c:.3f} | Dominance index |
        """)
        
    csv = df.drop(columns=["Color"]).to_csv(index=False)
    st.download_button("⬇️ Download weights CSV", csv.encode("utf-8"), 
                       file_name=f"qdwa_weights_{hashlib.md5(query.encode()).hexdigest()[:8]}.csv", 
                       mime="text/csv")


# ============================================================================
# LLM QUERY PANEL (Sidebar)
# ============================================================================
def render_llm_query_panel(ontology: Any, expander: DynamicOntologyExpander, full_graph: nx.Graph) -> Optional[QueryAnalysisResult]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 LLM-Guided Query")
    st.sidebar.caption("Ask a question to dynamically expand the ontology and focus the graph")

    session = QuerySessionManager.get_session()
    mode = st.sidebar.selectbox("Analysis Engine", ["auto", "fallback", "openai", "local"], index=["auto", "fallback", "openai", "local"].index(session.get("analyzer_mode", "auto")), key="llm_mode_select")
    session["analyzer_mode"] = mode

    api_key = None
    if mode in ("auto", "openai"):
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value=os.environ.get("OPENAI_API_KEY", ""), key="openai_key_input")

    local_model = None
    if mode in ("auto", "local"):
        st.sidebar.markdown("#### 🖥️ Local LLM Model")
        st.sidebar.caption("⚠️ Streamlit Cloud ≈1 GB RAM. Pick a small model or use Fallback.")
        LOCAL_LLM_REGISTRY = {
            "Fallback (Rule-based, no LLM)": None,
            "DistilGPT-2 (82M, fastest)": "distilgpt2",
            "GPT-Neo-125M (125M)": "EleutherAI/gpt-neo-125M",
            "Pythia-410M (410M, balanced)": "EleutherAI/pythia-410m",
            "BLOOM-560M (560M, multilingual)": "bigscience/bloom-560m",
            "Qwen2-0.5B-Instruct (500M, best JSON)": "Qwen/Qwen2-0.5B-Instruct",
            "Qwen2.5-0.5B-Instruct (500M, newest)": "Qwen/Qwen2.5-0.5B-Instruct",
        }
        model_display_names = list(LOCAL_LLM_REGISTRY.keys())
        selected_display = st.sidebar.selectbox("Select model:", options=model_display_names, index=0, key="local_model_select")
        local_model = LOCAL_LLM_REGISTRY[selected_display]
        st.session_state['selected_local_model'] = local_model

    example_queries = [q for pdef in GLACIER_PROBLEM_DEFINITIONS.values() for q in pdef.example_queries[:1]]
    selected_example = st.sidebar.selectbox("Or select an example:", [""] + example_queries, key="example_query_select")
    if selected_example:
        st.session_state['synced_query'] = selected_example

    query = st.sidebar.text_area(
        "Your question about glacial flow mechanics:", 
        value=st.session_state.get('synced_query', "How does Coulomb friction affect grounding line dynamics?"),
        height=100, key="llm_query_input", 
        placeholder="e.g., How does enthalpy solver couple with temperature?"
    )

    submitted = st.sidebar.button("🚀 Analyze & Expand Ontology", type="primary", key="llm_submit")
    if not submitted or not query.strip():
        return None

    st.session_state['synced_query'] = query

    factory = LLMQueryAnalyzerFactory()
    analyzer = factory.get_analyzer(mode=mode, api_key=api_key, local_model=local_model)

    if isinstance(analyzer, OpenAIQueryAnalyzer):
        st.sidebar.info("🤖 Using **OpenAI GPT-4o-mini**")
    elif isinstance(analyzer, LocalLLMQueryAnalyzer):
        st.sidebar.info("🖥️ Using **Local LLM**")
    else:
        st.sidebar.info("📋 Using **Rule-based fallback**")

    with st.sidebar.spinner("Analyzing query..."):
        analysis = analyzer.analyze_query(query, ontology)
    with st.sidebar.spinner("Expanding ontology..."):
        mutations = expander.apply_query_analysis(analysis, analyzer)

    if hasattr(analyzer, 'unload_model'):
        analyzer.unload_model()
    del analyzer
    gc.collect()

    whitelist = set(analysis.explicitly_mentioned)
    whitelist.update(analysis.inferred_concepts)
    whitelist.update(expander.session_concepts_added)
    whitelist.update(expander.query_bridge_concepts.keys())
    st.session_state['last_query_analysis'] = analysis
    st.session_state['last_query_text'] = query
    st.session_state['last_query_whitelist'] = whitelist
    st.session_state['last_query_dynamic_concepts'] = expander.session_concepts_added
    st.session_state['last_query_bridge_concepts'] = expander.query_bridge_concepts

    QuerySessionManager.record_query(query, analysis, mutations)

    st.sidebar.success(f"✅ Analysis complete (confidence: {analysis.confidence:.0%})")
    st.sidebar.caption(f"Primary problem: **{analysis.primary_problem.value}**")
    st.sidebar.caption(f"Explicit concepts: {len(analysis.explicitly_mentioned)} | Inferred: {len(analysis.inferred_concepts)}")
    if mutations["concepts_added"]:
        st.sidebar.warning(f"🆕 {len(mutations['concepts_added'])} new concept(s) added")
        for c in mutations["concepts_added"]:
            st.sidebar.markdown(f"  - `{c['name']}` ({c['type']})")
    if mutations["bridges_created"]:
        st.sidebar.info(f"🌉 {len(mutations['bridges_created'])} bridge concept(s) created")
        for b in mutations["bridges_created"]:
            st.sidebar.markdown(f"  - `{b['bridge']}` ← `{b['for']}`")

    if analysis is not None and getattr(analysis, "category_weights", None):
        render_category_weight_gauge(
            analysis.category_weights, analysis.concentration, container=st.sidebar
        )

    return analysis


def render_mutation_controls(expander: DynamicOntologyExpander) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Ontology Mutations")
    stats = expander.stats
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Concepts +", stats["concepts_added"])
    col2.metric("Relations +", stats["relationships_added"])
    if stats["total_mutations"] > 0:
        with st.sidebar.expander("📋 Mutation Log", expanded=False):
            for i, mut in enumerate(expander.mutation_log[-10:], 1):
                if mut["type"] == "add_concept":
                    st.sidebar.markdown(f"{i}. ➕ `{mut['concept']}`")
                elif mut["type"] == "add_relationship":
                    st.sidebar.markdown(f"{i}. 🔗 `{mut['source']}` → `{mut['target']}`")
                elif mut["type"] == "create_bridge":
                    st.sidebar.markdown(f"{i}. 🌉 `{mut['bridge_name']}`")
        col_undo, col_reset = st.sidebar.columns(2)
        if col_undo.button("↩️ Undo Last", key="undo_mutation"):
            undone = expander.undo_last_mutation()
            if undone:
                st.sidebar.toast(f"Undone: {undone['type']}")
                st.rerun()
        if col_reset.button("🔄 Reset All", key="reset_mutations"):
            result = expander.reset_to_base()
            st.sidebar.toast(f"Reset: {result['concepts_removed']} concepts, {result['relationships_removed']} relations removed")
            st.rerun()


def render_query_history() -> None:
    session = QuerySessionManager.get_session()
    if not session["query_history"]:
        return
    st.sidebar.markdown("---")
    with st.sidebar.expander("📜 Query History", expanded=False):
        for i, entry in enumerate(reversed(session["analysis_history"][-10:]), 1):
            st.sidebar.markdown(f"**{i}.** {entry['query'][:60]}...")
            st.sidebar.caption(f"  Problem: {entry['primary_problem']} | Type: {entry['query_type']} | Concepts: {entry['concepts_found']}")


def render_analysis_details(analysis: QueryAnalysisResult) -> None:
    if analysis is not None and getattr(analysis, "category_weights", None):
        render_category_weight_gauge(analysis.category_weights, analysis.concentration)

    st.markdown("## 📊 Query Analysis Results")
    with st.expander("🧠 Reasoning Chain", expanded=True):
        for step in analysis.reasoning_chain:
            st.markdown(f"→ {step}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Primary Problem", analysis.primary_problem.value.replace("_", " "))
    col2.metric("Query Type", analysis.query_type)
    col3.metric("Confidence", f"{analysis.confidence:.0%}")

    st.markdown("### Concept Priority Rankings")
    top = analysis.get_top_concepts(15)
    if top:
        df = pd.DataFrame([cp.to_dict() for cp in top])
        def highlight_row(row):
            if row.get("explicit", False):
                return ["background-color: #d4edda"] * len(row)
            elif row.get("inferred", False):
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)


def render_llm_qa_tab(analysis_data: Dict, ontology: Any):
    st.subheader("🤖 LLM-Guided Graph Q&A")
    st.markdown("Ask a specific scientific question about glacial flow mechanics (Elmer/Ice). The system will dynamically expand the ontology, extract a relevant subgraph, and generate a grounded answer using retrieved literature snippets.")

    if "qa_factory" not in st.session_state:
        st.session_state.qa_factory = LLMQueryAnalyzerFactory()
    if "qa_expander" not in st.session_state:
        st.session_state.qa_expander = DynamicOntologyExpander(ontology)
    if "qa_generator" not in st.session_state:
        st.session_state.qa_generator = GraphRAGAnswerGenerator(st.session_state.qa_factory.get_analyzer("auto"))

    factory = st.session_state.qa_factory
    expander = st.session_state.qa_expander
    generator = st.session_state.qa_generator

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your research question:", placeholder="e.g., How does enthalpy solver affect grounding line migration?")
    with col2:
        mode = st.selectbox("Engine", ["auto", "openai", "local", "fallback"], index=0)

    if st.button("🔍 Analyze & Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        local_model = st.session_state.get('selected_local_model')
        analyzer = factory.get_analyzer(mode=mode, local_model=local_model)
        generator.analyzer = analyzer

        with st.spinner("🧠 Analyzing query and expanding ontology..."):
            analysis = analyzer.analyze_query(query, ontology)
            mutations = expander.apply_query_analysis(analysis, analyzer)

            whitelist = set(analysis.explicitly_mentioned)
            whitelist.update(analysis.inferred_concepts)
            whitelist.update(expander.session_concepts_added)
            whitelist.update(expander.query_bridge_concepts.keys())
            st.session_state['last_query_analysis'] = analysis
            st.session_state['last_query_text'] = query
            st.session_state['last_query_whitelist'] = whitelist
            st.session_state['last_query_dynamic_concepts'] = expander.session_concepts_added
            st.session_state['last_query_bridge_concepts'] = expander.query_bridge_concepts

            if st.session_state.get('query_focused_build'):
                st.success(f"✅ Query analysis complete. Whitelist contains {len(whitelist)} concepts.")
                if st.button("🔧 Rebuild Graph for This Query", type="primary", key="rebuild_for_query_btn"):
                    st.session_state['force_rebuild'] = True
                    st.rerun()

        with st.spinner("🕸️ Extracting priority-guided subgraph..."):
            full_graph = analysis_data["nx_graph"]
            extractor = PriorityGuidedSubgraphExtractor(full_graph, ontology, expander)
            embed_model = analysis_data.get("embed_model")
            if embed_model is not None:
                st.session_state['embed_model'] = embed_model
            query_embedding = None
            if embed_model is not None:
                try:
                    with torch.no_grad():
                        query_embedding = embed_model.encode(query, convert_to_numpy=True)
                except Exception:
                    pass
            subgraph = extractor.extract(analysis, query_embedding)

        with st.spinner("📚 Retrieving evidence and generating answer..."):
            answer = generator.generate_ground_response(
                query=query, analysis=analysis, subgraph=subgraph,
                concept_abstract_map=analysis_data["concept_abstract_map"],
                all_texts=analysis_data.get("all_texts", []),
                max_docs_per_concept=2
            )

        if hasattr(analyzer, 'unload_model'):
            analyzer.unload_model()
        del analyzer
        gc.collect()

        st.markdown("### 💡 Generated Answer")
        st.markdown(answer)
        st.markdown("---")
        st.markdown("### 🕸️ Focused Subgraph Visualization")
        visualizer = QueryDrivenVisualizer(ontology)
        html = visualizer.render_pyvis(subgraph, analysis)
        st.components.v1.html(html, height=600, scrolling=True)
        with st.expander("🔧 Behind the Scenes: Ontology Mutations & Reasoning"):
            st.markdown("**Reasoning Chain:**")
            for step in analysis.reasoning_chain:
                st.markdown("- " + step)
            if mutations.get("concepts_added") or mutations.get("bridges_created"):
                st.markdown("**Dynamic Ontology Updates:**")
                for c in mutations.get("concepts_added", []):
                    st.markdown("➕ Added Concept: `" + c['name'] + "` (" + c['type'] + ")")
                for b in mutations.get("bridges_created", []):
                    st.markdown("🌉 Created Bridge: `" + b['bridge'] + "` for `" + b['for'] + "`")


# ============================================================================
# QUERY‑DRIVEN VISUALIZER
# ============================================================================
class QueryDrivenVisualizer:
    def __init__(self, ontology: Any):
        self.ontology = ontology
        self.type_colors = {
            "material": "#FF6B6B", "property": "#4ECDC4", "phenomenon": "#FFE66D",
            "method": "#95E1D3", "parameter": "#F38181", "process": "#AA96DA",
            "model": "#FCBAD3", "general": "#A8D8EA"
        }

    def render_pyvis(self, subgraph: nx.Graph, analysis: QueryAnalysisResult, height: str = "700px",
                     physics_enabled: bool = True,
                     gravity: float = -800.0,
                     central_gravity: float = 0.1,
                     spring_length: float = 120,
                     spring_strength: float = 0.02,
                     damping: float = 0.95) -> str:
        from pyvis.network import Network
        net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="remote")
        if physics_enabled:
            net.barnes_hut(
                gravity=gravity,
                central_gravity=central_gravity,
                spring_length=spring_length,
                spring_strength=spring_strength,
                damping=damping,
                overlap=0.1
            )
        else:
            net.set_options('{"physics": {"enabled": false}, "interaction": {"hover": true, "dragNodes": true, "dragView": true, "zoomView": true}}')
        for node, attrs in subgraph.nodes(data=True):
            concept_type = attrs.get("concept_type", "general")
            priority = attrs.get("priority_score", 0.2)
            is_explicit = attrs.get("is_explicit", False)
            is_llm_added = attrs.get("is_llm_added", False)
            size = 15 + priority * 35
            color = self.type_colors.get(concept_type, "#A8D8EA")
            if is_explicit:
                border_width, border_color, shape = 4, "#FF0000", "dot"
            elif is_llm_added:
                border_width, border_color, shape = 3, "#00FF00", "diamond"
            else:
                border_width, border_color, shape = 1, "#666666", "dot"
            title = "<b>" + node + "</b><br>Type: " + concept_type + "<br>Priority: " + str(round(priority, 2))
            if is_llm_added:
                title += "<br>⚠️ LLM-inferred concept"
            defn = attrs.get("definition", "")
            if defn:
                title += "<br><i>" + defn[:150] + "...</i>"
            net.add_node(node, label=node.replace("_", " ").title(), size=size, color=color,
                         border_width=border_width, border_color=border_color, shape=shape,
                         title=title, font={"size": 10 + priority * 6})
        for u, v, attrs in subgraph.edges(data=True):
            color = attrs.get("color", "#888888")
            width = attrs.get("width", 1.0)
            highlighted = any(len(p) >= 2 and ((p[0] == u and p[1] == v) or (p[1] == u and p[0] == v)) for p in analysis.highlight_paths)
            if highlighted:
                color, width = "#FF0000", max(width, 4.0)
            net.add_edge(u, v, color=color, width=width,
                         dashes=attrs.get("style") == "dashed" or attrs.get("inferred", False),
                         title=u + " → " + v + "<br>Type: " + attrs.get('edge_type','unknown'),
                         arrows="to")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            return Path(f.name).read_text(encoding='utf-8')


# ============================================================================
# SIDEBAR RENDERING
# ============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration v6.2 + QDWA (Glacier)")
        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0,
        )
        theme = THEME_PRESETS[st.session_state['theme']]
        st.subheader("🧊 Glacial Flow Mechanics Focus Areas")
        st.markdown(r"- **Rheology:** Glen's law, Arrhenius factor, enhancement, damage, fabric")
        st.markdown(r"- **Basal Sliding:** Coulomb/Weertman friction, effective pressure, hydrology")
        st.markdown(r"- **Numerical:** ParStokes, preconditioning, meshing, adjoint, free surface")
        st.markdown(r"- **Dynamics/Thermo:** Enthalpy, temperature, calving, SMB, basal melt")
        st.subheader("🧠 NLP Reasoning Options")
        st.session_state['use_ontology'] = st.checkbox(
            "Use ontology-based resolution", value=True,
            help="Maps synonyms like 'Glen's law', 'SSA' to canonical concepts",
        )
        st.session_state['use_embedding_resolution'] = st.checkbox(
            "Use embedding-based semantic equivalence", value=True,
            help="Detects semantic similarity >0.85 even for unseen variants",
        )
        st.session_state['use_relationship_extraction'] = st.checkbox(
            "Extract cause-effect relationships", value=True,
            help="Identifies causal links between glacial parameters and ice dynamics",
        )
        st.session_state['use_inference'] = st.checkbox(
            "Enable reasoning-based edge inference", value=True,
            help="Infers method→property chains even when not co-occurring",
        )
        st.session_state['context_window'] = st.slider(
            "Context window (chars)", 20, 200, 50,
            help="Window size for context-based disambiguation",
        )
        st.subheader("📊 Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:",
            ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"],
            index=0,
        )
        st.session_state['show_edge_weights'] = st.toggle(
            "Show edge weights", value=False,
            help="Display numerical weight labels on graph edges.",
        )
        st.session_state['edge_label_mode'] = st.selectbox(
            "Edge label mode:", ["hover", "threshold", "all"], index=0,
            help="hover=tooltip only, threshold=top 20% edges, all=all edges",
        )
        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:",
            options=list(SUPPORTED_COLORMAPS.keys()),
            index=0,
        )
        st.subheader("⚡ Physics & Layout")
        st.session_state['physics_preset'] = st.selectbox(
            "Physics preset:",
            options=list(PHYSICS_PRESETS.keys()),
            index=0,
        )
        preset = PHYSICS_PRESETS[st.session_state['physics_preset']]
        st.session_state['physics_enabled'] = st.checkbox(
            "Enable physics", value=(preset["gravity"] != 0),
        )
        with st.expander("Advanced Physics Overrides"):
            st.session_state['adv_damping'] = st.slider(
                "Damping", 0.05, 0.95, preset["damping"], step=0.05,
            )
            st.session_state['adv_gravity'] = st.slider(
                "Repulsion", -8000, -500, preset["gravity"], step=100,
            )
            st.session_state['adv_spring_length'] = st.slider(
                "Spring length", 40, 300, preset["spring_length"], step=10,
            )
            st.session_state['adv_spring_strength'] = st.slider(
                "Spring strength", 0.01, 0.20,
                preset["spring_strength"], step=0.01,
            )
            st.session_state['adv_central_gravity'] = st.slider(
                "Central gravity", 0.0, 0.5,
                preset["central_gravity"], step=0.05,
            )
            st.session_state['adv_stabilization'] = st.slider(
                "Stabilization iter", 0, 5000,
                preset["stabilization"], step=250,
            )
        base_preset = PHYSICS_PRESETS[
            st.session_state['physics_preset']
        ].copy()
        if st.session_state.get('adv_damping') is not None:
            base_preset["damping"] = st.session_state['adv_damping']
            base_preset["gravity"] = st.session_state['adv_gravity']
            base_preset["spring_length"] = st.session_state['adv_spring_length']
            base_preset["spring_strength"] = st.session_state['adv_spring_strength']
            base_preset["central_gravity"] = st.session_state['adv_central_gravity']
            base_preset["stabilization"] = st.session_state['adv_stabilization']
        st.session_state['effective_physics'] = base_preset
        st.subheader("📏 Display Limits")
        col_all1, col_slider1 = st.columns([0.3, 0.7])
        with col_all1:
            all_graph = st.checkbox("All", value=True, key="all_graph_chk")
        with col_slider1:
            st.session_state['top_n_graph'] = st.slider(
                "Max nodes", 10, 500, 200, step=10,
                disabled=all_graph, key="top_n_graph_slider",
            )
        if all_graph:
            st.session_state['top_n_graph'] = 0
        col_all2, col_slider2 = st.columns([0.3, 0.7])
        with col_all2:
            all_sun = st.checkbox("All", value=True, key="all_sun_chk")
        with col_slider2:
            st.session_state['top_n_sunburst'] = st.slider(
                "Max children/category", 10, 100, 40, step=10,
                disabled=all_sun, key="top_n_sunburst_slider",
            )
        if all_sun:
            st.session_state['top_n_sunburst'] = 0
        col_all3, col_slider3 = st.columns([0.3, 0.7])
        with col_all3:
            all_radar = st.checkbox("All", value=True, key="all_radar_chk")
        with col_slider3:
            st.session_state['top_n_radar'] = st.slider(
                "Top K for radar", 5, 30, 15,
                disabled=all_radar, key="top_n_radar_slider",
            )
        if all_radar:
            st.session_state['top_n_radar'] = 0
        st.subheader("🔧 Graph Parameters")
        st.session_state['min_freq'] = st.slider(
            "Min concept frequency", 1, 20, 1,
        )
        st.session_state['min_words'] = st.slider(
            "Min words per concept", 2, 5, 2,
        )
        st.session_state['sim_threshold'] = st.slider(
            "Semantic threshold", 0.6, 0.95, 0.85, step=0.05,
        )
        st.session_state['cooc_weight'] = st.slider(
            "Co-occurrence weight", 0.5, 1.0, 0.7, step=0.1,
        )
        st.session_state['sem_weight'] = st.slider(
            "Semantic weight", 0.0, 0.5, 0.2, step=0.1,
        )
        st.session_state['inf_weight'] = st.slider(
            "Inference weight", 0.0, 0.3, 0.1, step=0.05,
        )

        st.subheader("🔍 Query-Focused Graph Mode")
        st.checkbox("Build graph only for current query concepts", key="query_focused_build")
        if st.session_state.get('query_focused_build', False):
            whitelist = st.session_state.get('last_query_whitelist', set())
            if whitelist:
                st.success(f"Will extract {len(whitelist)} focused concepts")
                if st.session_state.get('batch_mode', False):
                    st.info("📦 Batch mode compatible — frequency threshold auto‑lowered.")
                with st.expander("Preview whitelisted concepts"):
                    st.write(sorted(whitelist))
            else:
                st.info("Ask a question in the 🤖 LLM-Guided Q&A tab to generate a whitelist.")

        render_batch_processing_controls()

        st.subheader("📈 Statistics")
        st.session_state['bootstrap_samples'] = st.slider(
            "Bootstrap samples", 100, 2000, 500, step=100,
        )
        st.session_state['alpha_level'] = st.selectbox(
            "Significance alpha", [0.01, 0.05, 0.10], index=1,
        )

        st.markdown("---")
        st.subheader("🎨 Visualization Customization")
        st.session_state['enable_node_highlight'] = st.checkbox(
            "🔍 Enable Node Selection Highlight & Descriptions",
            value=False,
            help=(
                "When enabled, clicking a node highlights connected nodes "
                "with gold borders and overlays edge weights/relationship descriptions."
            ),
        )
        with st.expander("Node & Label Settings"):
            st.session_state['node_label_size'] = st.slider(
                "Node label font size", 8, 24, 12, step=1,
            )
            st.session_state['node_label_position'] = st.selectbox(
                "Node label position",
                ["center", "top", "bottom", "left", "right"],
                index=0,
            )
            st.session_state['node_font_face'] = st.selectbox(
                "Node font family",
                [
                    "Inter, Segoe UI, Roboto, sans-serif",
                    "Arial, Helvetica, sans-serif",
                    "Georgia, serif",
                    "Courier New, monospace",
                    "Times New Roman, serif",
                ],
                index=0,
            )
            st.slider(
                "Node legend font size", 8, 20, 13, step=1,
                key="node_legend_font_size",
            )
        st.session_state['use_abbreviated_labels'] = st.checkbox(
            "Use short labels (N1, N2...) for long names",
            value=False,
        )
        if st.session_state['use_abbreviated_labels']:
            st.session_state['max_label_length'] = st.slider(
                "Max label length before abbreviation",
                min_value=2, max_value=50, value=15, step=1,
            )
        else:
            st.session_state['max_label_length'] = 15
        st.session_state['show_definitions'] = st.checkbox(
            "📖 Show concept definitions in tooltips",
            value=True,
        )
        st.slider(
            "Tooltip font size", 10, 20, 13, step=1,
            key="tooltip_font_size",
        )
        with st.expander("Edge Label Settings"):
            st.session_state['edge_label_size'] = st.slider(
                "Edge label font size", 6, 18, 10, step=1,
            )
            st.session_state['edge_label_color'] = st.color_picker(
                "Edge label color", value="#000000",
            )
            st.session_state['edge_label_position'] = st.selectbox(
                "Edge label position",
                ["middle", "top", "bottom", "from", "to"],
                index=0,
            )
        edge_color_value = st.session_state.get('edge_label_color')
        if not edge_color_value or edge_color_value == '':
            edge_color_value = '#000000'
        st.session_state['edge_label_color'] = edge_color_value

        with st.expander("Edge Color Customization"):
            st.selectbox(
                "Edge color mode",
                ["theme", "uniform_grey", "custom"],
                index=0,
                key="edge_color_mode",
            )
            if st.session_state['edge_color_mode'] == "custom":
                st.color_picker(
                    "Custom edge color", value="#AAAAAA",
                    key="custom_edge_color",
                )
            else:
                st.session_state['custom_edge_color'] = "#AAAAAA"
            st.slider(
                "Edge lightness (0=original, 1=white)", 0.0, 1.0, 0.6, step=0.05,
                key="edge_lightness",
            )

        st.markdown("---")
        st.subheader("✏️ Graph Editing")
        with st.expander("Remove Nodes"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                nodes_to_remove = st.multiselect(
                    "Select nodes to remove:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="remove_nodes_select",
                )
                st.session_state['nodes_to_remove'] = nodes_to_remove
            else:
                st.info("Build graph first to edit nodes.")
                st.session_state['nodes_to_remove'] = []
        with st.expander("Merge Nodes"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                nodes_to_merge = st.multiselect(
                    "Select nodes to merge:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="merge_nodes_select",
                )
                merge_name = st.text_input(
                    "New merged concept name:", key="merge_name_input",
                )
                st.session_state['nodes_to_merge'] = nodes_to_merge
                st.session_state['merge_name'] = merge_name
            else:
                st.info("Build graph first to merge nodes.")
                st.session_state['nodes_to_merge'] = []
                st.session_state['merge_name'] = ""
        with st.expander("Add Edge"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                all_concepts = st.session_state['analysis_data']['valid_concepts']
                edge_u = st.selectbox(
                    "Source concept:", options=all_concepts, key="edge_u_select",
                )
                edge_v = st.selectbox(
                    "Target concept:", options=all_concepts, key="edge_v_select",
                )
                edge_weight = st.number_input(
                    "Edge weight:", min_value=0.1, max_value=10.0,
                    value=1.0, step=0.1, key="edge_weight_input",
                )
                st.session_state['new_edge'] = (
                    (edge_u, edge_v) if edge_u != edge_v else None
                )
                st.session_state['new_edge_weight'] = edge_weight
            else:
                st.info("Build graph first to add edges.")
                st.session_state['new_edge'] = None
                st.session_state['new_edge_weight'] = 1.0
        with st.expander("Filter by Degree/Frequency"):
            st.session_state['filter_min_degree'] = st.slider(
                "Min degree", 0, 20, 0, key="filter_degree_slider",
            )
            st.session_state['filter_min_freq'] = st.slider(
                "Min frequency", 0, 50, 0, key="filter_freq_slider",
            )
        if (
            st.session_state.get('analysis_data')
            and st.session_state['analysis_data'].get('valid_concepts')
        ):
            if st.button("Apply Graph Edits", key="apply_edits_btn"):
                st.session_state['apply_edits'] = True
        if (
            st.session_state.get('analysis_data')
            and st.session_state.get('edit_history')
        ):
            col_undo, col_redo = st.columns(2)
            with col_undo:
                if (
                    st.button("↩️ Undo", key="undo_btn")
                    and st.session_state['edit_history'].can_undo()
                ):
                    snapshot = st.session_state['edit_history'].undo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Undo applied!")
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
            with col_redo:
                if (
                    st.button("↪️ Redo", key="redo_btn")
                    and st.session_state['edit_history'].can_redo()
                ):
                    snapshot = st.session_state['edit_history'].redo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Redo applied!")
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()

        st.markdown("---")
        st.subheader("☀️ Sunburst Chart Customization")
        st.session_state['sunburst_cmap'] = st.selectbox(
            "Colormap:",
            options=[
                "viridis", "plasma", "inferno", "magma", "cividis",
                "turbo", "rainbow", "hsv", "coolwarm", "RdBu", "Spectral",
                "tab10", "tab20", "Pastel1", "Set1", "Set2", "Set3",
                "YlOrRd", "PuBuGn", "GnBu", "YlGnBu",
            ],
            index=0,
            key="sunburst_cmap_select",
        )
        st.session_state['sunburst_font_family'] = st.selectbox(
            "Sunburst font family",
            [
                "Arial, sans-serif",
                "Inter, Segoe UI, Roboto, sans-serif",
                "Georgia, serif",
                "Courier New, monospace",
                "Times New Roman, serif",
            ],
            index=0,
            key="sunburst_font_family_select",
        )
        col_labels, col_values = st.columns(2)
        with col_labels:
            st.session_state['sunburst_show_labels'] = st.checkbox(
                "Show symbols", value=True,
                key="sunburst_show_labels_chk",
            )
        with col_values:
            st.session_state['sunburst_show_values'] = st.checkbox(
                "Show values", value=False,
                key="sunburst_show_values_chk",
            )
        st.session_state['sunburst_hover_info'] = st.selectbox(
            "Hover information:",
            options=["all", "minimal", "none"],
            index=0,
            key="sunburst_hover_select",
        )
        st.session_state['sunburst_branchvalues'] = st.selectbox(
            "Branch values mode:", ["total", "remainder"], index=0,
            key="sunburst_branch_mode",
        )
        col_w, col_h = st.columns(2)
        with col_w:
            st.session_state['sunburst_width'] = st.slider(
                "Chart width (px)", 600, 1400, 900, step=50,
                key="sunburst_width_slider",
            )
        with col_h:
            st.session_state['sunburst_height'] = st.slider(
                "Chart height (px)", 500, 1200, 700, step=50,
                key="sunburst_height_slider",
            )
        st.session_state['sunburst_label_size'] = st.slider(
            "Symbol font size", 8, 30, 20, step=1,
            key="sunburst_label_size_slider",
        )
        st.slider(
            "Sunburst legend font size", 8, 20, 12, step=1,
            key="sunburst_legend_font_size",
        )
        st.session_state['sunburst_show_legend'] = st.checkbox(
            "Show symbol legend", value=True,
            key="sunburst_show_legend_chk",
        )
        if (
            st.session_state.get('analysis_data')
            and st.session_state['analysis_data'].get('valid_concepts')
        ):
            all_cats = list(set(
                categorize_glacier_concept(c) for c in st.session_state['analysis_data']['valid_concepts']
            ))
            st.session_state['sunburst_categories'] = st.multiselect(
                "Filter categories:", options=all_cats,
                default=all_cats, key="sunburst_cat_filter",
            )
        else:
            st.info("Build graph first to filter categories.")
            st.session_state['sunburst_categories'] = []

        st.markdown("---")
        with st.expander("⚡ Performance Monitor"):
            if st.button("Show Timing Report"):
                report = PerformanceMonitor.get_report()
                if report:
                    st.code(report, language="text")
                else:
                    st.info("No timing data yet. Run analysis first.")
            if st.button("Reset Timings"):
                PerformanceMonitor.reset()
                st.success("Timing data reset!")

        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"Device: {gpu_info}")

        st.markdown("---")
        st.subheader("🔍 LLM-Guided Query (Pre-Build Scope)")
        st.caption("Ask a question BEFORE building to narrow the concept scope")

        ontology = st.session_state.ontology
        expander = st.session_state.qa_expander

        full_graph = (
            st.session_state.analysis_data.get("nx_graph")
            if st.session_state.get("analysis_data")
            else nx.Graph()
        )

        render_llm_query_panel(ontology, expander, full_graph)
        render_mutation_controls(expander)
        render_query_history()


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    st.title("🧊 Glacial Flow Mechanics (Elmer/Ice) Concept Graph v6.2 + QDWA")
    st.caption(
        "Multi-level reasoning concept graph for glacial flow mechanics with the Elmer/Ice finite element model | "
        "Focus: Rheology, basal sliding, numerical solvers, ice dynamics & thermodynamics | "
        "Memory-Safe | Batch Processing (≤ 1 GB) | Interactive Visualization | "
        "Ontology-aware resolution | Query Distillation & Weighted Allocation (QDWA)"
    )

    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology

    if 'qa_factory' not in st.session_state:
        st.session_state.qa_factory = LLMQueryAnalyzerFactory()
    if 'qa_expander' not in st.session_state:
        st.session_state.qa_expander = DynamicOntologyExpander(ontology)
    if 'qa_generator' not in st.session_state:
        st.session_state.qa_generator = GraphRAGAnswerGenerator(st.session_state.qa_factory.get_analyzer("auto"))

    render_sidebar()

    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None
    if "apply_edits" not in st.session_state:
        st.session_state.apply_edits = False
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = GraphEditHistory()
    if "burst_df" not in st.session_state:
        st.session_state.burst_df = None
    if "drift_df" not in st.session_state:
        st.session_state.drift_df = None
    if "genealogy_df" not in st.session_state:
        st.session_state.genealogy_df = None
    if "bridge_df" not in st.session_state:
        st.session_state.bridge_df = None
    if "motifs" not in st.session_state:
        st.session_state.motifs = {}
    if "last_query_analysis" not in st.session_state:
        st.session_state.last_query_analysis = None
    if "last_query_text" not in st.session_state:
        st.session_state.last_query_text = ""
    if "last_query_whitelist" not in st.session_state:
        st.session_state.last_query_whitelist = set()
    if "last_query_dynamic_concepts" not in st.session_state:
        st.session_state.last_query_dynamic_concepts = set()
    if "last_query_bridge_concepts" not in st.session_state:
        st.session_state.last_query_bridge_concepts = {}
    if "query_focused_build" not in st.session_state:
        st.session_state.query_focused_build = False

    st.header("📁 Data Loading")
    st.info(f"Place JSON/BibTeX/CSV files in: `{JSON_METADATA_DIR}`")
    with st.spinner("Scanning json_metadatabase..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)

    if not file_records:
        st.warning("No .json/.bib/.csv files found in the directory.")
        st.info("Please place your metadata files in the `json_metadatabase/` folder.")
        return
    successful_files = [f for f in file_records if f[1]]
    if not successful_files:
        st.error("Files found but none could be parsed. Check error messages above.")
        return
    st.success(f"Loaded {len(successful_files)} file(s) | {len(df)} record(s)")
    file_names = [f[0] for f in successful_files]
    selected_files = st.multiselect("Filter by source file", file_names, default=file_names)
    if selected_files:
        df_filtered = df[df["_source_file"].isin(selected_files)].copy()
    else:
        df_filtered = df.copy()
    st.write(f"Working with **{len(df_filtered)}** records")
    with st.expander("Preview Data Structure"):
        st.dataframe(df_filtered.head(5), use_container_width=True)
        st.markdown("**Available columns:**")
        st.write(list(df_filtered.columns))

    text_cols = [
        c for c in df_filtered.columns
        if any(k in c.lower() for k in ['abstract', 'title', 'summary', 'text', 'content', 'description'])
    ]
    if not text_cols:
        text_cols = [c for c in df_filtered.columns if df_filtered[c].dtype == 'object']
    selected_text_cols = st.multiselect(
        "Select text columns for concept extraction:",
        options=text_cols,
        default=text_cols[:2] if len(text_cols) >= 2 else text_cols,
    )
    if not selected_text_cols:
        st.error("Please select at least one text column.")
        return

    build_clicked = st.button(
        "🚀 Build Concept Graph with Reasoning",
        type="primary",
        use_container_width=True,
    )

    force_rebuild = st.session_state.pop("force_rebuild", False)
    should_build = build_clicked or force_rebuild
    batch_trigger = st.session_state.pop("batch_trigger", None)
    batch_mode_on = st.session_state.get("batch_mode", False)

    if batch_mode_on and (should_build or batch_trigger):
        run_batch_analysis(
            df_filtered=df_filtered,
            selected_text_cols=selected_text_cols,
            ontology=ontology,
            run_mode=(batch_trigger or "all"),
        )
    elif should_build:
        progress_bar = st.progress(0.0)
        status = st.status("Initializing advanced NLP analysis...", expanded=True)
        overall_start = time.perf_counter()
        try:
            with status:
                metrics = get_metrics_logger()
                metrics.reset()
                metrics.set("mode", "standard")
                metrics.set("total_docs", len(df_filtered))
                live_panel = st.empty()

                st.write("Preparing text corpus...")
                all_texts = []
                for idx, row in df_filtered.iterrows():
                    text = " ".join([str(row[col]) for col in selected_text_cols if col in row and pd.notna(row[col])])
                    all_texts.append(text)
                num_abstracts = len(all_texts)
                st.write(f"Prepared {num_abstracts} documents")
                progress_bar.progress(0.05)

                st.write("Loading embedding model...")
                embed_model = load_embedding_model()
                st.success("Embedding model loaded")
                progress_bar.progress(0.10)

                config = get_adaptive_config(num_abstracts)
                config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)
                config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
                config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
                config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.7)
                config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.2)
                config["INFERENCE_WEIGHT"] = st.session_state.get('inf_weight', 0.1)
                progress_bar.progress(0.15)

                whitelist = build_query_whitelist(st.session_state)
                is_query_focused = st.session_state.get('query_focused_build', False) and whitelist is not None and len(whitelist) > 0
                if is_query_focused:
                    st.info(f"🎯 Query-focused build: {len(whitelist)} concepts whitelisted. MIN_CONCEPT_FREQ lowered.")
                    config["MIN_CONCEPT_FREQ"] = 1 if len(whitelist) <= 15 else 2
                use_ontology = st.session_state.get('use_ontology', True)
                use_inference = st.session_state.get('use_inference', True)

                if use_ontology:
                    st.write("Initializing ontology-based concept resolver...")
                    resolver = AdvancedConceptResolver(ontology, embed_model)
                    extractor = EnhancedConceptExtractor(ontology, resolver)
                    st.session_state.resolver = resolver
                    st.session_state.extractor = extractor
                    st.success("Ontology and resolver initialized")
                else:
                    resolver = None
                    extractor = None
                progress_bar.progress(0.20)

                st.write("Extracting concepts from abstracts (Parallel)...")
                all_concepts = [None] * len(df_filtered)
                all_metrics = [None] * len(df_filtered)

                def _process_single_row(idx, row):
                    text = " ".join([str(row[col]) for col in selected_text_cols if col in row and pd.notna(row[col])])
                    if use_ontology and extractor is not None:
                        concepts = extractor.extract_from_text(
                            text, idx, allowed_concepts=whitelist
                        )
                    else:
                        concepts = extract_glacier_concepts_from_text(text)
                        if whitelist is not None:
                            concepts = [c for c in concepts if c in whitelist]
                    doc_metrics: Dict[str, Any] = {}
                    vel_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:m/a|m yr-1|m yr\^-1)', text, re.I)
                    if vel_matches: doc_metrics['velocity_m_per_yr'] = [float(m) for m in vel_matches]
                    temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', text, re.I)
                    if temp_matches: doc_metrics['temperature'] = [float(m) for m in temp_matches]
                    press_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mpa|kpa|pa)', text, re.I)
                    if press_matches: doc_metrics['pressure'] = [float(m) for m in press_matches]
                    return idx, concepts, doc_metrics

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(_process_single_row, idx, row): idx for idx, row in df_filtered.iterrows()}
                    completed = 0
                    total = len(futures)
                    for future in as_completed(futures):
                        idx, concepts, doc_metrics = future.result()
                        all_concepts[idx] = concepts
                        all_metrics[idx] = doc_metrics
                        completed += 1
                        if completed % 10 == 0 or completed == total:
                            progress_bar.progress(0.20 + (completed / total) * 0.15)
                            st.write(f"Extracted {completed}/{total} documents...")

                all_concepts = [c if c is not None else [] for c in all_concepts]
                all_metrics = [m if m is not None else {} for m in all_metrics]

                if use_ontology and extractor is not None:
                    concept_freq = extractor.get_concept_frequencies()
                    valid_concepts = [c for c, f in concept_freq.items() if f >= config.get("MIN_CONCEPT_FREQ", 2)]
                    concept_abstract_map = defaultdict(list)
                    for doc_idx, concepts in enumerate(all_concepts):
                        for c in set(concepts): concept_abstract_map[c].append(doc_idx)
                else:
                    concept_freq = defaultdict(int)
                    for concepts in all_concepts:
                        for c in concepts: concept_freq[c] += 1
                    valid_concepts = [c for c, f in concept_freq.items() if f >= config.get("MIN_CONCEPT_FREQ", 2)]
                    concept_abstract_map = defaultdict(list)
                    for doc_idx, concepts in enumerate(all_concepts):
                        for c in set(concepts): concept_abstract_map[c].append(doc_idx)

                valid_concepts = sorted(valid_concepts, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)
                top_n = config.get("TOP_N_CONCEPTS", 1000)
                if len(valid_concepts) > top_n: valid_concepts = valid_concepts[:top_n]
                concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
                id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
                progress_bar.progress(0.45)

                metrics.start_step("Concept extraction")
                metrics.end_step({"concepts_found": len(valid_concepts)})
                render_live_metrics_panel(metrics, live_panel)

                if len(valid_concepts) < 5:
                    st.error("Too few concepts extracted. Try lowering frequency thresholds.")
                    return

                metrics.start_step("Graph construction")
                st.write("Building concept graph...")
                if use_ontology and use_inference:
                    graph_builder = ReasoningEnhancedGraphBuilder(ontology, extractor)
                    nx_graph = graph_builder.build_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config)
                else:
                    nx_graph = build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config, ontology)

                metrics.end_step({"nodes": len(valid_concepts), "edges": nx_graph.number_of_edges()})
                render_live_metrics_panel(metrics, live_panel)

                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, valid_concepts, concept_to_id, config)
                progress_bar.progress(0.55)

                try:
                    with torch.no_grad():
                        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64, convert_to_numpy=True)
                    node_features = torch.tensor(embeddings, dtype=torch.float32)
                except Exception:
                    node_features = torch.randn(len(valid_concepts), 384)
                progress_bar.progress(0.65)

                def training_progress(epoch, loss):
                    progress = 0.65 + (epoch / 50) * 0.15
                    progress_bar.progress(min(1.0, progress))

                metrics.start_step("GNN training")
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, training_progress
                )
                metrics.end_step({"embedding_dim": final_emb.shape[1] if hasattr(final_emb, 'shape') else 0})
                render_live_metrics_panel(metrics, live_panel)
                progress_bar.progress(0.80)

                concept_properties = {}
                for concept in valid_concepts:
                    doc_indices = concept_abstract_map.get(concept, [])
                    values = []
                    for idx in doc_indices:
                        if idx < len(all_metrics):
                            metric_dict = all_metrics[idx]
                            if metric_dict:
                                for metric_values in metric_dict.values(): values.extend(metric_values)
                    concept_properties[concept] = float(np.median(values)) if values else 0.0

                X_feat, y_target = [], []
                for u, v in nx_graph.edges():
                    pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
                ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target)) if len(X_feat) > 5 else None

                top_scores = compute_research_direction_scores(
                    gnn_model, node_features, final_emb, nx_graph, valid_concepts, concept_properties, ridge, embed_model
                )
                distill_df = compute_concept_distillation(valid_concepts, concept_abstract_map, all_texts)

                burst_df = detect_keyword_bursts(df_filtered, valid_concepts, concept_abstract_map, selected_text_cols)
                drift_df = detect_semantic_drift(df_filtered, valid_concepts, concept_abstract_map, selected_text_cols)
                genealogy_df = build_concept_genealogy(nx_graph, valid_concepts, concept_abstract_map)
                bridge_df = detect_cross_domain_bridges(nx_graph, valid_concepts, concept_abstract_map)
                motifs = analyze_network_motifs(nx_graph)

                st.session_state.burst_df = burst_df
                st.session_state.drift_df = drift_df
                st.session_state.genealogy_df = genealogy_df
                st.session_state.bridge_df = bridge_df
                st.session_state.motifs = motifs

                total_time = time.perf_counter() - overall_start
                metrics.set("total_runtime_s", total_time)
                status.update(label=f"Analysis complete! ({total_time:.1f}s)", state="complete", expanded=False)
                log_path = metrics.save_log()
                st.caption(f"📊 Metrics log saved: `{log_path}`")

                analysis_data = {
                    "valid_concepts": valid_concepts, "concept_to_id": concept_to_id,
                    "id_to_concept": id_to_concept, "concept_abstract_map": concept_abstract_map,
                    "nx_graph": nx_graph, "concept_properties": concept_properties,
                    "ridge": ridge, "top_scores": top_scores, "distill_df": distill_df,
                    "gnn_model": gnn_model, "final_emb": final_emb, "embed_model": embed_model,
                    "all_metrics": all_metrics, "all_texts": all_texts, "config": config,
                    "df_filtered": df_filtered, "selected_text_cols": selected_text_cols,
                }
                if use_ontology:
                    analysis_data.update({
                        "ontology": ontology, "resolver": resolver, "extractor": extractor,
                        "graph_builder": graph_builder if use_inference else None,
                        "reasoning_paths": graph_builder.reasoning_paths if use_inference else [],
                    })
                st.session_state.analysis_data = analysis_data
                st.session_state.edit_history = GraphEditHistory()
                st.session_state.edit_history.save_snapshot(nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map)

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            with st.expander("Traceback"): st.code(traceback.format_exc())
            return
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if st.session_state.get('apply_edits') and st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        st.session_state.edit_history.save_snapshot(
            data["nx_graph"], data["valid_concepts"], data["concept_to_id"], data["id_to_concept"], data["concept_abstract_map"]
        )
        nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, edited = apply_graph_edits(
            data["nx_graph"], data["valid_concepts"], data["concept_to_id"], data["id_to_concept"], data["concept_abstract_map"],
            nodes_to_remove=st.session_state.get('nodes_to_remove', []),
            nodes_to_merge=st.session_state.get('nodes_to_merge', []),
            merge_name=st.session_state.get('merge_name', None),
            new_edge=st.session_state.get('new_edge', None),
            new_edge_weight=st.session_state.get('new_edge_weight', 1.0),
            min_degree=st.session_state.get('filter_min_degree', 0),
            min_freq=st.session_state.get('filter_min_freq', 0),
        )
        if edited:
            st.session_state.analysis_data["nx_graph"] = nx_graph
            st.session_state.analysis_data["valid_concepts"] = valid_concepts
            st.session_state.analysis_data["concept_to_id"] = concept_to_id
            st.session_state.analysis_data["id_to_concept"] = id_to_concept
            st.session_state.analysis_data["concept_abstract_map"] = concept_abstract_map
            st.session_state['apply_edits'] = False
            st.rerun()

    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        valid_concepts = data["valid_concepts"]
        concept_abstract_map = data["concept_abstract_map"]
        nx_graph = data["nx_graph"]
        top_scores = data["top_scores"]
        distill_df = data["distill_df"]
        df_filtered = data.get("df_filtered", pd.DataFrame())
        selected_text_cols = data.get("selected_text_cols", [])
        cmap = st.session_state.get('cmap_name', 'viridis')
        has_reasoning = "ontology" in data

        tab_names = [
            "🎯 QDWA Weights",
            "📊 Visualization",
            "🧪 Distillation",
            "🎯 Research Directions",
            "✅ Validation",
            "📥 Export",
            "📈 Extra Viz",
            "🔬 Advanced Analytics"
        ]
        if has_reasoning:
            tab_names.append("🧠 Reasoning Dashboard")
        tab_names.append("🧠 Microtransformer #2")
        tab_names.append("🤖 LLM-Guided Q&A")
        tab_names.append("📊 Computational Metrics")
        tabs = st.tabs(tab_names)

        with tabs[0]:
            st.header("🎯 Query Distillation & Weighted Allocation")
            st.caption("Enter a query about glacial flow mechanics to see how QDWA allocates relevance weight across the four domain categories.")
            encoder = load_embedding_model()
            ontology = st.session_state.ontology
            if "qdwa_allocator" not in st.session_state:
                st.session_state.qdwa_allocator = init_qdwa_allocator(ontology, encoder)
            qdwa = st.session_state.qdwa_allocator

            qdwa_query = st.text_input(
                "QDWA Query",
                value=st.session_state.get('synced_query', "How does Coulomb friction affect grounding line dynamics?"),
                key="qdwa_query_input",
            )

            if not qdwa_query.strip():
                st.info("Type a query above to compute QDWA weights.")
            else:
                render_qdwa_tab(qdwa, qdwa_query.strip(), get_metrics_logger())

        with tabs[1]:
            st.subheader("Interactive Concept Graph")
            if nx_graph.number_of_nodes() == 0:
                st.warning("No nodes to display.")
            viz_choice = st.session_state.get('viz_backend', 'PyVis (Interactive)')
            theme = THEME_PRESETS.get(st.session_state.get('theme', 'Bright (Default)'), THEME_PRESETS["Bright (Default)"])

            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(
                    nx_graph, concept_abstract_map,
                    physics_enabled=st.session_state.get('physics_enabled', True),
                    cmap_name=cmap, top_n_nodes=st.session_state.get('top_n_graph', 0),
                    theme=theme, physics_preset=st.session_state.get('effective_physics', PHYSICS_PRESETS["Stable (Default)"]),
                    show_edge_weights=st.session_state.get('show_edge_weights', False),
                    edge_label_mode=st.session_state.get('edge_label_mode', 'hover'),
                    node_label_size=st.session_state.get('node_label_size') or 12,
                    node_label_position=st.session_state.get('node_label_position') or 'center',
                    node_font_face=st.session_state.get('node_font_face') or 'Inter, Segoe UI, Roboto, sans-serif',
                    edge_label_size=st.session_state.get('edge_label_size') or 10,
                    edge_label_color=st.session_state.get('edge_label_color') or None,
                    edge_label_position=st.session_state.get('edge_label_position') or 'middle',
                    use_abbreviated_labels=st.session_state.get('use_abbreviated_labels', False),
                    max_label_length=st.session_state.get('max_label_length', 15),
                    enable_node_highlight=st.session_state.get('enable_node_highlight', False),
                    show_definitions=st.session_state.get('show_definitions', True),
                    edge_lightness=st.session_state.get('edge_lightness', 0.6),
                    edge_color_mode=st.session_state.get('edge_color_mode', 'theme'),
                    custom_edge_color=st.session_state.get('custom_edge_color', '#AAAAAA'),
                    tooltip_font_size=st.session_state.get('tooltip_font_size', 13),
                    node_legend_font_size=st.session_state.get('node_legend_font_size', 13),
                )
            elif viz_choice == "Plotly 2D":
                render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=st.session_state.get('top_n_graph', 0), theme=theme)
            elif viz_choice == "Plotly 3D":
                render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=st.session_state.get('top_n_graph', 0), theme=theme)
            else:
                render_graph_fallback(nx_graph, concept_abstract_map, theme=theme)

            with st.expander("Graph Metrics"):
                display_metric_dashboard(compute_graph_metrics(nx_graph), theme=theme)
            with st.expander("Domain Hierarchy (Sunburst)"):
                labels, parents, values = build_category_hierarchy(valid_concepts, concept_abstract_map, top_n_per_category=st.session_state.get('top_n_sunburst', 0))
                render_sunburst_chart(labels, parents, values, cmap_name=st.session_state.get('sunburst_cmap', cmap), theme=theme, branchvalues=st.session_state.get('sunburst_branchvalues', 'total'), label_size=st.session_state.get('sunburst_label_size') or 20, width=st.session_state.get('sunburst_width') or 900, height=st.session_state.get('sunburst_height') or 700, show_labels=st.session_state.get('sunburst_show_labels', True), show_values=st.session_state.get('sunburst_show_values', False), hover_info=st.session_state.get('sunburst_hover_info', 'all'), font_family=st.session_state.get('sunburst_font_family', 'Inter, Segoe UI, Roboto, sans-serif'), legend_font_size=st.session_state.get('sunburst_legend_font_size', 12))
            with st.expander("Concept Radar"):
                c_r1, c_r2 = st.columns(2)
                with c_r1:
                    radar_label = st.slider("Label size", 8, 24, 12, key="radar_label_slider")
                with c_r2:
                    radar_font = st.slider("Tick/Legend size", 8, 20, 11, key="radar_font_slider")
                radar_k = st.session_state.get('top_n_radar', 15)
                render_radar_chart(
                    distill_df,
                    top_k=radar_k if radar_k > 0 else 15,
                    cmap_name=st.session_state.get('cmap_name', 'viridis'),
                    theme=theme,
                    label_size=radar_label,
                    font_size=radar_font
                )

        with tabs[2]:
            st.subheader("Concept Distillation Efficiency")
            display_df = distill_df.head(st.slider("Show Top N", 10, min(200, len(distill_df)), 50, key="distill_top_n"))
            st.dataframe(display_df, use_container_width=True)
            st.bar_chart(display_df.set_index('concept')[['distillation_efficiency']])

        with tabs[3]:
            st.subheader("Top Research Direction Recommendations")
            if not top_scores.empty:
                st.dataframe(top_scores[['concept_u', 'concept_v', 'composite_score', 'gnn_affinity', 'semantic_novelty', 'expected_property_gain', 'feasibility_score']].head(20), use_container_width=True)

        with tabs[4]:
            st.subheader("Mathematical Validation")
            val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Modularity", f"{val_metrics.get('modularity', 0):.3f}")
            col2.metric("Silhouette", f"{val_metrics.get('silhouette_score', 0):.3f}")
            col3.metric("Communities", val_metrics.get('n_communities', 0))
            col4.metric("Significant Edges", val_metrics.get('edge_significant_count', 0))

        with tabs[5]:
            st.subheader("Export & Post-Processing")
            if st.button("Generate Markdown Report"):
                report = generate_analysis_report(nx_graph, valid_concepts, concept_abstract_map, top_scores, distill_df, st.session_state.get('burst_df'), st.session_state.get('drift_df'), st.session_state.get('genealogy_df'), st.session_state.get('bridge_df'), st.session_state.get('motifs'), val_metrics, df_filtered)
                st.download_button("📄 Download Report", data=report.encode('utf-8'), file_name="glacier_report.md", mime="text/markdown")

        with tabs[6]:
            st.subheader("Extra Visualizations")
            render_concept_timeline(df_filtered, valid_concepts, concept_abstract_map, theme=theme)
            render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map, theme=theme)

        with tabs[7]:
            st.subheader("Advanced Analytics")
            if st.session_state.get('burst_df') is not None:
                st.dataframe(st.session_state.burst_df.head(20), use_container_width=True)

        idx = 8
        if has_reasoning:
            with tabs[idx]:
                render_reasoning_dashboard(nx_graph, valid_concepts, data.get("ontology"), data.get("extractor"))
            idx += 1

        with tabs[idx]:
            render_microtransformer_kg_rag_tab(st.session_state.analysis_data, st.session_state.ontology)
        idx += 1

        with tabs[idx]:
            if "ontology" in st.session_state.analysis_data:
                render_llm_qa_tab(st.session_state.analysis_data, st.session_state.analysis_data["ontology"])
            else:
                st.info("Please build the concept graph with ontology enabled first.")
        idx += 1

        with tabs[idx]:
            render_metrics_dashboard(get_metrics_logger())

if __name__ == "__main__":
    main()
