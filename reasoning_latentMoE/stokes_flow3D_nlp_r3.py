# ============================================================================
# ███ GLACIER INTELLIGENT ElmerSolver RECOMMENDER (v10.1.1-glacier)       ███
# ███ 1:1 architectural port of the nt-Cu Latent-MoE AI v10.1.1 codebase  ███
# ███ 3 substitutions only:                                              ███
# ███   1. Ontology:      ρ₀/μ/γ̇₀/m/σ₀  →  n/A₁/A₂/Q₁/Q₂/T*/E/γ̇_c/ρ/g   ███
# ███   2. Regime tables: RHO0/GAMMA0  →  GLEN_N / ARRHENIUS / FABRIC   ███
# ███   3. Solver:        FFT phase-field →  Elmer SIF + subprocess     ███
# ███ Every other subsystem is byte-faithful to the nt-Cu port:         ███
# ███   · 6-key FINE provenance taxonomy  →  3-key COARSE rollup        ███
# ███   · _norm_provenance total function (order of hints LOAD-BEARING) ███
# ███   · INCUMBENT_ROUTES pin-list  +  _is_context union cascade       ███
# ███   · 10-expert Latent MoE (Material/Thermal/Strain/Method/…)        ███
# ███   · Framework(A)/Genesis(B)/Thermo-Mech(C)/Corpus-Density threads ███
# ███   · PhysicsRegimeExpert  ∧  TheoryRegimeExpert gating             ███
# ███   · Exact-XAI stacked bar chart (no SHAP/LIME approximation)      ███
# ███   · Publication visuals dashboard (radar/bar/Sankey/Treemap/Hist) ███
# ███   · Journal templates (Nature/Science/AdvMat/PRL/custom)          ███
# ███   · Ollama three-tier cascade (grounded LLM ∧ regex ∧ prior)      ███
# ███   · FAISS + lexical hybrid retriever (RRF fusion)                 ███
# ███   · Side-note markdown table parser (rejects non-positive)        ███
# ███   · Exact-XAI stacked-MoE chart with show-context toggle          ███
# ███   · 'Best match' legend entry is OPT-IN (default OFF)             ███
# ███   · '⭐ BEST' icon GONE — thin orange underline marks the winner   ███
# ============================================================================
# QUICK-START
#   1.  (optional)  ollama serve && ollama pull qwen2.5:7b
#   2.  (optional)  mkdir json_metadatabase  &&  drop *.json there
#   3.  streamlit run glacier_elmer_ai.py
#   4.  Sidebar → AI Recommender → Analyse.  Adopt.  SIF tab.  Run Elmer.
#
# REGRESSION SUITE
#   GLACIER_REGRESSION=1 streamlit run glacier_elmer_ai.py
#
# WORKING DIRECTORY
#   SIF is written to the sidebar "Working directory" and ElmerSolver is
#   launched there.  Corpus is read from "Corpus folder" (default
#   ./json_metadatabase).  SIF-context parsing reads an existing .sif
#   (optional) to seed the recommender with current n, A, Q, T*, E, ρ.
# ============================================================================

import os
import re
import io
import sys
import json
import math
import time
import uuid
import glob
import html
import base64
import shutil
import hashlib
import logging
import pathlib
import pickle
import sqlite3
import zipfile
import tempfile
import threading
import subprocess
import traceback
import warnings
import unicodedata
from io import BytesIO, StringIO
from datetime import datetime
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Any, Optional, Tuple, Iterable, Set, Callable, Union
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# ── Matplotlib (mandatory) ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator,
                               FormatStrFormatter, FuncFormatter, NullLocator)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb, LinearSegmentedColormap, \
                             ListedColormap
import matplotlib.animation as animation

# ── Plotly (mandatory for interactive charts) ───────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Optional AI dependencies (soft-imported) ────────────────────────────
try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None
    REQUESTS_AVAILABLE = False

try:
    import faiss as _faiss
    FAISS_AVAILABLE = True
except ImportError:
    _faiss = None
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None
    SBERT_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False

# ── Warnings + logging ──────────────────────────────────────────────────
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ███ PHYSICAL CONSTANTS                                                 ███
# ============================================================================
SECONDS_PER_YEAR = 31556926.0        # Julian-tropical mean, 365.2422 d
DAYS_PER_YEAR    = 365.2422
SECONDS_PER_DAY  = 86400.0
R_GAS            = 8.314462618       # J mol^-1 K^-1
MPA_PER_GPA      = 1.0e3
PA_PER_MPA       = 1.0e6
PA_PER_GPA       = 1.0e9
NM_PER_M         = 1.0e9
GRAVITY_STANDARD = 9.80665           # m s^-2
RHO_ICE_STD      = 917.0             # kg m^-3 (PATERSON, 3rd ed., 1994)
T0_CELSIUS       = 273.15
T_TRIPLE_K       = 273.16


# ============================================================================
# ███ ERROR HANDLING DECORATOR                                           ███
# ============================================================================
def handle_errors(func):
    """Wrap any Streamlit-callback or physics function to surface a clean
    Streamlit error and log a traceback instead of blanking the app."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"❌ Error in {func.__name__}: {e}"
            try:
                st.error(msg)
            except Exception:
                pass
            logger.error("%s\n%s", msg, traceback.format_exc())
            return None
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# ============================================================================
# ███ SECTION 1 — GLACIER / ELMER ONTOLOGY                              ███
# ============================================================================
# The ontology is the SINGLE SOURCE OF TRUTH for:
#   • parameter labels / symbols / units
#   • UI display units (ui_unit) and ui_scale (SI → UI)
#   • plausible_range (gatekeeper hard bounds)
#   • soft_range     (radar / display normalization bounds)
#   • per-material defaults
#   • expected JSON metadata file
#
# Adding a parameter here automatically wires it into the recommender,
# cascade, MoE scorer, bar/radar/Sankey/Treemap charts, and SIF generator
# (provided SOLVER_KEY_MAP has a mapping or a passthrough name).
# ============================================================================

GLACIER_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "glen_n": {
        "label": "Glen Flow-Law Exponent",
        "symbol": "n",
        "aliases": [
            "glen exponent", "glen flow-law exponent", "glen flow law exponent",
            "stress exponent", "creep exponent", "flow law exponent",
            "flow-law exponent", "power law exponent", "power-law exponent",
            "n exponent", "n glen", "creep power law", "creep power-law",
            "glen n", "stress exponent n", "flow-law stress exponent",
            "ice creep exponent", "dislocation creep exponent",
            "inverse strain-rate sensitivity index",
            "srs index", "creep exponent n",
        ],
        "unit": "dimensionless", "ui_unit": "–", "ui_scale": 1.0,
        "valid_range": (0.5, 6.0),
        "soft_range": (1.5, 4.5),
        "defaults": {
            "ice": 3.0, "temperate_ice": 3.0, "cold_ice": 3.0,
            "polythermal_ice": 3.0, "polar_ice": 3.0, "firn": 3.0,
            "greenland": 3.0, "antarctica": 3.0, "himalaya": 3.0,
        },
        "expected_file": "glen_exponent_metadatabase.json",
        "derivation": "dislocation creep n≈3; GBS n≈1.8; high-stress n≈4",
    },
    "rate_factor_A1": {
        "label": "Rate Factor A₁ (cold regime)",
        "symbol": "A_1",
        "aliases": [
            "rate factor", "arrhenius rate factor", "pre-exponential factor",
            "pre-exponential", "softness parameter", "creep rate factor",
            "flow law coefficient", "A factor", "A1 factor", "A_1",
            "cold rate factor", "a1", "rate coefficient",
            "glen rate factor", "ice softness", "temperature-dependent rate factor",
            "arrhenius coefficient",
        ],
        "unit": "MPa^-n yr^-1", "ui_unit": "MPa⁻ⁿ yr⁻¹", "ui_scale": 1.0,
        "valid_range": (1e-10, 1e30),
        "soft_range": (1e-6, 1e25),
        "defaults": {
            "ice": 1.258e13, "temperate_ice": 2.4e24, "cold_ice": 1.14e-5,
            "polythermal_ice": 1.258e13, "polar_ice": 1.14e-5,
            "firn": 1.258e13, "greenland": 1.258e13, "antarctica": 1.14e-5,
            "himalaya": 1.258e13,
        },
        "expected_file": "rate_factor_metadatabase.json",
        "derivation": "Cuffey-Paterson A_cold = 3.5e-25 Pa^-3 s^-1 (T<-10°C)",
    },
    "rate_factor_A2": {
        "label": "Rate Factor A₂ (warm regime)",
        "symbol": "A_2",
        "aliases": [
            "warm rate factor", "A2 factor", "A_2", "a2",
            "temperate rate factor", "warm softness",
            "high-temperature rate factor",
        ],
        "unit": "MPa^-n yr^-1", "ui_unit": "MPa⁻ⁿ yr⁻¹", "ui_scale": 1.0,
        "valid_range": (1e-10, 1e40),
        "soft_range": (1e5, 1e33),
        "defaults": {
            "ice": 6.046e28, "temperate_ice": 6.046e28, "cold_ice": 6.046e28,
            "polythermal_ice": 6.046e28, "polar_ice": 6.046e28,
            "firn": 6.046e28, "greenland": 6.046e28, "antarctica": 6.046e28,
            "himalaya": 6.046e28,
        },
        "expected_file": "rate_factor_metadatabase.json",
        "derivation": "Cuffey-Paterson A_warm = 2.4e-24 Pa^-3 s^-1 (T≥-10°C)",
    },
    "activation_energy_Q1": {
        "label": "Activation Energy Q₁ (cold)",
        "symbol": "Q_1",
        "aliases": [
            "activation energy", "creep activation energy",
            "arrhenius activation energy", "Q activation",
            "activation enthalpy", "creep activation enthalpy",
            "cold activation energy", "Q1 activation", "Q_1", "q1",
            "glen activation energy",
        ],
        "unit": "J mol^-1", "ui_unit": "J mol⁻¹", "ui_scale": 1.0,
        "valid_range": (1e3, 3e5),
        "soft_range": (4e4, 1e5),
        "defaults": {
            "ice": 60000.0, "temperate_ice": 60000.0, "cold_ice": 60000.0,
            "polythermal_ice": 60000.0, "polar_ice": 60000.0,
            "firn": 60000.0, "greenland": 60000.0, "antarctica": 60000.0,
            "himalaya": 60000.0,
        },
        "expected_file": "activation_energy_metadatabase.json",
        "derivation": "Cuffey-Paterson Q_cold = 60 kJ/mol (T<-10°C)",
    },
    "activation_energy_Q2": {
        "label": "Activation Energy Q₂ (warm)",
        "symbol": "Q_2",
        "aliases": [
            "warm activation energy", "Q2 activation", "Q_2", "q2",
            "temperate activation energy", "high-temperature activation",
            "melt-regime activation energy",
        ],
        "unit": "J mol^-1", "ui_unit": "J mol⁻¹", "ui_scale": 1.0,
        "valid_range": (6e4, 3e5),
        "soft_range": (1.1e5, 1.7e5),
        "defaults": {
            "ice": 139000.0, "temperate_ice": 139000.0, "cold_ice": 139000.0,
            "polythermal_ice": 139000.0, "polar_ice": 139000.0,
            "firn": 139000.0, "greenland": 139000.0, "antarctica": 139000.0,
            "himalaya": 139000.0,
        },
        "expected_file": "activation_energy_metadatabase.json",
        "derivation": "Cuffey-Paterson Q_warm = 139 kJ/mol (T≥-10°C)",
    },
    "limit_temperature": {
        "label": "Limit Temperature (regime switch)",
        "symbol": "T*",
        "aliases": [
            "limit temperature", "regime switch temperature",
            "critical temperature", "transition temperature",
            "T-limit", "T_star", "arrhenius transition temperature",
            "crossover temperature", "breakpoint temperature",
        ],
        "unit": "degC", "ui_unit": "°C", "ui_scale": 1.0,
        "valid_range": (-60.0, 10.0),
        "soft_range": (-25.0, -5.0),
        "defaults": {
            "ice": -10.0, "temperate_ice": -10.0, "cold_ice": -10.0,
            "polythermal_ice": -10.0, "polar_ice": -10.0, "firn": -10.0,
            "greenland": -10.0, "antarctica": -10.0, "himalaya": -10.0,
        },
        "expected_file": "limit_temperature_metadatabase.json",
        "derivation": "Cuffey-Paterson canonical T* = -10°C",
    },
    "constant_temperature": {
        "label": "Constant Temperature (isothermal)",
        "symbol": "T",
        "aliases": [
            "constant temperature", "isothermal temperature",
            "ice temperature", "temperature", "ice body temperature",
            "mean temperature", "bulk temperature", "thermal state",
        ],
        "unit": "degC", "ui_unit": "°C", "ui_scale": 1.0,
        "valid_range": (-60.0, 1.0),
        "soft_range": (-35.0, -0.5),
        "defaults": {
            "ice": -3.0, "temperate_ice": -0.5, "cold_ice": -30.0,
            "polythermal_ice": -8.0, "polar_ice": -30.0, "firn": -5.0,
            "greenland": -15.0, "antarctica": -25.0, "himalaya": -5.0,
        },
        "expected_file": "constant_temperature_metadatabase.json",
        "derivation": "Mean column temperature from borehole thermometry",
    },
    "glen_enhancement": {
        "label": "Glen Enhancement Factor",
        "symbol": "E",
        "aliases": [
            "enhancement factor", "glen enhancement", "enhancement",
            "E factor", "fabric enhancement", "anisotropy enhancement",
            "flow enhancement", "softening factor", "hardening factor",
            "fabric softening", "shear margin enhancement",
        ],
        "unit": "dimensionless", "ui_unit": "–", "ui_scale": 1.0,
        "valid_range": (0.05, 50.0),
        "soft_range": (0.3, 12.0),
        "defaults": {
            "ice": 1.0, "temperate_ice": 1.0, "cold_ice": 1.0,
            "polythermal_ice": 1.0, "polar_ice": 1.0, "firn": 1.0,
            "greenland": 1.0, "antarctica": 1.0, "himalaya": 1.0,
        },
        "expected_file": "glen_enhancement_metadatabase.json",
        "derivation": "isotropic E=1; single-max fabric E≈3; ice-stream margin E≈5-10",
    },
    "critical_shear_rate": {
        "label": "Critical Shear Rate (regularisation floor)",
        "symbol": "γ̇_c",
        "aliases": [
            "critical shear rate", "critical strain rate",
            "regularisation shear rate", "shear rate floor",
            "strain rate floor", "viscosity floor",
            "glen regularisation rate",
        ],
        "unit": "s^-1", "ui_unit": "s⁻¹", "ui_scale": 1.0,
        "valid_range": (1e-25, 1e-2),
        "soft_range": (1e-16, 1e-7),
        "defaults": {
            "ice": 1.0e-10, "temperate_ice": 1.0e-10, "cold_ice": 1.0e-10,
            "polythermal_ice": 1.0e-10, "polar_ice": 1.0e-10, "firn": 1.0e-10,
            "greenland": 1.0e-10, "antarctica": 1.0e-10, "himalaya": 1.0e-10,
        },
        "expected_file": "critical_shear_rate_metadatabase.json",
        "derivation": "Regularisation of effective viscosity in Stokes solvers",
    },
    "ice_density": {
        "label": "Ice Density",
        "symbol": "ρ",
        "aliases": [
            "ice density", "density", "rho ice", "mass density",
            "bulk density", "firn density", "glacier ice density",
        ],
        "unit": "kg m^-3", "ui_unit": "kg m⁻³", "ui_scale": 1.0,
        "valid_range": (300.0, 1000.0),
        "soft_range": (850.0, 925.0),
        "defaults": {
            "ice": 917.0, "temperate_ice": 917.0, "cold_ice": 917.0,
            "polythermal_ice": 917.0, "polar_ice": 917.0, "firn": 600.0,
            "greenland": 917.0, "antarctica": 917.0, "himalaya": 900.0,
        },
        "expected_file": "ice_density_metadatabase.json",
        "derivation": "Bubble-free ice 917 kg/m³; firn 500-830 kg/m³",
    },
    "gravity": {
        "label": "Gravity (scaled to years)",
        "symbol": "g",
        "aliases": [
            "gravity", "gravitational acceleration", "g",
            "gravitational constant", "gravitational field strength",
        ],
        "unit": "m s^-2", "ui_unit": "m s⁻²", "ui_scale": 1.0,
        "valid_range": (9.0, 11.0),
        "soft_range": (9.7, 9.9),
        "defaults": {
            "ice": 9.81, "temperate_ice": 9.81, "cold_ice": 9.81,
            "polythermal_ice": 9.81, "polar_ice": 9.81, "firn": 9.81,
            "greenland": 9.82, "antarctica": 9.83, "himalaya": 9.79,
        },
        "expected_file": "gravity_metadatabase.json",
        "derivation": "Standard gravity adjusted for latitude",
    },
}

# ── Canonical parameter ordering used EVERYWHERE (UI, charts, exports) ──
GLACIER_PARAM_ORDER: List[str] = [
    "glen_n", "rate_factor_A1", "rate_factor_A2",
    "activation_energy_Q1", "activation_energy_Q2",
    "limit_temperature", "constant_temperature",
]

# Context parameters (derived, not solver-consumed directly)
CONTEXT_PARAM_ORDER: List[str] = [
    "glen_enhancement", "critical_shear_rate",
    "ice_density", "gravity",
]

ALL_PARAMS: List[str] = GLACIER_PARAM_ORDER + CONTEXT_PARAM_ORDER

# Two additional convenience groupings used by the sidebar and Lab tabs
SIDEBAR_PARAM_ORDER: List[str] = [
    "glen_n", "rate_factor_A1", "activation_energy_Q1",
    "limit_temperature", "constant_temperature",
]

ADVANCED_PARAM_ORDER: List[str] = [
    "rate_factor_A2", "activation_energy_Q2",
]

LAB_PARAM_ORDER: List[str] = [
    "glen_enhancement", "critical_shear_rate",
    "ice_density", "gravity",
]


# ============================================================================
# ███ SECTION 2 — PROVENANCE TAXONOMY (byte-faithful port)               ███
# ============================================================================
# Six FINE keys roll up into three COARSE buckets.  The fine keys are
# canonical and appear in the audit legend; the coarse buckets are the
# default main-text legend.  The mapping is exhaustive and total.
#
#   llm_extract    ─┐
#   llm_reasoned   ─┴→  llm_grounded
#   llm_prior      ──→  llm_prior
#   regex_ner      ─┐
#   regime_prior   ─┤  →  deterministic
#   physics_inferred┘
# ============================================================================

FINE_PROVENANCE_KEYS: Tuple[str, ...] = (
    'llm_extract', 'llm_reasoned', 'regex_ner',
    'regime_prior', 'llm_prior', 'physics_inferred',
)

PROVENANCE_GROUPS: Dict[str, Tuple[str, ...]] = {
    'llm_grounded':  ('llm_extract', 'llm_reasoned'),
    'llm_prior':     ('llm_prior',),
    'deterministic': ('regex_ner', 'regime_prior', 'physics_inferred'),
}
_FINE_TO_GROUP: Dict[str, str] = {
    f: g for g, fs in PROVENANCE_GROUPS.items() for f in fs
}
COARSE_PROVENANCE_KEYS: Tuple[str, ...] = tuple(PROVENANCE_GROUPS.keys())

# ── Coarse (default) legend markers — SHAPE only ────────────────────────
LEGEND_MARKERS: Dict[str, Dict[str, Any]] = {
    'llm_grounded':  dict(marker='D', ms=6.0, fill=True,
                          label='LLM (corpus-grounded)'),
    'llm_prior':     dict(marker='o', ms=6.5, fill=True,
                          label='LLM prior (parametric)'),
    'deterministic': dict(marker='^', ms=7.5, fill=True,
                          label='Deterministic (regex · physics-inferred)'),
}

# ── Fine (audit) legend markers — SHAPE = method, FILL = evidence ───────
PROVENANCE_MARKERS: Dict[str, Dict[str, Any]] = {
    'llm_extract':   dict(marker='D', ms=6.0, fill=True,
                          label='LLM extraction (verbatim, grounded)'),
    'llm_reasoned':  dict(marker='p', ms=6.5, fill=True,
                          label='LLM chain-of-thought (grounded)'),
    'llm_prior':     dict(marker='o', ms=6.5, fill=False,
                          label='LLM prior (no corpus evidence)'),
    'regex_ner':     dict(marker='^', ms=7.5, fill=True,
                          label='Regex NER (deterministic)'),
    'regime_prior':  dict(marker='s', ms=6.5, fill=False,
                          label='Regime classifier (lookup table)'),
    'physics_inferred': dict(marker='v', ms=7.0, fill=False,
                             label='Physics-inferred (Glen / Arrhenius)'),
}

# ── The pin-list: for each parameter, which COARSE bucket(s) win ────────
# A candidate whose bucket is not in this list for its parameter enters
# as a greyed NON-RANKED context bar (union-cascade mode).  The pin-list
# is the mechanism that makes findings STRUCTURALLY UNCHANGEABLE — a
# high-score context candidate can never outrank a pinned incumbent.
INCUMBENT_ROUTES: Dict[str, Tuple[str, ...]] = {
    'glen_n':               ('deterministic',),
    'rate_factor_A1':       ('deterministic',),
    'rate_factor_A2':       ('deterministic',),
    'activation_energy_Q1': ('deterministic',),
    'activation_energy_Q2': ('deterministic',),
    'limit_temperature':    ('llm_grounded',),
    'constant_temperature': ('llm_grounded',),
    'glen_enhancement':     ('deterministic',),
    'critical_shear_rate':  ('deterministic',),
    'ice_density':          ('deterministic',),
    'gravity':              ('deterministic',),
}

# ── Backward-compat hint tables for `_norm_provenance` ──────────────────
# The ORDER of the checks inside `_norm_provenance` is LOAD-BEARING.
_DERIVED_HINTS: Tuple[str, ...] = (
    'physics_inferred', 'derived',
    'arrhenius_inversion', 'glen_inversion',
    'flow_law_inversion', 'activation_inversion',
    'enhancement_inversion', 'solver_law',
    'model_inversion', 'consensus',
)
_REGIME_HINTS: Tuple[str, ...] = (
    'regime_prior', 'regime_inference', 'regime_classifier',
    'physics_regime', 'theory_regime', 'glacier_regime',
    'arrhenius_regime', 'glen_regime', 'fabric_regime',
)
_PRIOR_HINTS: Tuple[str, ...] = (
    'prior', 'estimate', 'no evidence', 'world knowledge', 'parametric',
)
_REASONED_HINTS: Tuple[str, ...] = (
    'reasoned', 'chain_of_thought', 'chain-of-thought', 'cot',
)
_REGEX_HINTS: Tuple[str, ...] = (
    'regex', 'heuristic', 'side_note', 'side-note', 'default_fallback',
)
_LLM_HINTS: Tuple[str, ...] = ('llm', 'inferred', 'model', 'explicit')


def _norm_provenance(p: Any) -> str:
    """Total function: ANY input → exactly one of FINE_PROVENANCE_KEYS.

    Ordering is LOAD-BEARING:
        1. exact fine key
        2. physics inference      — 'model_inversion' etc. NEVER LLM
        3. regime classifier      — MUST precede prior, else compounds
                                     like 'glen_regime_prior' become LLM
                                     priors (the mislabel class this
                                     ordering exists to kill)
        4. LLM prior
        5. LLM chain-of-thought
        6. regex NER
        7. grounded LLM            — 'ai' matched on word boundary only
        8. safe default 'regex_ner'

    Plain 'regime' is deliberately NOT a hint — Tier-3 LLM-within-band
    strings still classify as llm_prior, which is correct.
    """
    s = str(p).strip().lower()
    if s in FINE_PROVENANCE_KEYS:
        return s
    if any(k in s for k in _DERIVED_HINTS):
        return 'physics_inferred'
    if any(k in s for k in _REGIME_HINTS):
        return 'regime_prior'
    if any(k in s for k in _PRIOR_HINTS):
        return 'llm_prior'
    if any(k in s for k in _REASONED_HINTS):
        return 'llm_reasoned'
    if any(k in s for k in _REGEX_HINTS):
        return 'regex_ner'
    if any(k in s for k in _LLM_HINTS) or re.search(r'\bai\b', s):
        return 'llm_extract'
    return 'regex_ner'


_VALID_GRANULARITIES: Tuple[str, ...] = ('coarse', 'fine')


def _norm_granularity(g: Any) -> str:
    """Single normalization point — invalid → 'coarse'."""
    s = str(g).strip().lower() if g is not None else 'coarse'
    if s not in _VALID_GRANULARITIES:
        logger.warning("legend_granularity %r invalid — defaulting to 'coarse'", g)
        return 'coarse'
    return s


def _legend_key(prov: Any, granularity: str = 'coarse') -> str:
    """Roll any provenance string up to a legend key.  Total function."""
    granularity = _norm_granularity(granularity)
    fine = _norm_provenance(prov)
    if granularity == 'fine':
        return fine
    return _FINE_TO_GROUP.get(fine, 'deterministic')


def _stamp_provenance(ext: Dict[str, Any], fine_key: str) -> None:
    """Tag an extraction dict with its canonical fine provenance key."""
    if fine_key not in FINE_PROVENANCE_KEYS:
        raise ValueError(f"not a fine key: {fine_key!r}")
    ext['_provenance'] = fine_key


def _is_context(param: str, prov: Any) -> bool:
    """Return True iff a candidate with provenance `prov` for `param`
    should be treated as non-ranked context in union-cascade mode.

    A candidate is context iff:
      (a) `param` is pinned in INCUMBENT_ROUTES, AND
      (b) its coarse bucket is not among the pinned buckets for `param`."""
    bucket = _FINE_TO_GROUP.get(_norm_provenance(prov), 'deterministic')
    pinned = INCUMBENT_ROUTES.get(param, ())
    return bool(pinned) and bucket not in pinned


def _default_reasoning(c: Any) -> str:
    """Fallback reasoning string when a candidate carries none."""
    prov = _norm_provenance(getattr(c, "provenance", "")
                            or getattr(c, "method", ""))
    if prov == "llm_extract":
        return ("Tier-1 grounded LLM extraction — verbatim span, "
                "corpus-anchored, unit-coerced by the gatekeeper.")
    if prov == "llm_reasoned":
        return ("Tier-1 grounded LLM chain-of-thought — evidence span "
                "plus an explicit step-by-step derivation, corpus-anchored.")
    if prov == "regex_ner":
        return ("Tier-2 deterministic regex — sci-notation scanner + "
                "param-aware unit tokens; no LLM involved.")
    if prov == "llm_prior":
        return ("Tier-3 LLM prior inference — no verbatim corpus evidence; "
                "confidence capped at 0.5, bounded by GLACIER_ONTOLOGY.")
    if prov == "regime_prior":
        return ("Tier-3b curated glacier-regime classifier — reproducible "
                "lookup table, no LLM involved.")
    if prov == "physics_inferred":
        return ("Physics-inferred from measured side-note context — "
                "Glen flow-law / Arrhenius / inverse-formula.")
    return ""


# ============================================================================
# ███ SECTION 3 — PARAM_META (chart labels)                              ███
# ============================================================================
# Parallel to nt-Cu PARAM_META.  Titles and symbols used in chart axes,
# colorbar labels, and the stacked-MoE chart.  `unit=None` suppresses the
# unit suffix in axis labels.
# ============================================================================

PARAM_META: Dict[str, Dict[str, Any]] = {
    'glen_n':               dict(title='Glen Flow-Law Exponent',
                                 symbol='n', unit=None),
    'rate_factor_A1':       dict(title='Rate Factor A₁ (cold)',
                                 symbol='A_1', unit='MPa^{-n}\\,\\mathrm{yr}^{-1}'),
    'rate_factor_A2':       dict(title='Rate Factor A₂ (warm)',
                                 symbol='A_2', unit='MPa^{-n}\\,\\mathrm{yr}^{-1}'),
    'activation_energy_Q1': dict(title='Activation Energy Q₁ (cold)',
                                 symbol='Q_1', unit='J\\,\\mathrm{mol}^{-1}'),
    'activation_energy_Q2': dict(title='Activation Energy Q₂ (warm)',
                                 symbol='Q_2', unit='J\\,\\mathrm{mol}^{-1}'),
    'limit_temperature':    dict(title='Limit Temperature',
                                 symbol='T^{*}', unit='^{\\circ}\\mathrm{C}'),
    'constant_temperature': dict(title='Constant Temperature',
                                 symbol='T', unit='^{\\circ}\\mathrm{C}'),
    'glen_enhancement':     dict(title='Glen Enhancement Factor',
                                 symbol='E', unit=None),
    'critical_shear_rate':  dict(title='Critical Shear Rate',
                                 symbol='\\dot{\\gamma}_{c}',
                                 unit='\\mathrm{s}^{-1}'),
    'ice_density':          dict(title='Ice Density',
                                 symbol='\\rho', unit='\\mathrm{kg}\\,\\mathrm{m}^{-3}'),
    'gravity':              dict(title='Gravity',
                                 symbol='g', unit='\\mathrm{m}\\,\\mathrm{s}^{-2}'),
}


# ============================================================================
# ███ SECTION 4 — SAFE MATHTEXT + COLOR HELPERS                          ███
# ============================================================================

def normalize_tex(s: Any) -> Any:
    """Collapse accidental double-backslashes in LaTeX strings."""
    if not isinstance(s, str):
        return s
    if '\\' not in s:
        return s
    return re.sub(r'\\{2,}', r'\\', s)


def strip_tex(s: Any) -> str:
    """Return a plain-text fallback for a possibly-malformed mathtext."""
    if s is None:
        return ''
    return re.sub(r'[\\{}$]', '', str(s))


_MTX = MathTextParser('path')


def safe_mathtext(s: Any, fallback: Optional[str] = None) -> str:
    """Return `s` if it renders cleanly; otherwise a stripped fallback."""
    if s is None:
        return ''
    s = normalize_tex(str(s))
    if '$' not in s:
        return s
    try:
        _MTX.parse(s, dpi=100, prop=FontProperties())
        return s
    except Exception:
        logger.warning("safe_mathtext: refusing malformed mathtext %r", s)
        return fallback if fallback is not None else strip_tex(s)


def sci_tex(value: Any, unit: Optional[str] = None,
            decimals: int = 1) -> str:
    """Format a number as mathtext (10^{e} form when out of [-1, 3])."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '$?$'
    if v == 0:
        s = '0'
    else:
        e = int(np.floor(np.log10(abs(v))))
        if -1 <= e <= 3:
            s = f'{v:g}'
        else:
            m = f'{v / 10.0 ** e:.{decimals}f}'.rstrip('0').rstrip('.')
            s = f'10^{{{e}}}' if m == '1' else f'{m}\\times10^{{{e}}}'
    if unit:
        s += f'\\,\\mathrm{{{unit}}}'
    return f'${s}$'


def _color_to_rgb(c: Any) -> Tuple[float, float, float]:
    """Coerce any matplotlib-acceptable colour to an RGB triple."""
    if c is None:
        return (0.7, 0.7, 0.7)
    if isinstance(c, (tuple, list, np.ndarray)) and len(c) >= 3:
        try:
            return (float(c[0]), float(c[1]), float(c[2]))
        except Exception:
            return (0.7, 0.7, 0.7)
    if isinstance(c, str):
        try:
            return to_rgb(c)
        except Exception:
            pass
    try:
        return to_rgb(str(c))
    except Exception:
        return (0.7, 0.7, 0.7)


def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    """Rec. 601 luminance — used to pick white vs black text on a bar."""
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


# ============================================================================
# ███ SECTION 5 — GLACIER-SPECIFIC UNIT CONVERSION                        ███
# ============================================================================
# Elmer's Glen flow-law uses MPa^-n yr^-1 for A.  The catalogue may store
# A in Pa^-n s^-1, MPa^-n s^-1, kPa^-n yr^-1, bar^-n yr^-1, etc.  The
# conversion factor depends on the exponent n, so it MUST receive n
# from the user's current n-candidate (or its default).
# ============================================================================

def _norm_unit_token(u: str) -> str:
    """Canonicalise a unit token for substring matching."""
    u = (u or "").strip().lower()
    u = u.replace("μ", "u").replace("µ", "u")
    u = u.replace("·", ".").replace("•", ".").replace("*", ".")
    u = u.replace("×", "x").replace(" ", "")
    u = u.replace("⁻", "-").replace("⁰", "0").replace("¹", "1")
    u = u.replace("²", "2").replace("³", "3").replace("⁴", "4")
    u = u.replace("⁵", "5").replace("⁶", "6").replace("⁷", "7")
    u = u.replace("⁸", "8").replace("⁹", "9")
    return u


def _detect_pressure_multiplier_to_MPa(u: str) -> float:
    """Return the factor m such that 1 <unit> = m MPa.

    Used for A (pressure exponent n) → v_MPa = v_unit × m^n
    Used for Q (no pressure)              → v_J   = v_unit × 1
    """
    if "gpa" in u:
        return 1.0e3
    if "mpa" in u:
        return 1.0
    if "kpa" in u:
        return 1.0e-3
    if "hpa" in u:
        return 1.0e-7
    if "bar" in u and "mbar" not in u:
        return 0.1
    if "mbar" in u:
        return 1.0e-4
    if "pa" in u:
        return 1.0e-6
    return 1.0


def _detect_time_multiplier_to_yr(u: str) -> float:
    """Return the factor m such that 1 <unit> = m yr.

    Used for A (time exponent −1) → v_yr = v_unit / m
    """
    if "yr" in u or "year" in u or "a-1" in u or "a^-1" in u \
            or u.endswith("/a") or u.endswith("a"):
        return 1.0
    if "day" in u or "/d" in u:
        return 1.0 / DAYS_PER_YEAR
    if "hr" in u or "hour" in u:
        return 1.0 / (DAYS_PER_YEAR * 24.0)
    if "s-1" in u or "s^-1" in u or "/s" in u or "sec" in u:
        return 1.0 / SECONDS_PER_YEAR
    return 1.0


def _detect_energy_multiplier_to_J_per_mol(u: str) -> float:
    """Return factor such that 1 <unit> = factor J/mol for Q."""
    if "kj" in u:
        return 1.0e3
    if "kcal" in u:
        return 4184.0
    if "cal" in u and "kcal" not in u:
        return 4.184
    if "ev" in u:
        return 96485.3321
    return 1.0


def normalize_glacier_unit(value: float, unit: str, param_key: str,
                           n: float = 3.0,
                           t_ref_c: Optional[float] = None) -> float:
    """Convert `value` from `unit` to the SI-flavoured canonical unit.

    Canonical units:
        glen_n               dimensionless
        rate_factor_A1/A2    MPa^-n yr^-1
        activation_energy_*  J mol^-1
        limit/constant_T     degC
        glen_enhancement     dimensionless
        critical_shear_rate  s^-1
        ice_density          kg m^-3
        gravity              m s^-2

    All temperature conversions target degC (subtract 273.15 for Kelvin).
    `n` is the Glen exponent (defaults to 3 if unknown); `t_ref_c` is
    unused here but reserved for temperature-dependent transformations
    (e.g. adding a Kelvin offset when the corpus uses °K for T).
    """
    u = _norm_unit_token(unit)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")

    # ── Zero-collapse guard for strictly-positive parameters ────────────
    if v == 0.0 and param_key in _POSITIVE_LOWER_BOUND_PARAMS:
        return float("nan")

    if param_key == "glen_n":
        return v
    if param_key in ("rate_factor_A1", "rate_factor_A2"):
        pres_mult = _detect_pressure_multiplier_to_MPa(u) ** n
        time_mult = _detect_time_multiplier_to_yr(u)
        # A has dimensions pressure^-n × time^-1
        # 1 <unit_A> = (pres_mult)^n × (1 / time_mult) MPa^-n yr^-1
        return v * pres_mult / max(time_mult, 1e-30)
    if param_key in ("activation_energy_Q1", "activation_energy_Q2"):
        return v * _detect_energy_multiplier_to_J_per_mol(u)
    if param_key in ("limit_temperature", "constant_temperature"):
        if u == "k" or u.startswith("kelvin") or "°k" in u or "degk" in u:
            return v - T0_CELSIUS
        return v
    if param_key == "glen_enhancement":
        return v
    if param_key == "critical_shear_rate":
        time_mult = _detect_time_multiplier_to_yr(u)
        return v / max(time_mult, 1e-30)
    if param_key == "ice_density":
        if "g/cm3" in u or "g.cm-3" in u or "gcm-3" in u:
            return v * 1.0e3
        if "kg/m3" in u or "kg.m-3" in u or "kgm-3" in u:
            return v
        return v
    if param_key == "gravity":
        return v
    return v


# ── Positive-lower-bound registry for the zero-collapse guard ───────────
_POSITIVE_LOWER_BOUND_PARAMS = frozenset({
    "glen_n", "rate_factor_A1", "rate_factor_A2",
    "activation_energy_Q1", "activation_energy_Q2",
    "glen_enhancement", "critical_shear_rate",
    "ice_density", "gravity",
})


def _glacier_clamp(value: float, param: str) -> Tuple[float, bool]:
    """Clamp to GLACIER_ONTOLOGY[param]['valid_range']; return clamped?"""
    lo, hi = GLACIER_ONTOLOGY[param]["valid_range"]
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


def glacier_fmt(param: str, si_value: float) -> str:
    """Format an SI value of `param` in the UI unit."""
    spec = GLACIER_ONTOLOGY[param]
    ui_val = si_value / spec["ui_scale"]
    if param in ("rate_factor_A1", "rate_factor_A2"):
        return f"{ui_val:.3e} {spec['ui_unit']}"
    if param in ("activation_energy_Q1", "activation_energy_Q2"):
        return f"{ui_val:.0f} {spec['ui_unit']}"
    if param == "glen_n":
        return f"{ui_val:.3f} {spec['ui_unit']}"
    if param == "glen_enhancement":
        return f"{ui_val:.3f} {spec['ui_unit']}"
    if param in ("limit_temperature", "constant_temperature"):
        return f"{ui_val:.1f} {spec['ui_unit']}"
    if param == "critical_shear_rate":
        return f"{ui_val:.2e} {spec['ui_unit']}"
    if param == "ice_density":
        return f"{ui_val:.1f} {spec['ui_unit']}"
    if param == "gravity":
        return f"{ui_val:.3f} {spec['ui_unit']}"
    return f"{ui_val:.4g} {spec['ui_unit']}"


# ============================================================================
# ███ SECTION 6 — GAZETTEER + ENTITY PATTERNS                            ███
# ============================================================================
# The gazetteer maps an entity type to a list of canonical aliases.  The
# regex compiler applies word-boundary rules so that "ice" doesn't match
# "police" and "n" doesn't match "nan".  Each alias can include spaces and
# underscores; the compiler interleaves `[\s_\-]{0,2}` between adjacent
# alnum characters to tolerate light punctuation noise.
# ============================================================================

GAZETTEER: Dict[str, List[str]] = {
    "material": [
        "ice", "glacier ice", "polar ice", "temperate ice", "cold ice",
        "polythermal ice", "firn", "ice sheet", "ice stream",
        "ice shelf", "marine ice", "debris-covered ice",
        "greenland ice", "antarctic ice", "arctic ice",
        "greenland", "antarctica", "himalaya", "himalayan",
        "alpine glacier", "valley glacier", "piedmont glacier",
        "ice cap", "ice field",
    ],
    "property": [
        "glen exponent", "glen flow law", "glen flow-law",
        "glen flow law exponent", "glen flow-law exponent",
        "flow law exponent", "flow-law exponent",
        "stress exponent", "creep exponent",
        "power law exponent", "power-law exponent",
        "rate factor", "arrhenius rate factor",
        "pre-exponential factor", "pre-exponential",
        "flow law coefficient", "creep rate factor",
        "softness parameter", "ice softness",
        "activation energy", "creep activation energy",
        "activation enthalpy", "arrhenius activation energy",
        "creep activation enthalpy",
        "enhancement factor", "glen enhancement", "fabric enhancement",
        "anisotropy enhancement",
        "critical shear rate", "regularisation shear rate",
        "shear rate floor", "strain rate floor",
        "ice density", "glacier ice density", "firn density",
        "gravity", "gravitational acceleration",
        "limit temperature", "transition temperature",
        "regime switch temperature",
        "ice temperature", "isothermal temperature",
    ],
    "property_weak": [
        "n", "a", "q", "e", "gamma_dot_c", "gdot_c",
        "n_glen", "a_rate", "a1", "a2", "q1", "q2", "t_star",
        "rho_ice", "rho_i", "g_acc",
    ],
    "temperature_regime": [
        "cold", "cold ice", "cold-based", "polar", "polar ice",
        "temperate", "temperate ice", "warm", "warm-based",
        "polythermal", "isothermal", "pressure-melting",
        "pressure melting", "sub-temperate",
    ],
    "stress_regime": [
        "low stress", "high stress", "quasi-static", "fast flow",
        "slow flow", "creep", "basal sliding", "shear-dominated",
        "extensional", "compressive", "ice divide", "ice dome",
        "ice stream", "shear margin", "outlet glacier",
        "fast-flow", "slow-flow", "surge", "quiescent",
    ],
    "ice_architecture": [
        "columnar", "columnar grain", "columnar ice",
        "equiaxed", "equiaxed grain", "equiaxed ice",
        "fabric", "anisotropic", "isotropic",
        "single-maximum fabric", "single maximum fabric",
        "multi-maximum fabric", "multi maximum fabric",
        "random fabric", "random texture",
        "c-axis fabric", "c axis fabric",
        "lattice-preferred", "lattice preferred",
        "lpo", "cpo", "crystal preferred orientation",
        "grain size", "crystal size", "crystallite size",
        "highly oriented", "aligned grains",
    ],
    "method": [
        "creep test", "uniaxial compression", "simple shear",
        "torsion test", "lab experiment", "laboratory experiment",
        "field measurement", "field observation",
        "borehole", "borehole thermometry", "borehole deformation",
        "seismic", "seismic reflection", "seismic refraction",
        "radar", "radio echo", "res phase",
        "ice core", "ice-core", "ice core analysis",
        "inverse model", "adjoint", "data assimilation",
        "glen", "glen 1955", "nye", "nye 1953",
        "lliboutry", "lliboutry 1963", "lliboutry 1969",
        "goldsby-kohlstedt", "goldsby kohlstedt",
        "budd-jacka", "budd jacka",
        "cuffey-paterson", "cuffey paterson",
        "arrhenius fit", "arrhenius regression",
        "power-law fit", "power law fit",
        "grain-boundary sliding", "gb sliding",
        "dislocation creep", "diffusion creep",
        "basal slip",
    ],
    "theory_framework": [
        "glen flow law", "glen", "glen 1955",
        "nye", "nye 1953",
        "power law", "power-law",
        "arrhenius", "arrhenius law",
        "goldsby-kohlstedt", "goldsby kohlstedt",
        "grain-boundary sliding", "gb sliding",
        "dislocation creep", "diffusion creep",
        "basal slip",
        "anisotropic flow law", "enhanced flow law",
        "budd-jacka", "budd jacka",
        "cuffey-paterson", "cuffey paterson",
        "shallow ice approximation", "sia",
        "higher-order", "higher order",
        "full stokes", "full-stokes",
        "sos", "shelf-ocean",
        "first-order", "first order",
    ],
    "unit": [
        "MPa", "kPa", "Pa", "GPa", "bar",
        "MPa^-n yr^-1", "MPa^-3 yr^-1", "MPa^-n s^-1",
        "Pa^-n s^-1", "Pa^-3 s^-1", "MPa^-3 s^-1",
        "s^-1", "s-1", "/s", "yr^-1", "yr-1", "a^-1", "a-1",
        "J/mol", "J mol-1", "J/mol", "kJ/mol", "kJ mol-1",
        "kJ mol^-1", "kJmol^-1",
        "degC", "°C", "C", "K", "kelvin",
        "kg/m3", "kg m-3", "kgm-3",
        "m/s2", "m s-2", "m s^-2",
    ],
    "synthesis_method": [
        "field-based", "field based", "field-derived",
        "lab-derived", "lab derived", "laboratory-derived",
        "ice-core", "ice core derived",
        "borehole-derived", "borehole derived",
        "seismic-inferred", "seismic inferred",
        "radar-inferred", "radar inferred",
        "inverse-model", "inverse model derived",
        "data-assimilation", "data assimilation derived",
    ],
    "grain_architecture": [
        "columnar", "equiaxed", "single-maximum", "single-max",
        "multi-maximum", "multi-max", "random-fabric",
        "random fabric", "anisotropic", "isotropic",
        "lpo", "cpo", "fabric-enhanced", "fabric enhanced",
        "fabric-weakened", "fabric weakened",
        "c-axis-aligned", "c-axis aligned",
    ],
}


# ── Unicode folding table for `norm_text` ───────────────────────────────
_CHAR_FOLD = {
    "μ": "mu", "µ": "mu", "ρ": "rho", "–": "-", "—": "-",
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
    "γ": "gamma", "σ": "sigma", "λ": "lambda",
    "θ": "theta", "φ": "phi", "η": "eta",
    "δ": "delta", "ε": "epsilon", "ν": "nu",
    "°": "deg", "×": "x", "÷": "/", "±": "+-",
    "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
    "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}


def norm_text(s: Any) -> str:
    """Aggressively normalise text for entity matching."""
    s = unicodedata.normalize("NFKC", str(s))
    for k, v in _CHAR_FOLD.items():
        s = s.replace(k, v)
    s = re.sub(r"[{}$\\]", "", s)
    s = re.sub(r"_(?=\d)", "", s)
    s = re.sub(r"(?<=[a-z])\s+(?=\d)", "", s.lower())
    return s


def alias_pattern(alias: str) -> str:
    """Build a word-bounded regex from an alias string.

    Adjacent alnum characters are separated by `[\\s_\\-]{0,2}` so that
    the alias tolerates light punctuation noise.  Leading/trailing
    lookarounds ensure we don't match inside larger tokens."""
    a = norm_text(alias)
    out = []
    for i, ch in enumerate(a):
        if i > 0 and ch.isalnum() and out and out[-1][-1].isalnum():
            out.append(r"[\s_\-]{0,2}")
        out.append(re.escape(ch))
    return rf"(?<![a-z0-9]){''.join(out)}(?![a-z0-9])"


ENTITY_PATTERNS: Dict[str, List[re.Pattern]] = {
    k: [re.compile(alias_pattern(a)) for a in v]
    for k, v in GAZETTEER.items()
}


# ============================================================================
# ███ SECTION 7 — PARAM_CANON (gatekeeper bounds + aliases)              ███
# ============================================================================
# PARAM_CANON is a lighter-weight view onto the ontology used by the
# LLM prompts and gatekeeper.  It defines plausible hard bounds (the
# gatekeeper rejects out-of-bound values) plus a shorter alias list.
# ============================================================================

PARAM_CANON: Dict[str, Dict[str, Any]] = {
    "glen_n": dict(
        aliases=[
            "glen exponent", "glen flow-law exponent",
            "stress exponent", "creep exponent", "flow law exponent",
            "n exponent", "n glen", "srs index",
        ],
        unit="dimensionless",
        plausible=(0.5, 6.0),
    ),
    "rate_factor_A1": dict(
        aliases=[
            "rate factor", "arrhenius rate factor", "pre-exponential",
            "softness parameter", "flow law coefficient", "A1", "A_1",
            "cold rate factor", "glen rate factor",
        ],
        unit="MPa^-n yr^-1",
        plausible=(1e-10, 1e30),
    ),
    "rate_factor_A2": dict(
        aliases=[
            "warm rate factor", "A2", "A_2",
            "temperate rate factor", "warm softness",
        ],
        unit="MPa^-n yr^-1",
        plausible=(1e-10, 1e40),
    ),
    "activation_energy_Q1": dict(
        aliases=[
            "activation energy", "creep activation energy",
            "activation enthalpy", "Q1", "Q_1",
            "cold activation energy", "arrhenius activation energy",
        ],
        unit="J mol^-1",
        plausible=(1e3, 3e5),
    ),
    "activation_energy_Q2": dict(
        aliases=[
            "warm activation energy", "Q2", "Q_2",
            "temperate activation energy", "high-temperature activation",
        ],
        unit="J mol^-1",
        plausible=(6e4, 3e5),
    ),
    "limit_temperature": dict(
        aliases=[
            "limit temperature", "regime switch temperature",
            "critical temperature", "transition temperature",
            "T_star", "T-limit", "arrhenius transition temperature",
        ],
        unit="degC",
        plausible=(-60.0, 10.0),
    ),
    "constant_temperature": dict(
        aliases=[
            "constant temperature", "isothermal temperature",
            "ice temperature", "temperature", "mean temperature",
        ],
        unit="degC",
        plausible=(-60.0, 1.0),
    ),
    "glen_enhancement": dict(
        aliases=[
            "enhancement factor", "glen enhancement", "enhancement",
            "E factor", "fabric enhancement", "softening factor",
        ],
        unit="dimensionless",
        plausible=(0.05, 50.0),
    ),
    "critical_shear_rate": dict(
        aliases=[
            "critical shear rate", "critical strain rate",
            "regularisation shear rate", "shear rate floor",
        ],
        unit="s^-1",
        plausible=(1e-25, 1e-2),
    ),
    "ice_density": dict(
        aliases=[
            "ice density", "density", "rho ice", "mass density",
            "bulk density", "firn density",
        ],
        unit="kg m^-3",
        plausible=(300.0, 1000.0),
    ),
    "gravity": dict(
        aliases=[
            "gravity", "gravitational acceleration", "g",
        ],
        unit="m s^-2",
        plausible=(9.0, 11.0),
    ),
}


# ── Canonical alias → PARAM_CANON key map ───────────────────────────────
_PARAM_ALIASES: Dict[str, str] = {
    # glen_n
    "n": "glen_n", "glen_n": "glen_n", "glen exponent": "glen_n",
    "stress exponent": "glen_n", "creep exponent": "glen_n",
    "flow law exponent": "glen_n", "flow-law exponent": "glen_n",
    "power law exponent": "glen_n", "power-law exponent": "glen_n",
    "n_glen": "glen_n", "srs index": "glen_n",
    # rate_factor_A1
    "a1": "rate_factor_A1", "a_1": "rate_factor_A1",
    "rate factor": "rate_factor_A1", "rate_factor": "rate_factor_A1",
    "arrhenius rate factor": "rate_factor_A1",
    "pre-exponential": "rate_factor_A1",
    "pre_exponential": "rate_factor_A1",
    "pre-exponential factor": "rate_factor_A1",
    "softness parameter": "rate_factor_A1",
    "flow law coefficient": "rate_factor_A1",
    "cold rate factor": "rate_factor_A1",
    "glen rate factor": "rate_factor_A1",
    "a_rate": "rate_factor_A1",
    # rate_factor_A2
    "a2": "rate_factor_A2", "a_2": "rate_factor_A2",
    "warm rate factor": "rate_factor_A2",
    "temperate rate factor": "rate_factor_A2",
    "warm softness": "rate_factor_A2",
    # activation_energy_Q1
    "q1": "activation_energy_Q1", "q_1": "activation_energy_Q1",
    "activation energy": "activation_energy_Q1",
    "creep activation energy": "activation_energy_Q1",
    "activation enthalpy": "activation_energy_Q1",
    "cold activation energy": "activation_energy_Q1",
    "arrhenius activation energy": "activation_energy_Q1",
    # activation_energy_Q2
    "q2": "activation_energy_Q2", "q_2": "activation_energy_Q2",
    "warm activation energy": "activation_energy_Q2",
    "temperate activation energy": "activation_energy_Q2",
    "high-temperature activation": "activation_energy_Q2",
    # limit_temperature
    "limit temperature": "limit_temperature",
    "limit_temperature": "limit_temperature",
    "regime switch temperature": "limit_temperature",
    "critical temperature": "limit_temperature",
    "transition temperature": "limit_temperature",
    "t_star": "limit_temperature", "t-limit": "limit_temperature",
    "arrhenius transition temperature": "limit_temperature",
    # constant_temperature
    "constant temperature": "constant_temperature",
    "constant_temperature": "constant_temperature",
    "isothermal temperature": "constant_temperature",
    "ice temperature": "constant_temperature",
    "temperature": "constant_temperature",
    "mean temperature": "constant_temperature",
    # glen_enhancement
    "enhancement factor": "glen_enhancement",
    "glen_enhancement": "glen_enhancement",
    "enhancement": "glen_enhancement",
    "e factor": "glen_enhancement", "e": "glen_enhancement",
    "fabric enhancement": "glen_enhancement",
    "softening factor": "glen_enhancement",
    # critical_shear_rate
    "critical shear rate": "critical_shear_rate",
    "critical_shear_rate": "critical_shear_rate",
    "critical strain rate": "critical_shear_rate",
    "regularisation shear rate": "critical_shear_rate",
    "shear rate floor": "critical_shear_rate",
    "strain rate floor": "critical_shear_rate",
    "gamma_dot_c": "critical_shear_rate",
    # ice_density
    "ice density": "ice_density", "ice_density": "ice_density",
    "density": "ice_density", "rho ice": "ice_density",
    "mass density": "ice_density", "bulk density": "ice_density",
    "firn density": "ice_density",
    "rho_ice": "ice_density", "rho_i": "ice_density",
    # gravity
    "gravity": "gravity", "gravitational acceleration": "gravity",
    "g": "gravity", "g_acc": "gravity",
}


def _canonicalize_param(raw_p: str) -> Optional[str]:
    """Best-effort canonicalisation of a raw param name to a GLACIER key."""
    if raw_p is None:
        return None
    p = str(raw_p).strip()
    if not p:
        return None
    if p in GLACIER_ONTOLOGY:
        return p
    if p in _PARAM_ALIASES:
        return _PARAM_ALIASES[p]
    low = p.lower().replace(" ", "_").replace("-", "_")
    if low in GLACIER_ONTOLOGY:
        return low
    if low in _PARAM_ALIASES:
        return _PARAM_ALIASES[low]
    for canon in sorted(GLACIER_ONTOLOGY, key=len, reverse=True):
        if canon in low:
            return canon
    for alias in sorted(_PARAM_ALIASES, key=len, reverse=True):
        if alias and alias in low:
            return _PARAM_ALIASES[alias]
    return None


# ============================================================================
# ███ SECTION 8 — VALUE COERCION + HASHING                               ███
# ============================================================================

def _coerce_value(item: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Coerce `item['value']` to a float even if it's a sci-notation
    string like "3.5 × 10^-25".  Also returns the trailing unit string
    if the value's unit is embedded in the string."""
    raw_v = item.get("value")
    explicit_unit = str(item.get("unit") or "").strip()
    if isinstance(raw_v, (int, float)) and not isinstance(raw_v, bool):
        return float(raw_v), explicit_unit
    s = str(raw_v or "").strip()
    if not s:
        return None, explicit_unit
    m = re.search(
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        r"(?:\s*[×xX\*]\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?)?"
        r"(.*)", s)
    if not m:
        return None, explicit_unit
    try:
        mantissa = float(m.group(1))
    except (TypeError, ValueError):
        return None, explicit_unit
    exponent = int(m.group(2)) if m.group(2) else 0
    value = mantissa * (10 ** exponent)
    trailing = m.group(3).strip()
    unit = explicit_unit or trailing
    return value, unit


def _glacier_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ============================================================================
# ███ SECTION 9 — SIF PARSER (context extraction from existing files)    ███
# ============================================================================

def parse_elmer_sif_context(sif_file_path: str) -> Dict[str, Any]:
    """Read an existing Elmer SIF and extract parameters + boundary IDs
    to seed the recommender as CONTEXT (never as a verbatim hit)."""
    if not os.path.exists(sif_file_path):
        return {"error": "File not found", "path": sif_file_path}
    try:
        with open(sif_file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return {"error": str(e), "path": sif_file_path}

    def _m(pattern: str, cast: Callable = float, default: Any = None):
        mm = re.search(pattern, content, re.IGNORECASE)
        if not mm:
            return default
        try:
            return cast(mm.group(1))
        except (TypeError, ValueError, IndexError):
            return default

    mesh_db = re.search(r'Mesh DB\s+"[^"]*"\s+"([^"]+)"', content)
    boundaries = re.findall(r'Target Boundaries\s*=\s*([0-9\s]+)', content)
    flat_ids: List[int] = []
    for group in boundaries:
        for tok in re.split(r'\s+', group.strip()):
            try:
                flat_ids.append(int(tok))
            except ValueError:
                pass

    density_expr = None
    dm = re.search(r'Density\s*=\s*Real\s+\$([^\n]+)', content)
    if dm:
        density_expr = dm.group(1).strip()
    else:
        dm = re.search(r'Density\s*=\s*Real\s+([0-9.eE+\-]+)', content)
        if dm:
            density_expr = dm.group(1).strip()

    gravity_expr = None
    gm = re.search(r'Flow BodyForce 3\s*=\s*Real\s+\$([^\n]+)', content)
    if gm:
        gravity_expr = gm.group(1).strip()
    else:
        gm = re.search(r'Flow BodyForce 3\s*=\s*Real\s+([0-9.eE+\-]+)', content)
        if gm:
            gravity_expr = gm.group(1).strip()

    return {
        "path": sif_file_path,
        "size_bytes": len(content),
        "mesh_db": mesh_db.group(1) if mesh_db else None,
        "boundary_ids": sorted(set(flat_ids)),
        "glen_n": _m(r'Glen Exponent\s*=\s*Real\s+([0-9.]+)', float),
        "rate_factor_1": _m(r'Rate Factor 1\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "rate_factor_2": _m(r'Rate Factor 2\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "activation_energy_1": _m(r'Activation Energy 1\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "activation_energy_2": _m(r'Activation Energy 2\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "limit_temperature": _m(r'Limit Temperature\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "constant_temperature": _m(r'Constant Temperature\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "glen_enhancement": _m(r'Glen Enhancement Factor\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "critical_shear_rate": _m(r'Critical Shear Rate\s*=\s*Real\s+([0-9.eE+\-]+)', float),
        "density_expr": density_expr,
        "gravity_expr": gravity_expr,
        "steady_state_max_iter": _m(r'Steady State Max Iterations\s*=\s*(\d+)', int),
        "output_file": (re.search(r'Output File\s*=\s*"([^"]+)"', content)
                        .group(1) if 'Output File' in content else None),
        "post_file": (re.search(r'Post File\s*=\s*"([^"]+)"', content)
                      .group(1) if 'Post File' in content else None),
        "raw_snippet": content[:1500],
    }


# ============================================================================
# ███ SECTION 10 — NUMERIC SCANNER + UNIT REGEXES                        ███
# ============================================================================

_NUM_ANY = re.compile(
    r"(?<![a-z0-9.])"
    r"(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"(?:\s*[x×*]\s*10\s*\^?\(?\s*(?P<exp>[+-]?\d+)\s*\)?)?"
)

UNIT_TOKENS: Dict[str, str] = {
    "GPa":  r"gpa",
    "MPa":  r"mpa",
    "kPa":  r"kpa",
    "Pa":   r"pa",
    "bar":  r"bar",
    "s^-1": r"s\^?\(?-1\)?|s-1|/s|per\s+second|sec-1",
    "yr^-1": r"yr\^?\(?-1\)?|yr-1|/yr|a\^?-?1|per\s+year|annual",
    "MPa^-n yr^-1": (
        r"mpa\^?-?n\s*[·•\*x×]?\s*yr\^?-?1|"
        r"mpa\^?-?n\s*[·•\*x×]?\s*a-1|"
        r"mpa\^-?3\s*[·•\*x×]?\s*yr\^?-?1|"
        r"mpa\^-?3\s*[·•\*x×]?\s*a-1"
    ),
    "Pa^-n s^-1": (
        r"pa\^?-?n\s*[·•\*x×]?\s*s\^?-?1|"
        r"pa\^-?3\s*[·•\*x×]?\s*s\^?-?1"
    ),
    "MPa^-n s^-1": (
        r"mpa\^?-?n\s*[·•\*x×]?\s*s\^?-?1|"
        r"mpa\^-?3\s*[·•\*x×]?\s*s\^?-?1"
    ),
    "J/mol": r"j\s*/?\s*mol|jmol",
    "kJ/mol": r"kj\s*/?\s*mol|kjm",
    "degC": r"°?c|deg\s*c|celsius",
    "K":    r"\bk\b|kelvin",
    "kg/m3": r"kg\s*/?\s*m\^?-?3|kg\s*m-3|kgm-3",
    "m/s2": r"m\s*/?\s*s\^?-?2|m\s*s-2",
}


def unit_regex_for(param_key: str) -> Optional[str]:
    """Return the alternation regex that matches any unit acceptable for
    `param_key`.  Used by `find_value_after` to disambiguate hits."""
    spec = GLACIER_ONTOLOGY.get(param_key, {})
    u = spec.get("unit")
    if u is None and param_key not in ("glen_n", "glen_enhancement"):
        return None
    if param_key in ("glen_n", "glen_enhancement"):
        return None
    if param_key in ("rate_factor_A1", "rate_factor_A2"):
        return "|".join(UNIT_TOKENS[x] for x in
                        ("MPa^-n yr^-1", "MPa^-n s^-1",
                         "Pa^-n s^-1", "MPa", "Pa", "yr^-1", "s^-1"))
    if param_key in ("activation_energy_Q1", "activation_energy_Q2"):
        return "|".join(UNIT_TOKENS[x] for x in ("J/mol", "kJ/mol"))
    if param_key in ("limit_temperature", "constant_temperature"):
        return "|".join(UNIT_TOKENS[x] for x in ("degC", "K"))
    if param_key == "critical_shear_rate":
        return "|".join(UNIT_TOKENS[x] for x in ("s^-1", "yr^-1"))
    if param_key == "ice_density":
        return UNIT_TOKENS["kg/m3"]
    if param_key == "gravity":
        return UNIT_TOKENS["m/s2"]
    return UNIT_TOKENS.get(u)


def _num_value(m: "re.Match") -> float:
    v = float(m.group("val"))
    return v * 10.0 ** int(m.group("exp")) if m.group("exp") else v


def find_value_after(text: str, pos: int, param_key: str,
                     window: int = 90) -> Optional[Dict[str, Any]]:
    """Return the best numeric candidate following `pos` in `text`."""
    ure = unit_regex_for(param_key)
    seg = text[pos: pos + window]
    best = None
    for m in _NUM_ANY.finditer(seg):
        um = (re.match(rf"\s*({ure})\b", seg[m.end(): m.end() + 14])
              if ure else None)
        sc = (2.0 if um else (1.0 if ure is None else 0.6)) - 0.01 * m.start()
        if best is None or sc > best[0]:
            best = (sc, _num_value(m), um.group(1) if um else None)
    if best is None:
        return None
    return dict(value=best[1], unit=best[2])


def _value_from_cell(key: str, val: str, param_key: str
                     ) -> Tuple[Optional[float], Optional[str]]:
    """Extract (value, unit) from a structured key/value cell."""
    kt = norm_text(key)
    vt = norm_text(val)
    ure = unit_regex_for(param_key)
    best = None
    for m in _NUM_ANY.finditer(vt):
        um = (re.match(rf"\s*({ure})\b", vt[m.end(): m.end() + 14])
              if ure else None)
        sc = 2.0 if um else (1.0 if (ure is None or
                                     (ure and re.search(rf"\b{ure}\b", kt)))
                             else 0.0)
        if sc and (best is None or sc > best[0]):
            best = (sc, _num_value(m), um.group(1) if um else None)
    return (best[1], best[2]) if best else (None, None)


# ============================================================================
# ███ SECTION 11 — SIDE-NOTE TABLE PARSER                                ███
# ============================================================================
# Parses markdown table rows of the form:
#     | shear modulus | 47.19 | GPa |
# and emits a dict with (param, value, unit, evidence, method).
# REJECTS values ≤ 0 — this kills the "0.000 GPa" degenerate collapse that
# silently poisons downstream physics.
# ============================================================================

_SIDE_NOTE_ROW_RE = re.compile(
    r"\|\s*(?P<key>[A-Za-z][A-Za-z0-9_'λσνβ·\-\s]{0,40}?)\s*\|"
    r"\s*(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"(?:\s*[×xX\*]\s*10\s*\^?\s*\{?\s*[+-]?\d+\s*\}?)?)"
    r"\s*(?P<unit>[A-Za-zμÅ·/^(){}.0-9\-\s]{0,24}?)\s*\|",
    re.MULTILINE,
)

_SIDE_NOTE_KEY_MAP: Dict[str, str] = {
    # glen n
    "glen exponent":            "glen_n",
    "glen flow law exponent":   "glen_n",
    "flow law exponent":        "glen_n",
    "stress exponent":          "glen_n",
    "creep exponent":           "glen_n",
    "power law exponent":       "glen_n",
    # rate factor
    "rate factor":              "rate_factor_A1",
    "arrhenius rate factor":    "rate_factor_A1",
    "pre-exponential":          "rate_factor_A1",
    "pre exponential":          "rate_factor_A1",
    "softness parameter":       "rate_factor_A1",
    "flow law coefficient":     "rate_factor_A1",
    "warm rate factor":         "rate_factor_A2",
    # activation energy
    "activation energy":        "activation_energy_Q1",
    "activation enthalpy":      "activation_energy_Q1",
    "creep activation energy":  "activation_energy_Q1",
    "cold activation energy":   "activation_energy_Q1",
    "warm activation energy":   "activation_energy_Q2",
    # temperatures
    "limit temperature":        "limit_temperature",
    "transition temperature":   "limit_temperature",
    "regime switch temperature": "limit_temperature",
    "constant temperature":     "constant_temperature",
    "ice temperature":          "constant_temperature",
    "isothermal temperature":   "constant_temperature",
    "temperature":              "constant_temperature",
    # enhancement
    "enhancement factor":       "glen_enhancement",
    "glen enhancement":         "glen_enhancement",
    "fabric enhancement":       "glen_enhancement",
    # shear rate
    "critical shear rate":      "critical_shear_rate",
    "regularisation shear rate": "critical_shear_rate",
    "shear rate floor":         "critical_shear_rate",
    # density
    "ice density":              "ice_density",
    "glacier ice density":      "ice_density",
    "firn density":             "ice_density",
    "density":                  "ice_density",
    # gravity
    "gravity":                  "gravity",
    "gravitational acceleration": "gravity",
}


def parse_side_note_table(text: str) -> List[Dict[str, Any]]:
    """Parse markdown table rows; REJECT non-positive values."""
    out: List[Dict[str, Any]] = []
    for m in _SIDE_NOTE_ROW_RE.finditer(text):
        key_norm = norm_text(m.group("key")).strip(" -|").lower()
        param: Optional[str] = None
        for k in sorted(_SIDE_NOTE_KEY_MAP, key=len, reverse=True):
            if k in key_norm:
                param = _SIDE_NOTE_KEY_MAP[k]
                break
        if param is None:
            continue

        num_field = m.group("val").strip()
        mm = re.match(r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", num_field)
        if not mm:
            continue
        try:
            mantissa = float(mm.group(1))
        except (TypeError, ValueError):
            continue
        em = re.search(r"10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?", num_field)
        exponent = int(em.group(1)) if em else 0
        value = mantissa * (10 ** exponent)
        if value <= 0.0 and param not in ("limit_temperature",
                                          "constant_temperature"):
            logger.info("parse_side_note_table: rejecting %s ≤ 0", param)
            continue
        out.append({
            "param":    param,
            "value":    value,
            "unit":     m.group("unit").strip(),
            "evidence": m.group(0).strip(),
            "method":   "side_note_table",
        })
    return out


# ============================================================================
# ███ SECTION 12 — FLATTEN + RECORD TEXT                                 ███
# ============================================================================

def flatten_keyvals(node: Any, path: str = "") -> List[Tuple[str, str, str]]:
    """Flatten a nested dict/list into a list of (path, key, value) triples."""
    out: List[Tuple[str, str, str]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                out += flatten_keyvals(v, f"{path}.{k}")
            else:
                out.append((f"{path}.{k}", str(k),
                            "" if v is None else str(v)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += flatten_keyvals(v, f"{path}[{i}]")
    return out


def record_text(data: Any) -> str:
    """Render a metadata record as a flat text blob for retrieval."""
    return "; ".join(f"{k}: {v}" for _, k, v in flatten_keyvals(data))


# ============================================================================
# ███ SECTION 13 — HEURISTIC NER EXTRACTION                              ███
# ============================================================================

def heuristic_extract(records: List[Dict[str, Any]],
                      param_key: str,
                      target_unit: Optional[str] = None
                      ) -> List[Dict[str, Any]]:
    """Deterministic regex NER: three passes.

      (a) structured key/value cells
      (b) free-text scan next to each alias
      (c) side-note markdown table rows
    """
    canon = GLACIER_ONTOLOGY.get(param_key, {})
    tu = (target_unit or canon.get("unit") or "").lower()
    key_pats = [re.compile(alias_pattern(a))
                for a in canon.get("aliases", [])]
    out: List[Dict[str, Any]] = []

    for rec in records:
        text = rec.get("text_norm") or norm_text(rec.get("text", ""))
        raw = rec.get("text", text)

        # (a) structured cells
        for path, key, val in flatten_keyvals(rec.get("data", {})):
            if not any(p.search(norm_text(key)) for p in key_pats):
                continue
            v, u = _value_from_cell(key, val, param_key)
            if v is None:
                continue
            out.append({
                "value": float(v), "unit": u or tu,
                "property_label": key,
                "evidence_span": f"{key} = {val}",
                "source": rec.get("source", ""),
                "path": path,
            })

        # (b) free-text
        for p in key_pats:
            for m in p.finditer(text):
                hit = find_value_after(text, m.end(), param_key)
                if not hit:
                    continue
                try:
                    fv = float(hit["value"])
                except (TypeError, ValueError):
                    continue
                out.append({
                    "value": fv,
                    "unit": hit.get("unit") or tu,
                    "property_label": m.group(0),
                    "evidence_span": text[max(0, m.start() - 15):
                                          m.end() + 45],
                    "source": rec.get("source", ""),
                    "path": "",
                })

        # (c) side-note tables
        for row in parse_side_note_table(raw):
            if row["param"] != param_key:
                continue
            try:
                v_si = normalize_glacier_unit(row["value"], row["unit"],
                                              row["param"])
            except Exception:
                v_si = float(row["value"])
            if v_si is None or (isinstance(v_si, float) and math.isnan(v_si)):
                continue
            out.append({
                "value": float(v_si),
                "unit": canon.get("unit") or tu,
                "property_label": row["evidence"][:80],
                "evidence_span": row["evidence"],
                "source": rec.get("source", ""),
                "path": "side_note_table",
            })

    seen: Set[Tuple[float, str]] = set()
    ded: List[Dict[str, Any]] = []
    for e in out:
        kk = (round(e["value"], 6), str(e.get("unit")))
        if kk not in seen:
            seen.add(kk)
            ded.append(e)
    return ded


# ============================================================================
# ███ SECTION 14 — GLACIER REGIME TABLES                                 ███
# ============================================================================
# Three curated tables:
#   GLEN_N_REGIMES       keyed by (stress_regime, thermal_state, fabric)
#   ARRHENIUS_REGIMES    keyed by (thermal_state, impurity) for A1/A2 ratio
#   FABRIC_REGIMES       keyed by (fabric, thermal_state) for enhancement E
# ============================================================================

GLEN_N_REGIMES: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ("low",  "cold",      "isotropic"):   dict(
        low=1.0, high=2.0, bench=1.5,
        desc="Diffusion creep / low-stress Newtonian limit"),
    ("low",  "temperate", "isotropic"):   dict(
        low=1.0, high=2.0, bench=1.8,
        desc="Temperate low-stress diffusion-accommodated GBS"),
    ("low",  "polythermal","isotropic"):  dict(
        low=1.0, high=2.5, bench=1.8,
        desc="Polythermal low-stress mixed creep"),
    ("high", "cold",      "isotropic"):   dict(
        low=3.0, high=4.5, bench=3.0,
        desc="Canonical Glen n=3 dislocation creep (cold)"),
    ("high", "temperate", "isotropic"):   dict(
        low=2.5, high=4.0, bench=3.0,
        desc="Warm high-stress dislocation creep"),
    ("high", "polythermal","isotropic"):  dict(
        low=2.5, high=4.5, bench=3.0,
        desc="Polythermal high-stress dislocation creep"),
    ("high", "cold",      "anisotropic"): dict(
        low=3.0, high=4.5, bench=3.5,
        desc="Fabric-enhanced cold dislocation creep"),
    ("high", "temperate", "anisotropic"): dict(
        low=2.0, high=4.0, bench=3.0,
        desc="Warm fabric-enhanced tertiary creep"),
    ("low",  "temperate", "anisotropic"): dict(
        low=1.5, high=3.0, bench=2.0,
        desc="Fabric-enhanced diffusion-accommodated flow"),
}

ARRHENIUS_REGIMES: Dict[Tuple[str, str], Dict[str, Any]] = {
    # (thermal, impurity) → A-warm regime.  Cold A is fixed at the
    # Cuffey-Paterson reference (1.14e-5 MPa^-3 yr^-1 for A1 in Elmer's
    # year-based unit convention).
    ("cold",       "clean"):    dict(
        low=1e-8, high=1e-2, bench=1.14e-5,
        desc="Cold polar ice — very stiff (Cuffey-Paterson A_cold)"),
    ("cold",       "dust"):     dict(
        low=1e-7, high=1e-1, bench=1.14e-4,
        desc="Cold dusty ice — moderate softening"),
    ("temperate",  "clean"):    dict(
        low=1e16, high=1e25, bench=2.4e24,
        desc="Temperate ice at pressure-melting point"),
    ("temperate",  "dust"):     dict(
        low=1e17, high=1e26, bench=1.5e25,
        desc="Temperate dusty ice — enhanced flow"),
    ("polythermal","clean"):    dict(
        low=1e5, high=1e18, bench=1.258e13,
        desc="Polythermal Glen (Paterson A_warm at -10°C)"),
    ("polythermal","dust"):     dict(
        low=1e6, high=1e19, bench=1e14,
        desc="Polythermal dusty ice — enhanced flow"),
}

FABRIC_REGIMES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("isotropic",  "cold"):      dict(
        low=0.5, high=1.5, bench=1.0,
        desc="Random fabric — enhancement E ≈ 1"),
    ("isotropic",  "temperate"): dict(
        low=0.5, high=1.5, bench=1.0,
        desc="Random fabric temperate — E ≈ 1"),
    ("isotropic",  "polythermal"): dict(
        low=0.5, high=1.5, bench=1.0,
        desc="Random fabric polythermal — E ≈ 1"),
    ("anisotropic","cold"):      dict(
        low=2.0, high=5.0, bench=3.0,
        desc="Single-max c-axis fabric cold — E ≈ 3"),
    ("anisotropic","temperate"): dict(
        low=3.0, high=10.0, bench=5.0,
        desc="Single-max fabric temperate shear margin — E ≈ 5"),
    ("anisotropic","polythermal"): dict(
        low=2.0, high=6.0, bench=3.5,
        desc="Anisotropic polythermal — E ≈ 3.5"),
}

ARRHENIUS_Q_REGIMES: Dict[str, Dict[str, Any]] = {
    "cold":   dict(low=55000.0, high=65000.0, bench=60000.0,
                   desc="Below T* = -10°C (Cuffey-Paterson Q_cold)"),
    "warm":   dict(low=120000.0, high=160000.0, bench=139000.0,
                   desc="Above T* = -10°C (Cuffey-Paterson Q_warm)"),
    "transitional": dict(low=55000.0, high=140000.0, bench=80000.0,
                         desc="Near-transition — large uncertainty"),
}


# ── Regime normalisers ─────────────────────────────────────────────────

def _norm_stress_regime(s: Optional[str]) -> Optional[str]:
    s = (s or "").lower()
    if any(k in s for k in ("high", "fast", "shear-dominated",
                            "extensional", "compressive", "stream")):
        return "high"
    if any(k in s for k in ("low", "slow", "quasi-static", "creep",
                            "divide", "dome")):
        return "low"
    return None


def _norm_temp_regime(s: Optional[str]) -> Optional[str]:
    s = (s or "").lower()
    if any(k in s for k in ("cold", "polar", "cold-based")):
        return "cold"
    if any(k in s for k in ("temperate", "warm", "melting", "warm-based")):
        return "temperate"
    if "polythermal" in s:
        return "polythermal"
    return None


def _norm_fabric(s: Optional[str]) -> Optional[str]:
    s = (s or "").lower()
    if any(k in s for k in ("single-max", "single max", "multi-max",
                            "multi max", "anisotropic", "lpo", "cpo",
                            "c-axis", "c axis", "fabric")):
        return "anisotropic"
    if any(k in s for k in ("isotropic", "random", "equiaxed")):
        return "isotropic"
    return None


# ============================================================================
# ███ SECTION 15 — GLACIER REGIME CLASSIFIERS                            ███
# ============================================================================

def classify_glen_n_regime(stress: Optional[str],
                           temp: Optional[str],
                           fabric: Optional[str]) -> Dict[str, Any]:
    """Three-factor classifier: (stress, thermal, fabric) → Glen-n band."""
    s = _norm_stress_regime(stress)
    t = _norm_temp_regime(temp)
    f = _norm_fabric(fabric) or "isotropic"
    r = None
    if s and t and f:
        r = GLEN_N_REGIMES.get((s, t, f)) or \
            GLEN_N_REGIMES.get((s, t, "isotropic"))
    if r is None:
        # Relax each factor in turn (stress → thermal → fabric) to try a
        # partial-match fallback that still honours the most specific
        # known factor.  This is the v10.1.1 "regime-before-prior"
        # ordering — the classifier NEVER silently promotes to a
        # different regime family without recording it.
        if s and t:
            r = GLEN_N_REGIMES.get((s, t, "isotropic"))
        if r is None and s:
            r = GLEN_N_REGIMES.get((s, "cold", "isotropic"))
        if r is None and t:
            r = GLEN_N_REGIMES.get(("high", t, "isotropic"))
    if r is None:
        return dict(
            regime="unclassified — using PARAM_CANON plausible range",
            low=0.5, high=6.0, bench=3.0, inferred=3.0,
            source="fallback", factors=dict(stress=s, thermal=t, fabric=f))
    return dict(regime=r["desc"], low=r["low"], high=r["high"],
                bench=r["bench"], inferred=r["bench"],
                source=f"classifier({s},{t},{f})",
                factors=dict(stress=s, thermal=t, fabric=f))


def classify_rate_factor_regime(temp: Optional[str],
                                fabric: Optional[str],
                                impurity: str = "clean") -> Dict[str, Any]:
    """Two-factor classifier: (thermal, impurity) → A-warm band."""
    t = _norm_temp_regime(temp)
    f = _norm_fabric(fabric) or "isotropic"
    imp = impurity if impurity in ("clean", "dust") else "clean"
    r = ARRHENIUS_REGIMES.get((t, imp)) if t else None
    if r is None and t:
        r = ARRHENIUS_REGIMES.get((t, "clean"))
    if r is None:
        return dict(
            regime="unclassified — using PARAM_CANON plausible range",
            low=1e-8, high=1e40, bench=1.258e13, inferred=1.258e13,
            source="fallback", factors=dict(thermal=t, fabric=f, impurity=imp))
    return dict(regime=r["desc"], low=r["low"], high=r["high"],
                bench=r["bench"], inferred=r["bench"],
                source=f"classifier({t},{imp})",
                factors=dict(thermal=t, fabric=f, impurity=imp))


def classify_activation_q_regime(temp_c: Optional[float]) -> Dict[str, Any]:
    """Single-factor classifier: temperature (°C) → Q cold/warm/transitional."""
    if temp_c is None:
        r = ARRHENIUS_Q_REGIMES["transitional"]
        return dict(regime=r["desc"], low=r["low"], high=r["high"],
                    bench=r["bench"], inferred=r["bench"],
                    source="classifier(T=unknown)")
    if temp_c <= -10.0:
        key = "cold"
    elif temp_c >= -5.0:
        key = "warm"
    else:
        key = "transitional"
    r = ARRHENIUS_Q_REGIMES[key]
    return dict(regime=r["desc"], low=r["low"], high=r["high"],
                bench=r["bench"], inferred=r["bench"],
                source=f"classifier(T={temp_c:.1f})")


def classify_fabric_regime(fabric: Optional[str],
                           temp: Optional[str]) -> Dict[str, Any]:
    """Two-factor classifier for the enhancement factor E."""
    f = _norm_fabric(fabric) or "isotropic"
    t = _norm_temp_regime(temp) or "cold"
    r = FABRIC_REGIMES.get((f, t))
    if r is None:
        return dict(
            regime="unclassified — using PARAM_CANON plausible range",
            low=0.05, high=50.0, bench=1.0, inferred=1.0,
            source="fallback", factors=dict(fabric=f, thermal=t))
    return dict(regime=r["desc"], low=r["low"], high=r["high"],
                bench=r["bench"], inferred=r["bench"],
                source=f"classifier({f},{t})",
                factors=dict(fabric=f, thermal=t))


# ============================================================================
# ███ SECTION 16 — LLM PROMPT TEMPLATES                                   ███
# ============================================================================
# Three prompt families:
#   build_extraction_prompt         — Tier 1 verbatim
#   build_prior_inference_prompt    — Tier 3 parametric reasoning
#   build_regime_prior_prompt       — Tier 3b regime-bounded reasoning
#
# All three enforce: (1) SI units where possible, (2) JSON-array-only
# output, (3) confidence caps for inferred values.
# ============================================================================

def build_extraction_prompt(param_key: str, material: str,
                            records: List[Dict[str, Any]],
                            max_chars: int = 1200) -> str:
    meta = GLACIER_ONTOLOGY.get(param_key,
                                {"label": param_key, "symbol": param_key,
                                 "unit": ""})
    aliases = ", ".join(
        f'"{a}"' for a in GLACIER_ONTOLOGY.get(param_key, {}).get("aliases", [])[:8])
    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r.get('source', '?')}, "
        f"id={r.get('id', '?')})\n"
        f"{json.dumps(r.get('data', {}), ensure_ascii=False, default=str)[:max_chars]}"
        for i, r in enumerate(records))
    return f"""TASK: Extract the {meta['label']} ({meta['symbol']}, unit
{meta['unit']}) of {material} from the EVIDENCE records below.
The property may be labelled: {aliases}.

STRICT RULES
1. Report ONLY numbers that appear VERBATIM in an EVIDENCE record.
2. A number qualifies ONLY IF the SAME record contains (a) the material
   term, (b) a property label from the list, and (c) a unit compatible
   with {meta['unit']}.
3. Convert values to {meta['unit']} where possible:
   - A: MPa^-n s^-1 → ×31556926 yr^-1
   - A: Pa^-n s^-1  → ×(1e6)^n ×31556926 yr^-1
   - A: kPa^-n yr^-1→ ×(1e-3)^n
   - Q: kJ/mol      → ×1000 J/mol
   - T: K           → °C (subtract 273.15)
4. Never invent, average, or interpolate.
5. Answer with a JSON ARRAY ONLY. Schema per element:
   [{{"value": <number>, "unit": "{meta['unit']}",
      "property_label": "<short label>", "method": "explicit",
      "evidence_id": <int>, "evidence_span": "verbatim quote <=15 words",
      "confidence": <0.0-1.0>}}]
6. If nothing qualifies, answer [].

{ev}"""


def build_prior_inference_prompt(param_key: str, material: str,
                                 records: List[Dict[str, Any]],
                                 max_chars: int = 600) -> str:
    meta = GLACIER_ONTOLOGY.get(param_key,
                                {"label": param_key, "symbol": param_key,
                                 "unit": None})
    canon = GLACIER_ONTOLOGY.get(param_key, {})
    lo, hi = canon.get("valid_range", (float("-inf"), float("inf")))
    unit_str = canon.get("unit") or "(dimensionless)"
    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r.get('source', '?')})\n"
        f"{json.dumps(r.get('data', {}), ensure_ascii=False, default=str)[:max_chars]}"
        for i, r in enumerate(records)) or "(no records retrieved)"
    return f"""TASK: The evidence below does NOT contain an explicit value for
the {meta['label']} ({meta['symbol']}) of {material}. You are in PRIOR-
INFERENCE mode, which is DISTINCT from verbatim extraction:

1. Reason step by step toward a physically defensible value for {material},
   using established glaciology knowledge (Cuffey–Paterson, Glen 1955,
   Nye 1953, Budd–Jacka, Goldsby–Kohlstedt, Lliboutry).
2. Stay strictly inside [{lo:g}, {hi:g}] {unit_str}.
3. Propose 1–3 candidates, most defensible first.
4. Confidence MUST NOT exceed 0.5 for inferred candidates.
5. Output a JSON array ONLY. Schema per element:
   [{{"value": <number>, "unit": "{unit_str}",
      "property_label": "{meta['symbol']}", "method": "prior inference",
      "evidence_id": null, "evidence_span": null,
      "inference_basis": "<one-line justification>",
      "confidence": <0.0-0.5>}}]
6. Never present an inferred value as if it were measured.

EVIDENCE:
{ev}"""


def build_regime_prior_prompt(param_key: str, material: str,
                              temp_c: Optional[float],
                              stress: Optional[str],
                              fabric: Optional[str],
                              records: List[Dict[str, Any]]) -> str:
    meta = GLACIER_ONTOLOGY.get(param_key, {})
    if param_key == "glen_n":
        regime = classify_glen_n_regime(stress, temp_c, fabric)
    elif param_key in ("rate_factor_A1", "rate_factor_A2"):
        regime = classify_rate_factor_regime(temp_c, fabric)
    elif param_key in ("activation_energy_Q1", "activation_energy_Q2"):
        regime = classify_activation_q_regime(temp_c)
    elif param_key == "glen_enhancement":
        regime = classify_fabric_regime(fabric, temp_c)
    else:
        regime = dict(regime="N/A", low=0.0, high=0.0,
                      inferred=0.0, source="n/a")
    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r.get('source', '?')})\n"
        f"{json.dumps(r.get('data', {}), ensure_ascii=False, default=str)[:400]}"
        for i, r in enumerate(records[:4])) or "(no records)"
    return f"""TASK: Infer the {meta.get('label', param_key)} of {material}
inside the physically-bounded regime below. Walk a THREE-STEP chain.

INPUT CONTEXT
  • Temperature (°C):          {temp_c}
  • Stress regime:             {stress}
  • Fabric / ice architecture: {fabric}

PHYSICS PRIOR (from the three-factor classifier)
  Regime              : {regime['regime']}
  Physical range      : [{regime['low']:g}, {regime['high']:g}]
  Log-center estimate : {regime['inferred']:g}

STRICT RULES
  1. Confidence MUST NOT exceed 0.50.
  2. Propose 2–3 candidates INSIDE the regime.
  3. Never present an inferred value as if it were measured.
  4. Output JSON array ONLY:
     [{{"value": <number>, "unit": "{meta.get('unit', '')}",
        "property_label": "{meta.get('symbol', param_key)}",
        "method": "regime_inference",
        "inference_basis": "<one-line justification>",
        "reasoning": "<Step 1 … Step 3 chain, newline-separated>",
        "confidence": <0.0-0.5>}}]

EVIDENCE (context only):
{ev}"""


# ============================================================================
# ███ SECTION 17 — OLLAMA CLIENT                                          ███
# ============================================================================

class GlacierOllamaClient:
    """Thin wrapper around Ollama /api/generate returning parsed JSON.

    Retries with linear backoff; `format='json'` instructs Ollama to
    enforce JSON grammar.  Graceful fallback to None on all errors."""

    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b",
                 timeout: float = 120.0, max_retries: int = 2):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def is_available(url: str = "http://localhost:11434") -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        try:
            r = _requests.get(f"{url.rstrip('/')}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def list_models(url: str = "http://localhost:11434") -> List[str]:
        if not REQUESTS_AVAILABLE:
            return []
        try:
            r = _requests.get(f"{url.rstrip('/')}/api/tags", timeout=3.0)
            if r.status_code == 200:
                return sorted(m.get("name", "") for m in r.json().get("models", []))
        except Exception:
            pass
        return []

    def generate_json(self, prompt: str,
                      system: Optional[str] = None,
                      debug: bool = False) -> Optional[Any]:
        if not REQUESTS_AVAILABLE:
            return None
        payload: Dict[str, Any] = {
            "model": self.model, "prompt": prompt, "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 4096},
        }
        if system:
            payload["system"] = system
        for attempt in range(self.max_retries + 1):
            try:
                r = _requests.post(f"{self.url}/api/generate",
                                   json=payload, timeout=self.timeout)
                r.raise_for_status()
                raw = r.json().get("response", "")
                parsed = self._parse_lenient(raw)
                if debug:
                    logger.info("Ollama raw (first 800 chars): %s",
                                str(raw)[:800])
                return parsed
            except Exception as e:
                logger.warning("Ollama attempt %d failed: %s",
                               attempt + 1, e)
                if attempt == self.max_retries:
                    return None
                time.sleep(0.75 * (attempt + 1))
        return None

    @staticmethod
    def _parse_lenient(raw: str) -> Any:
        if raw is None:
            return None
        s = raw.strip()
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()
        for oc, cc in (("[", "]"), ("{", "}")):
            i, j = s.find(oc), s.rfind(cc)
            if i != -1 and j > i:
                try:
                    return json.loads(s[i: j + 1])
                except Exception:
                    continue
        try:
            return json.loads(s)
        except Exception:
            return None


def parse_llm_json(raw: str) -> list:
    """Robustly parse an LLM response into a list of dicts."""
    txt = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.S)
    txt = txt.replace("```json", "").replace("```", "")
    starts = [i for i in (txt.find("["), txt.find("{")) if i >= 0]
    if not starts:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(txt[min(starts):])
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        logger.warning("parse_llm_json: unparseable LLM output")
        return []


def ollama_extract(prompt: str, model: str = "qwen2.5:7b",
                   host: str = "http://localhost:11434",
                   timeout: int = 120) -> str:
    """Single-shot Ollama generate returning the raw response string."""
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests not installed")
    r = _requests.post(f"{host}/api/generate",
                       json={"model": model, "prompt": prompt,
                             "stream": False,
                             "options": {"temperature": 0}},
                       timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


# ============================================================================
# ███ SECTION 18 — CORPUS + HYBRID RETRIEVER                             ███
# ============================================================================

def load_corpus_folder(folder: str) -> Dict[str, Any]:
    """Load every *.json file in `folder` into a dict keyed by filename."""
    corpus: Dict[str, Any] = {}
    if not os.path.isdir(folder):
        return corpus
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".json"):
            try:
                with open(os.path.join(folder, fn),
                          encoding="utf-8-sig") as f:
                    corpus[fn] = json.load(f)
            except Exception as e:
                logger.warning("cannot load %s: %s", fn, e)
    return corpus


def iter_corpus_records(corpus: Any) -> List[Dict[str, Any]]:
    """Flatten any corpus dict/list into a list of records with an id."""
    records: List[Dict[str, Any]] = []
    if isinstance(corpus, dict):
        for fname, data in corpus.items():
            rows = data if isinstance(data, list) else [data]
            records += [{"id": f"{fname}[{i}]", "source": fname, "data": r}
                        for i, r in enumerate(rows)]
    else:
        records = [{"id": f"rec[{i}]", "source": "corpus", "data": r}
                   for i, r in enumerate(corpus)]
    return records


def _spans(text: str,
           patterns: List[re.Pattern]) -> List[Tuple[int, str]]:
    return [(m.start(), m.group(0))
            for p in patterns for m in p.finditer(text)]


def score_record(text: str, param_key: str, material: str = "ice",
                 value_hints: Optional[List[str]] = None
                 ) -> Tuple[float, List[str]]:
    """Lexical relevance score for a candidate record."""
    score, why = 0.0, []
    mat = _spans(text, ENTITY_PATTERNS["material"])
    props = _spans(text, ENTITY_PATTERNS["property"])
    units = _spans(text, ENTITY_PATTERNS["unit"])
    if mat:
        score += 2.0
        why.append("material")
    if props:
        score += 3.0 + 0.5 * min(len(props), 3)
        why.append("property")
    elif _spans(text, ENTITY_PATTERNS["property_weak"]) and (units or mat):
        score += 0.5
        why.append("property(weak)")

    for etype, w in (("temperature_regime", .7),
                     ("stress_regime",      .6),
                     ("ice_architecture",   .6),
                     ("method",             .8),
                     ("theory_framework",   .9),
                     ("grain_architecture", .5),
                     ("synthesis_method",   .5)):
        if _spans(text, ENTITY_PATTERNS[etype]):
            score += w
            why.append(etype)

    for v in (value_hints or []):
        m = re.search(rf"(?<![\d.]){re.escape(str(v))}(?!\d)", text)
        if m:
            near = (any(abs(m.start() - s) < 80 for s, _ in props) or
                    any(abs(m.start() - s) < 15 for s, _ in units))
            score += 4.0 if near else 1.0
            why.append(f"value:{v}{'*' if near else ''}")
    return score, why


def rrf(rank_lists: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion across multiple rank lists."""
    agg: Dict[str, float] = {}
    for lst in rank_lists:
        for r, rid in enumerate(lst):
            agg[rid] = agg.get(rid, 0.0) + 1.0 / (k + r + 1)
    return [rid for rid, _ in sorted(agg.items(), key=lambda kv: -kv[1])]


def build_query_texts(material: str, param_key: str,
                      value_hints: Optional[List[str]] = None
                      ) -> List[str]:
    """Build dense-retrieval query texts for a target parameter."""
    props = GLACIER_ONTOLOGY.get(param_key, {}).get("aliases", [param_key])[:3]
    texts = [f"{material} {p} glen flow law" for p in props]
    texts += [f"{material} {p} cuffey paterson" for p in props]
    texts += [f"{material} glacier creep {v}"
              for v in (value_hints or [])[:4]]
    return texts


class HybridRetriever:
    """Lexical + dense (FAISS/SBERT) retriever with RRF fusion.

    If FAISS or SBERT are unavailable, degrades gracefully to
    lexical-only mode without raising.  Score records can be filtered
    by `material`, `value_hints`, and `param_key`.
    """

    def __init__(self, corpus: Any, use_dense: bool = True):
        self.records = iter_corpus_records(corpus)
        for r in self.records:
            r["text_norm"] = norm_text(record_text(r["data"]))
        self.ids = [r["id"] for r in self.records]
        self._txt = {r["id"]: r["text_norm"] for r in self.records}
        self.dense = False
        self.model = None
        self.index = None
        if use_dense and FAISS_AVAILABLE and SBERT_AVAILABLE and self.records:
            try:
                self.model = _SentenceTransformer(
                    "all-MiniLM-L6-v2", device="cpu")
                emb = self.model.encode(
                    list(self._txt.values()),
                    normalize_embeddings=True,
                    show_progress_bar=False).astype(np.float32)
                self.index = _faiss.IndexFlatIP(emb.shape[1])
                self.index.add(np.ascontiguousarray(emb, dtype="float32"))
                self.dense = True
            except Exception as e:
                logger.warning("HybridRetriever dense channel disabled: %s", e)

    def search(self, param_key: str, material: str = "ice",
               value_hints: Optional[List[str]] = None,
               k: int = 6) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        lex = []
        for rid in self.ids:
            s, _ = score_record(self._txt[rid], param_key, material,
                                value_hints)
            if s > 0:
                lex.append((-s, rid))
        rank_lists = [[rid for _, rid in sorted(lex)]]
        if self.dense and self.model is not None and self.index is not None:
            for q in build_query_texts(material, param_key, value_hints)[:4]:
                qv = self.model.encode([q], normalize_embeddings=True
                                       ).astype(np.float32)
                _, I = self.index.search(
                    np.ascontiguousarray(qv, "float32"),
                    min(10, len(self.ids)))
                rank_lists.append([self.ids[i] for i in I[0]])
        id2rec = {r["id"]: r for r in self.records}
        return [id2rec[rid] for rid in rrf(rank_lists) if rid in id2rec][:k]


@st.cache_resource(show_spinner=False)
def get_retriever(folder: str = "json_metadatabase",
                  use_dense: bool = True) -> HybridRetriever:
    """Streamlit-cached hybrid retriever factory."""
    corpus = load_corpus_folder(folder)
    return HybridRetriever(corpus, use_dense=use_dense)


# ============================================================================
# ███ SECTION 19 — CANDIDATE DATACLASSES                                 ███
# ============================================================================

@dataclass
class ValueCandidate:
    """Raw tier output before MoE scoring."""
    value: float
    unit: str
    provenance: str
    property_label: str = ""
    method: str = ""
    evidence: str = ""
    source: str = ""
    confidence: float = 1.0
    reasoning: str = ""
    regime_tag: str = ""
    context: bool = False

    def to_display(self) -> Dict[str, Any]:
        d = {
            "value": round(self.value, 6),
            "unit": self.unit,
            "provenance": self.provenance,
            "property_label": self.property_label[:80],
            "method": self.method,
            "source": self.source[:80],
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:140],
            "reasoning": (self.reasoning[:200] + "…"
                          if len(self.reasoning) > 200
                          else self.reasoning),
        }
        if self.regime_tag:
            d["regime"] = self.regime_tag
        if self.context:
            d["context"] = True
        return d


@dataclass
class GlacierCandidate:
    """MoE-scored candidate ready for display and SIF generation."""
    param: str
    value_si: float
    raw_value: float
    raw_unit: str
    score: float
    confidence: float
    material: str
    temp_k: Optional[float]
    method: str
    source_file: str = ""
    source_title: str = ""
    evidence: str = ""
    reasoning: str = ""
    clamped: bool = False
    regime_tag: str = ""
    provenance: str = ""
    context: bool = False
    moe_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_display(self) -> Dict[str, Any]:
        spec = GLACIER_ONTOLOGY[self.param]
        d = {
            "value": self.value_si / spec["ui_scale"],
            "unit": spec["ui_unit"],
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "material": self.material or "n/a",
            "temp_K": self.temp_k,
            "method": self.method,
            "source": f"{self.source_file} — {self.source_title[:60]}",
            "clamped": self.clamped,
            "evidence": self.evidence[:140],
            "reasoning": (self.reasoning[:200] + "…"
                          if len(self.reasoning) > 200
                          else self.reasoning),
        }
        if self.regime_tag:
            d["regime"] = self.regime_tag
        if self.provenance:
            d["provenance"] = self.provenance
        if self.context:
            d["context"] = True
        if self.moe_breakdown:
            d["moe_breakdown"] = {k: round(v, 4)
                                  for k, v in self.moe_breakdown.items()}
        return d


# ============================================================================
# ███ SECTION 20 — GATEKEEPER                                            ███
# ============================================================================
# Gatekeep coerces units, rejects out-of-range and non-positive values,
# dedupes, applies confidence caps, and passes through `_context`.
# ============================================================================

def gatekeep(items: List[Dict[str, Any]], param_key: str,
             provenance: str = "regex_ner",
             conf_cap: Optional[float] = None,
             glen_n_for_conversion: float = 3.0
             ) -> List[ValueCandidate]:
    canon = GLACIER_ONTOLOGY.get(param_key, {})
    (lo, hi) = canon.get("valid_range", (float("-inf"), float("inf")))

    if conf_cap is None:
        if provenance in ("llm_prior", "regime_prior"):
            conf_cap = 0.5
        elif provenance == "physics_inferred":
            conf_cap = 0.75
        else:
            conf_cap = 1.0

    reject_non_positive = param_key in _POSITIVE_LOWER_BOUND_PARAMS
    out, seen = [], set()
    for it in items:
        try:
            raw = it.get("value")
            if isinstance(raw, str):
                v = float(re.sub(r"[^0-9eE+\-\.]", "", raw))
            else:
                v = float(raw)
        except (TypeError, ValueError):
            continue
        u = str(it.get("unit") or canon.get("unit") or "").lower()
        try:
            v = normalize_glacier_unit(v, u, param_key,
                                        n=glen_n_for_conversion)
        except Exception:
            pass

        if reject_non_positive and v <= 0.0:
            logger.info("gatekeep[%s]: rejected non-positive %s",
                        param_key, v)
            continue
        if not (lo <= v <= hi):
            logger.info("gatekeep[%s]: rejected %s (out of %s..%s)",
                        param_key, v, lo, hi)
            continue
        key = round(v, 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(ValueCandidate(
            value=v,
            unit=canon.get("unit") or "",
            provenance=provenance,
            property_label=str(it.get("property_label", "")),
            method=str(it.get("method", "")),
            evidence=str(it.get("evidence_span") or it.get("evidence")
                         or it.get("inference_basis") or ""),
            source=str(it.get("source", "")),
            confidence=min(float(it.get("confidence", 1.0) or 1.0),
                           conf_cap),
            reasoning=str(it.get("reasoning") or ""),
            regime_tag=str(it.get("regime") or it.get("dominant_factor")
                           or ""),
            context=bool(it.get("_context", False)),
        ))
    return sorted(out, key=lambda c: -c.confidence)


# ============================================================================
# ███ SECTION 21 — REGIME EXPERTS (Latent MoE internals)                 ███
# ============================================================================

class GlenNRegimeExpert:
    """Score a Glen-n candidate by proximity to its regime band center."""

    def __init__(self, falloff_decades: float = 1.0):
        self.falloff_decades = float(falloff_decades)

    def score(self, value_si: float,
              stress: Optional[str],
              temp: Optional[str],
              fabric: Optional[str]) -> float:
        try:
            v = float(value_si)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0
        regime = classify_glen_n_regime(stress, temp, fabric)
        if regime["low"] <= v <= regime["high"]:
            center = regime["inferred"]
            span = max(regime["high"] - regime["low"], 1e-6)
            sigma = max(span / 4.0, 0.25)
            return float(np.exp(-((v - center) ** 2) / (2 * sigma ** 2)))
        dist = min(abs(v - regime["low"]), abs(v - regime["high"]))
        return float(np.exp(-dist / self.falloff_decades))


class RateFactorRegimeExpert:
    """Log-Gaussian scorer for A1/A2 candidates."""

    def __init__(self, falloff_decades: float = 2.0):
        self.falloff_decades = float(falloff_decades)

    def score(self, value_si: float,
              temp: Optional[str],
              fabric: Optional[str]) -> float:
        try:
            v = float(value_si)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0
        regime = classify_rate_factor_regime(temp, fabric)
        log_v = np.log10(v)
        log_lo = np.log10(max(regime["low"], 1e-30))
        log_hi = np.log10(max(regime["high"], 1e-30))
        log_ctr = np.log10(max(regime["inferred"], 1e-30))
        if log_lo <= log_v <= log_hi:
            sigma = max((log_hi - log_lo) / 4.0, 0.25)
            return float(np.exp(-(log_v - log_ctr) ** 2
                                / (2 * sigma ** 2)))
        dist = min(abs(log_v - log_lo), abs(log_v - log_hi))
        return float(np.exp(-dist / self.falloff_decades))


class ActivationQRegimeExpert:
    """Gaussian scorer for Q cold/warm/transitional bands."""

    def __init__(self, falloff_frac: float = 0.25):
        self.falloff_frac = float(falloff_frac)

    def score(self, value_si: float, temp_c: Optional[float]) -> float:
        try:
            v = float(value_si)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0:
            return 0.0
        regime = classify_activation_q_regime(temp_c)
        lo, hi = regime["low"], regime["high"]
        span = max(hi - lo, 1.0)
        if lo <= v <= hi:
            center = regime["inferred"]
            sigma = max(span / 4.0, 1.0)
            return float(np.exp(-((v - center) ** 2) / (2 * sigma ** 2)))
        dist = min(abs(v - lo), abs(v - hi)) / span
        return float(np.exp(-dist / self.falloff_frac))


class FabricRegimeExpert:
    """Regime scorer for the enhancement factor E."""

    def __init__(self, falloff_decades: float = 1.0):
        self.falloff_decades = float(falloff_decades)

    def score(self, value_si: float,
              fabric: Optional[str],
              temp: Optional[str]) -> float:
        try:
            v = float(value_si)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0
        regime = classify_fabric_regime(fabric, temp)
        if regime["low"] <= v <= regime["high"]:
            center = regime["inferred"]
            span = max(regime["high"] - regime["low"], 1e-6)
            sigma = max(span / 4.0, 0.25)
            return float(np.exp(-((v - center) ** 2) / (2 * sigma ** 2)))
        dist = min(abs(v - regime["low"]), abs(v - regime["high"]))
        return float(np.exp(-dist / self.falloff_decades))


# ============================================================================
# ███ SECTION 22 — 10-EXPERT LATENT MoE SCORER                           ███
# ============================================================================
# Ten experts, linear weights, sum to 1.00:
#
#    Material         0.15
#    Thermal          0.10
#    Strain           0.05
#    Method           0.10
#    Confidence       0.05
#    Reasoning        0.10
#    Regime (A)       0.15    ← Glen-n / A / Q / E regime experts
#    Arrhenius (B)    0.15    ← Arrhenius coupling between T and A/Q
#    Fabric (C)       0.10    ← anisotropy vs enhancement consistency
#    Corpus Density   0.05    ← provenance-binned evidence density
#   ----------------------
#    Sum              1.00
#
# Because the aggregation is deterministic and linear, the per-expert
# breakdown IS the exact feature attribution — no SHAP/LIME needed.
# ============================================================================

class GlacierLatentMoEScorer:
    """Physics-informed Mixture-of-Experts ranker for glacier parameters."""

    EXPERT_ORDER: Tuple[str, ...] = (
        'Material', 'Thermal', 'Strain', 'Method', 'Confidence',
        'Reasoning', 'Regime (A)', 'Arrhenius (B)',
        'Fabric (C)', 'Corpus Density',
    )
    EXPERT_COLORS: Dict[str, str] = {
        'Material':       '#E69F00',
        'Thermal':        '#56B4E9',
        'Strain':         '#009E73',
        'Method':         '#CC79A7',
        'Confidence':     '#999999',
        'Reasoning':      '#F0E442',
        'Regime (A)':     '#0072B2',
        'Arrhenius (B)':  '#D55E00',
        'Fabric (C)':     '#000000',
        'Corpus Density': '#88CCEE',
    }

    def __init__(self, thermal_sigma: float = 15.0):
        self.w_material = 0.15
        self.w_thermal = 0.10
        self.w_strain = 0.05
        self.w_method = 0.10
        self.w_confidence = 0.05
        self.w_reasoning = 0.10
        self.w_regime = 0.15
        self.w_arrhenius = 0.15
        self.w_fabric = 0.10
        self.w_density = 0.05
        self.thermal_sigma = thermal_sigma

        self.glen_expert = GlenNRegimeExpert()
        self.rate_expert = RateFactorRegimeExpert()
        self.q_expert = ActivationQRegimeExpert()
        self.fabric_expert = FabricRegimeExpert()

    # ── Individual experts ───────────────────────────────────────────
    def _material_expert(self, ext_mat: str, target_mat: str) -> float:
        if not ext_mat:
            return 0.6
        a, b = ext_mat.lower().strip(), target_mat.lower().strip()
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.85
        ice_terms = ("ice", "glacier", "greenland", "antarctica",
                     "himalaya", "polar", "temperate", "firn")
        if any(k in a for k in ice_terms) and any(k in b for k in ice_terms):
            return 0.7
        return 0.3

    def _thermal_expert(self, ext_temp_c: Optional[float],
                        target_temp_c: Optional[float]) -> float:
        if ext_temp_c is None or target_temp_c is None:
            return 0.5
        try:
            t = float(ext_temp_c)
            diff = t - float(target_temp_c)
        except (TypeError, ValueError):
            return 0.5
        return float(np.exp(-(diff ** 2) / (2 * self.thermal_sigma ** 2)))

    def _strain_expert(self, stress_regime: Optional[str],
                       method: str) -> float:
        m = (method or "").lower()
        s = _norm_stress_regime(stress_regime)
        base = 0.5
        if "creep test" in m or "lab" in m or "laboratory" in m:
            base = 0.9
        elif "field" in m or "borehole" in m:
            base = 0.8
        elif "inverse" in m or "adjoint" in m or "assimilation" in m:
            base = 0.7
        elif "md" in m or "atomistic" in m:
            base = 0.6
        if s == "high" and ("creep" in m or "shear" in m or "field" in m):
            base = min(base + 0.1, 1.0)
        if s == "low" and ("torsion" in m or "uniaxial" in m):
            base = min(base + 0.1, 1.0)
        return float(base)

    @staticmethod
    def _method_expert(method: str) -> float:
        return {
            "explicit": 0.9,
            "heuristic": 0.35,
            "prior inference": 0.35,
            "llm_prior": 0.35,
            "default_fallback": 0.30,
            "regime_inference": 0.55,
            "physics_regime_inference": 0.60,
            "theory_regime_inference": 0.60,
            "glen_law_inversion": 0.60,
            "arrhenius_inversion": 0.60,
            "activation_inversion": 0.55,
            "enhancement_inversion": 0.55,
            "side_note_table": 0.55,
            "derived_consensus": 0.60,
            "sigma0_route_glen_inversion": 0.55,
            "sigma0_route_arrhenius": 0.60,
            "creep_test": 1.00,
            "field_measurement": 0.90,
            "ice_core": 0.85,
            "inverse_model": 0.80,
            "molecular_dynamics": 0.60,
            "dft": 0.55,
        }.get((method or "").lower(), 0.5)

    @staticmethod
    def _reasoning_expert(reasoning: str, method: str) -> float:
        method_l = (method or "").lower()
        cog_methods = (
            "prior inference", "llm_prior", "heuristic",
            "default_fallback", "regime_inference",
            "physics_regime_inference", "theory_regime_inference",
            "glen_law_inversion", "arrhenius_inversion",
            "side_note_table", "derived_consensus",
            "sigma0_route_glen_inversion", "sigma0_route_arrhenius",
        )
        if method_l not in cog_methods:
            return 0.5
        if not reasoning:
            return 0.2
        n_steps = reasoning.count("Step ") + reasoning.count("\n")
        cites = any(k in reasoning for k in [
            "Glen", "Arrhenius", "Cuffey", "Paterson", "Nye",
            "Goldsby", "Budd-Jacka", "Lliboutry",
            "MPa", "Pa", "J/mol", "s^-1", "yr^-1",
            "activation", "rate factor", "exponent", "n =",
            "temperature", "fabric", "regime", "enhancement",
        ])
        base = min(0.4 + 0.05 * n_steps, 0.8)
        if cites:
            base = min(base + 0.15, 0.85)
        return float(base)

    def _regime_expert(self, p: str, v_si: float,
                       temp_c: Optional[float],
                       stress: Optional[str],
                       fabric: Optional[str],
                       regime_active: bool) -> float:
        """Thread A — regime classifier."""
        if not regime_active:
            return 0.5
        if p == "glen_n":
            return self.glen_expert.score(v_si, stress, temp_c, fabric)
        if p in ("rate_factor_A1", "rate_factor_A2"):
            t = None
            if temp_c is not None:
                if temp_c <= -10.0:
                    t = "cold"
                elif temp_c >= -1.0:
                    t = "temperate"
                else:
                    t = "polythermal"
            return self.rate_expert.score(v_si, t, fabric)
        if p in ("activation_energy_Q1", "activation_energy_Q2"):
            return self.q_expert.score(v_si, temp_c)
        if p == "glen_enhancement":
            t = None
            if temp_c is not None:
                if temp_c <= -10.0:
                    t = "cold"
                elif temp_c >= -1.0:
                    t = "temperate"
                else:
                    t = "polythermal"
            return self.fabric_expert.score(v_si, fabric, t)
        return 0.5

    @staticmethod
    def _arrhenius_expert(p: str, v_si: float,
                          temp_c: Optional[float],
                          ext: Dict[str, Any]) -> float:
        """Thread B — Arrhenius consistency between T and A or Q.

        A candidate is rewarded when its temperature and its value fall
        on the same side of the -10 °C transition (canonical T*).
        """
        if p not in ("rate_factor_A1", "rate_factor_A2",
                     "activation_energy_Q1", "activation_energy_Q2"):
            return 0.5
        if temp_c is None:
            return 0.5
        cold = temp_c <= -10.0
        warm = temp_c >= -5.0
        if p == "rate_factor_A1":
            # A1 is the cold value; punish when the ice is clearly warm
            return 0.95 if cold else (0.35 if warm else 0.60)
        if p == "rate_factor_A2":
            return 0.95 if warm else (0.35 if cold else 0.60)
        if p == "activation_energy_Q1":
            return 0.95 if cold else (0.35 if warm else 0.60)
        if p == "activation_energy_Q2":
            return 0.95 if warm else (0.35 if cold else 0.60)
        return 0.5

    @staticmethod
    def _fabric_expert(p: str, v_si: float,
                       fabric: Optional[str]) -> float:
        """Thread C — fabric / enhancement / shear-rate coupling."""
        if p == "glen_enhancement":
            f = _norm_fabric(fabric)
            if f == "isotropic":
                return 1.0 if 0.5 <= v_si <= 1.5 else 0.5
            if f == "anisotropic":
                return 1.0 if 2.0 <= v_si <= 8.0 else 0.5
            return 0.6
        if p == "critical_shear_rate":
            return 0.7
        if p == "ice_density":
            return 0.9 if 800.0 <= v_si <= 950.0 else 0.4
        if p == "gravity":
            return 1.0 if 9.5 <= v_si <= 10.0 else 0.5
        return 0.5

    @staticmethod
    def _density_expert(prov_fine: str) -> float:
        """Thread D — corpus evidence density (provenance-binned)."""
        if prov_fine in ("llm_extract", "llm_reasoned", "regex_ner"):
            return 1.0
        if prov_fine == "physics_inferred":
            return 0.7
        if prov_fine == "regime_prior":
            return 0.4
        return 0.2

    # ── Aggregate ───────────────────────────────────────────────────
    def score(self, extractions: List[Dict[str, Any]],
              target_material: str = "ice",
              target_temp_c: Optional[float] = None,
              stress_regime: Optional[str] = None,
              fabric: Optional[str] = None,
              target_strain_rate: float = 1e-3,
              top_k: int = 8
              ) -> Dict[str, List[GlacierCandidate]]:
        regime_active = (target_temp_c is not None
                         or bool(stress_regime)
                         or bool(fabric))

        buckets: Dict[str, List[GlacierCandidate]] = {
            p: [] for p in ALL_PARAMS
        }

        for ext in extractions:
            p = ext.get("param")
            if p not in buckets:
                continue
            try:
                v = float(ext["value"])
            except (TypeError, ValueError):
                continue
            # Use the current n for A conversion if available; else 3.0
            n_for_conv = 3.0
            if p in ("rate_factor_A1", "rate_factor_A2"):
                for cand in extractions:
                    if cand.get("param") == "glen_n":
                        try:
                            n_for_conv = float(cand["value"])
                            break
                        except (TypeError, ValueError):
                            pass
            try:
                v_si = normalize_glacier_unit(v, ext.get("unit", ""), p,
                                               n=n_for_conv)
            except Exception:
                continue
            if v_si is None or not np.isfinite(v_si):
                continue
            v_si_clamped, was_clamped = _glacier_clamp(v_si, p)

            # Normalise ext temp to °C if it looks like K
            ext_temp = ext.get("temp")
            if ext_temp is not None:
                try:
                    ext_temp = (float(ext_temp) - T0_CELSIUS
                                if float(ext_temp) > 100
                                else float(ext_temp))
                except (TypeError, ValueError):
                    ext_temp = None

            s_mat = self._material_expert(ext.get("material", ""),
                                          target_material)
            s_temp = self._thermal_expert(ext_temp, target_temp_c)
            s_str = self._strain_expert(stress_regime,
                                        ext.get("method", ""))
            s_meth = self._method_expert(ext.get("method", "unknown"))
            s_conf = float(ext.get("confidence", 0.5) or 0.5)
            s_reas = self._reasoning_expert(
                ext.get("reasoning", ""), ext.get("method", ""))
            s_reg = self._regime_expert(p, v_si_clamped, target_temp_c,
                                        stress_regime, fabric, regime_active)
            s_arr = self._arrhenius_expert(p, v_si_clamped, target_temp_c, ext)
            s_fab = self._fabric_expert(p, v_si_clamped, fabric)
            prov_fine = _norm_provenance(
                ext.get("_provenance", "") or ext.get("method", ""))
            s_den = self._density_expert(prov_fine)

            breakdown = {
                "Material":       self.w_material   * s_mat,
                "Thermal":        self.w_thermal    * s_temp,
                "Strain":         self.w_strain     * s_str,
                "Method":         self.w_method     * s_meth,
                "Confidence":     self.w_confidence * s_conf,
                "Reasoning":      self.w_reasoning  * s_reas,
                "Regime (A)":     self.w_regime     * s_reg,
                "Arrhenius (B)":  self.w_arrhenius  * s_arr,
                "Fabric (C)":     self.w_fabric     * s_fab,
                "Corpus Density": self.w_density    * s_den,
            }
            score = float(sum(breakdown.values()))

            cand = GlacierCandidate(
                param=p,
                value_si=v_si_clamped,
                raw_value=float(ext["value"]),
                raw_unit=str(ext.get("unit", "")),
                score=score,
                confidence=s_conf,
                material=str(ext.get("material", "")),
                temp_k=ext.get("temp"),
                method=str(ext.get("method", "unknown")),
                source_file=str(ext.get("_source_file", "")),
                source_title=str(ext.get("_source_title", "")),
                evidence=str(ext.get("evidence", "")),
                reasoning=str(ext.get("reasoning", "")),
                clamped=was_clamped,
                regime_tag=str(ext.get("regime") or ""),
                provenance=str(ext.get("_provenance", "")),
                context=bool(ext.get("_context", False)),
                moe_breakdown=breakdown,
            )
            buckets[p].append(cand)

        for p in buckets:
            buckets[p].sort(key=lambda c: c.score, reverse=True)
            buckets[p] = buckets[p][:top_k]
        return buckets


# ============================================================================
# ███ SECTION 23 — THREE-TIER CASCADE                                    ███
# ============================================================================

def recommend_param_values(
    param_key: str,
    material: str = "ice",
    temp_c: Optional[float] = None,
    stress_regime: Optional[str] = None,
    fabric: Optional[str] = None,
    k: int = 6,
    use_llm: bool = True,
    ollama_model: str = "qwen2.5:7b",
    retriever: Optional[HybridRetriever] = None,
    allow_prior_inference: bool = True,
    cascade_mode: str = 'union',
    glen_n_for_conversion: float = 3.0,
) -> Tuple[List[ValueCandidate], List[Dict[str, Any]], Dict[str, Any]]:
    """Generic three-tier cascade for a glacier parameter.

    Tier 1  grounded LLM extraction (verbatim, conf ≤ 1.0)
    Tier 2  deterministic regex NER
    Tier 3  LLM prior inference (conf ≤ 0.5)
    Tier 3b regime classifier (deterministic, conf 0.35)

    cascade_mode='union'    all viable tiers run; non-incumbent routes
                            are tagged `_context=True`.
    cascade_mode='fallback' short-circuit; first successful tier wins.
    """
    if retriever is None:
        retriever = get_retriever()

    diag: Dict[str, Any] = dict(
        param=param_key, records=0, n_llm=0, n_heuristic=0, n_prior=0,
        tier="none", reason="", cascade_mode=cascade_mode)

    records = retriever.search(param_key, material, k=k)
    if not records:
        records = retriever.search(param_key, "ice", k=2 * k)
        diag["reason"] += "relaxed retrieval (material term dropped); "
    diag["records"] = len(records)

    cands: List[ValueCandidate] = []

    # ── Tier 1 ──────────────────────────────────────────────────────
    t1 = len(cands)
    if use_llm and records:
        try:
            prompt = build_extraction_prompt(param_key, material, records)
            raw = ollama_extract(prompt, model=ollama_model)
            items = parse_llm_json(raw)
            for it in items:
                if isinstance(it, dict):
                    _stamp_provenance(it, 'llm_extract')
                    it['_context'] = _is_context(param_key, 'llm_extract')
            cands.extend(gatekeep(items, param_key,
                                  provenance="llm_extract",
                                  glen_n_for_conversion=glen_n_for_conversion))
        except Exception as e:
            diag["reason"] += f"llm failed: {e}; "
            logger.warning("Tier-1 LLM failed (%s): %s", param_key, e)
    diag["n_llm"] = len(cands) - t1

    # ── Tier 2 ──────────────────────────────────────────────────────
    t2 = len(cands)
    if records and (cascade_mode == 'union' or not cands):
        hits = heuristic_extract(records, param_key)
        for h in hits:
            if isinstance(h, dict):
                _stamp_provenance(h, 'regex_ner')
                h['_context'] = _is_context(param_key, 'regex_ner')
        cands.extend(gatekeep(hits, param_key, provenance="regex_ner",
                              glen_n_for_conversion=glen_n_for_conversion))
    diag["n_heuristic"] = len(cands) - t2

    # ── Tier 3 ──────────────────────────────────────────────────────
    t3 = len(cands)
    if use_llm and allow_prior_inference and \
            (cascade_mode == 'union' or not cands):
        try:
            prompt = build_prior_inference_prompt(param_key, material, records)
            raw = ollama_extract(prompt, model=ollama_model)
            items = parse_llm_json(raw)
            for it in items:
                if isinstance(it, dict):
                    _stamp_provenance(it, 'llm_prior')
                    it['_context'] = _is_context(param_key, 'llm_prior')
            cands.extend(gatekeep(items, param_key,
                                  provenance="llm_prior",
                                  conf_cap=0.5,
                                  glen_n_for_conversion=glen_n_for_conversion))
        except Exception as e:
            diag["reason"] += f"prior failed: {e}; "
            logger.warning("Tier-3 prior failed (%s): %s", param_key, e)
    diag["n_prior"] = len(cands) - t3

    # ── Tier 3b: regime classifier ──────────────────────────────────
    if cascade_mode == 'union' or not cands:
        try:
            if param_key == "glen_n":
                regime = classify_glen_n_regime(stress_regime, temp_c, fabric)
            elif param_key in ("rate_factor_A1", "rate_factor_A2"):
                regime = classify_rate_factor_regime(temp_c, fabric)
            elif param_key in ("activation_energy_Q1", "activation_energy_Q2"):
                regime = classify_activation_q_regime(temp_c)
            elif param_key == "glen_enhancement":
                regime = classify_fabric_regime(fabric, temp_c)
            else:
                regime = None
            if regime and regime["inferred"] > 0:
                rc = ValueCandidate(
                    value=regime["inferred"],
                    unit=GLACIER_ONTOLOGY[param_key]["unit"],
                    provenance="regime_prior",
                    property_label=f"{param_key} (regime classifier)",
                    method="regime_inference",
                    evidence=f"Regime: {regime['regime']}",
                    source=regime["source"],
                    confidence=0.35,
                    reasoning=(f"Step 1: classifier says {regime['regime']}.\n"
                               f"Step 2: range [{regime['low']:g}, "
                               f"{regime['high']:g}].\n"
                               f"Step 3: log-center "
                               f"{regime['inferred']:g}."),
                    regime_tag=regime["regime"],
                    context=_is_context(param_key, "regime_prior"),
                )
                if not any(abs(c.value - rc.value) /
                           max(abs(rc.value), 1e-30) < 1e-3 for c in cands):
                    cands.append(rc)
        except Exception as e:
            logger.warning("Tier-3b classifier failed (%s): %s",
                           param_key, e)

    diag["tier"] = ("llm" if diag["n_llm"]
                    else "heuristic" if diag["n_heuristic"]
                    else "llm_prior" if diag["n_prior"]
                    else "none")
    return cands, records, diag


# ============================================================================
# ███ SECTION 24 — SIF GENERATION                                        ███
# ============================================================================

def generate_glacier_sif(p: Dict[str, Any]) -> str:
    """Generate a complete Elmer SIF for a 3D glacier Stokes solve.

    Uses pre-deformed mesh.nodes verbatim (no `Surface` / `include`
    directive), so Elmer consumes the mesh as-is.

    CRITICAL: `Density` and `Flow BodyForce 3` are Elmer expressions
    beginning with `$`, so the leading `$` is INTENTIONALLY preserved in
    the template (this is Elmer syntax — see ElmerSolver docs §"Real
    valued parameters with expressions").
    """
    return f"""! ──────────────────────────────────────────────────────────────────
! Auto-generated by Glacier Intelligent ElmerSolver Recommender
! Generated at: {datetime.now().isoformat()}
! ──────────────────────────────────────────────────────────────────
Header
  !CHECK KEYWORDS Warn
  Mesh DB "." "{p['mesh_db']}"
  Include Path ""
  Results Directory ""
End

Simulation
  Max Output Level = 4
  Coordinate System = "Cartesian 3D"
  Coordinate Mapping(3) = 1 2 3
  Simulation Type = "Steady"
  Steady State Max Iterations = {p['steady_state_max_iter']}
  Output Intervals = {p['output_intervals']}
  Output File = "{p['output_file']}"
  Post File = "{p['post_file']}"
  Initialize Dirichlet Conditions = Logical False
End

Constants
  Stefan Boltzmann = 5.67e-08
End

Body 1
  Name = "Glacier"
  Body Force = 1
  Equation = 1
  Material = 1
  Initial Condition = 1
End

Equation 1
  Name = "Equation1"
  Convection = "computed"
  Flow Solution Name = String "Flow Solution"
  Active Solvers(1) = 2
End

Initial Condition 1
  Velocity 1 = 0.0
  Velocity 2 = 0.0
  Velocity 3 = 0.0
  Pressure = 0.0
  Depth = Real 0.0
End

! ──────────────────────────────────────────────────────────────────
! Solver 1: HeightDepth for postprocessing — Exec Solver = Never
! ──────────────────────────────────────────────────────────────────
Solver 1
  Exec Solver = "Never"
  Equation = "HeightDepth"
  Procedure = "StructuredProjectToPlane" "StructuredProjectToPlane"
  Active Coordinate = Integer 3
  Operator 1 = depth
  Operator 2 = height
  Dot Product Tolerance = Real 0.9
End

! ──────────────────────────────────────────────────────────────────
! Solver 2: Stokes (Navier-Stokes flow model)
! ──────────────────────────────────────────────────────────────────
Solver 2
  Equation = "Navier-Stokes"
  Optimize Bandwidth = Logical True
  Linear System Solver = Direct
  Linear System Direct Method = "UMFPACK"
  Linear System Max Iterations = {p['solver2_linear_max_iter']}
  Linear System Convergence Tolerance = {p['solver2_linear_tol']}
  Linear System Abort Not Converged = False
  Linear System Preconditioning = "ILU1"
  Linear System Residual Output = 1
  Flow Model = Stokes
  Steady State Convergence Tolerance = {p['solver2_steady_tol']}
  Stabilization Method = Stabilized
  Nonlinear System Convergence Tolerance = {p['solver2_nonlinear_tol']}
  Nonlinear System Convergence Measure = Solution
  Nonlinear System Max Iterations = {p['solver2_nonlinear_max_iter']}
  Nonlinear System Newton After Iterations = 3
  Nonlinear System Newton After Tolerance = 1.0E-01
  Exported Variable 1 = -dofs 3 "Mesh Velocity"
End

! ──────────────────────────────────────────────────────────────────
! Material — Glen flow law with two Arrhenius regimes
! Density is a $-expression: SI kg/m³ × (year_s)^-2
! ──────────────────────────────────────────────────────────────────
Material 1
  Name = "{p['material_name']}"
  Density = Real ${p['density_formula']}
  Viscosity Model = String "Glen"
  Viscosity = Real {p['viscosity']}
  Glen Exponent = Real {p['glen_exponent']}
  Critical Shear Rate = Real {p['critical_shear_rate']}
  Rate Factor 1 = Real {p['rate_factor_1']}
  Rate Factor 2 = Real {p['rate_factor_2']}
  Activation Energy 1 = Real {p['activation_energy_1']}
  Activation Energy 2 = Real {p['activation_energy_2']}
  Glen Enhancement Factor = Real {p['glen_enhancement_factor']}
  Limit Temperature = Real {p['limit_temperature']}
  Constant Temperature = Real {p['constant_temperature']}
End

! ──────────────────────────────────────────────────────────────────
! Body Force — gravity scaled to year units
! ──────────────────────────────────────────────────────────────────
Body Force 1
  Name = "BodyForce1"
  Heat Source = 1
  Flow BodyForce 1 = Real 0.0
  Flow BodyForce 2 = Real 0.0
  Flow BodyForce 3 = Real ${p['gravity_formula']}
End

! ──────────────────────────────────────────────────────────────────
! Boundary conditions
! ──────────────────────────────────────────────────────────────────

Boundary Condition 1
  Name = "bedrock"
  Target Boundaries = {p['bedrock_target']}
  Velocity 1 = Real 0.0e0
  Velocity 2 = Real 0.0e0
  Velocity 3 = Real 0.0e0
End

Boundary Condition 2
  Name = "sides"
  Target Boundaries = {p['sides_target']}
  Velocity 1 = Real 0.0e0
  Velocity 2 = Real 0.0e0
End

Boundary Condition 3
  Name = "surface"
  Target Boundaries = {p['surface_target']}
End
"""


# ============================================================================
# ███ SECTION 25 — JOURNAL TEMPLATES                                     ███
# ============================================================================

class JournalTemplates:
    """Journal-specific matplotlib rcParams presets."""

    @staticmethod
    def get_journal_styles() -> Dict[str, Dict[str, Any]]:
        return {
            'nature': {
                'figure_width_single': 8.9, 'figure_width_double': 18.3,
                'font_family': 'Arial', 'font_size_small': 7,
                'font_size_medium': 8, 'font_size_large': 9,
                'line_width': 0.5, 'axes_linewidth': 0.5,
                'tick_width': 0.5, 'tick_length': 2,
                'grid_alpha': 0.1, 'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                '#bcbd22', '#17becf'],
            },
            'science': {
                'figure_width_single': 5.5, 'figure_width_double': 11.4,
                'font_family': 'Helvetica', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 0.75, 'axes_linewidth': 0.75,
                'tick_width': 0.75, 'tick_length': 3,
                'grid_alpha': 0.15, 'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                                '#77AC30', '#4DBEEE', '#A2142F', '#FF00FF',
                                '#00FFFF', '#FFA500'],
            },
            'advanced_materials': {
                'figure_width_single': 8.6, 'figure_width_double': 17.8,
                'font_family': 'Arial', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4,
                'grid_alpha': 0.2, 'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000',
                                '#44AA99', '#882255', '#117733', '#999933',
                                '#AA4499', '#88CCEE'],
            },
            'prl': {
                'figure_width_single': 3.4, 'figure_width_double': 7.0,
                'font_family': 'Times New Roman', 'font_size_small': 8,
                'font_size_medium': 10, 'font_size_large': 12,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4,
                'grid_alpha': 0, 'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                                '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                                '#999999', '#FFFFFF'],
            },
            'jgr_earth_surface': {
                'figure_width_single': 7.0, 'figure_width_double': 14.0,
                'font_family': 'Helvetica', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 0.8, 'axes_linewidth': 0.8,
                'tick_width': 0.8, 'tick_length': 3,
                'grid_alpha': 0.15, 'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                                '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
            },
            'custom': {
                'figure_width_single': 6.0, 'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans', 'font_size_small': 10,
                'font_size_medium': 12, 'font_size_large': 14,
                'line_width': 1.5, 'axes_linewidth': 1.5,
                'tick_width': 1.0, 'tick_length': 5,
                'grid_alpha': 0.3, 'dpi': 300,
                'color_cycle': plt.cm.Set2(np.linspace(0, 1, 10)),
            },
        }

    @staticmethod
    def apply_journal_style(fig, axes, journal_name: str = 'nature'):
        styles = JournalTemplates.get_journal_styles()
        style = styles.get(journal_name, styles['nature'])
        rcParams.update({
            'font.family': style['font_family'],
            'font.size': style['font_size_medium'],
            'axes.linewidth': style['axes_linewidth'],
            'axes.labelsize': style['font_size_medium'],
            'axes.titlesize': style['font_size_large'],
            'xtick.labelsize': style['font_size_small'],
            'ytick.labelsize': style['font_size_small'],
            'legend.fontsize': style['font_size_small'],
            'figure.titlesize': style['font_size_large'],
            'lines.linewidth': style['line_width'],
            'lines.markersize': 4,
            'xtick.major.width': style['tick_width'],
            'ytick.major.width': style['tick_width'],
            'xtick.minor.width': style['tick_width'] * 0.5,
            'ytick.minor.width': style['tick_width'] * 0.5,
            'xtick.major.size': style['tick_length'],
            'ytick.major.size': style['tick_length'],
            'xtick.minor.size': style['tick_length'] * 0.6,
            'ytick.minor.size': style['tick_length'] * 0.6,
            'axes.grid': False,
            'savefig.dpi': style['dpi'],
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.prop_cycle': plt.cycler(color=style['color_cycle']),
        })
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        for ax in axes_flat:
            if ax is None:
                continue
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_linewidth(style['axes_linewidth'] * 0.5)
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=style['tick_length'])
            ax.tick_params(which='minor', length=style['tick_length'] * 0.6)
        return fig, style


# ============================================================================
# ███ SECTION 26 — COLOURMAP LIBRARY + PALETTES                          ███
# ============================================================================

COLORMAPS = {
    'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno',
    'magma': 'magma', 'cividis': 'cividis', 'hot': 'hot', 'cool': 'cool',
    'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn',
    'winter': 'winter', 'copper': 'copper', 'bone': 'bone', 'gray': 'gray',
    'pink': 'pink', 'afmhot': 'afmhot', 'gist_heat': 'gist_heat',
    'binary': 'binary', 'coolwarm': 'coolwarm', 'bwr': 'bwr',
    'seismic': 'seismic', 'RdBu': 'RdBu', 'RdBu_r': 'RdBu_r',
    'RdGy': 'RdGy', 'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG',
    'PuOr': 'PuOr', 'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted', 'hsv': 'hsv', 'tab10': 'tab10',
    'tab20': 'tab20', 'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3',
    'Paired': 'Paired', 'Accent': 'Accent', 'Dark2': 'Dark2', 'jet': 'jet',
    'turbo': 'turbo', 'rainbow': 'rainbow',
    'nipy_spectral': 'nipy_spectral', 'gist_ncar': 'gist_ncar',
    'gist_earth': 'gist_earth', 'ocean': 'ocean', 'terrain': 'terrain',
    'gnuplot': 'gnuplot', 'cubehelix': 'cubehelix', 'brg': 'brg',
    'rocket': 'rocket', 'mako': 'mako', 'crest': 'crest', 'flare': 'flare',
    'icefire': 'icefire', 'vlag': 'vlag',
}

cmap_list = list(COLORMAPS.keys())

BAR_CMAP_CATEGORIES: Dict[str, List[str]] = {
    "🌈 Sequential (score gradient)": [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'rocket', 'mako', 'crest', 'flare',
        'Blues', 'Greens', 'Reds', 'Purples', 'Oranges',
        'YlOrRd', 'YlGnBu', 'BuGn', 'PuRd',
    ],
    "🔄 Diverging (low↔high)": [
        'coolwarm', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
        'seismic', 'bwr', 'vlag', 'icefire',
    ],
    "🌊 Cyclic / Perceptual": [
        'twilight', 'twilight_shifted', 'hsv', 'turbo',
    ],
    "🎯 Categorical (distinct)": [
        'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
        'Paired', 'Accent', 'Dark2', 'Pastel1', 'Pastel2',
    ],
    "🖤 Classic": [
        'Greys', 'gray', 'bone', 'copper', 'hot', 'afmhot',
        'gist_heat', 'binary',
    ],
}
BAR_CMAP_FLAT: List[str] = [n for cat in BAR_CMAP_CATEGORIES.values() for n in cat]

PUBLICATION_PALETTES: Dict[str, List[str]] = {
    'okabe_ito': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                  '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
    'tol_bright': ['#4477AA', '#EE6677', '#228833', '#CCBB44',
                   '#66CCEE', '#AA3377', '#BBBBBB'],
    'tol_muted': ['#CC6677', '#332288', '#DDCC77', '#117733',
                  '#88CCEE', '#882255', '#44AA99', '#999933',
                  '#AA4499', '#DDDDDD'],
    'ibm_carbon': ['#6929c4', '#1192e8', '#005d5d', '#9f1853',
                   '#fa4d56', '#570408', '#198038', '#002d9c',
                   '#ee538b', '#b28600'],
    'nature_classic': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf'],
    'science_aaas': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                     '#77AC30', '#4DBEEE', '#A2142F'],
    'ieee': ['#0000FF', '#FF0000', '#00AA00', '#FF8000',
             '#800080', '#00AAAA', '#800000'],
    'jgr': ['#000000', '#E69F00', '#56B4E9', '#009E73',
            '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
}

FONT_FAMILIES: List[str] = [
    'Arial', 'Helvetica', 'Times New Roman', 'Courier New',
    'DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono',
    'Calibri', 'Cambria', 'Georgia', 'Verdana', 'Tahoma',
    'Trebuchet MS', 'Garamond', 'Palatino', 'Bookman Old Style',
    'sans-serif', 'serif', 'monospace',
]


# ============================================================================
# ███ SECTION 27 — BAR-CHART TYPOGRAPHY HELPERS                          ███
# ============================================================================

def _resolve_bar_typography(style: Dict[str, Any],
                            annotation_fontsize: Optional[float],
                            legend_fontsize: Optional[float],
                            colorbar_label_fontsize: Optional[float],
                            colorbar_tick_fontsize: Optional[float]
                            ) -> Tuple[float, float, float, float]:
    base = float(style.get('font_size_small', 8))
    ann_fs = base if annotation_fontsize is None else max(2.0, float(annotation_fontsize))
    leg_fs = base if legend_fontsize is None else max(2.0, float(legend_fontsize))
    cbar_fs = base if colorbar_label_fontsize is None else max(2.0, float(colorbar_label_fontsize))
    ctick_fs = (max(cbar_fs - 1.0, 2.0)
                if colorbar_tick_fontsize is None
                else max(2.0, float(colorbar_tick_fontsize)))
    return ann_fs, leg_fs, cbar_fs, ctick_fs


def _compute_min_safe_headroom(fig_height_in: float,
                               label_pad_pt: float,
                               annotation_fontsize_pt: float,
                               axes_fraction: float = 0.85) -> float:
    clearance_pt = float(label_pad_pt) + 1.5 * float(annotation_fontsize_pt)
    axes_pt = max(1.0, axes_fraction * float(fig_height_in) * 72.0)
    clearance_frac = min(0.85, clearance_pt / axes_pt)
    return 1.0 / max(1e-6, 1.0 - clearance_frac)


# ============================================================================
# ███ SECTION 28 — V9.x CANDIDATE-SCORE BAR CHART                        ███
# ============================================================================

def plot_candidate_scores(candidates, scores, provenance,
                          best_idx=None, *,
                          param_title: Optional[str] = None,
                          param_symbol: Optional[str] = None,
                          unit: Optional[str] = None,
                          score_label: str = 'LatentMoE score',
                          journal: str = 'nature',
                          fig_size: Tuple[float, float] = (5.2, 3.4),
                          bar_width: float = 0.62,
                          label_pad: float = 9,
                          headroom: float = 1.22,
                          despine: bool = True,
                          title: Optional[str] = None,
                          bar_colormap: str = 'viridis',
                          color_by: str = 'score',
                          best_color_override: Optional[str] = None,
                          cmap_vmin: Optional[float] = None,
                          cmap_vmax: Optional[float] = None,
                          show_colorbar: bool = False,
                          annotation_fontsize: Optional[float] = None,
                          legend_fontsize: Optional[float] = None,
                          colorbar_label_fontsize: Optional[float] = None,
                          colorbar_tick_fontsize: Optional[float] = None,
                          colorbar_label: str = 'Score',
                          legend_anchor_y: float = 1.02,
                          legend_columnspacing: float = 1.4,
                          legend_handletextpad: float = 0.4,
                          legend_borderaxespad: float = 0.0,
                          legend_granularity: str = 'coarse',
                          legend_show_counts: bool = False,
                          context_flags: Optional[List[bool]] = None,
                          show_best_legend: bool = False):
    """Publication bar chart of candidate scores."""
    n = len(candidates)
    assert n == len(scores) == len(provenance), 'length mismatch'

    param_symbol = normalize_tex(param_symbol)
    granularity = _norm_granularity(legend_granularity)

    order = np.argsort(candidates)
    xs, cands = np.arange(n), [candidates[i] for i in order]
    scs = [scores[i] for i in order]
    provs = [_legend_key(provenance[i], granularity) for i in order]
    ctxs = ([bool(context_flags[i]) for i in order]
            if context_flags is not None else [False] * n)

    best_x = (int(np.where(order == best_idx)[0][0])
              if (best_idx is not None and best_idx in order) else None)

    style = JournalTemplates.get_journal_styles().get(journal,
                                                     {'font_size_small': 8})
    ann_fs, leg_fs, cbar_fs, ctick_fs = _resolve_bar_typography(
        style, annotation_fontsize, legend_fontsize,
        colorbar_label_fontsize, colorbar_tick_fontsize)

    min_safe = _compute_min_safe_headroom(
        fig_height_in=float(fig_size[1]),
        label_pad_pt=float(label_pad),
        annotation_fontsize_pt=float(ann_fs))
    effective_headroom = max(float(headroom), min_safe * 1.05)

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    JournalTemplates.apply_journal_style(fig, ax, journal)

    if isinstance(bar_colormap, str):
        try:
            cmap = plt.get_cmap(bar_colormap)
        except ValueError:
            logger.warning("Unknown colormap %r — falling back", bar_colormap)
            cmap = plt.get_cmap('viridis')
    else:
        cmap = bar_colormap

    if color_by == 'score':
        vmin = (float(cmap_vmin) if cmap_vmin is not None
                else float(min(scs)))
        vmax = (float(cmap_vmax) if cmap_vmax is not None
                else float(max(scs)))
        span = max(vmax - vmin, 1e-9)
        norm_scores = [(float(s) - vmin) / span for s in scs]
        faces = [cmap(ns) for ns in norm_scores]
    elif color_by == 'index':
        indices = np.linspace(0.05, 0.95, max(n, 1))
        faces = [cmap(float(i)) for i in indices]
    else:
        faces = ['#BBBBBB'] * n

    if best_x is not None:
        if best_color_override:
            try:
                faces[best_x] = best_color_override
            except Exception:
                pass
        else:
            faces[best_x] = '#F0E442' if color_by == 'score' else '#0072B2'

    comp_x = [x for x in xs if not ctxs[x]]
    ctx_x = [x for x in xs if ctxs[x]]
    if comp_x:
        ax.bar(comp_x, [scs[x] for x in comp_x], width=bar_width,
               facecolor=[faces[x] for x in comp_x],
               edgecolor='black', linewidth=0.8, zorder=3)
    if ctx_x:
        ax.bar(ctx_x, [scs[x] for x in ctx_x], width=bar_width,
               facecolor=[faces[x] for x in ctx_x], alpha=0.55,
               edgecolor='0.35', linewidth=0.7, zorder=3)

    marker_table = (LEGEND_MARKERS if granularity == 'coarse'
                    else PROVENANCE_MARKERS)

    for x, s, p in zip(xs, scs, provs):
        st_ = marker_table[p]
        is_ctx = ctxs[x]
        bar_rgb = _color_to_rgb(faces[x])
        lum = _relative_luminance(bar_rgb)
        marker_edge = '0.35' if is_ctx else ('black' if lum > 0.55 else 'white')
        mfc = 'white' if st_.get('fill', True) else 'none'
        ms = st_['ms'] * (0.75 if is_ctx else 1.0)
        ax.plot([x], [s], marker=st_['marker'], markersize=ms,
                linestyle='none', markerfacecolor=mfc,
                markeredgecolor=marker_edge,
                markeredgewidth=1.2 if not is_ctx else 0.9,
                clip_on=False, zorder=6)

    for x, s in zip(xs, scs):
        bar_rgb = _color_to_rgb(faces[x])
        lum = _relative_luminance(bar_rgb)
        is_ctx = ctxs[x]
        if is_ctx:
            text_color = '0.45'
            halo_color = 'white'
        else:
            text_color = 'black' if lum > 0.55 else 'white'
            halo_color = 'white' if lum > 0.55 else 'black'
        ax.annotate(f'{s:.3f}', xy=(x, s),
                    xytext=(0, label_pad), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=ann_fs * (0.9 if is_ctx else 1.0),
                    fontweight=('bold' if (x == best_x and not is_ctx)
                                else 'normal'),
                    color=text_color,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor=halo_color,
                              edgecolor='none', alpha=0.85),
                    zorder=7)

    ax.set_ylim(0, max(scs) * effective_headroom)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{c:.3g}' for c in cands], rotation=30,
                       ha='right')
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.6, n - 0.4)

    parts = [p for p in (param_title,
                         f'${param_symbol}$' if param_symbol else '') if p]
    xlabel = ' '.join(parts)
    if unit:
        xlabel += f' ($\\mathrm{{{unit}}}$)'
    ax.set_xlabel(safe_mathtext(xlabel))
    ax.set_ylabel(safe_mathtext(score_label))
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.22)
    ax.set_axisbelow(True)

    if despine:
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(which='both', top=False, right=False)

    if show_colorbar and color_by == 'score':
        sm = ScalarMappable(cmap=cmap,
                            norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02,
                            orientation='vertical')
        cbar.set_label(colorbar_label, fontsize=cbar_fs)
        cbar.ax.tick_params(labelsize=ctick_fs)

    if best_x is not None and any(ctxs):
        ax.axhline(scs[best_x], color='0.35', linewidth=0.7,
                   linestyle=(0, (4, 3)), zorder=2)

    present_keys = {p for p in provs if p in marker_table}
    handles = []
    for key, st_ in marker_table.items():
        if key not in present_keys:
            continue
        lbl = st_['label']
        if legend_show_counts:
            n_ctx_only = sum(1 for i, p in enumerate(provs)
                             if p == key and not ctxs[i])
            if n_ctx_only != provs.count(key):
                lbl = f"{lbl} (n={n_ctx_only}+ctx)"
            else:
                lbl = f"{lbl} (n={provs.count(key)})"
        handles.append(Line2D([], [], marker=st_['marker'],
                              linestyle='none', markersize=st_['ms'],
                              markerfacecolor=('white'
                                               if st_.get('fill', True)
                                               else 'none'),
                              markeredgecolor='black', markeredgewidth=1.1,
                              label=lbl))
    if show_best_legend and best_x is not None and not ctxs[best_x]:
        handles.append(Patch(facecolor=faces[best_x], edgecolor='black',
                             linewidth=0.8, label='Best match'))
    if any(ctxs):
        handles.append(Patch(facecolor='0.75', edgecolor='0.4',
                             alpha=0.65, label='Context (non-ranked)'))
    if handles:
        ncol = min(4, len(handles))
        ax.legend(handles=handles, loc='lower left',
                  bbox_to_anchor=(0.0, legend_anchor_y),
                  ncol=ncol, frameon=False, fontsize=leg_fs,
                  columnspacing=legend_columnspacing,
                  handletextpad=legend_handletextpad,
                  borderaxespad=legend_borderaxespad)

    if title:
        title_pad = 34.0 + max(0.0, leg_fs - 8.0) * 2.0
        if handles and len(handles) > ncol:
            title_pad += leg_fs * 1.8
        ax.set_title(safe_mathtext(title), pad=title_pad,
                     fontsize=style['font_size_large'])
    return fig


# ============================================================================
# ███ SECTION 29 — EXACT-XAI STACKED LATENT-MoE CHART                    ███
# ============================================================================

def plot_stacked_latentmoe(candidates: List[GlacierCandidate],
                           param_title: str,
                           param_symbol: str,
                           unit: Optional[str],
                           journal: str = 'nature',
                           fig_size: Tuple[float, float] = (6.5, 4.0),
                           max_headroom: float = 1.18,
                           show_context: bool = False):
    """Exact XAI stacked bar chart — bar height = total MoE score; each
    coloured segment = one expert's exact weighted contribution.

    Because the MoE is deterministic and linear, this IS the exact
    feature attribution — no SHAP/LIME needed.

    v10.1.1: when `show_context=True`, non-incumbent candidates are
    included (greyed overlay); the pinned winner is marked by a thin
    orange underline (NO '⭐ BEST' icon)."""
    n = len(candidates)
    if n == 0:
        return None
    cands_sorted = sorted(candidates, key=lambda c: c.score)

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    JournalTemplates.apply_journal_style(fig, ax, journal)

    xs = np.arange(n)
    bar_width = 0.65
    bottoms = np.zeros(n)

    drawn_experts: List[str] = []
    for expert in GlacierLatentMoEScorer.EXPERT_ORDER:
        heights = np.array([
            float(c.moe_breakdown.get(expert, 0.0))
            for c in cands_sorted])
        if np.sum(heights) <= 1e-4:
            continue
        color = GlacierLatentMoEScorer.EXPERT_COLORS.get(expert, '#BBBBBB')
        ax.bar(xs, heights, bar_width, bottom=bottoms,
               color=color, edgecolor='black', linewidth=0.5,
               label=expert, zorder=3)
        bottoms += heights
        drawn_experts.append(expert)

    if show_context:
        for i, c in enumerate(cands_sorted):
            if getattr(c, 'context', False):
                ax.bar([xs[i]], [c.score], bar_width,
                       color='grey', alpha=0.30, edgecolor='none', zorder=4)

    winner_x: Optional[int] = None
    for i, c in enumerate(cands_sorted):
        if not getattr(c, 'context', False):
            winner_x = i

    for i, (x, c) in enumerate(zip(xs, cands_sorted)):
        is_ctx = bool(getattr(c, 'context', False))
        is_winner = (winner_x is not None and i == winner_x)
        ax.text(x, c.score + 0.012, f"{c.score:.2f}",
                ha='center', va='bottom', fontsize=7,
                fontweight='bold' if is_winner else 'normal',
                color='0.45' if is_ctx else 'black', zorder=6)
        if is_winner:
            ax.plot([x - bar_width / 2, x + bar_width / 2],
                    [-0.015, -0.015], color='#D55E00', linewidth=2.5,
                    solid_capstyle='butt', clip_on=False, zorder=6)

    ax.set_ylim(0, max_headroom)
    ax.set_ylabel('Total LatentMoE score', fontsize=9)

    xlabel = f'{param_title}' if param_title else ''
    if param_symbol:
        xlabel += f' (${param_symbol}$)'
    if unit:
        xlabel += f' [$\\mathrm{{{unit}}}$]'
    if xlabel:
        ax.set_xlabel(safe_mathtext(xlabel), fontsize=9)

    spec = GLACIER_ONTOLOGY[cands_sorted[0].param]
    ui_vals = [c.value_si / spec["ui_scale"] for c in cands_sorted]

    def _fmt_xtick(v: float) -> str:
        if abs(v) > 1e4 or (v != 0 and abs(v) < 1e-2):
            return f"{v:.1e}"
        return f"{v:.3g}"

    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt_xtick(v) for v in ui_vals],
                       rotation=45, ha='right', fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.6, n - 0.4)

    if drawn_experts:
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02),
                  frameon=False, fontsize=7.5, ncol=2,
                  columnspacing=1.0, handletextpad=0.4,
                  borderaxespad=0.0)

    ax.yaxis.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(which='both', top=False, right=False)
    return fig


# ============================================================================
# ███ SECTION 30 — FIG-TO-BYTES HELPERS                                  ███
# ============================================================================

def _fig_to_bytes_matplotlib(fig, fmt: str = "png",
                             dpi: int = 300) -> Optional[bytes]:
    try:
        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning("Matplotlib export failed (%s): %s", fmt, e)
        return None


def _fig_to_bytes_plotly(fig, fmt: str = "png",
                         width: int = 1200, height: int = 800,
                         scale: float = 3.0) -> Optional[bytes]:
    try:
        return fig.to_image(format=fmt, width=width, height=height, scale=scale)
    except Exception as e:
        logger.warning("Plotly static export failed (%s): %s", fmt, e)
        return None


# ============================================================================
# ███ SECTION 31 — PHYSICS LAB: ARRHENIUS                                ███
# ============================================================================
# The Arrhenius Lab synthesises the temperature-dependent rate factor
# A(T) from the user's current cold/warm parameters and displays the
# sensitivity to T* and Q1/Q2.
# ============================================================================

def arrhenius_rate_factor(T_c: float, A1: float, A2: float,
                          Q1: float, Q2: float, T_star_c: float,
                          R: float = R_GAS) -> float:
    """Return A(T) in the same units as A1/A2 (MPa^-n yr^-1).

    Below T*: A(T) = A1 · exp[−Q1/R (1/T − 1/T*)]
    Above T*: A(T) = A2 · exp[−Q2/R (1/T − 1/T*)]

    T is in Kelvin internally.
    """
    T_K = T_c + T0_CELSIUS
    T_star_K = T_star_c + T0_CELSIUS
    if T_K <= 0 or T_star_K <= 0:
        return float("nan")
    if T_c <= T_star_c:
        return A1 * math.exp(-Q1 / R * (1.0 / T_K - 1.0 / T_star_K))
    return A2 * math.exp(-Q2 / R * (1.0 / T_K - 1.0 / T_star_K))


@handle_errors
def render_arrhenius_lab():
    st.markdown('### 🌡️ Arrhenius Temperature Lab')
    st.caption(
        'The Glen flow-law rate factor A(T) is defined piecewise around '
        'the limit temperature T*.  Below T* the cold (Q₁, A₁) pair '
        'applies; above, the warm (Q₂, A₂) pair.  The Arrhenius coupling '
        'is a first-class MoE thread in the v10.1.1-glacier scorer.')

    bundle = st.session_state.get('glacier_recommender_bundle')
    if bundle is None:
        st.info('Run the AI recommender in the sidebar first, then come '
                'back to this lab.  Meanwhile, defaults are shown.')
        # Fall back to defaults so the lab is interactive before Analyse.
        A1 = GLACIER_ONTOLOGY['rate_factor_A1']['defaults']['ice']
        A2 = GLACIER_ONTOLOGY['rate_factor_A2']['defaults']['ice']
        Q1 = GLACIER_ONTOLOGY['activation_energy_Q1']['defaults']['ice']
        Q2 = GLACIER_ONTOLOGY['activation_energy_Q2']['defaults']['ice']
        T_star = GLACIER_ONTOLOGY['limit_temperature']['defaults']['ice']
    else:
        def _best_val(p):
            b = bundle.best(p)
            return b.value_si if b is not None else bundle.defaults[p]
        A1 = _best_val('rate_factor_A1')
        A2 = _best_val('rate_factor_A2')
        Q1 = _best_val('activation_energy_Q1')
        Q2 = _best_val('activation_energy_Q2')
        T_star = _best_val('limit_temperature')

    col1, col2 = st.columns(2)
    with col1:
        A1_show = st.number_input(
            'A₁ (cold, MPa⁻ⁿ yr⁻¹)', value=float(A1), format='%.3e',
            key='arr_A1')
        Q1_show = st.number_input(
            'Q₁ (cold, J/mol)', value=float(Q1), format='%.0f',
            key='arr_Q1')
        T_star_show = st.number_input(
            'T* (°C)', value=float(T_star), format='%.2f',
            key='arr_Tstar')
    with col2:
        A2_show = st.number_input(
            'A₂ (warm, MPa⁻ⁿ yr⁻¹)', value=float(A2), format='%.3e',
            key='arr_A2')
        Q2_show = st.number_input(
            'Q₂ (warm, J/mol)', value=float(Q2), format='%.0f',
            key='arr_Q2')

    T_min, T_max = -50.0, 0.0
    Ts = np.linspace(T_min, T_max, 400)
    As = np.array([arrhenius_rate_factor(t, A1_show, A2_show,
                                          Q1_show, Q2_show,
                                          T_star_show)
                   for t in Ts])

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    JournalTemplates.apply_journal_style(fig, ax, 'nature')
    ax.semilogy(Ts, As, color='#0072B2', linewidth=1.5)
    ax.axvline(T_star_show, color='#D55E00', linewidth=1.0,
               linestyle='--', label=f'T* = {T_star_show:.1f} °C')
    # annotate cold/warm sides
    ax.text(T_star_show - 2, As.max() * 0.5, 'cold', ha='right',
            fontsize=8, color='#0072B2')
    ax.text(T_star_show + 2, As.max() * 0.5, 'warm', ha='left',
            fontsize=8, color='#D55E00')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('A(T)  [MPa⁻ⁿ yr⁻¹]')
    ax.set_title('Arrhenius rate factor A(T)')
    ax.grid(True, which='both', alpha=0.25, linewidth=0.5)
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('#### Derived diagnostics')
    A_at_minus10 = arrhenius_rate_factor(-10.0, A1_show, A2_show,
                                          Q1_show, Q2_show, T_star_show)
    A_at_minus30 = arrhenius_rate_factor(-30.0, A1_show, A2_show,
                                          Q1_show, Q2_show, T_star_show)
    A_at_minus1  = arrhenius_rate_factor(-1.0, A1_show, A2_show,
                                          Q1_show, Q2_show, T_star_show)
    ratio = A_at_minus1 / max(A_at_minus30, 1e-30)
    st.write(f"**A(−10 °C)** = `{A_at_minus10:.3e}` MPa⁻ⁿ yr⁻¹")
    st.write(f"**A(−30 °C)** = `{A_at_minus30:.3e}` MPa⁻ⁿ yr⁻¹")
    st.write(f"**A(−1 °C)**  = `{A_at_minus1:.3e}` MPa⁻ⁿ yr⁻¹")
    st.write(f"**Softening ratio A(−1)/A(−30)** = `{ratio:.2e}`")


# ============================================================================
# ███ SECTION 32 — PHYSICS LAB: GLEN FLOW-LAW                            ███
# ============================================================================
# The Glen Lab synthesises the effective viscosity from the full flow law
# ν_eff = (2 A E)^(−1/n) · ε̇^(1/n − 1) and lets the user sweep strain
# rate and temperature.
# ============================================================================

def glen_effective_viscosity(strain_rate: float, A_val: float,
                             n_val: float, E_val: float = 1.0
                             ) -> float:
    """Effective viscosity ν_eff = (2 A E)^(−1/n) · ε̇^(1/n − 1).

    Returns a value in MPa·yr (if A is in MPa^-n yr^-1 and strain rate in
    yr^-1), or SI units if the inputs are SI.  The formula is invariant.
    """
    if strain_rate <= 0 or A_val <= 0 or n_val <= 0 or E_val <= 0:
        return float("nan")
    pre = (2.0 * A_val * E_val) ** (-1.0 / n_val)
    return pre * (strain_rate ** (1.0 / n_val - 1.0))


@handle_errors
def render_glen_flow_law_lab():
    st.markdown('### 🧊 Glen Flow-Law Lab')
    st.caption(
        'Sweeps the effective viscosity ν_eff = (2 A E)^(−1/n) · ε̇^(1/n−1) '
        'over strain rate and temperature.  The Glen-n exponent and the '
        'enhancement factor E are FIRST-CLASS MoE threads.')

    bundle = st.session_state.get('glacier_recommender_bundle')
    if bundle is None:
        st.info('Run the AI recommender first for material-specific values.')
        n_val = 3.0
        E_val = 1.0
        A_val = 1.258e13
    else:
        def _best_val(p):
            b = bundle.best(p)
            return b.value_si if b is not None else bundle.defaults[p]
        n_val = _best_val('glen_n')
        E_val = _best_val('glen_enhancement')
        A_val = _best_val('rate_factor_A1')

    col1, col2, col3 = st.columns(3)
    with col1:
        n_show = st.number_input('n (Glen exponent)', value=float(n_val),
                                 step=0.1, format='%.3f', key='glen_n_in')
    with col2:
        E_show = st.number_input('E (enhancement)', value=float(E_val),
                                 step=0.1, format='%.3f', key='glen_E_in')
    with col3:
        A_show = st.number_input('A (MPa⁻ⁿ yr⁻¹)', value=float(A_val),
                                 format='%.3e', key='glen_A_in')

    eps_min, eps_max = 1e-6, 1e-1
    eps_sweep = np.logspace(np.log10(eps_min), np.log10(eps_max), 120)
    nu_arr = np.array([glen_effective_viscosity(e, A_show, n_show, E_show)
                      for e in eps_sweep])

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    JournalTemplates.apply_journal_style(fig, ax, 'nature')
    ax.loglog(eps_sweep, nu_arr, color='#0072B2', linewidth=1.5)
    ax.set_xlabel('Strain rate ε̇ [yr⁻¹]')
    ax.set_ylabel('ν_eff  [MPa · yr]')
    ax.set_title(f'Glen flow-law ν_eff  (n = {n_show:.2f}, E = {E_show:.2f})')
    ax.grid(True, which='both', alpha=0.25, linewidth=0.5)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('#### Derived diagnostics')
    nu_at_1e_3 = glen_effective_viscosity(1e-3, A_show, n_show, E_show)
    nu_at_1e_1 = glen_effective_viscosity(1e-1, A_show, n_show, E_show)
    st.write(f"**ν_eff at ε̇ = 1e-3 yr⁻¹** = `{nu_at_1e_3:.3e}` MPa·yr")
    st.write(f"**ν_eff at ε̇ = 1e-1 yr⁻¹** = `{nu_at_1e_1:.3e}` MPa·yr")
    st.write(f"**Shear-thinning index** = `{1.0/n_show - 1.0:+.3f}`")


# ============================================================================
# ███ SECTION 33 — PHYSICS LAB: ENHANCEMENT                              ███
# ============================================================================
# Enhancement / fabric lab correlates E with c-axis fabric strength via
# the empirical relation from Lliboutry / Thorsteinsson.
# ============================================================================

def enhancement_from_fabric(fabric_strength: float) -> float:
    """Empirical E(fabric_strength) after Lliboutry 1993.

    fabric_strength ∈ [0, 1] is the single-maximum pole-figure strength
    (0 = random, 1 = perfect single-max).
    """
    fs = float(np.clip(fabric_strength, 0.0, 1.0))
    # E ≈ 1 + 4 fs² + 0.5 fs⁴  (interpolates random→E=1, perfect→E≈5.5)
    return 1.0 + 4.0 * fs ** 2 + 0.5 * fs ** 4


@handle_errors
def render_enhancement_lab():
    st.markdown('### 🧵 Enhancement / Fabric Lab')
    st.caption(
        'Empirical enhancement vs c-axis fabric strength after '
        'Lliboutry (1993).  Fabric strength ∈ [0, 1]; 0 = random fabric, '
        '1 = perfect single-maximum c-axis cluster.')

    bundle = st.session_state.get('glacier_recommender_bundle')
    default_E = (bundle.best('glen_enhancement').value_si
                 if bundle and bundle.best('glen_enhancement')
                 else 1.0)

    fs = st.slider('Fabric strength (dimensionless)', 0.0, 1.0, 0.5, 0.01,
                   key='fab_fs')
    E_from_fabric = enhancement_from_fabric(fs)

    col1, col2 = st.columns(2)
    with col1:
        st.metric('E(fabric) Lliboutry', f'{E_from_fabric:.3f}')
    with col2:
        st.metric('E from recommender', f'{default_E:.3f}')

    fabric_strength_sweep = np.linspace(0, 1, 100)
    E_sweep = np.array([enhancement_from_fabric(f)
                       for f in fabric_strength_sweep])

    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    JournalTemplates.apply_journal_style(fig, ax, 'nature')
    ax.plot(fabric_strength_sweep, E_sweep, color='#009E73', linewidth=1.5)
    ax.axvline(fs, color='#D55E00', linewidth=1.0, linestyle='--',
               label=f'current fs = {fs:.2f}')
    ax.set_xlabel('Fabric strength (0 = random, 1 = single-max)')
    ax.set_ylabel('Enhancement factor E')
    ax.set_title('Enhancement from fabric — Lliboutry 1993')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    st.pyplot(fig)
    plt.close(fig)


# ============================================================================
# ███ SECTION 34 — PHYSICS LAB: BASAL / DRIVING STRESS                   ███
# ============================================================================
# Driving stress lab: τ_b = ρ g H sin(α)  and the Glen-inverted
# strain rate from the user's current parameters.
# ============================================================================

def basal_driving_stress(rho: float, g: float, H: float,
                         alpha_deg: float) -> float:
    """Basal driving stress τ_b = ρ g H sin(α) in Pa."""
    alpha_rad = math.radians(alpha_deg)
    return rho * g * H * math.sin(alpha_rad)


def glen_strain_rate(tau_pa: float, A_val: float, n_val: float,
                     E_val: float = 1.0) -> float:
    """Glen strain rate ε̇ = A · E · τ^n (τ in Pa; A in MPa^-n yr^-1 must
    be converted to Pa^-n yr^-1 first)."""
    tau_mpa = tau_pa / PA_PER_MPA
    return A_val * E_val * (tau_mpa ** n_val)


@handle_errors
def render_basal_stress_lab():
    st.markdown('### ⛰️ Basal / Driving Stress Lab')
    st.caption(
        'Driving stress τ_b = ρ g H sin α and the Glen-inverted strain '
        'rate ε̇ = A · E · τ^n.  Useful for sanity-checking the '
        'recommended A against a first-order ice-sheet balance.')

    bundle = st.session_state.get('glacier_recommender_bundle')
    default_rho = (bundle.best('ice_density').value_si
                   if bundle and bundle.best('ice_density') else 917.0)
    default_g = (bundle.best('gravity').value_si
                 if bundle and bundle.best('gravity') else 9.81)
    default_A = (bundle.best('rate_factor_A1').value_si
                 if bundle and bundle.best('rate_factor_A1') else 1.258e13)
    default_n = (bundle.best('glen_n').value_si
                 if bundle and bundle.best('glen_n') else 3.0)
    default_E = (bundle.best('glen_enhancement').value_si
                 if bundle and bundle.best('glen_enhancement') else 1.0)

    col1, col2 = st.columns(2)
    with col1:
        rho = st.number_input('ρ (kg/m³)', value=float(default_rho),
                              step=10.0, key='bs_rho')
        g = st.number_input('g (m/s²)', value=float(default_g),
                            step=0.01, format='%.3f', key='bs_g')
        H = st.number_input('Ice thickness H (m)', value=1000.0,
                            step=50.0, key='bs_H')
        alpha = st.number_input('Surface slope α (deg)', value=3.0,
                                step=0.5, key='bs_alpha')
    with col2:
        A_show = st.number_input('A (MPa⁻ⁿ yr⁻¹)', value=float(default_A),
                                 format='%.3e', key='bs_A')
        n_show = st.number_input('n', value=float(default_n), step=0.1,
                                 format='%.3f', key='bs_n')
        E_show = st.number_input('E', value=float(default_E), step=0.1,
                                 format='%.3f', key='bs_E')

    tau_b = basal_driving_stress(rho, g, H, alpha)
    eps_dot = glen_strain_rate(tau_b, A_show, n_show, E_show)
    col1, col2, col3 = st.columns(3)
    col1.metric('Driving stress τ_b', f'{tau_b/1e3:.2f} kPa')
    col2.metric('Glen strain rate ε̇', f'{eps_dot:.3e} yr⁻¹')
    col3.metric('Equivalent velocity (1 km col)',
                f'{eps_dot * 1000.0:.3f} m/yr')


# ============================================================================
# ███ SECTION 35 — RECOMMENDER ORCHESTRATOR                               ███
# ============================================================================

@dataclass
class GlacierRecommendationBundle:
    """All outputs of a recommender run."""
    material: str
    temp_c: Optional[float]
    stress_regime: Optional[str]
    fabric: Optional[str]
    candidates: Dict[str, List[GlacierCandidate]]
    defaults: Dict[str, float]
    retrieval_backend: str
    llm_used: bool
    timestamp: float = field(default_factory=time.time)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    sif_context: Dict[str, Any] = field(default_factory=dict)

    def best(self, param: str) -> Optional[GlacierCandidate]:
        lst = self.candidates.get(param, [])
        if not lst:
            return None
        ranked = [c for c in lst if not getattr(c, 'context', False)]
        if not ranked:
            logger.warning("bundle.best[%s]: all candidates are context — "
                           "returning top-scored anyway", param)
            ranked = list(lst)
        return max(ranked, key=lambda c: c.score)

    def coverage(self) -> int:
        return sum(1 for p in GLACIER_PARAM_ORDER
                   if self.candidates.get(p))

    def coverage_five(self) -> int:
        return sum(1 for p in SIDEBAR_PARAM_ORDER
                   if self.candidates.get(p))


class GlacierRecommender:
    """Orchestrates the full three-tier cascade across every parameter."""

    def __init__(self, db_dir: str = "json_metadatabase",
                 ollama_model: str = "qwen2.5:7b",
                 use_llm: bool = True,
                 debug_llm: bool = False,
                 use_grounded: bool = True,
                 allow_prior_inference: bool = True,
                 cascade_mode: str = 'union'):
        self.db_dir = db_dir
        self.client = GlacierOllamaClient(model=ollama_model)
        self.llm_available = use_llm and GlacierOllamaClient.is_available()
        self.hybrid: Optional[HybridRetriever] = None
        if use_grounded:
            try:
                self.hybrid = get_retriever(db_dir, use_dense=True)
            except Exception as e:
                logger.warning("Hybrid retriever init failed: %s", e)
                self.hybrid = None
        self.scorer = GlacierLatentMoEScorer()
        self.debug_llm = debug_llm
        self.allow_prior_inference = allow_prior_inference
        self.cascade_mode = (cascade_mode
                             if cascade_mode in ('union', 'fallback')
                             else 'union')

    def recommend(self,
                  material: str,
                  temp_c: Optional[float],
                  stress_regime: Optional[str] = None,
                  fabric: Optional[str] = None,
                  sif_context: Optional[Dict[str, Any]] = None,
                  progress_callback: Optional[Callable] = None
                  ) -> GlacierRecommendationBundle:
        if self.hybrid is None:
            self.hybrid = get_retriever(self.db_dir, use_dense=True)

        extractions: List[Dict[str, Any]] = []
        diagnostics: List[Dict[str, Any]] = []
        backend = ("hybrid(lexical+faiss)" if self.hybrid.dense
                   else "hybrid(lexical-only)")

        for i, param in enumerate(GLACIER_PARAM_ORDER):
            if progress_callback:
                progress_callback(i + 1, len(GLACIER_PARAM_ORDER),
                                  f"cascade[{param}]")
            cands, records, diag = recommend_param_values(
                param_key=param, material=material, temp_c=temp_c,
                stress_regime=stress_regime, fabric=fabric,
                use_llm=self.llm_available,
                ollama_model=self.client.model,
                retriever=self.hybrid,
                allow_prior_inference=self.allow_prior_inference,
                cascade_mode=self.cascade_mode)
            diagnostics.append(diag)
            for c in cands:
                prov_fine = _norm_provenance(getattr(c, "provenance", "")
                                             or getattr(c, "method", ""))
                method = ("explicit"
                          if prov_fine in ("llm_extract", "llm_reasoned")
                          else "heuristic" if prov_fine == "regex_ner"
                          else "prior inference")
                if c.method == "regime_inference":
                    method = "regime_inference"
                extractions.append({
                    "param": param,
                    "value": c.value,
                    "unit": c.unit,
                    "material": material,
                    "temp": (temp_c + T0_CELSIUS
                             if temp_c is not None else None),
                    "method": method,
                    "confidence": c.confidence,
                    "evidence": c.evidence,
                    "reasoning": (getattr(c, "reasoning", "")
                                  or _default_reasoning(c)),
                    "regime": getattr(c, "regime_tag", ""),
                    "_source_file": c.source,
                    "_source_title": c.property_label,
                    "_provenance": prov_fine,
                    "_context": bool(getattr(c, "context", False)),
                })

        # Inject context parameter defaults (density/gravity/shear rate)
        for param in CONTEXT_PARAM_ORDER:
            if not any(e["param"] == param for e in extractions):
                default_val = GLACIER_ONTOLOGY[param]["defaults"].get(
                    "ice", 1.0)
                extractions.append({
                    "param": param,
                    "value": default_val,
                    "unit": GLACIER_ONTOLOGY[param]["unit"],
                    "material": material,
                    "temp": (temp_c + T0_CELSIUS
                             if temp_c is not None else None),
                    "method": "default_fallback",
                    "confidence": 0.3,
                    "evidence": "",
                    "reasoning": ("No corpus candidate — using ontology "
                                  "default."),
                    "_source_file": "ontology_default",
                    "_source_title": "built-in",
                    "_provenance": "regex_ner",
                    "_context": False,
                })

        candidates = self.scorer.score(
            extractions, target_material=material, target_temp_c=temp_c,
            stress_regime=stress_regime, fabric=fabric,
            target_strain_rate=1e-3, top_k=8)

        defaults = {
            p: GLACIER_ONTOLOGY[p]["defaults"].get("ice", 1.0)
            for p in ALL_PARAMS
        }

        return GlacierRecommendationBundle(
            material=material, temp_c=temp_c,
            stress_regime=stress_regime, fabric=fabric,
            candidates=candidates, defaults=defaults,
            retrieval_backend=backend, llm_used=self.llm_available,
            diagnostics=diagnostics, sif_context=sif_context or {})


# ============================================================================
# ███ SECTION 36 — SIF GENERATOR HELPERS                                 ███
# ============================================================================

def build_sif_params_from_bundle(bundle: Optional[GlacierRecommendationBundle],
                                 overrides: Dict[str, Any],
                                 ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge recommender bundle + UI overrides + parsed SIF context into
    the parameter dict consumed by `generate_glacier_sif`."""
    def _pick(param, fallback):
        if param in overrides:
            return overrides[param]
        if bundle is not None:
            b = bundle.best(param)
            if b is not None:
                return b.value_si
            return bundle.defaults.get(param, fallback)
        return fallback

    glen_n = _pick('glen_n', 3.0)
    A1 = _pick('rate_factor_A1', 1.258e13)
    A2 = _pick('rate_factor_A2', 6.046e28)
    Q1 = _pick('activation_energy_Q1', 60000.0)
    Q2 = _pick('activation_energy_Q2', 139000.0)
    T_star = _pick('limit_temperature', -10.0)
    T_const = _pick('constant_temperature', -3.0)
    E = _pick('glen_enhancement', 1.0)
    gcs = _pick('critical_shear_rate', 1.0e-10)
    rho = _pick('ice_density', 917.0)
    g = _pick('gravity', 9.81)

    # Density expression: Elmer expects an expression in the SIF unit
    # system.  The recommender gives SI kg/m³.  The Elmer SIF uses a
    # year-based time unit, so ρ_Elmer = ρ_SI × (1 yr)^-2 × ... i.e. we
    # apply the year-to-second conversion in the expression to keep
    # units coherent.
    density_expr = f"{rho:.2f} * (1.0E-06) * ({SECONDS_PER_YEAR:.6e})^(-2.0)"
    gravity_expr = f"-{g:.4f} * ({SECONDS_PER_YEAR:.6e})^(2.0)"

    return {
        "mesh_db": ctx.get("mesh_db", "glacier_mesh"),
        "output_file": ctx.get("output_file", "glacier.result"),
        "post_file": ctx.get("post_file", "glacier.vtu"),
        "steady_state_max_iter": ctx.get("steady_state_max_iter", 1) or 1,
        "output_intervals": 1,
        "solver2_linear_max_iter": 5000,
        "solver2_linear_tol": 1.0e-6,
        "solver2_steady_tol": 1.0e-5,
        "solver2_nonlinear_max_iter": 50,
        "solver2_nonlinear_tol": 1.0e-4,
        "material_name": "ice",
        "viscosity": 1.0,
        "glen_exponent": glen_n,
        "critical_shear_rate": gcs,
        "rate_factor_1": A1,
        "rate_factor_2": A2,
        "activation_energy_1": Q1,
        "activation_energy_2": Q2,
        "glen_enhancement_factor": E,
        "limit_temperature": T_star,
        "constant_temperature": T_const,
        "density_formula": density_expr,
        "gravity_formula": gravity_expr,
        "bedrock_target": ctx.get("bedrock_target", 2),
        "sides_target": ctx.get("sides_target", 1),
        "surface_target": ctx.get("surface_target", 3),
    }


# ============================================================================
# ███ SECTION 37 — SIDEBAR RECOMMENDER UI                                ███
# ============================================================================

_PLR = "gl_rec_"


def _plr_get(key, default=None):
    return st.session_state.get(_PLR + key, default)


def _plr_set(key, value):
    st.session_state[_PLR + key] = value


def _plr_reset():
    for p in ALL_PARAMS:
        for suffix in ("choice", "manual", "value_si", "radio",
                       "manual_input"):
            st.session_state.pop(f"{_PLR}{p}_{suffix}", None)
    for k in ("bundle", "diagnostics", "cascade_mode"):
        st.session_state.pop(_PLR + k, None)
    st.session_state.pop("glacier_overrides", None)
    st.session_state.pop("glacier_recommender_bundle", None)
    st.session_state.pop("_glacier_live_recommender", None)


def _plr_render_parameter_selector(
    param: str,
    bundle: GlacierRecommendationBundle,
) -> float:
    spec = GLACIER_ONTOLOGY[param]
    st.markdown(f"**{spec['symbol']} — {spec['label']}**")
    options: List[Tuple[str, float, Optional[GlacierCandidate]]] = []
    best = bundle.best(param)
    if best is not None:
        prov_fine = _norm_provenance(getattr(best, "provenance", "")
                                     or getattr(best, "method", ""))
        tag = {
            "llm_extract":      "✓ grounded",
            "llm_reasoned":     "~ llm-cot",
            "regex_ner":        "· regex",
            "regime_prior":     "🧭 regime",
            "llm_prior":        "∅ prior",
            "physics_inferred": "△ physics",
        }.get(prov_fine, best.method)
        options.append((
            f"⭐ Recommended: {glacier_fmt(param, best.value_si)} "
            f"(score {best.score:.2f} · {tag})",
            best.value_si, best))
    n_alt = 0
    for c in bundle.candidates.get(param, []):
        if getattr(c, 'context', False):
            continue
        if c is best:
            continue
        n_alt += 1
        options.append((
            f"   Alt {n_alt}: {glacier_fmt(param, c.value_si)} "
            f"(score {c.score:.2f})",
            c.value_si, c))
    default_si = bundle.defaults.get(param, 1.0)
    options.append((f"⚙️ Default: {glacier_fmt(param, default_si)}",
                    default_si, None))
    options.append(("✏️ Manual override…", float("nan"), None))

    labels = [o[0] for o in options]
    prev = _plr_get(f"{param}_choice", labels[0])
    if prev not in labels:
        prev = labels[0]
    idx = labels.index(prev)
    choice = st.radio(f"Select {param}", labels, index=idx,
                      key=f"{_PLR}{param}_radio",
                      label_visibility="collapsed")
    _plr_set(f"{param}_choice", choice)
    sel = labels.index(choice)
    _, value_si, chosen = options[sel]
    if math.isnan(value_si):
        ui_default = default_si / spec["ui_scale"]
        manual = st.number_input(
            f"Manual {param} ({spec['ui_unit']})",
            value=float(_plr_get(f"{param}_manual", ui_default)),
            format="%.6g", key=f"{_PLR}{param}_manual_input")
        _plr_set(f"{param}_manual", manual)
        value_si = manual * spec["ui_scale"]
        chosen = None

    if chosen is not None:
        st.caption(
            f"📚 {chosen.source_file} — {chosen.source_title[:70]} | "
            f"method={chosen.method}, conf={chosen.confidence:.2f}"
            + (" | ⚠️ clamped" if chosen.clamped else ""))
        if chosen.evidence:
            st.code(chosen.evidence, language="text")
        if chosen.reasoning:
            with st.expander("🧠 Reasoning chain", expanded=False):
                st.markdown(chosen.reasoning)

    n_ctx = sum(1 for c in bundle.candidates.get(param, [])
                if getattr(c, 'context', False))
    if n_ctx > 0:
        with st.expander(f"🔎 Supporting routes shown as context ({n_ctx})",
                         expanded=False):
            st.caption("Greyed context bars in the chart. Non-ranked — "
                       "the recommended value above is unaffected.")
            ctx_rows = []
            for c in bundle.candidates.get(param, []):
                if not getattr(c, 'context', False):
                    continue
                ctx_rows.append({
                    "value": glacier_fmt(param, c.value_si),
                    "method": c.method,
                    "provenance": c.provenance or c.method,
                    "confidence": round(c.confidence, 3),
                    "evidence": c.evidence[:80],
                })
            if ctx_rows:
                st.dataframe(pd.DataFrame(ctx_rows),
                             use_container_width=True, hide_index=True)

    st.session_state[f"{_PLR}{param}_value_si"] = value_si
    st.markdown("---")
    return value_si


def render_recommender_sidebar(default_material: str = "ice",
                               default_temp_c: float = -10.0,
                               default_stress: str = "high",
                               default_fabric: str = "isotropic",
                               ollama_model: str = "qwen2.5:7b"):
    st.subheader("🤖 Glacier Intelligent Recommender v10.1.1")
    st.caption(
        "**10-expert Latent MoE** · **union cascade + pin-list** · "
        "**6→3 provenance rollup** · **exact-XAI stacked chart**.")

    col1, col2 = st.columns(2)
    with col1:
        material = st.text_input("Target material", value=default_material,
                                 key=f"{_PLR}material")
    with col2:
        temp_c = st.number_input("Ice temperature (°C)", value=float(default_temp_c),
                                 min_value=-60.0, max_value=1.0, step=1.0,
                                 key=f"{_PLR}temp")

    stress_options = ["(unknown)", "low stress / slow flow",
                      "high stress / fast flow"]
    stress = st.selectbox("Stress regime", stress_options, index=2,
                          key=f"{_PLR}stress")
    _stress_arg = None if stress.startswith("(") else \
        ("low" if "low" in stress else "high")

    fabric_options = ["(unknown)",
                      "isotropic (random fabric)",
                      "anisotropic (single-max c-axis)",
                      "columnar (highly oriented)",
                      "equiaxed (nanocrystalline)"]
    fabric = st.selectbox("Ice fabric / architecture", fabric_options,
                          index=1, key=f"{_PLR}fabric")
    if fabric.startswith("("):
        _fabric_arg = None
    elif fabric.startswith("isotropic"):
        _fabric_arg = "isotropic"
    elif fabric.startswith("columnar"):
        _fabric_arg = "anisotropic"
    elif fabric.startswith("equiaxed"):
        _fabric_arg = "isotropic"
    else:
        _fabric_arg = "anisotropic"

    # Regime previews
    st.markdown('#### 🧭 Live regime preview')
    g_n = classify_glen_n_regime(_stress_arg, temp_c, _fabric_arg)
    r_A = classify_rate_factor_regime(temp_c, _fabric_arg)
    q_reg = classify_activation_q_regime(temp_c)
    e_reg = classify_fabric_regime(_fabric_arg, temp_c)
    st.caption(f"**n** — {g_n['regime']} · "
               f"range [{g_n['low']:g}, {g_n['high']:g}] · "
               f"center {g_n['inferred']:g}")
    st.caption(f"**A** — {r_A['regime']} · "
               f"range [{r_A['low']:.1e}, {r_A['high']:.1e}] · "
               f"center {r_A['inferred']:.2e}")
    st.caption(f"**Q** — {q_reg['regime']} · "
               f"range [{q_reg['low']:.0f}, {q_reg['high']:.0f}] · "
               f"center {q_reg['inferred']:.0f}")
    st.caption(f"**E** — {e_reg['regime']} · "
               f"range [{e_reg['low']:.2f}, {e_reg['high']:.2f}] · "
               f"center {e_reg['inferred']:.2f}")

    # LLM model selection
    _models = ["(disable LLM)"] + GlacierOllamaClient.list_models() or \
              ["(disable LLM)", "qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b",
               "mistral:7b", "gemma2:9b"]
    sel_model = st.selectbox("Ollama model", _models, index=0,
                             key=f"{_PLR}ollama_sel")
    _llm_avail = (GlacierOllamaClient.is_available()
                  if sel_model != "(disable LLM)" else False)
    st.caption(f"{'✅' if _llm_avail else '⚠️'} Ollama "
               f"{'available' if _llm_avail else 'unreachable / disabled'}")

    use_grounded = st.checkbox("🧭 Use grounded NER pipeline", value=True,
                               key=f"{_PLR}use_grounded")
    allow_prior = st.checkbox("🧠 Allow LLM prior inference when corpus "
                              "evidence is missing", value=True,
                              key=f"{_PLR}allow_prior")

    cascade_mode = st.radio(
        "Cascade mode",
        options=['union', 'fallback'],
        format_func=lambda x: {
            'union': '🔀 Union — all routes as context',
            'fallback': '➡️ Fallback — first tier only'}[x],
        index=0, key=f"{_PLR}cascade_mode", horizontal=True)

    debug_llm = st.checkbox("🔍 Debug LLM output", value=False,
                            key=f"{_PLR}debug")

    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        run_btn = st.button("🔍 Analyse", use_container_width=True,
                            type="primary")
    with btn2:
        refresh_btn = st.button("🔄 Reload corpus",
                                use_container_width=True)
    with btn3:
        if st.button("♻️ Reset", use_container_width=True):
            _plr_reset()
            st.rerun()

    if refresh_btn:
        try:
            st.cache_resource.clear()
            st.success("Cache purged. Next Analyse runs fresh.")
        except Exception as e:
            st.warning(f"Purge failed: {e}")

    if run_btn:
        recommender = GlacierRecommender(
            ollama_model=(sel_model if sel_model != "(disable LLM)"
                          else "qwen2.5:7b"),
            use_llm=(sel_model != "(disable LLM)" and _llm_avail),
            debug_llm=debug_llm,
            use_grounded=use_grounded,
            allow_prior_inference=allow_prior,
            cascade_mode=cascade_mode)
        st.session_state["_glacier_live_recommender"] = recommender
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(i, n, title):
            progress.progress(i / max(n, 1))
            status.caption(f"Scanning {i}/{n}: {title}…")

        try:
            with st.spinner("Retrieving candidates + running cascade…"):
                bundle = recommender.recommend(
                    material=material, temp_c=float(temp_c),
                    stress_regime=_stress_arg, fabric=_fabric_arg,
                    sif_context=st.session_state.get("sif_context", {}),
                    progress_callback=_cb)
            _plr_set("bundle", bundle)
            st.session_state['glacier_recommender_bundle'] = bundle
            progress.empty()
            n_ctx = sum(
                sum(1 for c in bundle.candidates.get(p, [])
                    if getattr(c, 'context', False))
                for p in ALL_PARAMS)
            status.success(
                f"✅ Retrieved "
                f"{sum(len(v) for v in bundle.candidates.values())} "
                f"candidates ({n_ctx} context) · "
                f"backend `{bundle.retrieval_backend}` · "
                f"LLM {'yes' if bundle.llm_used else 'no'} · "
                f"mode `{cascade_mode}` · "
                f"coverage {bundle.coverage_five()}/5")
        except Exception as e:
            progress.empty()
            status.error(f"Recommendation failed: {e}")
            st.exception(e)

    bundle: Optional[GlacierRecommendationBundle] = _plr_get("bundle")
    if bundle is None:
        return

    st.caption(f"Retrieval: **{bundle.retrieval_backend}** · "
               f"LLM: **{'yes' if bundle.llm_used else 'no (heuristic)'}**")
    st.metric("Coverage (5-target)",
              f"{bundle.coverage_five()}/5",
              help="n, A₁, Q₁, T*, T_const")

    st.markdown("### Choose target values")
    for param in SIDEBAR_PARAM_ORDER:
        _plr_render_parameter_selector(param, bundle)

    with st.expander("🔧 Advanced parameters (A₂, Q₂, context)",
                     expanded=False):
        for param in ADVANCED_PARAM_ORDER + LAB_PARAM_ORDER:
            _plr_render_parameter_selector(param, bundle)

    if st.button("✅ Adopt for SIF generator", type="primary",
                 use_container_width=True):
        overrides = {}
        for param in ALL_PARAMS:
            v = st.session_state.get(f"{_PLR}{param}_value_si")
            if v is not None:
                overrides[param] = float(v)
        st.session_state["glacier_overrides"] = overrides
        st.success(f"Applied {len(overrides)} parameters. "
                   "Switch to the SIF Generator tab.")


# ============================================================================
# ███ SECTION 38 — MAIN STREAMLIT UI                                     ███
# ============================================================================

def main():
    st.set_page_config(page_title="Glacier Intelligent ElmerSolver",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem; color: #1E3A8A; text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] {
        height: 3rem; white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px; padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🏔️ Glacier Intelligent '
                'ElmerSolver Recommender (v10.1.1-glacier)</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #F0F9FF; padding: 1.5rem;
                border-radius: 10px; border-left: 5px solid #3B82F6;
                margin-bottom: 1rem;">
    <strong>✅ 1:1 architectural port of the nt-Cu v10.1.1 AI stack:</strong><br>
    • <span style="color: green;">🧠 10-EXPERT Latent MoE:</span>
      Material / Thermal / Strain / Method / Confidence / Reasoning +
      Regime (A) / Arrhenius (B) / Fabric (C) / Corpus Density.<br>
    • <span style="color: green;">🔀 UNION CASCADE + PIN-LIST:</span>
      every viable route runs; non-incumbent routes are greyed context.<br>
    • <span style="color: green;">🎯 6→3 PROVENANCE:</span>
      fine (llm_extract/reasoned/regex_ner/regime_prior/llm_prior/
      physics_inferred) → coarse (llm_grounded/llm_prior/deterministic).<br>
    • <span style="color: green;">📊 EXACT-XAI STACKED CHART:</span>
      no SHAP/LIME — every segment IS one expert's weighted contribution.<br>
    • <span style="color: green;">📚 PUBLICATION DASHBOARD:</span>
      radar / bar / Sankey / Treemap / histograms + journal presets.<br>
    • <span style="color: green;">🌡️ ARRHENIUS · GLEN · ENHANCEMENT LABS:</span>
      interactive physics exploration on top of the recommendation.<br>
    • <span style="color: green;">📄 SIF GENERATOR + ElmerSolver RUNNER:</span>
      write SIF → subprocess → live log.<br>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Execution Settings")
        sif_name = st.text_input("SIF filename", value="glacier_elmer.sif",
                                 key="side_sif_name")
        work_dir = st.text_input("Working directory",
                                 value=str(pathlib.Path.cwd()),
                                 key="side_work_dir")
        elmer_cmd = st.text_input("ElmerSolver command",
                                  value="ElmerSolver",
                                  key="side_elmer_cmd")

        st.markdown("---")
        st.header("📚 Corpus & LLM")
        corpus_dir = st.text_input(
            "Corpus folder", value="json_metadatabase",
            help="Directory containing JSON metadatabases.",
            key="side_corpus_dir")

        st.markdown("---")
        st.header("📂 Parse existing SIF (optional)")
        sif_context_path = st.text_input(
            "Path to .sif", value="",
            help="If provided, parsed values seed the recommender context.",
            key="side_sif_ctx_path")
        if st.button("📖 Parse SIF", use_container_width=True,
                     key="side_sif_parse_btn"):
            if not sif_context_path:
                st.warning("Enter a path first.")
            else:
                ctx = parse_elmer_sif_context(sif_context_path)
                st.session_state["sif_context"] = ctx
                if "error" in ctx:
                    st.error(f"SIF parse error: {ctx['error']}")
                else:
                    st.success(
                        f"Parsed SIF: mesh_db={ctx.get('mesh_db')}, "
                        f"n={ctx.get('glen_n')}, "
                        f"A1={ctx.get('rate_factor_1')}, "
                        f"Q1={ctx.get('activation_energy_1')}")

        st.markdown("---")
        with st.expander("🧠 AI Glacier Recommender v10.1.1",
                         expanded=False):
            render_recommender_sidebar(
                default_material="ice",
                default_temp_c=-10.0,
                default_stress="high",
                default_fabric="isotropic",
                ollama_model="qwen2.5:7b")

    # ── Main tabs ────────────────────────────────────────────────────
    tab_rec, tab_sif, tab_run, tab_arr, tab_glen, tab_enh, tab_tau, \
        tab_vis, tab_diag = st.tabs([
            "🤖 Recommender",
            "📄 SIF Generator",
            "🚀 Run ElmerSolver",
            "🌡️ Arrhenius Lab",
            "🧊 Glen Flow-Law Lab",
            "🧵 Enhancement Lab",
            "⛰️ Basal Stress Lab",
            "📊 Visuals Dashboard",
            "🔧 Diagnostics",
        ])

    with tab_rec:
        st.subheader("🤖 AI Recommender Summary")
        bundle = st.session_state.get("glacier_recommender_bundle")
        if bundle is None:
            st.info("Run the recommender from the sidebar (🔍 Analyse).")
        else:
            st.markdown("### Recommended values")
            cols = st.columns(len(SIDEBAR_PARAM_ORDER))
            for i, p in enumerate(SIDEBAR_PARAM_ORDER):
                with cols[i % len(cols)]:
                    best = bundle.best(p)
                    if best is None:
                        st.metric(GLACIER_ONTOLOGY[p]['label'], "—")
                    else:
                        st.metric(
                            GLACIER_ONTOLOGY[p]['label'],
                            glacier_fmt(p, best.value_si),
                            f"score {best.score:.2f}")
            st.markdown("### Coverage")
            n_ctx_total = sum(
                sum(1 for c in bundle.candidates.get(p, [])
                    if getattr(c, 'context', False))
                for p in ALL_PARAMS)
            st.caption(
                f"Backend: `{bundle.retrieval_backend}` · "
                f"LLM: `{'yes' if bundle.llm_used else 'no'}` · "
                f"candidates: `{sum(len(v) for v in bundle.candidates.values())}` · "
                f"context: `{n_ctx_total}` · "
                f"coverage: `{bundle.coverage_five()}/5`")
            if bundle.sif_context and "error" not in bundle.sif_context:
                with st.expander("📂 SIF context", expanded=False):
                    st.json(bundle.sif_context)

    with tab_sif:
        st.subheader("📄 Elmer SIF Generator")
        ov = st.session_state.get("glacier_overrides", {})
        ctx = st.session_state.get("sif_context", {})
        if not ov:
            st.info("No overrides adopted yet — using defaults. "
                    "Run the recommender and click 'Adopt for SIF "
                    "generator' in the sidebar.")
        params = build_sif_params_from_bundle(
            st.session_state.get("glacier_recommender_bundle"),
            ov, ctx)
        sif_content = generate_glacier_sif(params)
        with st.expander("📄 View Generated SIF", expanded=True):
            st.code(sif_content, language="plaintext")
        st.download_button("⬇️ Download SIF", data=sif_content,
                           file_name=sif_name, mime="text/plain")

    with tab_run:
        st.subheader("🚀 ElmerSolver Launcher")
        ov = st.session_state.get("glacier_overrides", {})
        ctx = st.session_state.get("sif_context", {})
        params = build_sif_params_from_bundle(
            st.session_state.get("glacier_recommender_bundle"),
            ov, ctx)
        sif_content = generate_glacier_sif(params)

        col1, col2 = st.columns([2, 5])
        with col1:
            run_btn = st.button("🚀 Run ElmerSolver", type="primary",
                                use_container_width=True)
        with col2:
            st.caption(
                f"Writes `{sif_name}` to `{work_dir}` and launches "
                f"`{elmer_cmd} {sif_name}`.")

        if run_btn:
            work_path = pathlib.Path(work_dir)
            sif_path = work_path / sif_name
            try:
                sif_path.write_text(sif_content)
                st.success(f"SIF written to `{sif_path}`")
            except Exception as e:
                st.error(f"Failed to write SIF: {e}")
                st.stop()
            cmd = [elmer_cmd, sif_name]
            st.info(f"Running: `{' '.join(cmd)}` in `{work_dir}`")
            output_area = st.empty()
            collected_lines: List[str] = []
            try:
                proc = subprocess.Popen(
                    cmd, cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                for line in proc.stdout:
                    collected_lines.append(line)
                    output_area.code("".join(collected_lines[-80:]),
                                     language="plaintext")
                proc.wait()
                if proc.returncode == 0:
                    st.success("✅ ElmerSolver finished successfully.")
                else:
                    st.error(f"❌ ElmerSolver exited with code "
                             f"{proc.returncode}")
            except FileNotFoundError:
                st.error(f"`{elmer_cmd}` not found.  Check your PATH or "
                         "provide an absolute path.")

    with tab_arr:
        render_arrhenius_lab()

    with tab_glen:
        render_glen_flow_law_lab()

    with tab_enh:
        render_enhancement_lab()

    with tab_tau:
        render_basal_stress_lab()

    with tab_vis:
        st.subheader("📊 Recommender Visuals Dashboard")
        bundle = st.session_state.get("glacier_recommender_bundle")
        if bundle is None:
            st.info("Run the recommender first.")
        else:
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart (v9.x provenance)",
                 "Stacked Latent-MoE (exact-XAI)",
                 "Radar",
                 "Sankey",
                 "Treemap",
                 "Histograms"],
                index=1, key="vis_chart_type")

            if chart_type == "Bar Chart (v9.x provenance)":
                param = st.selectbox("Parameter", ALL_PARAMS,
                                     key="vis_bar_param")
                cands = bundle.candidates.get(param, [])
                if cands:
                    legend_gran = st.radio(
                        "Legend granularity", ["coarse", "fine"],
                        index=0, horizontal=True, key="vis_bar_gran")
                    show_best_legend = st.checkbox(
                        "Show 'Best match' legend entry", value=False,
                        key="vis_bar_best_legend")
                    show_all_competitive = st.checkbox(
                        "🔓 Show all routes as competitive (audit)",
                        value=False, key="vis_bar_audit")
                    candidates = [c.value_si / GLACIER_ONTOLOGY[param]['ui_scale']
                                  for c in cands]
                    scores = [float(c.score) for c in cands]
                    provenance = [(c.provenance or c.method) for c in cands]
                    if show_all_competitive:
                        ctx_flags = [False] * len(cands)
                    else:
                        ctx_flags = [bool(getattr(c, 'context', False))
                                     for c in cands]
                    non_ctx = [(s if not f else -1e18)
                               for s, f in zip(scores, ctx_flags)]
                    best_idx = int(np.argmax(non_ctx)) \
                        if any(not f for f in ctx_flags) \
                        else int(np.argmax(scores))
                    fig = plot_candidate_scores(
                        candidates, scores, provenance, best_idx,
                        param_title=PARAM_META[param]['title'],
                        param_symbol=PARAM_META[param]['symbol'],
                        unit=PARAM_META[param]['unit'],
                        legend_granularity=legend_gran,
                        context_flags=ctx_flags,
                        show_best_legend=show_best_legend)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No candidates for this parameter.")

            elif chart_type == "Stacked Latent-MoE (exact-XAI)":
                param = st.selectbox("Parameter", ALL_PARAMS,
                                     key="vis_stack_param")
                cands = bundle.candidates.get(param, [])
                show_context = st.checkbox(
                    "🩶 Show context candidates in stacked chart",
                    value=False, key="vis_stack_ctx")
                if show_context:
                    ranked = list(cands)
                else:
                    ranked = [c for c in cands
                              if not getattr(c, 'context', False)]
                    if not ranked:
                        ranked = list(cands)
                fig = plot_stacked_latentmoe(
                    candidates=ranked,
                    param_title=PARAM_META[param]['title'],
                    param_symbol=PARAM_META[param]['symbol'],
                    unit=PARAM_META[param]['unit'],
                    show_context=show_context)
                if fig is not None:
                    st.pyplot(fig)
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=300,
                                bbox_inches='tight', pad_inches=0.05,
                                facecolor=fig.get_facecolor())
                    plt.close(fig)
                    st.download_button(
                        "⬇️ Download PNG (300 dpi)", data=buf.getvalue(),
                        file_name=f"stacked_moe_{param}.png",
                        mime="image/png")
                else:
                    st.warning("Stacked chart could not be generated.")

            elif chart_type == "Radar":
                categories = [GLACIER_ONTOLOGY[p]["symbol"]
                              for p in ALL_PARAMS]
                # Normalise to soft_range
                def _norm(p, v):
                    lo, hi = GLACIER_ONTOLOGY[p]["soft_range"]
                    if p in ("rate_factor_A1", "rate_factor_A2",
                             "critical_shear_rate"):
                        if v <= 0:
                            return 0.0
                        l_lo = np.log10(max(lo, 1e-30))
                        l_hi = np.log10(max(hi, 1e-30))
                        l_v = np.log10(max(v, 1e-30))
                        return float(np.clip((l_v - l_lo) / max(l_hi - l_lo, 1e-9),
                                             0, 1))
                    return float(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1))

                defaults_norm = [_norm(p, bundle.defaults[p])
                                 for p in ALL_PARAMS]
                bests_norm = []
                for p in ALL_PARAMS:
                    b = bundle.best(p)
                    bests_norm.append(
                        _norm(p, b.value_si if b is not None
                              else bundle.defaults[p]))
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=defaults_norm + [defaults_norm[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name='Defaults',
                    line=dict(color='#94a3b8', dash='dot')))
                fig.add_trace(go.Scatterpolar(
                    r=bests_norm + [bests_norm[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name='AI Recommended',
                    line=dict(color='#0072B2')))
                fig.update_layout(polar=dict(radialaxis=dict(
                    visible=True, range=[0, 1])),
                    height=600, width=700)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Sankey":
                sources = set()
                for p in ALL_PARAMS:
                    for c in bundle.candidates.get(p, []):
                        if c.source_file:
                            sources.add(c.source_file)
                source_list = sorted(sources) if sources else ["(none)"]
                param_list = ALL_PARAMS
                selected_list = [f"✅ {p}" for p in ALL_PARAMS]
                all_nodes = source_list + param_list + selected_list
                idx = {n: i for i, n in enumerate(all_nodes)}
                links_src, links_tgt, links_val = [], [], []
                for p in ALL_PARAMS:
                    for c in bundle.candidates.get(p, []):
                        if c.source_file in idx:
                            links_src.append(idx[c.source_file])
                            links_tgt.append(idx[p])
                            links_val.append(max(c.score, 0.01))
                for p in ALL_PARAMS:
                    b = bundle.best(p)
                    links_src.append(idx[p])
                    links_tgt.append(idx[f"✅ {p}"])
                    links_val.append(1.0)
                node_colors = ["#a5b4fc" if n in source_list else
                               "#fbbf24" if n in param_list else "#34d399"
                               for n in all_nodes]
                fig = go.Figure(data=[go.Sankey(
                    node=dict(pad=18, thickness=22, label=all_nodes,
                              color=node_colors),
                    link=dict(source=links_src, target=links_tgt,
                              value=links_val))])
                fig.update_layout(height=800, width=1200,
                                  title="Parameter source flow")
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Treemap":
                ids, labels, parents, values, colors = [], [], [], [], []
                palette = PUBLICATION_PALETTES['okabe_ito']
                for i, p in enumerate(ALL_PARAMS):
                    spec = GLACIER_ONTOLOGY[p]
                    ids.append(p)
                    labels.append(f"<b>{spec['symbol']}</b><br>{spec['label']}")
                    parents.append("")
                    values.append(0)
                    colors.append(palette[i % len(palette)])
                    for j, c in enumerate(bundle.candidates.get(p, [])):
                        c_id = f"{p}_{j}"
                        ids.append(c_id)
                        ctx = " 🩶" if getattr(c, 'context', False) else ""
                        labels.append(f"{glacier_fmt(p, c.value_si)}"
                                      f"{ctx}<br>{c.method}")
                        parents.append(p)
                        values.append(max(c.score, 0.01))
                        if getattr(c, 'context', False):
                            colors.append('rgb(150,150,150)')
                        else:
                            colors.append(f'rgb({int(200-80*c.score)},'
                                          f'{int(180-60*c.score)},'
                                          f'{int(80+120*c.score)})')
                fig = go.Figure(go.Treemap(
                    ids=ids, labels=labels, parents=parents,
                    values=values, branchvalues="total",
                    marker=dict(colors=colors)))
                fig.update_layout(height=800, width=1200)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histograms":
                available = [p for p in ALL_PARAMS
                             if bundle.candidates.get(p)]
                if available:
                    n_cols = min(3, len(available))
                    n_rows = (len(available) + n_cols - 1) // n_cols
                    fig = make_subplots(rows=n_rows, cols=n_cols,
                                        subplot_titles=[
                                            GLACIER_ONTOLOGY[p]['label']
                                            for p in available])
                    for i, p in enumerate(available):
                        r = i // n_cols + 1
                        c = i % n_cols + 1
                        spec = GLACIER_ONTOLOGY[p]
                        ui_vals = [cd.value_si / spec['ui_scale']
                                   for cd in bundle.candidates[p]]
                        fig.add_trace(go.Histogram(
                            x=ui_vals,
                            nbinsx=max(4, min(20, len(ui_vals) * 2))),
                            row=r, col=c)
                    fig.update_layout(height=350 * n_rows,
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tab_diag:
        st.subheader("🔧 Diagnostics")
        bundle = st.session_state.get("glacier_recommender_bundle")
        if bundle is None:
            st.info("Run the recommender first.")
        else:
            st.markdown("### Cascade diagnostics")
            st.dataframe(pd.DataFrame(bundle.diagnostics),
                         use_container_width=True, hide_index=True)
            st.markdown("### LLM cache")
            _rec = st.session_state.get("_glacier_live_recommender")
            if _rec is not None and hasattr(_rec, "client"):
                st.caption(f"Model: `{_rec.client.model}` · "
                           f"URL: `{_rec.client.url}`")


# ============================================================================
# ███ SECTION 39 — REGRESSION TEST SUITE                                 ███
# ============================================================================
# Gated behind GLACIER_REGRESSION=1 so module-level code re-execution on
# every Streamlit widget interaction doesn't run the suite repeatedly.
# ============================================================================

def _regression_test_v881() -> None:
    """Retriever + heuristic round-trip."""
    corpus = {"glacier.json": [
        {"material": "ice", "quantity": "glen exponent n",
         "value": 3.0, "unit": "dimensionless", "method": "creep test"},
    ]}
    r = HybridRetriever(corpus, use_dense=False)
    recs = r.search("glen_n", "ice", value_hints=["3.0"], k=3)
    assert recs, "HybridRetriever returned no records"
    hits = heuristic_extract(recs, "glen_n")
    assert hits, "heuristic_extract returned no candidates"
    assert any(abs(h["value"] - 3.0) < 1e-6 for h in hits)
    cands = gatekeep(hits, "glen_n", provenance="regex_ner")
    assert cands and abs(cands[0].value - 3.0) < 1e-6
    assert hasattr(cands[0], "reasoning")
    logger.info("v8.8.1 regression test: PASS")


def _regression_test_v882() -> None:
    """Sci-notation scanner + confidence cap."""
    scan_text = "A1 = 3.5e-25 Pa^-3 s^-1 and rate factor 2.4×10^-24 Pa^-3 s^-1"
    hits = list(_NUM_ANY.finditer(scan_text))
    assert any(abs(_num_value(m) - 3.5e-25) < 1e-30 for m in hits)
    fake = [{"value": 1.2e13, "unit": "MPa^-3 yr^-1", "confidence": 0.9,
             "property_label": "rate factor", "source": "test"}]
    prior_cands = gatekeep(fake, "rate_factor_A1",
                            provenance="llm_prior", conf_cap=0.5)
    assert prior_cands and prior_cands[0].confidence == 0.5
    assert _norm_provenance("llm_prior") == "llm_prior"
    logger.info("v8.8.2 regression test: PASS")


def _regression_test_v883() -> None:
    """Provenance taxonomy completeness."""
    for fine in FINE_PROVENANCE_KEYS:
        assert _legend_key(fine, 'coarse') in LEGEND_MARKERS
        assert _legend_key(fine) in COARSE_PROVENANCE_KEYS
        assert _legend_key(fine, 'fine') == fine
    assert _legend_key('heuristic') == 'deterministic'
    assert _legend_key('llm') == 'llm_grounded'
    assert _legend_key('derived') == 'deterministic'
    assert _norm_provenance('derived') == 'physics_inferred'
    logger.info("v8.8.3 regression test: PASS")


def _regression_test_v884() -> None:
    """Dataclass field ordering invariants."""
    import dataclasses as _dc
    names = [f.name for f in _dc.fields(GlacierCandidate)]
    for required in ("moe_breakdown", "context", "provenance",
                     "regime_tag", "reasoning"):
        assert required in names, f"GlacierCandidate missing {required!r}"

    # ValueCandidate must have `regime_tag` before `context`
    vc_names = [f.name for f in _dc.fields(ValueCandidate)]
    assert vc_names.index("regime_tag") < vc_names.index("context")
    logger.info("v8.8.4 regression test: PASS")


def _regression_test_v890() -> None:
    """Side-note parser must reject non-positive values; regimes must be
    ordered correctly; the MoE must prefer in-regime to out-of-regime."""
    bad_table = (
        "| Field | Value |\n"
        "| Rate factor | 0.000 |\n"
        "| Rate factor | 0 | MPa^-3 yr^-1 |\n"
    )
    rows = parse_side_note_table(bad_table)
    assert all(r["value"] > 0 for r in rows), \
        f"parse_side_note_table emitted a non-positive value: {rows}"

    r_cold = classify_glen_n_regime("high", "cold", "isotropic")
    r_low = classify_glen_n_regime("low", "cold", "isotropic")
    assert r_cold["bench"] > r_low["bench"], \
        "high-stress regime should sit at higher n than low-stress"

    exp = GlenNRegimeExpert()
    s_in = exp.score(3.0, "high", "cold", "isotropic")
    s_out = exp.score(1.0, "high", "cold", "isotropic")
    assert s_in > 0.5, f"in-regime score too low: {s_in}"
    assert s_out < 0.5, f"out-of-regime score too high: {s_out}"
    logger.info("v8.9.0 regression test: PASS")


def _regression_test_v90() -> None:
    """Regime classifier monotonicity."""
    r_small = classify_rate_factor_regime("cold", "isotropic")
    r_warm = classify_rate_factor_regime("temperate", "isotropic")
    assert r_warm["bench"] > r_small["bench"], \
        "warm A should exceed cold A"
    r_iso = classify_fabric_regime("isotropic", "cold")
    r_ani = classify_fabric_regime("anisotropic", "cold")
    assert r_ani["bench"] > r_iso["bench"], \
        "anisotropic E should exceed isotropic E"
    q_cold = classify_activation_q_regime(-30.0)
    q_warm = classify_activation_q_regime(-1.0)
    assert q_warm["bench"] > q_cold["bench"], \
        "warm Q should exceed cold Q"
    logger.info("v9.0 regression test: PASS")


def _regression_test_v91() -> None:
    """Arrhenius coupling: A(T) must be continuous at T*."""
    A1 = 1.14e-5; A2 = 6.046e28
    Q1 = 60000.0; Q2 = 139000.0
    T_star = -10.0
    A_cold_side = arrhenius_rate_factor(T_star - 1e-3, A1, A2, Q1, Q2, T_star)
    A_warm_side = arrhenius_rate_factor(T_star + 1e-3, A1, A2, Q1, Q2, T_star)
    # Both must be within a factor of ~1.001 of the crossover value
    ratio = max(A_cold_side, A_warm_side) / min(A_cold_side, A_warm_side)
    assert ratio < 1.01, f"A(T) discontinuous at T*: ratio {ratio}"

    # Temperature monotonicity: A increases with T
    A_cold = arrhenius_rate_factor(-40.0, A1, A2, Q1, Q2, T_star)
    A_warm = arrhenius_rate_factor(-1.0, A1, A2, Q1, Q2, T_star)
    assert A_warm > A_cold, "A(T) should increase with temperature"
    logger.info("v9.1 regression test: PASS")


def _regression_test_v92() -> None:
    """3-bucket rollup: total, exhaustive, ≤ 3 coarse entries."""
    for fine in FINE_PROVENANCE_KEYS:
        assert _legend_key(fine, 'coarse') in LEGEND_MARKERS
        assert _legend_key(fine) in COARSE_PROVENANCE_KEYS
        assert _legend_key(fine, 'fine') == fine

    for junk in ('', '   ', 'nonsense', 'Ω≈ç√∫', None, 42, 3.14):
        assert _legend_key(junk, 'coarse') in COARSE_PROVENANCE_KEYS

    fig = plot_candidate_scores(
        [3.0, 3.5, 2.5, 4.0], [0.80, 0.85, 0.60, 0.90],
        ['heuristic', 'llm', 'llm_prior', 'physics_inferred'],
        best_idx=3, legend_granularity='coarse',
        param_title='Glen n', param_symbol='n', unit=None,
        show_best_legend=True)
    texts = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert 'LLM (corpus-grounded)' in texts
    assert 'LLM prior (parametric)' in texts
    assert 'Deterministic (regex · physics-inferred)' in texts
    plt.close(fig)
    logger.info("✅ _regression_test_v92 passed")


def _regression_test_v921() -> None:
    """Ordering: physics_inferred > regime > prior > reasoned > regex."""
    for s in ('glacier_regime_prior', 'regime_prior_fallback',
              'arrhenius_regime_prior', 'fabric_regime'):
        assert _norm_provenance(s) == 'regime_prior', s
    assert _norm_provenance('arrhenius_inversion') == 'physics_inferred'
    assert _norm_provenance('glen_law_inversion') == 'physics_inferred'
    assert _norm_provenance('explicit') == 'llm_extract'
    assert _norm_provenance('derived') == 'physics_inferred'
    logger.info("✅ _regression_test_v921 passed")


def _regression_test_v922() -> None:
    """The `_provenance` tag passes through `gatekeep` cleanly."""
    fake = [{"value": 3.0, "unit": "dimensionless", "confidence": 0.9,
             "property_label": "glen exponent", "source": "test",
             "_context": True}]
    cands = gatekeep(fake, "glen_n", provenance="llm_extract")
    assert cands and cands[0].context is True

    fake2 = [{"value": 3.0, "unit": "dimensionless", "confidence": 0.9,
              "property_label": "glen exponent", "source": "test"}]
    cands2 = gatekeep(fake2, "glen_n", provenance="llm_extract")
    assert cands2 and cands2[0].context is False
    logger.info("✅ _regression_test_v922 passed")


def _regression_test_v930() -> None:
    """Union cascade invariants: pin-list, context flags."""
    for param, buckets in INCUMBENT_ROUTES.items():
        for b in buckets:
            assert b in COARSE_PROVENANCE_KEYS, \
                f"INCUMBENT_ROUTES[{param}] references unknown bucket {b!r}"

    for param, pinned in INCUMBENT_ROUTES.items():
        for b in COARSE_PROVENANCE_KEYS:
            for fk in PROVENANCE_GROUPS[b]:
                is_ctx = _is_context(param, fk)
                if b in pinned:
                    assert not is_ctx, \
                        f"_is_context({param!r}, {fk!r}) should be False"
                else:
                    assert is_ctx, \
                        f"_is_context({param!r}, {fk!r}) should be True"

    c_inc = GlacierCandidate(
        param='glen_n', value_si=3.0, raw_value=3.0, raw_unit='–',
        score=0.50, confidence=0.70, material='ice', temp_k=None,
        method='explicit', source_file='', source_title='', evidence='',
        provenance='regex_ner', context=False)
    c_ctx = GlacierCandidate(
        param='glen_n', value_si=3.5, raw_value=3.5, raw_unit='–',
        score=0.99, confidence=0.99, material='ice', temp_k=None,
        method='explicit', source_file='', source_title='', evidence='',
        provenance='llm_extract', context=True)
    from_bundle = GlacierRecommendationBundle(
        material='ice', temp_c=-10.0, stress_regime='high',
        fabric='isotropic',
        candidates={'glen_n': [c_inc, c_ctx]},
        defaults={'glen_n': 3.0},
        retrieval_backend='test', llm_used=False)
    winner = from_bundle.best('glen_n')
    assert winner is c_inc, \
        "context candidate must not win — pin-list guarantee"
    logger.info("✅ _regression_test_v930 passed")


def _regression_test_v931() -> None:
    """Honest naming + audit toggle semantics."""
    assert 'physics_inferred' in FINE_PROVENANCE_KEYS
    assert 'derived' not in FINE_PROVENANCE_KEYS
    assert 'physics_inferred' in PROVENANCE_GROUPS['deterministic']
    assert 'physics_inferred' in PROVENANCE_MARKERS

    assert _norm_provenance('derived') == 'physics_inferred'
    assert _norm_provenance('arrhenius_inversion') == 'physics_inferred'
    assert _norm_provenance('glen_law_inversion') == 'physics_inferred'
    assert _legend_key('derived', 'coarse') == 'deterministic'
    logger.info("✅ _regression_test_v931 passed")


def _regression_test_v10() -> None:
    """v10 MoE invariants: weights sum to 1.00; experts have colors."""
    sc = GlacierLatentMoEScorer()
    w_sum = (sc.w_material + sc.w_thermal + sc.w_strain + sc.w_method
             + sc.w_confidence + sc.w_reasoning + sc.w_regime
             + sc.w_arrhenius + sc.w_fabric + sc.w_density)
    assert abs(w_sum - 1.00) < 1e-9, \
        f"weight vector must sum to 1.00, got {w_sum}"
    for expert in GlacierLatentMoEScorer.EXPERT_ORDER:
        assert expert in GlacierLatentMoEScorer.EXPERT_COLORS, \
            f"expert {expert!r} has no palette entry"
    logger.info("✅ _regression_test_v10 passed")


def _regression_test_v1011() -> None:
    """v10.1.1 fixes: routes non-context, no '⭐ BEST', opt-in Best match."""
    # 1. σ₀-equivalent physics_inferred must be the incumbent for all
    #    deterministic pinned params
    for param in ('glen_n', 'rate_factor_A1', 'activation_energy_Q1',
                  'glen_enhancement', 'critical_shear_rate',
                  'ice_density', 'gravity'):
        assert _is_context(param, 'physics_inferred') is False, \
            f"{param}: physics_inferred must be non-context"

    # 2. Stacked chart contains no '⭐ BEST'
    sc = GlacierLatentMoEScorer()
    b = sc.score(
        [{"param": "glen_n", "value": 3.0, "unit": "dimensionless",
          "material": "ice", "temp": 263.0, "method": "explicit",
          "confidence": 0.9, "reasoning": "", "evidence": "",
          "_source_file": "t", "_source_title": "t",
          "_provenance": "llm_extract", "_context": False}],
        target_material="ice", target_temp_c=-10.0)
    bucket = b.get("glen_n", [])
    fig = plot_stacked_latentmoe(
        candidates=bucket, param_title='Glen n', param_symbol='n',
        unit=None)
    assert fig is not None
    all_text = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not any('BEST' in t for t in all_text), \
        f"stacked chart still has '⭐ BEST': {all_text}"
    plt.close(fig)

    # 3. 'Best match' legend entry is opt-in
    fig = plot_candidate_scores(
        [1.0, 2.0, 3.0], [0.8, 0.9, 0.7],
        ['regex_ner', 'physics_inferred', 'llm_prior'],
        best_idx=1, legend_granularity='coarse',
        show_best_legend=False)
    texts = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert 'Best match' not in texts, \
        f"'Best match' should be suppressed by default: {texts}"
    plt.close(fig)

    fig = plot_candidate_scores(
        [1.0, 2.0, 3.0], [0.8, 0.9, 0.7],
        ['regex_ner', 'physics_inferred', 'llm_prior'],
        best_idx=1, legend_granularity='coarse',
        show_best_legend=True)
    texts = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert 'Best match' in texts, \
        f"'Best match' should appear when requested: {texts}"
    plt.close(fig)
    logger.info("✅ _regression_test_v1011 passed")


def _run_regression_suite() -> None:
    """Run the whole battery; each test logs PASS/FAIL independently."""
    suite = [
        ('v8.8.1',  '_regression_test_v881'),
        ('v8.8.2',  '_regression_test_v882'),
        ('v8.8.3',  '_regression_test_v883'),
        ('v8.8.4',  '_regression_test_v884'),
        ('v8.9.0',  '_regression_test_v890'),
        ('v9.0',    '_regression_test_v90'),
        ('v9.1',    '_regression_test_v91'),
        ('v9.2.0',  '_regression_test_v92'),
        ('v9.2.1',  '_regression_test_v921'),
        ('v9.2.2',  '_regression_test_v922'),
        ('v9.3.0',  '_regression_test_v930'),
        ('v9.3.1',  '_regression_test_v931'),
        ('v10.0.0', '_regression_test_v10'),
        ('v10.1.1', '_regression_test_v1011'),
    ]
    for tag, name in suite:
        fn = globals().get(name)
        if fn is None:
            logger.warning("regression %s missing (%s undefined)", tag, name)
            continue
        try:
            fn()
        except AssertionError as ae:
            logger.error("❌ regression %s FAILED: %s", tag, ae)
        except Exception as e:
            logger.error("❌ regression %s ERRORED: %s\n%s",
                         tag, e, traceback.format_exc())


if os.environ.get("GLACIER_REGRESSION", "").lower() in ("1", "true", "yes"):
    _run_regression_suite()


if __name__ == "__main__":
    main()
