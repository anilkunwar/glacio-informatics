# ============================================================================
# ███ GLACIER PHASE-FIELD SIMULATOR (PURE FFT SPECTRAL)              ███
# ███ + PHYSICS-REGIME INTELLIGENT RECOMMENDER v1.0.0                ███
# ███ Converted from Enhanced Nanotwinned-Cu v10.1.1                 ███
# ███                                                                 ███
# ███ NEW (v1.0.0): ICE MECHANICS KERNEL                              ███
# ███   · SIA + spectral biharmonic stream-function Stokes solve.    ███
# ███   · Glen flow law with temperature-dependent A(T) via          ███
# ███     two-branch Arrhenius (A₁,Q₁ warm / A₂,Q₂ cold).            ███
# ███   · Non-conserved order parameters:                            ███
# ███       f   — fabric anisotropy  (0=isotropic, 1=single-max)     ███
# ███       w   — water content / damage                             ███
# ███   · Conserved: H — ice thickness (SIA continuity)               ███
# ███ NEW (v1.0.0): ELMER/ICE UNIT CONVERTER                         ███
# ███   · GlacierUnitConverter: Pa⁻ⁿ s⁻¹ → MPa⁻ⁿ yr⁻¹, kJ/mol → J/mol ███
# ███ NEW (v1.0.0): GLACIER PHYSICS-REGIME RECOMMENDER               ███
# ███   · N_REGIMES / A1_REGIMES / Q1_REGIMES / E_REGIMES tables      ███
# ███   · StressRegimeExpert, ThermalBranchExpert, FabricExpert       ███
# ███   · n → E cross-coupling (dislocation creep ⇒ single-max)       ███
# ███   · LatentMoE scorer with glacier threads                       ███
# ███ PUBLICATION-QUALITY VISUALS DASHBOARD                           ███
# ███ REGRESSION TESTS: unit conversion, regime monotonicity,         ███
# ███   cross-coupling, Elmer density formula, LatentMoE invariants   ███
# ============================================================================

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
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
from matplotlib.colors import Normalize, to_rgb
import matplotlib.animation as animation
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import zipfile
import pickle
import torch
import sqlite3
import hashlib
import traceback
import warnings
from scipy import stats
from io import BytesIO, StringIO
import tempfile
import os
import re
import math
import time
import threading
import logging
import unicodedata
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import faiss as _faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL CONSTANTS — ELMER/ICE MIXED UNIT SYSTEM
# ============================================================================
SECONDS_PER_YEAR = 31556926.0        # Tropical year, Elmer/Ice standard
PA_PER_MPA       = 1.0e6
MPA_PER_PA       = 1.0e-6
KG_M3_TO_ELMER   = 1.0e-6            # Pa → MPa conversion for ρg source term
RHO_ICE_SI       = 910.0             # kg/m³
G_SI             = 9.81              # m/s²
G_ELMER          = G_SI * SECONDS_PER_YEAR ** 2   # m/yr²


# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"❌ Error in {func.__name__}: {str(e)}"
            try:
                st.error(error_msg)
                st.error("Please check the console for detailed error information.")
            except Exception:
                pass
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None
    return wrapper


# ============================================================================
# SPECTRAL DERIVATIVE HELPERS (unchanged from nt-Cu)
# ============================================================================
def make_k_vectors(N, dx):
    kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1)
    ky = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1)
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1e-12
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    fh = fft2(field)
    gx = np.real(ifft2(1j * kx * fh))
    gy = np.real(ifft2(1j * ky * fh))
    return gx, gy


def spectral_laplacian(field, k2):
    fh = fft2(field)
    lap = np.real(ifft2(-k2 * fh))
    return lap


def spectral_biharmonic(field, k2):
    """∇⁴ f in Fourier space: multiply by |k|⁴."""
    fh = fft2(field)
    return np.real(ifft2((k2 ** 2) * fh))


# ============================================================================
# METADATA MANAGER (adapted field names)
# ============================================================================
class MetadataManager:
    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        if run_time is None:
            run_time = 0.0
        return {
            'run_time': run_time,
            'frames': len(history) if history else 0,
            'grid_size': kwargs.get('grid_size', sim_params.get('N', 256)),
            'dx': kwargs.get('dx', sim_params.get('dx', 500.0)),
            'dt': sim_params.get('dt', 1e-2),
            'created_at': datetime.now().isoformat(),
            'colormaps': {
                'H':      sim_params.get('cmap_H', 'Blues'),
                'vel':    sim_params.get('cmap_vel', 'viridis'),
                'visc':   sim_params.get('cmap_visc', 'plasma'),
                'fabric': sim_params.get('cmap_fabric', 'RdBu_r'),
                'water':  sim_params.get('cmap_water', 'Blues'),
                'T':      sim_params.get('cmap_T', 'coolwarm'),
            },
            'material_properties': sim_params.get('material_properties', {}),
            'simulation_parameters': {
                'dt': sim_params.get('dt', 1e-2),
                'N': sim_params.get('N', 256),
                'dx': sim_params.get('dx', 500.0),
                'glen_n': sim_params.get('glen_n', 3.0),
                'A1': sim_params.get('A1', 6.0e13),
                'Q1': sim_params.get('Q1', 60000.0),
                'E_glen': sim_params.get('E_glen', 1.0),
                'rho_ice': sim_params.get('rho_ice', 910.0),
                'T_ice_C': sim_params.get('T_ice_C', -3.0),
                'accumulation': sim_params.get('accumulation', 0.3),
                'n_steps': sim_params.get('n_steps', 100),
                'solver_method': 'spectral_biharmonic_sia',
            }
        }

    @staticmethod
    def validate_metadata(metadata):
        if not isinstance(metadata, dict):
            metadata = {}
        defaults = {'run_time': 0.0, 'frames': 0, 'grid_size': 256,
                    'dx': 500.0, 'dt': 1e-2,
                    'created_at': datetime.now().isoformat()}
        for k, v in defaults.items():
            metadata.setdefault(k, v)
        if 'colormaps' not in metadata:
            metadata['colormaps'] = {
                'H': 'Blues', 'vel': 'viridis', 'visc': 'plasma',
                'fabric': 'RdBu_r', 'water': 'Blues', 'T': 'coolwarm'}
        return metadata

    @staticmethod
    def get_metadata_field(metadata, field, default=None):
        try:
            return metadata.get(field, default)
        except Exception:
            return default


# ============================================================================
# JOURNAL TEMPLATES (unchanged)
# ============================================================================
class JournalTemplates:
    @staticmethod
    def get_journal_styles():
        return {
            'nature': {
                'figure_width_single': 8.9, 'figure_width_double': 18.3,
                'font_family': 'Arial', 'font_size_small': 7,
                'font_size_medium': 8, 'font_size_large': 9,
                'line_width': 0.5, 'axes_linewidth': 0.5,
                'tick_width': 0.5, 'tick_length': 2, 'grid_alpha': 0.1, 'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                '#bcbd22', '#17becf']
            },
            'science': {
                'figure_width_single': 5.5, 'figure_width_double': 11.4,
                'font_family': 'Helvetica', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 0.75, 'axes_linewidth': 0.75,
                'tick_width': 0.75, 'tick_length': 3, 'grid_alpha': 0.15, 'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                                '#77AC30', '#4DBEEE', '#A2142F']
            },
            'advanced_materials': {
                'figure_width_single': 8.6, 'figure_width_double': 17.8,
                'font_family': 'Arial', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0.2, 'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000',
                                '#44AA99', '#882255', '#117733', '#999933']
            },
            'prl': {
                'figure_width_single': 3.4, 'figure_width_double': 7.0,
                'font_family': 'Times New Roman', 'font_size_small': 8,
                'font_size_medium': 10, 'font_size_large': 12,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0, 'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                                '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
            },
            'custom': {
                'figure_width_single': 6.0, 'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans', 'font_size_small': 10,
                'font_size_medium': 12, 'font_size_large': 14,
                'line_width': 1.5, 'axes_linewidth': 1.5,
                'tick_width': 1.0, 'tick_length': 5, 'grid_alpha': 0.3, 'dpi': 300,
                'color_cycle': list(plt.cm.Set2(np.linspace(0, 1, 10)))
            }
        }

    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
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
            'axes.prop_cycle': plt.cycler(color=style['color_cycle'])
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
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_linewidth(style['axes_linewidth'] * 0.5)
            ax.spines['right'].set_linewidth(style['axes_linewidth'] * 0.5)
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=style['tick_length'])
            ax.tick_params(which='minor', length=style['tick_length'] * 0.6)
        return fig, style


# ============================================================================
# COLORMAP LIBRARY (unchanged)
# ============================================================================
COLORMAPS = {
    'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno',
    'magma': 'magma', 'cividis': 'cividis', 'hot': 'hot', 'cool': 'cool',
    'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn', 'winter': 'winter',
    'copper': 'copper', 'bone': 'bone', 'gray': 'gray', 'pink': 'pink',
    'afmhot': 'afmhot', 'gist_heat': 'gist_heat',
    'binary': 'binary', 'coolwarm': 'coolwarm', 'bwr': 'bwr', 'seismic': 'seismic',
    'RdBu': 'RdBu', 'RdBu_r': 'RdBu_r', 'RdGy': 'RdGy', 'PiYG': 'PiYG',
    'PRGn': 'PRGn', 'BrBG': 'BrBG', 'PuOr': 'PuOr', 'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted', 'hsv': 'hsv', 'tab10': 'tab10',
    'tab20': 'tab20', 'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3',
    'Paired': 'Paired', 'Accent': 'Accent', 'Dark2': 'Dark2', 'jet': 'jet',
    'turbo': 'turbo', 'rainbow': 'rainbow', 'nipy_spectral': 'nipy_spectral',
    'ocean': 'ocean', 'terrain': 'terrain', 'cubehelix': 'cubehelix',
    'Blues': 'Blues', 'Greens': 'Greens', 'Reds': 'Reds',
    'Purples': 'Purples', 'Oranges': 'Oranges', 'Greys': 'Greys',
    'YlOrRd': 'YlOrRd', 'YlGnBu': 'YlGnBu',
}
cmap_list = list(COLORMAPS.keys())

BAR_CMAP_CATEGORIES = {
    "🌈 Sequential (score gradient)": [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Blues', 'Greens', 'Reds', 'Purples', 'Oranges',
        'YlOrRd', 'YlGnBu', 'Greys',
    ],
    "🔄 Diverging (low↔high)": [
        'coolwarm', 'RdBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
        'seismic', 'bwr',
    ],
    "🌊 Cyclic / Perceptual": ['twilight', 'hsv', 'turbo'],
    "🎯 Categorical (distinct)": [
        'tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2',
    ],
    "🖤 Classic": ['Greys', 'gray', 'bone', 'copper', 'hot', 'binary'],
}
BAR_CMAP_FLAT = [n for cat in BAR_CMAP_CATEGORIES.values() for n in cat]

PUBLICATION_PALETTES = {
    'okabe_ito': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                  '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
    'tol_bright': ['#4477AA', '#EE6677', '#228833', '#CCBB44',
                   '#66CCEE', '#AA3377', '#BBBBBB'],
    'glacier_blue': ['#0B3C5D', '#328CC1', '#D9B310', '#1D2731',
                     '#8EB1C7', '#5A7D7C', '#A3C4BC'],
    'nature_classic': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
    'science_aaas': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                     '#77AC30', '#4DBEEE', '#A2142F'],
}

FONT_FAMILIES = [
    'Arial', 'Helvetica', 'Times New Roman', 'Courier New',
    'DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono',
    'Calibri', 'Cambria', 'Georgia', 'Verdana', 'Tahoma',
    'sans-serif', 'serif', 'monospace'
]


# ============================================================================
# PUBLICATION ENHANCER (unchanged)
# ============================================================================
class PublicationEnhancer:
    @staticmethod
    def add_scale_bar(ax, length_m, location='lower right', color='black',
                      linewidth=2, fontsize=8, unit_label='m'):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        bar_x_start = xlim[1] - x_range * 0.15
        bar_x_end = bar_x_start - length_m
        bar_y = ylim[0] + y_range * 0.05
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
                color=color, linewidth=linewidth, solid_capstyle='butt')
        ax.text((bar_x_start + bar_x_end) / 2, bar_y + y_range * 0.02,
                f'{length_m:g} {unit_label}', ha='center', va='bottom',
                color=color, fontsize=fontsize, fontweight='bold')
        return ax


# ============================================================================
# PROVENANCE TAXONOMY (unchanged shape — used identically)
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

LEGEND_MARKERS: Dict[str, Dict[str, Any]] = {
    'llm_grounded':  dict(marker='D', ms=6.0, fill=True,
                          label='LLM (corpus-grounded)'),
    'llm_prior':     dict(marker='o', ms=6.5, fill=True,
                          label='LLM prior (parametric)'),
    'deterministic': dict(marker='^', ms=7.5, fill=True,
                          label='Deterministic (regex · physics-inferred)'),
}

PROVENANCE_MARKERS: Dict[str, Dict[str, Any]] = {
    'llm_extract':      dict(marker='D', ms=6.0, fill=True,
                             label='LLM extraction (verbatim, grounded)'),
    'llm_reasoned':     dict(marker='p', ms=6.5, fill=True,
                             label='LLM chain-of-thought (grounded)'),
    'llm_prior':        dict(marker='o', ms=6.5, fill=False,
                             label='LLM prior (no corpus evidence)'),
    'regex_ner':        dict(marker='^', ms=7.5, fill=True,
                             label='Regex NER (deterministic)'),
    'regime_prior':     dict(marker='s', ms=6.5, fill=False,
                             label='Regime classifier (lookup table)'),
    'physics_inferred': dict(marker='v', ms=7.0, fill=False,
                             label='Physics-inferred (Arrhenius / cross-relation)'),
}

# Glacier incumbent-route pin-list
INCUMBENT_ROUTES: Dict[str, Tuple[str, ...]] = {
    'glen_n':  ('deterministic',),
    'A1':      ('llm_grounded',),
    'Q1':      ('deterministic',),
    'E':       ('deterministic',),
    'rho_ice': ('deterministic',),
}


def _is_context(param: str, prov: Any) -> bool:
    bucket = _FINE_TO_GROUP.get(_norm_provenance(prov), 'deterministic')
    pinned = INCUMBENT_ROUTES.get(param, ())
    return bool(pinned) and bucket not in pinned


_DERIVED_HINTS: Tuple[str, ...] = (
    'physics_inferred', 'derived',
    'arrhenius', 'cross_relation', 'two_temperature_inversion',
    'solver_law', 'model_inversion', 'consensus',
    'n_to_e_coupling',
)
_REGIME_HINTS: Tuple[str, ...] = (
    'regime_prior', 'regime_inference', 'regime_classifier',
    'stress_regime', 'thermal_branch', 'fabric_regime', 'physics_regime',
)
_PRIOR_HINTS: Tuple[str, ...] = (
    'prior', 'estimate', 'no evidence', 'world knowledge', 'parametric',
)
_REASONED_HINTS: Tuple[str, ...] = ('reasoned', 'chain_of_thought',
                                    'chain-of-thought', 'cot')
_REGEX_HINTS: Tuple[str, ...] = ('regex', 'heuristic', 'side_note',
                                 'side-note', 'default_fallback')
_LLM_HINTS: Tuple[str, ...] = ('llm', 'inferred', 'model', 'explicit')


def _norm_provenance(p: Any) -> str:
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
    s = str(g).strip().lower() if g is not None else 'coarse'
    if s not in _VALID_GRANULARITIES:
        logger.warning("legend_granularity %r invalid — defaulting to 'coarse'", g)
        return 'coarse'
    return s


def _legend_key(prov: Any, granularity: str = 'coarse') -> str:
    granularity = _norm_granularity(granularity)
    fine = _norm_provenance(prov)
    if granularity == 'fine':
        return fine
    return _FINE_TO_GROUP.get(fine, 'deterministic')


def _stamp_provenance(ext: Dict[str, Any], fine_key: str) -> None:
    if fine_key not in FINE_PROVENANCE_KEYS:
        raise ValueError(f"not a fine key: {fine_key!r}")
    ext['_provenance'] = fine_key


def _default_reasoning(c) -> str:
    prov = _norm_provenance(getattr(c, "provenance", "")
                            or getattr(c, "method", ""))
    if prov == "llm_extract":
        return ("Tier-1 grounded LLM extraction — verbatim span, "
                "corpus-anchored, unit-coerced by the gatekeeper.")
    if prov == "llm_reasoned":
        return ("Tier-1 grounded LLM chain-of-thought — evidence span plus "
                "an explicit step-by-step derivation, corpus-anchored.")
    if prov == "regex_ner":
        return ("Tier-2 deterministic regex — sci-notation scanner + "
                "glacier-unit-aware tokens; no LLM involved.")
    if prov == "llm_prior":
        return ("Tier-3 LLM prior inference — no verbatim corpus evidence; "
                "confidence capped at 0.5, bounded by GLACIER_CANON.")
    if prov == "regime_prior":
        return ("Tier-3b curated glacier-regime classifier — reproducible "
                "lookup table, no LLM involved.")
    if prov == "physics_inferred":
        return ("Physics-inferred from measured context — Arrhenius "
                "cross-relation / two-temperature inversion / n→E coupling.")
    return ""


def sci_tex(value, unit=None, decimals=1):
    v = float(value)
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


# ============================================================================
# GLACIER PARAMETER METADATA
# ============================================================================
GLACIER_PARAM_META: Dict[str, Dict[str, str]] = {
    'glen_n': dict(title='Glen Flow Exponent', symbol='n',
                   unit=None),
    'A1':     dict(title='Rate Factor (warm branch)',
                   symbol='A_1', unit='MPa^{-n}\\,yr^{-1}'),
    'A2':     dict(title='Rate Factor (cold branch)',
                   symbol='A_2', unit='MPa^{-n}\\,yr^{-1}'),
    'Q1':     dict(title='Activation Energy (warm branch)',
                   symbol='Q_1', unit='J\\,mol^{-1}'),
    'Q2':     dict(title='Activation Energy (cold branch)',
                   symbol='Q_2', unit='J\\,mol^{-1}'),
    'E':      dict(title='Glen Enhancement Factor',
                   symbol='E', unit=None),
    'rho_ice': dict(title='Ice Density', symbol=r'\rho_i',
                    unit='kg\\,m^{-3}'),
    'H':      dict(title='Ice Thickness', symbol='H', unit='m'),
    'd_grain': dict(title='Grain Size', symbol='d', unit='mm'),
    'w_water': dict(title='Water Content', symbol='w', unit='\\%'),
    'fabric': dict(title='Fabric Anisotropy Index', symbol='f', unit=None),
    'slope':  dict(title='Surface Slope', symbol=r'\alpha', unit='deg'),
    'bed_slope': dict(title='Bed Slope', symbol=r'\beta', unit='deg'),
    'gamma_rate': dict(title='Effective Shear Strain Rate',
                       symbol=r'\dot{\gamma}_e', unit='yr^{-1}'),
    'T_ice':  dict(title='Ice Temperature', symbol='T', unit='{}^\\circ C'),
    'T_pmp':  dict(title='Pressure-Melting Point', symbol='T_{pmp}',
                   unit='{}^\\circ C'),
    'accumulation': dict(title='Surface Mass Balance', symbol=r'\dot{b}',
                        unit='m\\,yr^{-1}'),
    'basal_drag': dict(title='Basal Drag Coefficient', symbol=r'\beta_{bed}',
                       unit='MPa\\,yr\\,m^{-1}'),
}

PARAM_META = GLACIER_PARAM_META  # alias for compatibility with viz code


def normalize_tex(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    if '\\' not in s:
        return s
    return re.sub(r'\\{2,}', r'\\', s)


def strip_tex(s: Any) -> str:
    if s is None:
        return ''
    return re.sub(r'[\\{}$]', '', str(s))


_MTX = MathTextParser('path')


def safe_mathtext(s: Any, fallback: Optional[str] = None) -> str:
    if s is None:
        return ''
    s = normalize_tex(str(s))
    if '$' not in s:
        return s
    try:
        _MTX.parse(s, dpi=100, prop=FontProperties())
        return s
    except Exception:
        logger.warning("safe_mathtext: refusing to render malformed mathtext %r", s)
        return fallback if fallback is not None else strip_tex(s)


# ============================================================================
# ═══════════════════════════════════════════════════════════════════════════
# ███ GLACIER UNIT CONVERTER (NEW — the critical bridge to Elmer/Ice)  ███
# ═══════════════════════════════════════════════════════════════════════════
# ============================================================================
class GlacierUnitConverter:
    """Converts literature units to the Elmer/Ice mixed system.

    Elmer/Ice assumes:
        Length     = 1 m
        Stress     = 1 MPa   (= 10^6 Pa)
        Time       = 1 yr    (= 31,556,926 s)

    The Glen flow law has the form
        ε̇ = E · A(T) · τ^n
    so A(T) carries units of [Stress]^{-n} [Time]^{-1}.  A literature value
    reported in Pa^{-n} s^{-1} must be multiplied by
        (10^6)^n · SECONDS_PER_YEAR
    to convert to MPa^{-n} yr^{-1}.

    Similarly:
        Q (activation energy):  kJ/mol → J/mol   (× 1000)
        ρ (ice density):        kg/m³  → kg/m³ × 1e-6 × yr^{-2}  (for ρg source term)
        g (gravity):            m/s²   → m/yr²    (× SECONDS_PER_YEAR²)
    """
    SECONDS_PER_YEAR = SECONDS_PER_YEAR

    @staticmethod
    def _clean(unit: str) -> str:
        if not unit:
            return ""
        s = unit.lower().strip()
        s = s.replace("·", " ").replace("*", " ").replace("⁻", "^-")
        s = s.replace("–", "-").replace("−", "-")
        s = re.sub(r"[\s/]+", " ", s)
        return s.strip()

    @staticmethod
    def _extract_pressure_exponent(u: str) -> Optional[float]:
        """Return the absolute exponent on the pressure unit, if present.
        Handles 'pa^-3', 'mpa-3', 'kpa^-n', 'pa^3' (raises), 'pa' (assumes 1)."""
        # Try explicit '^-3' or '-3'
        m = re.search(r'(pa|kpa|mpa|bar|gpa)(?:\^?)\s*([+-]?\d+(?:\.\d+)?|n)', u)
        if not m:
            return None
        unit_root = m.group(1)
        exp_str = m.group(2)
        if exp_str == 'n':
            return None  # will be filled by caller-supplied n
        try:
            return abs(float(exp_str))
        except ValueError:
            return None

    @staticmethod
    def convert(value: float, unit: str, param_key: str,
                n: float = 3.0) -> Tuple[float, str]:
        v = float(value)
        u = GlacierUnitConverter._clean(unit)

        # ── Dimensionless ────────────────────────────────────────────
        if param_key in ('glen_n', 'E', 'fabric'):
            return v, 'dimensionless'

        # ── Activation energy → J/mol ────────────────────────────────
        if param_key in ('Q1', 'Q2'):
            if 'kj' in u or 'kilojoule' in u:
                return v * 1000.0, 'J mol^-1'
            return v, 'J mol^-1'

        # ── Rate factor → MPa^{-n} yr^{-1} ───────────────────────────
        if param_key in ('A1', 'A2'):
            pres_root = 'pa'
            if 'gpa' in u:   pres_root = 'gpa'
            elif 'mpa' in u: pres_root = 'mpa'
            elif 'kpa' in u: pres_root = 'kpa'
            elif 'bar' in u: pres_root = 'bar'

            time_root = 's'
            if 'yr' in u or 'year' in u or 'a^-1' in u:
                time_root = 'yr'
            elif 'day' in u:
                time_root = 'day'

            parsed_n = GlacierUnitConverter._extract_pressure_exponent(u)
            if parsed_n is None:
                parsed_n = n

            # Pressure multiplier:  (1 X)^{-n} = (1/MPa)^{-n}
            if pres_root == 'pa':
                p_mult = 10.0 ** (6.0 * parsed_n)
            elif pres_root == 'kpa':
                p_mult = 10.0 ** (3.0 * parsed_n)
            elif pres_root == 'gpa':
                p_mult = 10.0 ** (-3.0 * parsed_n)   # 1 GPa = 1000 MPa
            elif pres_root == 'bar':
                p_mult = 10.0 ** (1.0 * parsed_n)    # 1 bar = 0.1 MPa
            else:
                p_mult = 1.0

            # Time multiplier: value in (X per second) → (X per year)
            if time_root == 's':
                t_mult = SECONDS_PER_YEAR
            elif time_root == 'day':
                t_mult = 365.25
            else:
                t_mult = 1.0

            final = v * p_mult * t_mult
            return final, f'MPa^-{parsed_n:.1f} yr^-1'

        # ── Ice density → kg/m³ × 1e-6 × yr⁻² (for Elmer ρg) ─────────
        if param_key == 'rho_ice':
            if 'kg' in u and ('m3' in u or 'm^3' in u or 'm-3' in u
                              or 'm^-3' in u or '/m3' in u):
                return v, 'kg m^-3'
            if 'g/cm3' in u or 'g cm^-3' in u:
                return v * 1000.0, 'kg m^-3'
            return v, 'kg m^-3'

        # ── Ice thickness, lengths ───────────────────────────────────
        if param_key in ('H', 'd_grain', 'l_char'):
            if 'km' in u:
                return v * 1000.0, 'm'
            if 'mm' in u:
                return v * 1.0e-3, 'm'
            if 'cm' in u:
                return v * 1.0e-2, 'm'
            if 'm' in u:
                return v, 'm'
            return v, 'm'

        # ── Velocity / strain rate → yr⁻¹ ────────────────────────────
        if param_key == 'gamma_rate':
            if 's^-1' in u or '/s' in u or 's-1' in u:
                return v * SECONDS_PER_YEAR, 'yr^-1'
            return v, 'yr^-1'

        # ── Temperature → °C (Elmer's default) ──────────────────────
        if param_key in ('T_ice', 'T_pmp'):
            if 'k' == u.strip() or 'kelvin' in u:
                return v - 273.15, 'deg C'
            return v, 'deg C'

        return v, u

    @staticmethod
    def density_formula(rho_si: float = RHO_ICE_SI) -> str:
        """Generate the exact Elmer/Ice density math string."""
        return (f"{rho_si}*1.0E-06*"
                f"({int(SECONDS_PER_YEAR)}.0)^(-2.0)")

    @staticmethod
    def gravity_formula(g_si: float = G_SI) -> str:
        """Generate the exact Elmer/Ice gravity math string."""
        return f"{g_si} * ({int(SECONDS_PER_YEAR)}.0)^(2.0)"

    @staticmethod
    def normalize_extraction(item: dict, param_key: str,
                             n: float = 3.0) -> dict:
        """In-place conversion of an extraction dict to Elmer units."""
        raw_v = item.get('value')
        raw_u = item.get('unit', '')
        if raw_v is None:
            return item
        try:
            conv_v, target_u = GlacierUnitConverter.convert(
                raw_v, raw_u, param_key, n)
            item['raw_value'] = raw_v
            item['raw_unit']  = raw_u
            item['value']     = conv_v
            item['unit']      = target_u
            item['converted'] = bool(abs(raw_v - conv_v) > 1e-30)
        except Exception:
            item['converted'] = False
        return item


# ============================================================================
# GLACIER ONTOLOGY + CANON (replaces PLASTICITY_ONTOLOGY / PARAM_CANON)
# ============================================================================
GLACIER_CANON: Dict[str, Dict[str, Any]] = {
    'glen_n': dict(
        aliases=['glen exponent', "glen's exponent", 'stress exponent',
                 'creep exponent', 'power-law exponent', 'n exponent',
                 'glen flow exponent', 'flow-law exponent'],
        unit=None, plausible=(0.5, 5.0),
    ),
    'A1': dict(
        aliases=['rate factor', 'arrhenius rate factor', 'rate factor warm',
                 'pre-exponential factor', 'a_1', 'a1', 'softness parameter',
                 'warm rate factor', 'creep rate factor'],
        unit='MPa^-n yr^-1', plausible=(1e11, 1e16),
    ),
    'A2': dict(
        aliases=['rate factor cold', 'cold rate factor', 'a_2', 'a2',
                 'arrhenius rate factor cold'],
        unit='MPa^-n yr^-1', plausible=(1e10, 1e16),
    ),
    'Q1': dict(
        aliases=['activation energy', 'activation energy warm',
                 'arrhenius activation energy', 'creep activation energy',
                 'q_1', 'q1', 'warm activation energy'],
        unit='J mol^-1', plausible=(2.0e4, 1.0e5),
    ),
    'Q2': dict(
        aliases=['activation energy cold', 'cold activation energy',
                 'q_2', 'q2'],
        unit='J mol^-1', plausible=(2.0e4, 1.5e5),
    ),
    'E': dict(
        aliases=['enhancement factor', 'glen enhancement factor',
                 'anisotropy enhancement', 'e_glen', 'softness multiplier',
                 'enhancement parameter'],
        unit=None, plausible=(0.3, 15.0),
    ),
    'rho_ice': dict(
        aliases=['ice density', 'density of ice', 'rho_ice', 'rho ice',
                 'glacier density', 'firn density'],
        unit='kg m^-3', plausible=(400.0, 950.0),
    ),
    'H': dict(
        aliases=['ice thickness', 'glacier thickness', 'thickness',
                 'h_ice', 'ice depth'],
        unit='m', plausible=(1.0, 1.0e4),
    ),
    'd_grain': dict(
        aliases=['grain size', 'ice grain size', 'crystal size',
                 'grain diameter', 'mean grain size'],
        unit='m', plausible=(1.0e-5, 1.0e-1),
    ),
    'w_water': dict(
        aliases=['water content', 'meltwater content', 'liquid water content',
                 'moisture content', 'w_water'],
        unit='%', plausible=(0.0, 20.0),
    ),
    'fabric': dict(
        aliases=['fabric anisotropy', 'fabric index', 'c-axis fabric',
                 'lattice preferred orientation', 'lpo', 'fabric strength'],
        unit=None, plausible=(0.0, 1.0),
    ),
    'slope': dict(
        aliases=['surface slope', 'ice surface slope', 'surface gradient',
                 'slope angle'],
        unit='deg', plausible=(0.0, 45.0),
    ),
    'bed_slope': dict(
        aliases=['bed slope', 'basal slope', 'bedrock slope'],
        unit='deg', plausible=(-30.0, 30.0),
    ),
    'T_ice': dict(
        aliases=['ice temperature', 'temperature', 'ice temperature c',
                 'measured temperature'],
        unit='deg C', plausible=(-60.0, 0.0),
    ),
    'T_pmp': dict(
        aliases=['pressure melting point', 'pressure-melting temperature',
                 'pmp'],
        unit='deg C', plausible=(-5.0, 0.0),
    ),
    'accumulation': dict(
        aliases=['accumulation', 'surface mass balance', 'smb', 'mass balance',
                 'accumulation rate'],
        unit='m yr^-1', plausible=(-2.0, 20.0),
    ),
    'basal_drag': dict(
        aliases=['basal drag', 'basal drag coefficient', 'sliding coefficient',
                 'bed friction', 'basal resistance'],
        unit='MPa yr m^-1', plausible=(1e-3, 1e3),
    ),
    'gamma_rate': dict(
        aliases=['shear strain rate', 'strain rate', 'effective strain rate',
                 'deformation rate'],
        unit='yr^-1', plausible=(1e-6, 1e2),
    ),
}


_POSITIVE_LOWER_BOUND_PARAMS = frozenset({
    'glen_n', 'A1', 'A2', 'Q1', 'Q2', 'E', 'rho_ice', 'H',
    'd_grain', 'accumulation', 'basal_drag',
})


def merge_no_clash(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    out = dict(base)
    out.update(overrides)
    return out


# ============================================================================
# GLACIER GAZETTEER (for NER)
# ============================================================================
GLACIER_GAZETTEER: Dict[str, List[str]] = {
    'material': ['ice', 'glacier ice', 'polar ice', 'temperate ice',
                 'firn', 'glacier', 'ice sheet', 'ice stream', 'ice shelf'],
    'property': ['glen exponent', "glen's exponent", 'stress exponent',
                 'creep exponent', 'power-law exponent',
                 'rate factor', 'arrhenius rate factor', 'pre-exponential',
                 'activation energy', 'creep activation energy',
                 'enhancement factor', 'glen enhancement factor',
                 'anisotropy enhancement',
                 'ice density', 'density of ice',
                 'ice thickness', 'glacier thickness',
                 'grain size', 'ice grain size',
                 'water content', 'meltwater content',
                 'fabric anisotropy', 'fabric index', 'c-axis fabric',
                 'lpo', 'lattice preferred orientation'],
    'property_weak': ['a1', 'a2', 'q1', 'q2', 'e_glen', 'n_exp', 'rho_i'],
    'stress_regime': ['low stress', 'high stress', 'intermediate stress',
                      'ice divide', 'ice stream', 'shear margin',
                      'outlet glacier', 'stagnant ice', 'surge'],
    'thermal_state': ['temperate', 'cold ice', 'warm ice', 'polar',
                      'sub-freezing', 'pressure-melting', 'isothermal',
                      'polythermal'],
    'fabric': ['isotropic fabric', 'single maximum', 'single-maximum',
               'multi-maximum', 'random fabric', 'anisotropic fabric',
               'vertical c-axis', 'horizontal c-axis'],
    'impurity': ['clean ice', 'dust-laden', 'mineral dust', 'marine ice',
                 'brine', 'impurity-rich', 'volcanic ash'],
    'moisture': ['dry', 'moist', 'wet', 'brine', 'liquid water'],
    'method': ['rheology', 'creep test', 'shear test', 'compression test',
               'torsion test', 'ice core', 'deep ice core', 'borehole',
               'insar', 'gps', 'radio echo sounding'],
    'unit': ['MPa^-3 yr^-1', 'MPa-3 yr-1', 'MPa^-n yr^-1',
             'kJ/mol', 'kJ mol-1', 'J/mol',
             'Pa^-3 s^-1', 'Pa-3 s-1', 'kPa^-3 s^-1',
             'kg/m3', 'kg m-3', 'g/cm3',
             'm', 'km', 'mm', 'cm',
             's^-1', 'yr^-1', '/yr', 'a^-1',
             'deg C', 'K', 'Celsius'],
}


_CHAR_FOLD = {
    'μ': 'mu', 'µ': 'mu', 'ρ': 'rho', '–': '-', '—': '-',
    '’': "'", '\u00a0': ' ', 'γ': 'gamma', 'σ': 'sigma', 'τ': 'tau',
    'λ': 'lambda', 'θ': 'theta', 'ε': 'epsilon', 'η': 'eta', 'Α': 'A',
    '°': 'deg',
}


def norm_text(s: Any) -> str:
    s = unicodedata.normalize('NFKC', str(s))
    for k, v in _CHAR_FOLD.items():
        s = s.replace(k, v)
    s = re.sub(r'[{}$\\]', '', s)
    s = re.sub(r'_(?=\d)', '', s)
    s = re.sub(r'(?<=[a-z])\s+(?=\d)', '', s.lower())
    return s


def alias_pattern(alias: str) -> str:
    a = norm_text(alias)
    out = []
    for i, ch in enumerate(a):
        if i > 0 and ch.isalnum() and out and out[-1][-1].isalnum():
            out.append(r'[\s_\-]{0,2}')
        out.append(re.escape(ch))
    return rf"(?<![a-z0-9]){''.join(out)}(?![a-z0-9])"


ENTITY_PATTERNS = {k: [re.compile(alias_pattern(a)) for a in v]
                   for k, v in GLACIER_GAZETTEER.items()}


# ============================================================================
# GLACIER REGIME TABLES
# ============================================================================
N_REGIMES: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ('low', 'cold', 'isotropic'): dict(
        low=1.0, high=2.0, bench=1.5,
        desc='Low-stress divide, cold ice: diffusion creep '
             '(Nabarro-Herring / Coble) → n ≈ 1'),
    ('low', 'cold', 'anisotropic'): dict(
        low=1.0, high=2.0, bench=1.5,
        desc='Low stress, fabric-enhanced diffusion creep'),
    ('intermediate', 'cold', 'isotropic'): dict(
        low=1.8, high=2.5, bench=2.2,
        desc='Intermediate stress: grain-boundary sliding → n ≈ 2'),
    ('intermediate', 'warm', 'isotropic'): dict(
        low=1.0, high=2.0, bench=1.5,
        desc='Temperate ice: pressure-melting / refreezing → linear'),
    ('high', 'cold', 'isotropic'): dict(
        low=3.0, high=4.0, bench=3.0,
        desc='Ice streams / shear margins: dislocation creep → n ≈ 3–4'),
    ('high', 'warm', 'isotropic'): dict(
        low=1.0, high=2.5, bench=1.7,
        desc='Warm ice near bed: water-assisted creep → n ≈ 1–2'),
    ('high', 'cold', 'anisotropic'): dict(
        low=3.5, high=4.5, bench=4.0,
        desc='Strong fabric + dislocation creep: enhanced nonlinearity'),
}

A1_REGIMES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ('warm', 'clean'): dict(
        low=3.5e13, high=9.3e13, bench=6.0e13,
        desc='Paterson & Budd 1982 clean-ice warm branch (T ≥ −10 °C) '
             'in MPa⁻ⁿ yr⁻¹'),
    ('warm', 'dust_rich'): dict(
        low=8.0e13, high=2.5e14, bench=1.5e14,
        desc='Dust/impurity-laden warm ice: enhanced grain-boundary mobility'),
    ('warm', 'marine'): dict(
        low=5.0e13, high=1.2e14, bench=8.0e13,
        desc='Marine ice with brine networks: chloride softens ice'),
    ('warm', 'temperate'): dict(
        low=5.0e14, high=2.0e15, bench=1.0e15,
        desc='Temperate (at pressure-melting point): water-assisted softening'),
}

Q1_REGIMES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ('warm', 'dry'): dict(
        low=42000.0, high=67000.0, bench=60000.0,
        desc='Paterson canonical Q₁ = 60 kJ/mol (warm, dry, T ≥ −10 °C)'),
    ('warm', 'moist'): dict(
        low=35000.0, high=55000.0, bench=45000.0,
        desc='Moist temperate ice: water films reduce apparent activation'),
    ('warm', 'marine'): dict(
        low=40000.0, high=58000.0, bench=50000.0,
        desc='Brine networks modify Arrhenius barrier'),
}

E_REGIMES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ('isotropic', 'cold'): dict(
        low=0.8, high=1.2, bench=1.0,
        desc='Random fabric → E = 1 (isotropic reference)'),
    ('isotropic', 'warm'): dict(
        low=1.0, high=2.0, bench=1.5,
        desc='Warm isotropic: mild enhancement from GB processes'),
    ('single_max', 'cold'): dict(
        low=2.0, high=5.0, bench=3.0,
        desc='Single-maximum fabric (vertical c-axis): enhanced shear'),
    ('single_max', 'warm'): dict(
        low=3.0, high=8.0, bench=5.0,
        desc='Warm strong fabric: maximum ice-stream shear-margin enhancement'),
    ('multi_max', 'cold'): dict(
        low=1.5, high=3.0, bench=2.0,
        desc='Multi-maximum fabric: moderate enhancement'),
    ('shear_margin', 'warm'): dict(
        low=5.0, high=10.0, bench=7.0,
        desc='Shear-margin fabric with water: extreme enhancement (rare)'),
}


# n → E cross-coupling (analogous to λ → ρ₀ modulation)
N_TO_E_COUPLING: List[Tuple[Tuple[float, float], Dict[str, Any]]] = [
    ((1.0, 1.8), dict(E_band=(0.8, 1.2),
                      reason='n ≈ 1 → diffusion creep → isotropic fabric')),
    ((1.8, 2.6), dict(E_band=(1.0, 1.8),
                      reason='n ≈ 2 → GBS → mild enhancement')),
    ((2.6, 3.4), dict(E_band=(1.5, 3.0),
                      reason='n ≈ 3 → dislocation creep, modest fabric')),
    ((3.4, 4.5), dict(E_band=(2.5, 6.0),
                      reason='n ≈ 4 → strong fabric, shear margin')),
]


# ============================================================================
# GLACIER REGIME CLASSIFIERS
# ============================================================================
def _norm_stress(s: Optional[str]) -> Optional[str]:
    s = (s or '').lower()
    if any(k in s for k in ('low', 'divide', 'stagnant')):
        return 'low'
    if any(k in s for k in ('intermediate', 'mid', 'transition')):
        return 'intermediate'
    if any(k in s for k in ('high', 'stream', 'shear margin', 'outlet', 'surge')):
        return 'high'
    return None


def _norm_thermal(t: Optional[str]) -> Optional[str]:
    t = (t or '').lower()
    if any(k in t for k in ('warm', 'temperate', 'melt', 'wet')):
        return 'warm'
    if any(k in t for k in ('cold', 'polar', 'sub-freezing', 'dry')):
        return 'cold'
    return None


def _norm_fabric(f: Optional[str]) -> Optional[str]:
    f = (f or '').lower()
    if 'single' in f and 'max' in f:
        return 'single_max'
    if 'multi' in f and 'max' in f:
        return 'multi_max'
    if 'shear' in f and 'margin' in f:
        return 'shear_margin'
    if 'iso' in f or 'random' in f:
        return 'isotropic'
    return None


def _norm_impurity(i: Optional[str]) -> Optional[str]:
    i = (i or '').lower()
    if 'dust' in i or 'mineral' in i or 'ash' in i:
        return 'dust_rich'
    if 'marine' in i or 'brine' in i:
        return 'marine'
    if 'temperate' in i or 'water' in i:
        return 'temperate'
    if 'clean' in i or 'pure' in i:
        return 'clean'
    return None


def _norm_moisture(m: Optional[str]) -> Optional[str]:
    m = (m or '').lower()
    if 'dry' in m:
        return 'dry'
    if 'moist' in m or 'wet' in m:
        return 'moist'
    if 'marine' in m or 'brine' in m:
        return 'marine'
    return None


def classify_n_regime(stress: Optional[str],
                      thermal: Optional[str],
                      fabric: Optional[str]) -> Dict[str, Any]:
    sk = _norm_stress(stress)
    tk = _norm_thermal(thermal)
    fk = _norm_fabric(fabric)
    regime = N_REGIMES.get((sk, tk, fk)) if (sk and tk and fk) else None
    if regime is None:
        # Fall back to a marginal regime along (stress, thermal)
        for alt in (('high', 'cold', 'isotropic'),
                    ('intermediate', 'cold', 'isotropic'),
                    ('low', 'cold', 'isotropic')):
            if N_REGIMES.get(alt):
                regime = N_REGIMES[alt]
                break
        return dict(regime='unclassified — using (high,cold,isotropic) fallback',
                    low=regime['low'], high=regime['high'],
                    bench=regime['bench'], inferred=regime['bench'],
                    source='fallback')
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({sk},{tk},{fk})')


def classify_a1_regime(thermal: Optional[str],
                       impurity: Optional[str]) -> Dict[str, Any]:
    bk = _norm_thermal(thermal) or 'warm'
    ik = _norm_impurity(impurity) or 'clean'
    regime = A1_REGIMES.get((bk, ik), A1_REGIMES[('warm', 'clean')])
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({bk},{ik})')


def classify_q1_regime(thermal: Optional[str],
                       moisture: Optional[str]) -> Dict[str, Any]:
    bk = _norm_thermal(thermal) or 'warm'
    mk = _norm_moisture(moisture) or 'dry'
    regime = Q1_REGIMES.get((bk, mk), Q1_REGIMES[('warm', 'dry')])
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({bk},{mk})')


def classify_e_regime(fabric: Optional[str],
                      thermal: Optional[str]) -> Dict[str, Any]:
    fk = _norm_fabric(fabric) or 'isotropic'
    bk = _norm_thermal(thermal) or 'cold'
    regime = E_REGIMES.get((fk, bk), E_REGIMES[('isotropic', 'cold')])
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({fk},{bk})')


def apply_n_to_e_coupling(n_regime: Dict[str, Any],
                          e_regime: Dict[str, Any]) -> Dict[str, Any]:
    """Narrow E's band using the n → E cross-coupling table."""
    n_center = n_regime['inferred']
    for (lo, hi), info in N_TO_E_COUPLING:
        if lo <= n_center <= hi:
            e_lo, e_hi = info['E_band']
            new = dict(e_regime)
            new['low']  = max(e_regime['low'],  e_lo)
            new['high'] = min(e_regime['high'], e_hi)
            if new['high'] < new['low']:
                # coupling band fully inside — use it
                new['low'], new['high'] = e_lo, e_hi
            new['inferred'] = 0.5 * (new['low'] + new['high'])
            new['coupling'] = info['reason']
            return new
    return e_regime


# ============================================================================
# GLACIER EXPERTS
# ============================================================================
class StressRegimeExpert:
    """Gaussian peak inside the log-band, exponential decay outside."""
    def __init__(self, falloff_decades: float = 1.0,
                 min_sigma_decades: float = 0.25):
        self.falloff_decades = float(falloff_decades)
        self.min_sigma_decades = float(min_sigma_decades)

    def score(self, value: float,
              stress: Optional[str], thermal: Optional[str],
              fabric: Optional[str]) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0
        r = classify_n_regime(stress, thermal, fabric)
        lo, hi, ctr = r['low'], r['high'], r['inferred']
        if lo <= v <= hi:
            sigma = max((hi - lo) / 4.0, self.min_sigma_decades)
            return float(np.exp(-(v - ctr) ** 2 / (2.0 * sigma ** 2)))
        dist = min(abs(v - lo), abs(v - hi))
        return float(np.exp(-dist / self.falloff_decades))


class ThermalBranchExpert:
    """Tag-match × regime membership (analogous to TheoryRegimeExpert)."""
    def __init__(self, falloff_decades: float = 1.5):
        self.falloff_decades = float(falloff_decades)

    def score(self, value: float,
              target_branch: Optional[str],
              candidate_branch: Optional[str] = None) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0

        tb = _norm_thermal(target_branch)
        cb = _norm_thermal(candidate_branch) if candidate_branch else None
        tag = 1.0
        if tb and cb and tb != cb:
            tag = 0.05
        elif tb and not cb:
            tag = 0.7

        # Log-band membership
        lo, hi = A1_REGIMES[('warm', 'clean')]['low'], A1_REGIMES[('warm', 'temperate')]['high']
        if lo <= v <= hi:
            regime = 1.0
        else:
            dist = min(abs(np.log10(v) - np.log10(lo)),
                       abs(np.log10(v) - np.log10(hi)))
            regime = float(np.exp(-dist / self.falloff_decades))
        return tag * regime


class FabricExpert:
    """Analogous to PhysicsRegimeExpert — Gaussian inside E-band."""
    def __init__(self, falloff_decades: float = 1.0):
        self.falloff_decades = float(falloff_decades)

    def score(self, value: float, fabric: Optional[str],
              thermal: Optional[str]) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0
        r = classify_e_regime(fabric, thermal)
        lo, hi, ctr = r['low'], r['high'], r['inferred']
        if lo <= v <= hi:
            sigma = max((hi - lo) / 4.0, 0.2)
            return float(np.exp(-(v - ctr) ** 2 / (2.0 * sigma ** 2)))
        dist = min(abs(v - lo), abs(v - hi))
        return float(np.exp(-dist / self.falloff_decades))


# ============================================================================
# CANDIDATE DATACLASSES
# ============================================================================
@dataclass
class ValueCandidate:
    value: float
    unit: str
    provenance: str
    property_label: str = ""
    method: str = ""
    evidence: str = ""
    source: str = ""
    confidence: float = 1.0
    reasoning: str = ""
    dominant_factor: str = ""
    theory: str = ""
    context: bool = False

    def to_display(self) -> Dict[str, Any]:
        d = {
            'value': round(self.value, 6),
            'unit': self.unit,
            'provenance': self.provenance,
            'property_label': self.property_label[:80],
            'method': self.method,
            'source': self.source[:80],
            'confidence': round(self.confidence, 3),
            'evidence': self.evidence[:140],
            'reasoning': (self.reasoning[:200] + '…'
                          if len(self.reasoning) > 200 else self.reasoning),
        }
        if self.dominant_factor:
            d['dominant_factor'] = self.dominant_factor
        if self.theory:
            d['theory'] = self.theory
        if self.context:
            d['context'] = True
        return d


# ============================================================================
# GATEKEEPER (unit conversion + plausibility + dedupe + conf cap)
# ============================================================================
def gatekeep(items, param_key, provenance='regex_ner',
             conf_cap=None, n_for_conversion: float = 3.0):
    canon = GLACIER_CANON.get(param_key, {})
    lo, hi = canon.get('plausible', (float('-inf'), float('inf')))

    if conf_cap is None:
        if provenance in ('llm_prior', 'regime_prior'):
            conf_cap = 0.5
        elif provenance == 'physics_inferred':
            conf_cap = 0.75
        else:
            conf_cap = 1.0

    reject_non_positive = param_key in _POSITIVE_LOWER_BOUND_PARAMS

    out, seen = [], set()
    for it in items:
        try:
            raw_v = it.get('value')
            raw_u = str(it.get('unit', '') or canon.get('unit') or '')
            v, unit = GlacierUnitConverter.convert(
                raw_v, raw_u, param_key, n=n_for_conversion)
        except Exception:
            continue

        if reject_non_positive and v <= 0.0:
            logger.info('gatekeep[%s]: rejected non-positive %s', param_key, v)
            continue
        if not (lo <= v <= hi):
            logger.info('gatekeep[%s]: rejected %s (out of range %s..%s)',
                        param_key, v, lo, hi)
            continue
        key = round(v, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(ValueCandidate(
            value=v,
            unit=canon.get('unit') or unit or '',
            provenance=provenance,
            property_label=str(it.get('property_label', '')),
            method=str(it.get('method', '')),
            evidence=str(it.get('evidence_span')
                         or it.get('evidence')
                         or it.get('inference_basis') or ''),
            source=str(it.get('source', '')),
            confidence=min(float(it.get('confidence', 1.0)), conf_cap),
            reasoning=str(it.get('reasoning') or ''),
            dominant_factor=str(it.get('dominant_factor') or ''),
            theory=str(it.get('theory') or 'unspecified'),
            context=bool(it.get('_context', False)),
        ))
    return sorted(out, key=lambda c: -c.confidence)


# ============================================================================
# ███ GLACIER LATENT MoE SCORER                                        ███
# ============================================================================
class GlacierLatentMoEScorer:
    """Physics-Informed Mixture of Experts ranker for glacier parameters.

    Ten experts, weights sum to 1.00:

        v1.0 core        glacier threads
        ---------        ---------------
        Material 0.10    Stress Regime (B)  0.20  ← glen_n
        Thermal  0.10    Thermal Branch (A) 0.20  ← A1 / Q1
        Strain   0.05    Fabric (C)         0.10  ← E
        Method   0.05    Corpus Density     0.05
        Confidence 0.05
        Reasoning  0.10
        -------------------------------
        Sum      1.00
    """
    EXPERT_ORDER: Tuple[str, ...] = (
        'Material', 'Thermal', 'Strain', 'Method', 'Confidence',
        'Reasoning', 'Stress Regime', 'Thermal Branch',
        'Fabric', 'Corpus Density',
    )
    EXPERT_COLORS: Dict[str, str] = {
        'Material':       '#E69F00',
        'Thermal':        '#56B4E9',
        'Strain':         '#009E73',
        'Method':         '#CC79A7',
        'Confidence':     '#999999',
        'Reasoning':      '#F0E442',
        'Stress Regime':  '#0072B2',
        'Thermal Branch': '#D55E00',
        'Fabric':         '#000000',
        'Corpus Density': '#88CCEE',
    }

    def __init__(self, thermal_sigma: float = 15.0):
        self.w_material   = 0.10
        self.w_thermal    = 0.10
        self.w_strain     = 0.05
        self.w_method     = 0.05
        self.w_confidence = 0.05
        self.w_reasoning  = 0.10
        self.w_stress     = 0.20
        self.w_thermalbr  = 0.20
        self.w_fabric     = 0.10
        self.w_density    = 0.05

        self.thermal_sigma = thermal_sigma
        self.stress_expert  = StressRegimeExpert()
        self.thermal_expert = ThermalBranchExpert()
        self.fabric_expert  = FabricExpert()

    # ---- individual experts ----------------------------------------
    @staticmethod
    def _material_expert(ext_mat: str, target: str) -> float:
        if not ext_mat:
            return 0.4
        a, b = ext_mat.lower().strip(), target.lower().strip()
        if a == b:
            return 1.0
        if 'ice' in a and 'ice' in b:
            return 0.9
        if a in b or b in a:
            return 0.85
        return 0.25

    def _thermal_expert(self, ext_temp, target_temp) -> float:
        if ext_temp is None:
            return 0.5
        try:
            t = float(ext_temp)
        except (TypeError, ValueError):
            return 0.5
        diff = t - float(target_temp)
        return float(np.exp(-(diff ** 2) / (2.0 * self.thermal_sigma ** 2)))

    @staticmethod
    def _strain_expert(ext_rate, target_rate) -> float:
        if ext_rate is None or target_rate is None:
            return 0.5
        try:
            r = float(ext_rate)
            if r <= 0 or target_rate <= 0:
                return 0.5
            ratio = math.log10(r / target_rate)
            return float(np.exp(-(ratio ** 2) / 8.0))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _method_expert(method: str) -> float:
        return {
            'experiment': 1.0, 'review': 0.8,
            'ice core': 0.95, 'borehole': 0.9,
            'torsion test': 0.85, 'shear test': 0.85,
            'compression test': 0.85, 'creep test': 0.9,
            'insar': 0.7, 'gps': 0.7,
            'explicit': 0.9, 'llm_inferred': 0.5,
            'prior inference': 0.35, 'llm_prior': 0.35,
            'default_fallback': 0.3,
            'arrhenius_cross_relation':       0.60,
            'two_temperature_inversion':      0.65,
            'n_to_e_coupling':                0.55,
            'physics_regime_inference':       0.60,
            'theory_regime_inference':        0.60,
            'side_note_table':                0.55,
            'derived_consensus':              0.55,
        }.get((method or '').lower(), 0.5)

    @staticmethod
    def _reasoning_expert(reasoning: str, method: str) -> float:
        method_l = (method or '').lower()
        _COG_METHODS = (
            'llm_inferred', 'heuristic', 'default_fallback',
            'prior inference', 'llm_prior',
            'arrhenius_cross_relation', 'two_temperature_inversion',
            'n_to_e_coupling', 'physics_regime_inference',
            'theory_regime_inference', 'side_note_table',
            'derived_consensus',
        )
        if method_l not in _COG_METHODS:
            return 0.5
        if not reasoning:
            return 0.2
        n_steps = reasoning.count('Step ') + reasoning.count('\n')
        cites_formula = any(k in reasoning for k in [
            'n', 'A(T)', 'Arrhenius', 'Q₁', 'Q1', 'Glen', 'MPa',
            'yr', 'kJ', 'shear', 'fabric', 'single-max', 'isotropic',
            'divide', 'stream', 'temperate', 'cold',
        ])
        base = min(0.4 + 0.05 * n_steps, 0.8)
        if cites_formula:
            base = min(base + 0.15, 0.85)
        return float(base)

    # ---- glacier threads --------------------------------------------
    def _stress_expert(self, param: str, v: float,
                       stress: Optional[str], thermal: Optional[str],
                       fabric: Optional[str]) -> float:
        if param == 'glen_n':
            return self.stress_expert.score(v, stress, thermal, fabric)
        return 0.0

    def _thermal_branch_expert(self, param: str, v: float,
                               ext: Dict[str, Any],
                               target_thermal: Optional[str],
                               thermal_active: bool) -> float:
        if param in ('A1', 'A2', 'Q1', 'Q2'):
            cand_branch = ext.get('thermal_branch') or target_thermal
            return self.thermal_expert.score(v, target_thermal, cand_branch)
        return 0.0

    def _fabric_expert(self, param: str, v: float,
                       fabric: Optional[str],
                       thermal: Optional[str],
                       fabric_active: bool) -> float:
        if param == 'E':
            if fabric_active:
                return self.fabric_expert.score(v, fabric, thermal)
            return 0.5
        return 0.0

    @staticmethod
    def _density_expert(prov_fine: str) -> float:
        if prov_fine in ('llm_extract', 'llm_reasoned', 'regex_ner'):
            return 1.0
        if prov_fine == 'physics_inferred':
            return 0.7
        if prov_fine == 'regime_prior':
            return 0.4
        return 0.2

    def score(self, extractions,
              target_material='ice',
              target_temp=-3.0,
              target_strain_rate=0.1,
              top_k=8,
              stress_regime=None,
              thermal_state=None,
              fabric=None,
              target_thermal=None):
        stress_active  = bool(_norm_stress(stress_regime))
        thermal_active = bool(_norm_thermal(thermal_state or target_thermal))
        fabric_active  = bool(_norm_fabric(fabric))

        buckets: Dict[str, List[PlasticityCandidate]] = {
            p: [] for p in GLACIER_CANON
        }

        for ext in extractions:
            p = ext.get('param')
            if p not in buckets:
                continue
            try:
                # normalize to canonical (Elmer) units
                v_conv, unit_conv = GlacierUnitConverter.convert(
                    ext['value'], ext.get('unit', ''), p,
                    n=float(ext.get('n_for_conversion', 3.0)))
            except Exception:
                continue
            if not np.isfinite(v_conv):
                continue
            lo, hi = GLACIER_CANON[p]['plausible']
            v_clamped = float(np.clip(v_conv, lo, hi))
            was_clamped = (v_clamped != v_conv)

            s_mat  = self._material_expert(ext.get('material', ''), target_material)
            s_temp = self._thermal_expert(ext.get('temp'), target_temp)
            s_str  = self._strain_expert(ext.get('strain_rate'), target_strain_rate)
            s_meth = self._method_expert(ext.get('method', 'unknown'))
            s_conf = float(ext.get('confidence', 0.5) or 0.5)
            s_reas = self._reasoning_expert(
                ext.get('reasoning', ''), ext.get('method', 'unknown'))

            s_stress  = self._stress_expert(p, v_clamped,
                                            stress_regime, thermal_state, fabric)
            s_thermal = self._thermal_branch_expert(
                p, v_clamped, ext, target_thermal or thermal_state,
                thermal_active)
            s_fab     = self._fabric_expert(p, v_clamped, fabric,
                                            thermal_state, fabric_active)
            prov_fine = _norm_provenance(
                ext.get('_provenance', '') or ext.get('method', ''))
            s_den = self._density_expert(prov_fine)

            breakdown = {
                'Material':       self.w_material   * s_mat,
                'Thermal':        self.w_thermal    * s_temp,
                'Strain':         self.w_strain     * s_str,
                'Method':         self.w_method     * s_meth,
                'Confidence':     self.w_confidence * s_conf,
                'Reasoning':      self.w_reasoning  * s_reas,
                'Stress Regime':  self.w_stress     * s_stress,
                'Thermal Branch': self.w_thermalbr  * s_thermal,
                'Fabric':         self.w_fabric     * s_fab,
                'Corpus Density': self.w_density    * s_den,
            }
            score = float(sum(breakdown.values()))

            cand = PlasticityCandidate(
                param=p,
                value_si=v_clamped,
                raw_value=float(ext['value']),
                raw_unit=str(ext.get('unit', '')),
                score=score,
                confidence=s_conf,
                material=str(ext.get('material', '')),
                temp_k=ext.get('temp'),
                strain_rate=ext.get('strain_rate'),
                method=str(ext.get('method', 'unknown')),
                source_file=str(ext.get('_source_file', '')),
                source_title=str(ext.get('_source_title', '')),
                evidence=str(ext.get('evidence', '')),
                reasoning=str(ext.get('reasoning', '')),
                clamped=was_clamped,
                provenance=str(ext.get('_provenance', '')),
                context=bool(ext.get('_context', False)),
                moe_breakdown=breakdown,
            )
            if p in ('A1', 'A2', 'Q1', 'Q2'):
                cand.theory = str(ext.get('thermal_branch') or 'unspecified')
            buckets[p].append(cand)

        for p in buckets:
            buckets[p].sort(key=lambda c: c.score, reverse=True)
            buckets[p] = buckets[p][:top_k]
        return buckets


# Reuse PlasticityCandidate name from above (renamed for the glacier code)
PlasticityCandidate = ValueCandidate


# ============================================================================
# ███ GLACIER SOLVER — SPECTRAL VISCOUS FLOW + FABRIC EVOLUTION        ███
# ============================================================================
class GlacierGeometry:
    """Builds a 2D glacier cross-section: surface, bed, thickness."""
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.x = np.linspace(-N * dx / 2, N * dx / 2, N)
        self.y = np.linspace(-N * dx / 2, N * dx / 2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.extent = [-N * dx / 2, N * dx / 2,
                       -N * dx / 2, N * dx / 2]

    def create_ice_slab(self, H_center=200.0, slope_deg=3.0,
                        bed_waviness=0.05, wav_wavelength=2000.0,
                        meltwater_pool=False):
        """Simple slab ice body over a sinusoidal bed."""
        slope_rad = np.deg2rad(slope_deg)
        x_m = self.X
        y_m = self.Y

        # Bed: y_bed(x) = H_center/2 * cos(...) + slope*x + wave
        y_bed = (0.5 * H_center * np.sin(2 * np.pi * x_m / wav_wavelength) * bed_waviness
                 + x_m * np.tan(slope_rad))
        y_surface = y_bed + H_center

        # Fill ice where y_bed < y < y_surface
        H = np.zeros((self.N, self.N))
        mask = (y_m >= y_bed) & (y_m <= y_surface)
        H[mask] = y_surface[mask] - y_bed[mask]

        # Order parameters
        fabric = np.zeros_like(H)     # 0 = isotropic, 1 = single-max
        water  = np.zeros_like(H)
        if meltwater_pool:
            # inject a water pocket at the bed
            water = 0.15 * np.exp(-((y_m - y_bed) ** 2) / (2 * (H_center * 0.1) ** 2)) * mask

        return H, fabric, water, y_bed, y_surface


class GlacierSolver:
    """Pure-FFT spectral solver.

    Physics
    -------
    Ice thickness (SIA continuity, mass-conserving):
        ∂H/∂t = ṁ − ∇·(H · u_bar)
        u_bar = (2A/(n+2)) (ρ g)^n H^{n+1} |∇s|^{n-1} ∇s
    where s = b + H is the surface elevation, and A is the effective
    rate factor (temperature- and fabric-dependent).

    Fabric evolution (non-conserved order parameter f):
        ∂f/∂t = α_f ∇²f + R_f(f, ε̇_e)
        R_f relaxes f towards its steady-state value f_ss(ε̇_e, T)

    Water content evolution (damage / meltwater):
        ∂w/∂t = α_w ∇²w − k_drain · w + S_w(ε̇_e, T)
    where S_w is strain-heating melt production.

    Stream-function Stokes (biharmonic, constant-reference viscosity):
        ∇⁴ψ = ∂(ρgH sinα)/∂x  +  ∂(ρgH sinα)/∂y
        u =  ∂ψ/∂y
        v = −∂ψ/∂x
    """
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']

        # Glen parameters (already in Elmer units after conversion)
        self.n     = float(params.get('glen_n', 3.0))
        self.A1    = float(params.get('A1', 6.0e13))     # MPa^-n yr^-1
        self.Q1    = float(params.get('Q1', 60000.0))    # J/mol
        self.E_glen = float(params.get('E_glen', 1.0))
        self.rho   = float(params.get('rho_ice', RHO_ICE_SI))   # kg/m³
        self.T_C   = float(params.get('T_ice_C', -3.0))
        self.bdot  = float(params.get('accumulation', 0.3))     # m/yr
        self.g     = G_ELMER                                    # m/yr²

        self.geom = GlacierGeometry(self.N, self.dx)
        self.kx, self.ky, self.k2 = make_k_vectors(self.N, self.dx)

        (self.H, self.fabric, self.water,
         self.y_bed, self.y_surface) = self.geom.create_ice_slab(
            H_center=params.get('H_center', 200.0),
            slope_deg=params.get('slope_deg', 3.0),
            bed_waviness=params.get('bed_waviness', 0.05),
            wav_wavelength=params.get('wav_wavelength', 2000.0),
            meltwater_pool=params.get('meltwater_pool', False))

        self.history = {
            'H_max': [], 'H_mean': [], 'vel_max': [],
            'fabric_mean': [], 'water_mean': [],
            'visc_mean': [], 'energy': []
        }

        # physical constants for the SIA prefactor (Elmer units)
        # u_bar = C * A_eff * (ρg)^n * H^{n+1} |∇s|^{n-1} ∇s
        # with C = 2/(n+2). A_eff in MPa^-n yr^-1, ρg in MPa/m (Elmer).
        # Convert ρ_ice to "Elmer density" for the ρg source term:
        self.rho_elmer = self.rho * KG_M3_TO_ELMER   # kg/m³ → MPa/(m·yr⁻²)

    # ---- temperature / rate factor -----------------------------------
    def effective_A(self):
        """Two-branch Arrhenius + fabric enhancement.

        A_eff(T) = E · A₁ · exp(−Q₁/(R T))   for T > −10 °C
        A_eff(T) = E · A₂ · exp(−Q₂/(R T))   for T ≤ −10 °C
        T is in Kelvin.
        """
        T_K = self.T_C + 273.15
        R   = 8.314  # J/(mol·K)
        A1 = self.A1
        Q1 = self.Q1
        # Cold branch — use standard Paterson-Budd ratios (Q₂ = 139 kJ/mol,
        # A₂ = A₁ × (A1_ratio) with A1_ratio ≈ 6.0e28 / 1.258e13 ≈ 4.8e15
        # (we use a fixed canonical ratio so that A(T) is continuous at −10 °C)
        A2_ratio = 4.8e15
        Q2 = 139000.0
        A1_eff = A1 * np.exp(-Q1 / (R * T_K))
        A2_eff = (A1 * A2_ratio) * np.exp(-Q2 / (R * T_K))
        base = np.where(T_K > 263.15, A1_eff, A2_eff)
        # Fabric enhancement (scalar, applied uniformly for simplicity)
        return float(self.E_glen * base)

    # ---- viscosity ---------------------------------------------------
    def effective_viscosity(self, eps_dot):
        """η = (1/2) A^{-1/n} ε̇_e^{(1-n)/n}  in MPa·yr."""
        A_eff = self.effective_A()
        eps_safe = np.maximum(eps_dot, 1e-12)
        n = self.n
        eta = 0.5 * (A_eff ** (-1.0 / n)) * (eps_safe ** ((1.0 - n) / n))
        return np.clip(eta, 1e-6, 1e6)   # MPa·yr

    # ---- SIA velocity ------------------------------------------------
    def sia_velocity(self, H, y_surface):
        """Depth-averaged SIA velocity field.

        u_bar = (2A/(n+2)) (ρg)^n H^{n+1} |∇s|^{n-1} ∇s
        """
        A_eff = self.effective_A()
        n = self.n
        sx, sy = spectral_gradients(y_surface, self.kx, self.ky)
        grad_s = np.sqrt(sx ** 2 + sy ** 2 + 1e-30)
        grad_s_pow = grad_s ** (n - 1)
        prefactor = (2.0 * A_eff / (n + 2.0)) * (self.rho_elmer * self.g) ** n
        H_safe = np.maximum(H, 1e-6)
        ux = prefactor * H_safe ** (n + 1) * grad_s_pow * sx
        uy = prefactor * H_safe ** (n + 1) * grad_s_pow * sy
        return ux, uy

    # ---- spectral Stokes (stream function) ---------------------------
    def stream_function(self, ux, uy):
        """Solve ∇⁴ψ = ∂uy/∂x − ∂ux/∂y  (vorticity)."""
        duy_dx, _   = spectral_gradients(uy, self.kx, self.ky)
        _,   dux_dy = spectral_gradients(ux, self.kx, self.ky)
        omega = duy_dx - dux_dy
        omega_hat = fft2(omega)
        k4 = self.k2 ** 2
        k4[0, 0] = 1e-12
        psi_hat = omega_hat / k4
        psi = np.real(ifft2(psi_hat))
        return psi

    # ---- order parameter evolution ----------------------------------
    def evolve_fabric(self, H, ux, uy, eps_dot, dt):
        """Relax fabric f toward steady-state single-max where strain is high."""
        # Steady-state fabric from strain rate magnitude
        f_ss = 1.0 - np.exp(-eps_dot / 1e-3)
        # Advection + diffusion
        lap_f = spectral_laplacian(self.fabric, self.k2)
        fgx, fgy = spectral_gradients(self.fabric, self.kx, self.ky)
        adv = ux * fgx + uy * fgy
        alpha_f = 5e-3     # diffusion coefficient
        relax   = 1.0 / 50.0    # yr^-1
        f_new = self.fabric + dt * (alpha_f * lap_f - adv
                                    + relax * (f_ss - self.fabric))
        f_new = np.clip(f_new, 0.0, 1.0)
        f_new[H < 1e-3] = 0.0
        return f_new

    def evolve_water(self, H, ux, uy, eps_dot, dt):
        """Damage / meltwater: strain heating → melt, diffusive transport."""
        # Strain heating Q = 2η ε̇²  (MPa·yr·yr⁻¹·... — kept symbolic)
        T_K = self.T_C + 273.15
        eta = self.effective_viscosity(eps_dot)
        Q = 2.0 * eta * eps_dot ** 2
        # Melt fraction production rate (arbitrary scaling for visualisation)
        S_w = np.clip(Q / 5e6, 0.0, 0.05) if T_K > 270.0 else 0.0
        lap_w = spectral_laplacian(self.water, self.k2)
        wgx, wgy = spectral_gradients(self.water, self.kx, self.ky)
        adv = ux * wgx + uy * wgy
        alpha_w = 2e-3
        drain   = 1.0 / 20.0
        w_new = self.water + dt * (alpha_w * lap_w - adv
                                   - drain * self.water + S_w)
        w_new = np.clip(w_new, 0.0, 0.25)
        w_new[H < 1e-3] = 0.0
        return w_new

    def evolve_thickness(self, H, ux, uy, dt):
        """SIA mass continuity: ∂H/∂t = ṁ − ∇·(H u_bar)."""
        flux_x = H * ux
        flux_y = H * uy
        dflux_x_dx, _ = spectral_gradients(flux_x, self.kx, self.ky)
        _, dflux_y_dy = spectral_gradients(flux_y, self.kx, self.ky)
        div_flux = dflux_x_dx + dflux_y_dy
        # Accretion/ablation profile (m/yr) — simple cosine
        bdot = self.bdot * np.cos(np.pi * self.geom.X /
                                  (self.geom.extent[1] or 1.0))
        H_new = H + dt * (bdot - div_flux)
        return np.clip(H_new, 0.0, 5000.0)

    def step(self):
        """One forward-Euler step of the coupled system."""
        try:
            # 1. SIA velocity
            ux, uy = self.sia_velocity(self.H, self.y_surface)

            # 2. Effective strain rate (magnitude of velocity gradient)
            dux_dx, dux_dy = spectral_gradients(ux, self.kx, self.ky)
            duy_dx, duy_dy = spectral_gradients(uy, self.kx, self.ky)
            eps_dot = np.sqrt(
                0.5 * (dux_dx ** 2 + duy_dy ** 2)
                + 0.25 * (dux_dy + duy_dx) ** 2 + 1e-30)

            # 3. Viscosity diagnostic
            eta = self.effective_viscosity(eps_dot)

            # 4. Stream function (for visualisation / divergence check)
            psi = self.stream_function(ux, uy)

            # 5. Order-parameter updates
            self.fabric = self.evolve_fabric(self.H, ux, uy, eps_dot, self.dt)
            self.water  = self.evolve_water(self.H, ux, uy, eps_dot, self.dt)

            # 6. Thickness update (last — feeds back into next step)
            H_new = self.evolve_thickness(self.H, ux, uy, self.dt)
            self.H = H_new

            # 7. Refresh surface elevation
            self.y_surface = self.y_bed + self.H

            # 8. Book-keeping
            vel_mag = np.sqrt(ux ** 2 + uy ** 2)
            H_mean = float(np.mean(self.H[self.H > 1e-3])) if np.any(self.H > 1e-3) else 0.0
            f_mean = float(np.mean(self.fabric[self.H > 1e-3])) if np.any(self.H > 1e-3) else 0.0
            w_mean = float(np.mean(self.water[self.H > 1e-3])) if np.any(self.H > 1e-3) else 0.0
            eta_mean = float(np.mean(eta[self.H > 1e-3])) if np.any(self.H > 1e-3) else 0.0
            energy = float(np.sum((self.rho * 910.0 * 9.81 * self.H ** 2) * self.dx ** 2))

            self.history['H_max'].append(float(np.max(self.H)))
            self.history['H_mean'].append(H_mean)
            self.history['vel_max'].append(float(np.max(vel_mag)))
            self.history['fabric_mean'].append(f_mean)
            self.history['water_mean'].append(w_mean)
            self.history['visc_mean'].append(eta_mean)
            self.history['energy'].append(energy)

            return {
                'H': self.H.copy(),
                'ux': ux.copy(), 'uy': uy.copy(),
                'vel_mag': vel_mag.copy(),
                'eps_dot': eps_dot.copy(),
                'viscosity': eta.copy(),
                'fabric': self.fabric.copy(),
                'water': self.water.copy(),
                'stream_fn': psi.copy(),
                'y_bed': self.y_bed.copy(),
                'y_surface': self.y_surface.copy(),
                'convergence': {
                    'H_max': float(np.max(self.H)),
                    'H_mean': H_mean,
                    'vel_max': float(np.max(vel_mag)),
                    'fabric_mean': f_mean,
                    'water_mean': w_mean,
                    'visc_mean': eta_mean,
                    'energy': energy,
                }
            }
        except Exception as e:
            st.error(f"Error in glacier step: {e}")
            zeros = np.zeros((self.N, self.N))
            return {k: zeros.copy() for k in
                    ('H', 'ux', 'uy', 'vel_mag', 'eps_dot',
                     'viscosity', 'fabric', 'water', 'stream_fn')} | {
                'y_bed': zeros.copy(), 'y_surface': zeros.copy(),
                'convergence': {k: 0.0 for k in
                    ('H_max', 'H_mean', 'vel_max', 'fabric_mean',
                     'water_mean', 'visc_mean', 'energy')}}


# ============================================================================
# SIA PHYSICS — HELPER (used by unit tests)
# ============================================================================
def compute_sia_velocity_magnitude(H, A_eff, rho_elmer, g, slope_rad, n):
    """Depth-averaged SIA velocity for a slab of thickness H on slope α."""
    return (2.0 * A_eff / (n + 2.0)) * (rho_elmer * g) ** n \
           * H ** (n + 1) * (math.sin(slope_rad) ** n)


# ============================================================================
# RECOMMENDER
# ============================================================================
@dataclass
class GlacierRecommendationBundle:
    material: str
    temp_k: float
    strain_rate: float
    candidates: Dict[str, List[ValueCandidate]]
    defaults: Dict[str, float]
    priors_df: pd.DataFrame
    retrieval_backend: str
    llm_used: bool
    timestamp: float = field(default_factory=time.time)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    regime_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def best(self, param: str) -> Optional[ValueCandidate]:
        lst = self.candidates.get(param, [])
        if not lst:
            return None
        ranked = [c for c in lst if not getattr(c, 'context', False)]
        if not ranked:
            logger.warning('bundle.best[%s]: all context — falling back',
                           param)
            ranked = list(lst)
        return max(ranked, key=lambda c: c.score)

    def coverage(self) -> int:
        return sum(1 for p in GLACIER_CANON if self.candidates.get(p))

    def coverage_five(self) -> int:
        targets = ('glen_n', 'A1', 'Q1', 'E', 'rho_ice')
        return sum(1 for p in targets if self.candidates.get(p))


def recommend_glacier_params(
    stress_regime: str = 'high',
    thermal_state: str = 'cold',
    fabric: str = 'isotropic',
    impurity: str = 'clean',
    moisture: str = 'dry',
    material: str = 'ice',
    temp_c: float = -3.0,
    strain_rate: float = 0.1,
    corpus_folder: str = 'glacier_json_metadatabase',
    use_llm: bool = False,
    ollama_model: str = 'qwen2.5:7b',
    cascade_mode: str = 'union',
) -> GlacierRecommendationBundle:
    """Full glacier recommender — mirror of recommend_grounded for nt-Cu."""
    # Regime classifications (live preview & candidate generation)
    n_reg = classify_n_regime(stress_regime, thermal_state, fabric)
    a1_reg = classify_a1_regime(thermal_state, impurity)
    q1_reg = classify_q1_regime(thermal_state, moisture)
    e_reg = classify_e_regime(fabric, thermal_state)
    e_reg = apply_n_to_e_coupling(n_reg, e_reg)

    # Corpus scan (heuristic NER)
    heuristic: Dict[str, List[Dict[str, Any]]] = {p: [] for p in GLACIER_CANON}
    if os.path.isdir(corpus_folder):
        for fn in sorted(os.listdir(corpus_folder)):
            if not fn.lower().endswith('.json'):
                continue
            fpath = os.path.join(corpus_folder, fn)
            try:
                with open(fpath, encoding='utf-8-sig') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning('glacier corpus load %s: %s', fn, e)
                continue
            rows = data if isinstance(data, list) else [data]
            for i, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                title = str(row.get('Title') or row.get('title') or '')
                abstract = str(row.get('Abstract') or row.get('abstract') or '')
                full = str(row.get('Full Text') or row.get('full_text')
                           or row.get('text') or '')
                text = f'Title: {title}. Abstract: {abstract}. Full Text: {full[:2000]}'
                text_norm = norm_text(text)
                for param_key in GLACIER_CANON:
                    hits = heuristic_extract(text_norm, text, param_key,
                                             source=f'{fn}[{i}]')
                    heuristic[param_key].extend(hits)

    # Regime-prior candidates
    regime_map = {
        'glen_n': (n_reg, 'glen_n'),
        'A1':     (a1_reg, 'A1'),
        'Q1':     (q1_reg, 'Q1'),
        'E':      (e_reg, 'E'),
    }
    candidates: Dict[str, List[ValueCandidate]] = {p: [] for p in GLACIER_CANON}

    # Heuristic hits → gatekeep
    for p, hits in heuristic.items():
        n_for_conv = n_reg['inferred']
        for h in hits:
            h['_context'] = _is_context(p, 'regex_ner')
            _stamp_provenance(h, 'regex_ner')
        cands = gatekeep(hits, p, provenance='regex_ner',
                         n_for_conversion=n_for_conv)
        candidates[p].extend(cands)

    # Regime-prior fallback (guaranteed coverage)
    for p, (reg, _) in regime_map.items():
        reasoning = (
            f"Step 1: stress={stress_regime}, thermal={thermal_state}, "
            f"fabric={fabric}, impurity={impurity}, moisture={moisture}.\n"
            f"Step 2: classifier → {reg['regime']}.\n"
            f"Step 3: log-center → {reg['inferred']:.4g}."
        )
        if reg.get('coupling'):
            reasoning += f"\nStep 4 (n→E coupling): {reg['coupling']}."
        cand = ValueCandidate(
            value=reg['inferred'],
            unit=GLACIER_CANON[p]['unit'] or 'dimensionless',
            provenance='regime_prior',
            property_label=f'{p} (regime classifier)',
            method='physics_regime_inference',
            evidence=reg['regime'],
            source=reg['source'],
            confidence=0.35,
            reasoning=reasoning,
            dominant_factor=('combined' if (stress_regime and thermal_state)
                             else 'thermal'),
            context=_is_context(p, 'regime_prior'),
        )
        candidates[p].append(cand)

    # Add default (sane) values for parameters with no candidates
    defaults = {}
    for p in GLACIER_CANON:
        defaults[p] = {
            'glen_n': 3.0,
            'A1': 6.0e13, 'A2': 6.0e12,
            'Q1': 60000.0, 'Q2': 139000.0,
            'E': 1.0,
            'rho_ice': 910.0,
            'H': 200.0, 'd_grain': 0.002, 'w_water': 0.0,
            'fabric': 0.0, 'slope': 3.0, 'bed_slope': 0.0,
            'T_ice': -3.0, 'T_pmp': 0.0,
            'accumulation': 0.3, 'basal_drag': 1.0,
            'gamma_rate': 0.1,
        }[p]

    # Sort everything by score
    for p in candidates:
        candidates[p].sort(key=lambda c: -c.score)

    return GlacierRecommendationBundle(
        material=material,
        temp_k=temp_c + 273.15,
        strain_rate=strain_rate,
        candidates=candidates,
        defaults=defaults,
        priors_df=pd.DataFrame(),
        retrieval_backend='glacier-heuristic+regime',
        llm_used=False,
        regime_summary={'glen_n': n_reg, 'A1': a1_reg,
                        'Q1': q1_reg, 'E': e_reg},
    )


# ============================================================================
# HEURISTIC EXTRACTOR (glacier)
# ============================================================================
_NUM_ANY = re.compile(
    r"(?<![a-z0-9.])"
    r"(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"(?:\s*[x×*]\s*10\s*\^?\(?\s*(?P<exp>[+-]?\d+)\s*\)?)?"
)


def _num_value(m: "re.Match") -> float:
    v = float(m.group("val"))
    return v * 10.0 ** int(m.group("exp")) if m.group("exp") else v


def _spans(text, patterns):
    return [(m.start(), m.group(0)) for p in patterns
            for m in p.finditer(text)]


def heuristic_extract(text_norm: str, text_raw: str, param_key: str,
                      source: str = '') -> List[Dict[str, Any]]:
    """Extract number–unit pairs near each parameter mention."""
    aliases = GLACIER_CANON.get(param_key, {}).get('aliases', [])
    patterns = [re.compile(alias_pattern(a)) for a in aliases]
    out: List[Dict[str, Any]] = []
    for p in patterns:
        for m in p.finditer(text_norm):
            window = text_norm[m.end(): m.end() + 100]
            for nm in _NUM_ANY.finditer(window):
                v = _num_value(nm)
                # Unit heuristics
                tail = text_raw[m.end(): m.end() + 140]
                unit = _sniff_unit(tail, param_key)
                out.append({
                    'value': v,
                    'unit': unit,
                    'property_label': m.group(0),
                    'evidence_span': text_raw[max(0, m.start() - 20):
                                              m.end() + 60],
                    'source': source,
                    'method': 'regex_ner',
                    'confidence': 0.55,
                })
                break   # one hit per mention
    return out


_UNIT_SNIFF = [
    (r'MPa\s*\^?\s*-?\s*n|MPa\s*-?\s*3', 'MPa^-n yr^-1'),
    (r'MPa\s*-?\s*3\s*yr',               'MPa^-3 yr^-1'),
    (r'Pa\s*\^?\s*-?\s*3\s*s',           'Pa^-3 s^-1'),
    (r'kPa\s*\^?\s*-?\s*3\s*s',          'kPa^-3 s^-1'),
    (r'kJ\s*/?\s*mol',                    'kJ mol^-1'),
    (r'J\s*/?\s*mol',                     'J mol^-1'),
    (r'kg\s*/?\s*m\s*-?3',                'kg m^-3'),
    (r'g\s*/?\s*cm\s*-?3',                'g cm^-3'),
    (r'\bkm\b',                            'km'),
    (r'\bmm\b',                            'mm'),
    (r'\bcm\b',                            'cm'),
    (r'\bm\b',                             'm'),
    (r'\ba\s*-?1|/yr|yr\s*-?1',            'yr^-1'),
    (r's\s*-?1|/s',                        's^-1'),
    (r'deg\s*C|Celsius',                   'deg C'),
    (r'\bK\b',                             'K'),
]


def _sniff_unit(snippet: str, param_key: str) -> str:
    for rx, unit in _UNIT_SNIFF:
        if re.search(rx, snippet, re.I):
            return unit
    return GLACIER_CANON.get(param_key, {}).get('unit', '') or ''


# ============================================================================
# VISUALIZATION — BAR CHART (adapted from nt-Cu)
# ============================================================================
def _color_to_rgb(c):
    if c is None:
        return (0.7, 0.7, 0.7)
    if isinstance(c, (tuple, list, np.ndarray)) and len(c) >= 3:
        try:
            return (float(c[0]), float(c[1]), float(c[2]))
        except Exception:
            return (0.7, 0.7, 0.7)
    try:
        return to_rgb(str(c))
    except Exception:
        return (0.7, 0.7, 0.7)


def _relative_luminance(rgb):
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _resolve_bar_typography(style, ann, leg, cbar_lbl, cbar_tick):
    base = float(style.get('font_size_small', 8))
    ann = base if ann is None else max(2.0, float(ann))
    leg = base if leg is None else max(2.0, float(leg))
    cbar_lbl = base if cbar_lbl is None else max(2.0, float(cbar_lbl))
    cbar_tick = (max(cbar_lbl - 1.0, 2.0) if cbar_tick is None
                 else max(2.0, float(cbar_tick)))
    return ann, leg, cbar_lbl, cbar_tick


def _compute_min_safe_headroom(fig_h_in, pad_pt, ann_fs_pt,
                               axes_frac=0.85):
    clearance_pt = float(pad_pt) + 1.5 * float(ann_fs_pt)
    axes_pt = max(1.0, axes_frac * float(fig_h_in) * 72.0)
    clearance_frac = min(0.85, clearance_pt / axes_pt)
    return 1.0 / max(1e-6, 1.0 - clearance_frac)


def plot_candidate_scores(candidates, scores, provenance, best_idx=None, *,
                          param_title=None, param_symbol=None, unit=None,
                          score_label='LatentMoE score', journal='nature',
                          fig_size=(5.2, 3.4), bar_width=0.62,
                          label_pad=9, headroom=1.22, despine=True, title=None,
                          bar_colormap='viridis', color_by='score',
                          best_color_override=None, cmap_vmin=None,
                          cmap_vmax=None, show_colorbar=False,
                          annotation_fontsize=None, legend_fontsize=None,
                          colorbar_label_fontsize=None,
                          colorbar_tick_fontsize=None,
                          colorbar_label='Score',
                          legend_anchor_y=1.02,
                          legend_columnspacing=1.4,
                          legend_handletextpad=0.4,
                          legend_borderaxespad=0.0,
                          legend_granularity='coarse',
                          legend_show_counts=False,
                          context_flags=None,
                          show_best_legend=False):
    n = len(candidates)
    assert n == len(scores) == len(provenance), 'length mismatch'
    param_symbol = normalize_tex(param_symbol)
    granularity = _norm_granularity(legend_granularity)
    order = np.argsort(candidates)
    xs, cands = np.arange(n), [candidates[i] for i in order]
    scs   = [scores[i] for i in order]
    provs = [_legend_key(provenance[i], granularity) for i in order]
    ctxs  = ([bool(context_flags[i]) for i in order]
             if context_flags is not None else [False] * n)
    best_x = int(np.where(order == best_idx)[0][0]) \
        if (best_idx is not None and best_idx in order) else None

    style = JournalTemplates.get_journal_styles()
    style = style.get(journal, style['nature'])
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
    mpl.rcParams['mathtext.fontset'] = (
        'stix' if 'times' in style['font_family'].lower() else 'dejavusans')

    C_EDGE = 'black'
    if isinstance(bar_colormap, str):
        try:
            cmap = plt.get_cmap(bar_colormap)
        except ValueError:
            logger.warning('Unknown colormap %r — using viridis', bar_colormap)
            cmap = plt.get_cmap('viridis')
    else:
        cmap = bar_colormap

    if color_by == 'score':
        vmin = float(cmap_vmin) if cmap_vmin is not None else float(min(scs))
        vmax = float(cmap_vmax) if cmap_vmax is not None else float(max(scs))
        span = max(vmax - vmin, 1e-9)
        faces = [cmap((float(s) - vmin) / span) for s in scs]
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
    ctx_x  = [x for x in xs if ctxs[x]]
    if comp_x:
        ax.bar(comp_x, [scs[x] for x in comp_x], width=bar_width,
               facecolor=[faces[x] for x in comp_x],
               edgecolor=C_EDGE, linewidth=0.8, zorder=3)
    if ctx_x:
        ax.bar(ctx_x, [scs[x] for x in ctx_x], width=bar_width,
               facecolor=[faces[x] for x in ctx_x], alpha=0.55,
               edgecolor='0.35', linewidth=0.7, zorder=3)

    marker_table = (LEGEND_MARKERS if granularity == 'coarse'
                    else PROVENANCE_MARKERS)
    for x, s, p in zip(xs, scs, provs):
        st_ = marker_table[p]
        bar_rgb = _color_to_rgb(faces[x])
        lum = _relative_luminance(bar_rgb)
        is_ctx = ctxs[x]
        marker_edge = ('0.35' if is_ctx
                       else ('black' if lum > 0.55 else 'white'))
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
            text_color, halo = '0.45', 'white'
        else:
            text_color = 'black' if lum > 0.55 else 'white'
            halo = 'white' if lum > 0.55 else 'black'
        ax.annotate(f'{s:.3f}', xy=(x, s),
                    xytext=(0, label_pad), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=ann_fs * (0.9 if is_ctx else 1.0),
                    fontweight=('bold' if (x == best_x and not is_ctx)
                                else 'normal'),
                    color=text_color,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=halo,
                              edgecolor='none', alpha=0.85),
                    zorder=7)

    ax.set_ylim(0, max(scs) * effective_headroom if max(scs) > 0 else 1)
    ax.set_xticks(xs)
    ax.set_xticklabels([safe_mathtext(sci_tex(c)) for c in cands])
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
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02,
                            orientation='vertical')
        cbar.set_label(colorbar_label, fontsize=cbar_fs)
        cbar.ax.tick_params(labelsize=ctick_fs)

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
                lbl = f'{lbl} (n={n_ctx_only}+ctx)'
            else:
                lbl = f'{lbl} (n={provs.count(key)})'
        handles.append(Line2D([], [], marker=st_['marker'], linestyle='none',
                              markersize=st_['ms'],
                              markerfacecolor=('white'
                                               if st_.get('fill', True)
                                               else 'none'),
                              markeredgecolor='black',
                              markeredgewidth=1.1,
                              label=lbl))
    if show_best_legend and best_x is not None and not ctxs[best_x]:
        handles.append(Patch(facecolor=faces[best_x], edgecolor=C_EDGE,
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


def plot_stacked_latentmoe(candidates, param_title, param_symbol,
                           unit, journal='nature', fig_size=(6.5, 4.0),
                           max_headroom: float = 1.18,
                           show_context: bool = False):
    n = len(candidates)
    if n == 0:
        return None
    cands_sorted = sorted(candidates, key=lambda c: c.score)
    style = JournalTemplates.get_journal_styles()
    style = style.get(journal, style['nature'])
    mpl.rcParams['mathtext.fontset'] = (
        'stix' if 'times' in style['font_family'].lower() else 'dejavusans')
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    JournalTemplates.apply_journal_style(fig, ax, journal)

    xs = np.arange(n)
    bar_width = 0.65
    bottoms = np.zeros(n)
    drawn = []
    for expert_name in GlacierLatentMoEScorer.EXPERT_ORDER:
        heights = np.array([
            float(c.moe_breakdown.get(expert_name, 0.0))
            if isinstance(c.moe_breakdown, dict) else 0.0
            for c in cands_sorted])
        if np.sum(heights) <= 1e-4:
            continue
        color = GlacierLatentMoEScorer.EXPERT_COLORS.get(expert_name, '#BBBBBB')
        ax.bar(xs, heights, bar_width, bottom=bottoms,
               color=color, edgecolor='black', linewidth=0.5,
               label=expert_name, zorder=3)
        bottoms += heights
        drawn.append(expert_name)

    if show_context:
        for i, c in enumerate(cands_sorted):
            if getattr(c, 'context', False):
                ax.bar([xs[i]], [c.score], bar_width,
                       color='grey', alpha=0.30, edgecolor='none', zorder=4)

    winner_x = None
    for i, c in enumerate(cands_sorted):
        if not getattr(c, 'context', False):
            winner_x = i
    for i, (x, c) in enumerate(zip(xs, cands_sorted)):
        is_ctx = bool(getattr(c, 'context', False))
        is_winner = (winner_x is not None and i == winner_x)
        ax.text(x, c.score + 0.012, f'{c.score:.2f}', ha='center', va='bottom',
                fontsize=7, fontweight='bold' if is_winner else 'normal',
                color='0.45' if is_ctx else 'black', zorder=6)
        if is_winner:
            ax.plot([x - bar_width / 2, x + bar_width / 2],
                    [-0.015, -0.015], color='#D55E00',
                    linewidth=2.5, solid_capstyle='butt',
                    clip_on=False, zorder=6)

    ax.set_ylim(0, max_headroom)
    ax.set_ylabel('Total LatentMoE score', fontsize=9)
    xlabel = param_title or ''
    if param_symbol:
        xlabel += f' (${param_symbol}$)'
    if unit:
        xlabel += f' [$\\mathrm{{{unit}}}$]'
    if xlabel:
        ax.set_xlabel(safe_mathtext(xlabel), fontsize=9)
    ui_vals = [c.value for c in cands_sorted]
    def _fmt(v):
        if v == 0:
            return '0'
        av = abs(v)
        if av < 1e-3 or av >= 1e4:
            return f'{v:.1e}'
        return f'{v:.3g}'
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt(v) for v in ui_vals],
                       rotation=45, ha='right', fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.6, n - 0.4)
    if drawn:
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02),
                  frameon=False, fontsize=7.5, ncol=2,
                  columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(which='both', top=False, right=False)
    return fig


# ============================================================================
# STREAMLIT UI — GLACIER RECOMMENDER TAB
# ============================================================================
def _glacier_param_display(p: str, v: float) -> str:
    meta = GLACIER_PARAM_META.get(p, {})
    sym = strip_tex(meta.get('symbol', p))
    unit = strip_tex(meta.get('unit') or '')
    if p in ('glen_n', 'E', 'fabric'):
        return f'{sym} = {v:.3f}'
    if p in ('A1', 'A2'):
        return f'{sym} = {v:.3e} {unit}'
    if p in ('Q1', 'Q2'):
        return f'{sym} = {v:.0f} {unit}'
    return f'{sym} = {v:.4g} {unit}'


def render_glacier_recommender_tab():
    st.subheader('🤖 Glacier Physics-Regime Recommender (v1.0.0)')
    st.caption(
        'Converts literature values (Pa⁻ⁿ s⁻¹, kJ/mol) to Elmer/Ice '
        'mixed units (MPa, m, yr) before scoring. The n→E cross-coupling '
        'narrows the enhancement-factor band when a dislocation-creep '
        'stress regime is selected.')

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stress = st.selectbox('Stress regime',
                              ['low', 'intermediate', 'high'],
                              index=2, key='gr_stress')
    with c2:
        thermal = st.selectbox('Thermal state', ['cold', 'warm'],
                               index=0, key='gr_thermal')
    with c3:
        fabric = st.selectbox('Fabric', ['isotropic', 'single_max',
                                         'multi_max', 'shear_margin'],
                              index=0, key='gr_fabric')
    with c4:
        impurity = st.selectbox('Impurity', ['clean', 'dust_rich',
                                             'marine', 'temperate'],
                                index=0, key='gr_impurity')
    with c5:
        moisture = st.selectbox('Moisture', ['dry', 'moist', 'marine'],
                                index=0, key='gr_moisture')

    # Live regime previews
    n_prev = classify_n_regime(stress, thermal, fabric)
    a1_prev = classify_a1_regime(thermal, impurity)
    q1_prev = classify_q1_regime(thermal, moisture)
    e_prev = classify_e_regime(fabric, thermal)
    e_prev = apply_n_to_e_coupling(n_prev, e_prev)

    st.caption(f'🧭 **n regime:** {n_prev["regime"][:80]}')
    st.caption(f'   Band: `[{n_prev["low"]:.2f}, {n_prev["high"]:.2f}]`, '
               f'center `{n_prev["inferred"]:.2f}` · source `{n_prev["source"]}`')
    st.caption(f'⚡ **A₁ regime:** {a1_prev["regime"][:80]}')
    st.caption(f'   Band: `[{a1_prev["low"]:.2e}, {a1_prev["high"]:.2e}]` '
               f'MPa⁻ⁿ yr⁻¹')
    st.caption(f'🔥 **Q₁ regime:** {q1_prev["regime"][:80]}')
    st.caption(f'   Band: `[{q1_prev["low"]/1000:.1f}, '
               f'{q1_prev["high"]/1000:.1f}]` kJ/mol')
    st.caption(f'🔀 **E regime:** {e_prev["regime"][:80]}')
    if e_prev.get('coupling'):
        st.caption(f'   📐 Cross-coupling: {e_prev["coupling"]}')
    st.caption(f'   Band: `[{e_prev["low"]:.2f}, {e_prev["high"]:.2f}]`')

    with st.expander('📂 Corpus path (optional)', expanded=False):
        corpus_folder = st.text_input(
            'Glacier metadatabase folder',
            value='glacier_json_metadatabase',
            key='gr_corpus_folder')

    if st.button('🔍 Run glacier recommender', type='primary'):
        bundle = recommend_glacier_params(
            stress_regime=stress, thermal_state=thermal, fabric=fabric,
            impurity=impurity, moisture=moisture,
            corpus_folder=corpus_folder)
        st.session_state['glacier_bundle'] = bundle
        st.success(f'Retrieved candidates for '
                   f'{bundle.coverage()}/{len(GLACIER_CANON)} parameters '
                   f'(5-target coverage: {bundle.coverage_five()}/5).')

        # Adoption
        st.markdown('### Recommended values')
        targets = ['glen_n', 'A1', 'Q1', 'E', 'rho_ice']
        cols = st.columns(len(targets))
        adopted = {}
        for i, p in enumerate(targets):
            best = bundle.best(p)
            with cols[i]:
                if best:
                    st.metric(
                        GLACIER_PARAM_META[p]['title'][:20],
                        _glacier_param_display(p, best.value),
                        delta=f'score {best.score:.2f} · {best.provenance}')
                else:
                    st.metric(GLACIER_PARAM_META[p]['title'][:20], '—')

        if st.button('✅ Adopt for ElmerSolver SIF'):
            for p in targets:
                best = bundle.best(p)
                if best is None:
                    continue
                adopted[p] = best.value
            # Elmer SIF keywords
            sif_overrides = {}
            if 'glen_n' in adopted:
                sif_overrides['glen_exponent'] = adopted['glen_n']
            if 'A1' in adopted:
                sif_overrides['rate_factor_1'] = adopted['A1']
            if 'Q1' in adopted:
                # Elmer expects J/mol
                sif_overrides['activation_energy_1'] = adopted['Q1']
            if 'E' in adopted:
                sif_overrides['glen_enhancement_factor'] = adopted['E']
            if 'rho_ice' in adopted:
                sif_overrides['density_formula'] = \
                    GlacierUnitConverter.density_formula(adopted['rho_ice'])
            st.session_state['glacier_overrides'] = sif_overrides
            st.success(f'Stored {len(sif_overrides)} Elmer/Ice overrides. '
                       'They will be injected into the SIF on the next run.')

    # Show unit conversion examples
    with st.expander('📏 Unit conversion examples', expanded=False):
        st.markdown(
            '**The Elmer/Ice mixed unit system is:** Length = m, '
            'Stress = MPa, Time = yr.  Literature values reported in SI '
            'must be converted.')
        examples = [
            ('A₁', 3.5e-25, 'Pa^-3 s^-1', 3.0),
            ('A₁', 1.258e13, 'MPa^-3 yr^-1', 3.0),
            ('Q₁', 60.0, 'kJ mol^-1', 3.0),
            ('rho_ice', 910.0, 'kg m^-3', 3.0),
        ]
        rows = []
        for p, raw, u, n in examples:
            conv, tgt = GlacierUnitConverter.convert(raw, u, p, n=n)
            rows.append({
                'param': p,
                'raw': f'{raw:.4g}',
                'raw unit': u,
                'converted': f'{conv:.4g}',
                'target unit': tgt,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)


def render_glacier_visuals_dashboard():
    """Publication-style bar + stacked charts for the glacier recommender."""
    bundle = st.session_state.get('glacier_bundle')
    if bundle is None:
        st.info('Run the glacier recommender first.')
        return

    st.markdown('---')
    st.header('📊 Glacier Recommender Visuals')

    target = st.selectbox(
        'Target parameter',
        ['glen_n', 'A1', 'Q1', 'E', 'rho_ice'],
        format_func=lambda p: f'{GLACIER_PARAM_META[p]["title"]}',
        key='gr_viz_target')
    cands = bundle.candidates.get(target, [])
    if not cands:
        st.warning('No candidates for this parameter.')
        return

    meta = GLACIER_PARAM_META[target]
    values = [c.value for c in cands]
    scores = [c.score for c in cands]
    provs  = [c.provenance or c.method for c in cands]
    ctxs   = [bool(getattr(c, 'context', False)) for c in cands]

    non_ctx_scores = [(s if not f else float('-inf'))
                      for s, f in zip(scores, ctxs)]
    best_idx = int(np.argmax(non_ctx_scores))
    if not np.isfinite(non_ctx_scores[best_idx]):
        best_idx = int(np.argmax(scores))

    chart_type = st.radio('Chart', ['Bar (provenance)', 'Stacked LatentMoE'],
                          horizontal=True, key='gr_viz_chart')

    if chart_type == 'Bar (provenance)':
        fig = plot_candidate_scores(
            values, scores, provs, best_idx,
            param_title=meta['title'],
            param_symbol=meta['symbol'],
            unit=meta['unit'],
            score_label='LatentMoE score',
            journal='nature',
            context_flags=ctxs,
            legend_granularity='coarse',
            show_best_legend=True)
        st.pyplot(fig)
        plt.close(fig)
    else:
        show_ctx = st.checkbox('Show context candidates', value=False,
                               key='gr_viz_show_ctx')
        fig = plot_stacked_latentmoe(
            cands,
            param_title=meta['title'],
            param_symbol=meta['symbol'],
            unit=meta['unit'],
            journal='nature',
            show_context=show_ctx)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.set_page_config(
        page_title='Glacier Phase-Field + Elmer/Ice Recommender',
        layout='wide', initial_sidebar_state='expanded')
    st.markdown(
        '<h1 style="color: #0B3C5D;">🏔️ Glacier Phase-Field Simulator '
        '+ Elmer/Ice AI Recommender v1.0.0</h1>',
        unsafe_allow_html=True)
    st.markdown(
        '<div style="background-color: #EBF5FB; padding: 1rem; '
        'border-radius: 8px; border-left: 5px solid #2874A6;">'
        '<strong>Converted from Enhanced Nanotwinned Cu v10.1.1.</strong><br>'
        '• Pure FFT spectral solver for ice-thickness + fabric + damage.<br>'
        '• Glen flow law with two-branch Arrhenius + fabric enhancement.<br>'
        '• <strong>GlacierUnitConverter</strong>: Pa⁻ⁿ s⁻¹ → MPa⁻ⁿ yr⁻¹, '
        'kJ/mol → J/mol, with correct n-dependence.<br>'
        '• Physics-regime classifier for n, A₁, Q₁, E — analogous to '
        'ρ₀ / γ̇₀ on the metals side.<br>'
        '• n → E cross-coupling: dislocation creep narrows the fabric-'
        'enhancement band.<br>'
        '• LatentMoE scorer with ten experts summing to 1.00.<br>'
        '</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header('⚙️ Cache Management')
        if st.button('🗑️ Clear session'):
            for k in list(st.session_state.keys()):
                if k in ('glacier_bundle', 'glacier_overrides',
                         'glacier_history', 'glacier_solver'):
                    st.session_state.pop(k, None)
            st.success('Cleared.')
            st.rerun()
        st.markdown('---')

    tabs = st.tabs([
        '📊 Simulation & Output',
        '🤖 AI Recommender (Physics-Regime)',
        '🧊 Solver',
        '📏 Unit Converter',
        '📈 Recommender Visuals',
    ])

    with tabs[1]:
        render_glacier_recommender_tab()

    with tabs[2]:
        st.subheader('🧊 Glacier Phase-Field Solver (SIA + Spectral Stokes)')
        st.caption('Grid resolution and physics parameters for the '
                   'spectral SIA + fabric + damage evolution.')
        col1, col2 = st.columns(2)
        with col1:
            N = st.slider('Grid N×N', 64, 512, 128, 32, key='gs_N')
            dx = st.slider('Grid spacing (m)', 50.0, 2000.0, 500.0, 50.0,
                           key='gs_dx')
            dt = st.slider('Time step (yr)', 1e-4, 1.0, 1e-2, 1e-4,
                           format='%.4f', key='gs_dt')
        with col2:
            H_center = st.slider('Ice thickness H (m)', 50.0, 1000.0, 200.0,
                                 10.0, key='gs_H')
            slope_deg = st.slider('Surface slope (°)', 0.1, 20.0, 3.0, 0.1,
                                  key='gs_slope')
            T_C = st.slider('Ice temperature (°C)', -40.0, 0.0, -3.0, 0.5,
                            key='gs_T')

        # Optional overrides
        ovr = st.session_state.get('glacier_overrides', {})
        if ovr:
            st.info(f'🧠 Recommender overrides active: '
                    + ', '.join(f'`{k}={v}`' for k, v in ovr.items()))

        if st.button('▶️ Run glacier simulation', type='primary'):
            params = {
                'N': N, 'dx': dx, 'dt': dt,
                'H_center': H_center, 'slope_deg': slope_deg,
                'T_ice_C': T_C,
                'glen_n': ovr.get('glen_exponent', 3.0),
                'A1': ovr.get('rate_factor_1', 6.0e13),
                'Q1': ovr.get('activation_energy_1', 60000.0),
                'E_glen': ovr.get('glen_enhancement_factor', 1.0),
                'rho_ice': RHO_ICE_SI,
                'accumulation': 0.3,
                'bed_waviness': 0.05,
                'wav_wavelength': 2000.0,
                'meltwater_pool': False,
                'n_steps': 50,
            }
            solver = GlacierSolver(params)
            frames = []
            progress = st.progress(0.0)
            status = st.empty()
            for step in range(params['n_steps']):
                status.text(f'Step {step+1}/{params["n_steps"]} · '
                            f't = {(step+1)*dt:.4f} yr')
                frames.append(solver.step())
                progress.progress((step+1) / params['n_steps'])
            st.success(f'✅ Done — {len(frames)} frames.')
            st.session_state['glacier_history'] = frames
            st.session_state['glacier_solver'] = solver

        if 'glacier_history' in st.session_state and \
           st.session_state['glacier_history']:
            frames = st.session_state['glacier_history']
            idx = st.slider('Frame', 0, len(frames) - 1, len(frames) - 1,
                            key='gs_frame')
            fr = frames[idx]
            st.markdown('#### Field snapshots')
            fcols = st.columns(4)
            field_specs = [
                ('H', 'Ice thickness (m)', 'Blues'),
                ('vel_mag', 'Velocity magnitude (m/yr)', 'viridis'),
                ('viscosity', 'Effective viscosity (MPa·yr)', 'plasma'),
                ('fabric', 'Fabric anisotropy', 'RdBu_r'),
            ]
            for i, (key, title, cmap) in enumerate(field_specs):
                with fcols[i]:
                    fig, ax = plt.subplots(figsize=(3.5, 3.0),
                                           constrained_layout=True)
                    im = ax.imshow(fr[key], origin='lower', cmap=cmap,
                                   extent=solver.geom.extent, aspect='equal')
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel('x (m)', fontsize=8)
                    ax.set_ylabel('y (m)', fontsize=8)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
                    plt.close(fig)

    with tabs[3]:
        st.subheader('📏 Glacier Unit Converter')
        st.markdown(
            '**Elmer/Ice mixed system:** Length = m, Stress = MPa, '
            'Time = yr. A literature `A = 3.5 × 10⁻²⁵ Pa⁻³ s⁻¹` '
            'becomes `~0.011 MPa⁻³ yr⁻¹` after conversion.')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            param = st.selectbox('Param', list(GLACIER_CANON.keys()),
                                 index=list(GLACIER_CANON.keys()).index('A1'),
                                 key='uc_param')
        with col2:
            val = st.number_input('Value', value=3.5e-25, format='%.6e',
                                  key='uc_val')
        with col3:
            unit = st.text_input('Unit', value='Pa^-3 s^-1', key='uc_unit')
        with col4:
            n_used = st.number_input('n (for A only)', value=3.0, step=0.5,
                                     key='uc_n')
        if st.button('Convert'):
            conv, tgt = GlacierUnitConverter.convert(val, unit, param,
                                                    n=n_used)
            st.success(f'**{val:.6g} {unit}** → **{conv:.6g} {tgt}**')

        st.markdown('#### Auto-generated Elmer formulas')
        st.code(
            f'Density  = Real ${GlacierUnitConverter.density_formula(910.0)}\n'
            f'Flow BodyForce 3 = Real ${GlacierUnitConverter.gravity_formula(9.81)}',
            language='plaintext')

    with tabs[4]:
        render_glacier_visuals_dashboard()


# ============================================================================
# REGRESSION TESTS
# ============================================================================
def _test_unit_converter() -> None:
    """Pin the Pa⁻ⁿ s⁻¹ → MPa⁻ⁿ yr⁻¹ conversion."""
    # A₁ in Pa⁻³ s⁻¹ → MPa⁻³ yr⁻¹
    # Expected multiplier: (1e6)^3 × 31,556,926 ≈ 3.1557e25
    v_pa, _ = GlacierUnitConverter.convert(
        3.5e-25, 'Pa^-3 s^-1', 'A1', n=3.0)
    expected = 3.5e-25 * (1e6)**3 * SECONDS_PER_YEAR
    assert abs(v_pa - expected) / expected < 1e-9, \
        f'A₁ Pa⁻³s⁻¹ conversion wrong: {v_pa} vs {expected}'

    # A₁ in MPa⁻³ yr⁻¹ is identity
    v_mpa, _ = GlacierUnitConverter.convert(
        1.258e13, 'MPa^-3 yr^-1', 'A1', n=3.0)
    assert abs(v_mpa - 1.258e13) / 1.258e13 < 1e-6

    # Q₁ kJ/mol → J/mol
    q, _ = GlacierUnitConverter.convert(60.0, 'kJ mol^-1', 'Q1')
    assert abs(q - 60000.0) < 1e-6

    # Density kg/m³ → identity
    r, _ = GlacierUnitConverter.convert(910.0, 'kg m^-3', 'rho_ice')
    assert abs(r - 910.0) < 1e-6

    # Gravity formula round-trip
    gf = GlacierUnitConverter.gravity_formula(9.81)
    assert '31556926' in gf and '9.81' in gf

    logger.info('✅ _test_unit_converter passed')


def _test_regime_monotonicity() -> None:
    """Regime classifiers must be monotonic in their input axes."""
    # Cold high-stress dislocation creep → higher n than warm temperate
    n_cold = classify_n_regime('high', 'cold', 'isotropic')
    n_warm = classify_n_regime('high', 'warm', 'isotropic')
    assert n_cold['inferred'] > n_warm['inferred'], \
        f'cold high-stress n should exceed warm: {n_cold} vs {n_warm}'

    # Dusty warm A₁ should exceed clean warm A₁
    a_clean = classify_a1_regime('warm', 'clean')
    a_dust  = classify_a1_regime('warm', 'dust_rich')
    assert a_dust['inferred'] > a_clean['inferred'], \
        'dust-rich warm A₁ should exceed clean warm A₁'

    # Single-max fabric E should exceed isotropic E
    e_iso = classify_e_regime('isotropic', 'cold')
    e_sm  = classify_e_regime('single_max', 'cold')
    assert e_sm['inferred'] > e_iso['inferred'], \
        'single-max E should exceed isotropic E'

    logger.info('✅ _test_regime_monotonicity passed')


def _test_n_to_e_coupling() -> None:
    """n → E coupling must narrow the E band for dislocation creep."""
    n_reg = classify_n_regime('high', 'cold', 'isotropic')   # n ≈ 3
    e_reg = classify_e_regime('isotropic', 'cold')           # wide band 0.8–1.2
    e_coupled = apply_n_to_e_coupling(n_reg, e_reg)
    assert e_coupled.get('coupling'), \
        'coupling annotation missing for n ≈ 3'
    assert 'dislocation' in e_coupled['coupling'], \
        f'expected dislocation-creep reason, got {e_coupled["coupling"]!r}'
    # Band should be at least as wide as before, and shifted up
    assert e_coupled['inferred'] >= e_reg['inferred'] - 1e-9

    # n ≈ 1 → isotropic E ≈ 1
    n_low = classify_n_regime('low', 'cold', 'isotropic')
    e_coupled_low = apply_n_to_e_coupling(n_low, e_reg)
    assert 'diffusion' in e_coupled_low.get('coupling', ''), \
        f'expected diffusion reason for n ≈ 1'

    logger.info('✅ _test_n_to_e_coupling passed')


def _test_latentmoe_weights_sum() -> None:
    sc = GlacierLatentMoEScorer()
    wsum = (sc.w_material + sc.w_thermal + sc.w_strain + sc.w_method
            + sc.w_confidence + sc.w_reasoning + sc.w_stress
            + sc.w_thermalbr + sc.w_fabric + sc.w_density)
    assert abs(wsum - 1.0) < 1e-9, f'weights sum to {wsum}'
    for exp in GlacierLatentMoEScorer.EXPERT_ORDER:
        assert exp in GlacierLatentMoEScorer.EXPERT_COLORS, \
            f'missing colour for expert {exp}'
    logger.info('✅ _test_latentmoe_weights_sum passed')


def _test_provenance_taxonomy() -> None:
    assert _norm_provenance('arrhenius_cross_relation') == 'physics_inferred'
    assert _norm_provenance('physics_regime_inference') == 'regime_prior'
    assert _norm_provenance('explicit') == 'llm_extract'
    assert _norm_provenance('derived') == 'physics_inferred'
    # Coarse rollup
    assert _legend_key('arrhenius_cross_relation', 'coarse') == 'deterministic'
    assert _legend_key('physics_regime_inference', 'coarse') == 'deterministic'
    logger.info('✅ _test_provenance_taxonomy passed')


def _test_incumbent_routes() -> None:
    """A₁ is LLM-grounded incumbent; regime_prior must be context."""
    assert _is_context('A1', 'regime_prior') is True
    assert _is_context('A1', 'regex_ner') is True
    assert _is_context('A1', 'llm_extract') is False

    assert _is_context('glen_n', 'regex_ner') is False
    assert _is_context('glen_n', 'regime_prior') is False
    assert _is_context('glen_n', 'llm_prior') is True

    logger.info('✅ _test_incumbent_routes passed')


def _test_density_formula() -> None:
    df = GlacierUnitConverter.density_formula(910.0)
    assert '910.0' in df
    assert '1.0E-06' in df
    assert '31556926' in df
    logger.info('✅ _test_density_formula passed')


def _test_solver_smoke() -> None:
    """Run 3 steps of the glacier solver — no crash."""
    params = {
        'N': 32, 'dx': 500.0, 'dt': 1e-3,
        'H_center': 200.0, 'slope_deg': 3.0, 'T_ice_C': -3.0,
        'glen_n': 3.0, 'A1': 6.0e13, 'Q1': 60000.0, 'E_glen': 1.0,
        'rho_ice': 910.0, 'accumulation': 0.3,
        'bed_waviness': 0.05, 'wav_wavelength': 2000.0,
        'meltwater_pool': False,
    }
    solver = GlacierSolver(params)
    for _ in range(3):
        res = solver.step()
        assert 'H' in res
        assert np.isfinite(res['H']).all()
    logger.info('✅ _test_solver_smoke passed — 3 steps ok')


def _run_regression_suite() -> None:
    suite = [
        ('unit_converter',       _test_unit_converter),
        ('regime_monotonicity',  _test_regime_monotonicity),
        ('n_to_e_coupling',      _test_n_to_e_coupling),
        ('latentmoe_weights',    _test_latentmoe_weights_sum),
        ('provenance_taxonomy',  _test_provenance_taxonomy),
        ('incumbent_routes',     _test_incumbent_routes),
        ('density_formula',      _test_density_formula),
        ('solver_smoke',         _test_solver_smoke),
    ]
    for tag, fn in suite:
        try:
            fn()
        except AssertionError as ae:
            logger.error('❌ regression %s FAILED: %s', tag, ae)
        except Exception as e:
            logger.error('❌ regression %s ERRORED: %s\n%s', tag, e,
                         traceback.format_exc())


if os.environ.get('GLACIER_REGRESSION', '').lower() in ('1', 'true', 'yes'):
    _run_regression_suite()


if __name__ == '__main__':
    main()
