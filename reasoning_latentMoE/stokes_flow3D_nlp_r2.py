# ============================================================================
# ███ HIMALAYAN GLACIER 3D — ELMERSOLVER LAUNCHER + AI RECOMMENDER v1.1.0 ███
# ███                                                                  ███
# ███ ElmerSolver SIF generation + subprocess execution: INTACT.       ███
# ███ Full nt-Cu-derived recommender pipeline restored:               ███
# ███   · OllamaClient (graceful fallback if unreachable)             ███
# ███   · HybridRetriever (FAISS + SBERT, TF-IDF fallback)            ███
# ███   · 3-tier cascade (grounded LLM → regex NER → prior/regime)    ███
# ███   · Physics-regime classifiers for n, A₁, Q₁, E                 ███
# ███   · GlacierUnitConverter (Pa⁻ⁿ s⁻¹ → MPa⁻ⁿ yr⁻¹, kJ/mol → J/mol) ███
# ███   · 10-expert LatentMoE scorer summing to 1.00                  ███
# ███   · Provenance taxonomy (llm_extract / regex_ner / regime_prior /│███
# ███     physics_inferred / llm_prior)                               ███
# ███   · Full 6-fine-key / 3-coarse-bucket legend                    ███
# ███   · n → E cross-coupling                                         ███
# ███   · Union + fallback cascade modes                              ███
# ███                                                                  ███
# ███ FIX: AttributeError from PlasticityCandidate alias → proper     ███
# ███      GlacierCandidate dataclass with score + moe_breakdown +    ███
# ███      provenance + context + theory + reasoning + dominant_factor███
# ███ FIX: ElmerSolver execution block restored verbatim              ███
# ███ FIX: Material tab reads overrides, refreshes on adopt           ███
# ███ FIX: SIF generator reads st.session_state['glacier_overrides']  ███
# ============================================================================

import streamlit as st
import subprocess
import pathlib
import os
import re
import json
import math
import time
import hashlib
import logging
import unicodedata
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO

import numpy as np
import pandas as pd

# ── Optional dependencies ──────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS — ELMER/ICE MIXED UNIT SYSTEM
# ════════════════════════════════════════════════════════════════════════════
SECONDS_PER_YEAR = 31556926.0
KG_M3_TO_ELMER   = 1.0e-6
RHO_ICE_SI       = 910.0
G_SI             = 9.81
G_ELMER          = G_SI * SECONDS_PER_YEAR ** 2

# ════════════════════════════════════════════════════════════════════════════
# 1. GLACIER UNIT CONVERTER
# ════════════════════════════════════════════════════════════════════════════
class GlacierUnitConverter:
    """Converts literature units to the Elmer/Ice mixed system (MPa, m, yr)."""

    @staticmethod
    def _clean(unit: str) -> str:
        if not unit:
            return ''
        s = unit.lower().strip()
        s = s.replace('·', ' ').replace('*', ' ').replace('⁻', '^-')
        s = s.replace('–', '-').replace('−', '-')
        s = re.sub(r'[\s/]+', ' ', s)
        return s.strip()

    @staticmethod
    def _pressure_exponent(u: str, default_n: float) -> float:
        m = re.search(r'(pa|kpa|mpa|gpa|bar)(?:\^?)\s*([+-]?\d+(?:\.\d+)?|n)',
                      u)
        if not m:
            return default_n
        exp_str = m.group(2)
        if exp_str == 'n':
            return default_n
        try:
            return abs(float(exp_str))
        except ValueError:
            return default_n

    @staticmethod
    def convert(value: float, unit: str, param_key: str,
                n: float = 3.0) -> Tuple[float, str]:
        v = float(value)
        u = GlacierUnitConverter._clean(unit)

        if param_key in ('glen_n', 'E', 'fabric'):
            return v, 'dimensionless'

        if param_key in ('Q1', 'Q2'):
            if 'kj' in u or 'kilojoule' in u:
                return v * 1000.0, 'J mol^-1'
            return v, 'J mol^-1'

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

            parsed_n = GlacierUnitConverter._pressure_exponent(u, n)

            if pres_root == 'pa':
                p_mult = 10.0 ** (6.0 * parsed_n)
            elif pres_root == 'kpa':
                p_mult = 10.0 ** (3.0 * parsed_n)
            elif pres_root == 'gpa':
                p_mult = 10.0 ** (-3.0 * parsed_n)
            elif pres_root == 'bar':
                p_mult = 10.0 ** (1.0 * parsed_n)
            else:
                p_mult = 1.0

            if time_root == 's':
                t_mult = SECONDS_PER_YEAR
            elif time_root == 'day':
                t_mult = 365.25
            else:
                t_mult = 1.0

            return v * p_mult * t_mult, f'MPa^-{parsed_n:.1f} yr^-1'

        if param_key == 'rho_ice':
            if 'g/cm3' in u or 'g cm^-3' in u:
                return v * 1000.0, 'kg m^-3'
            return v, 'kg m^-3'

        if param_key in ('H', 'd_grain', 'l_char'):
            if 'km' in u: return v * 1000.0, 'm'
            if 'mm' in u: return v * 1.0e-3, 'm'
            if 'cm' in u: return v * 1.0e-2, 'm'
            return v, 'm'

        if param_key == 'gamma_rate':
            if 's^-1' in u or '/s' in u or 's-1' in u:
                return v * SECONDS_PER_YEAR, 'yr^-1'
            return v, 'yr^-1'

        if param_key in ('T_ice', 'T_pmp'):
            if u.strip() == 'k' or 'kelvin' in u:
                return v - 273.15, 'deg C'
            return v, 'deg C'

        return v, u

    @staticmethod
    def density_formula(rho_si: float = RHO_ICE_SI) -> str:
        return (f"{rho_si}*1.0E-06*"
                f"({int(SECONDS_PER_YEAR)}.0)^(-2.0)")

    @staticmethod
    def gravity_formula(g_si: float = G_SI) -> str:
        return f"{g_si} * ({int(SECONDS_PER_YEAR)}.0)^(2.0)"


# ════════════════════════════════════════════════════════════════════════════
# 2. GLACIER ONTOLOGY + CANON
# ════════════════════════════════════════════════════════════════════════════
GLACIER_CANON: Dict[str, Dict[str, Any]] = {
    'glen_n': dict(
        aliases=['glen exponent', "glen's exponent", 'stress exponent',
                 'creep exponent', 'power-law exponent', 'n exponent',
                 'glen flow exponent', 'flow-law exponent'],
        unit=None, plausible=(0.5, 5.0)),
    'A1': dict(
        aliases=['rate factor', 'arrhenius rate factor', 'rate factor warm',
                 'pre-exponential factor', 'a_1', 'a1', 'softness parameter',
                 'warm rate factor', 'creep rate factor'],
        unit='MPa^-n yr^-1', plausible=(1e10, 1e16)),
    'A2': dict(
        aliases=['rate factor cold', 'cold rate factor', 'a_2', 'a2',
                 'arrhenius rate factor cold'],
        unit='MPa^-n yr^-1', plausible=(1e9, 1e16)),
    'Q1': dict(
        aliases=['activation energy', 'activation energy warm',
                 'arrhenius activation energy', 'creep activation energy',
                 'q_1', 'q1', 'warm activation energy'],
        unit='J mol^-1', plausible=(2.0e4, 1.0e5)),
    'Q2': dict(
        aliases=['activation energy cold', 'cold activation energy',
                 'q_2', 'q2'],
        unit='J mol^-1', plausible=(2.0e4, 1.5e5)),
    'E': dict(
        aliases=['enhancement factor', 'glen enhancement factor',
                 'anisotropy enhancement', 'e_glen', 'softness multiplier',
                 'enhancement parameter'],
        unit=None, plausible=(0.3, 15.0)),
    'rho_ice': dict(
        aliases=['ice density', 'density of ice', 'rho_ice', 'rho ice',
                 'glacier density', 'firn density'],
        unit='kg m^-3', plausible=(400.0, 950.0)),
    'T_ice': dict(
        aliases=['ice temperature', 'temperature', 'ice temperature c'],
        unit='deg C', plausible=(-60.0, 0.0)),
    'accumulation': dict(
        aliases=['accumulation', 'surface mass balance', 'smb',
                 'mass balance', 'accumulation rate'],
        unit='m yr^-1', plausible=(-2.0, 20.0)),
    'H': dict(
        aliases=['ice thickness', 'glacier thickness', 'thickness',
                 'ice depth'],
        unit='m', plausible=(1.0, 1.0e4)),
    'd_grain': dict(
        aliases=['grain size', 'ice grain size', 'crystal size',
                 'grain diameter'],
        unit='m', plausible=(1.0e-5, 1.0e-1)),
    'w_water': dict(
        aliases=['water content', 'meltwater content',
                 'liquid water content', 'moisture content'],
        unit='%', plausible=(0.0, 20.0)),
}

_POSITIVE_LOWER_BOUND = frozenset({
    'glen_n', 'A1', 'A2', 'Q1', 'Q2', 'E', 'rho_ice', 'H',
    'd_grain', 'accumulation',
})

GLACIER_PARAM_META: Dict[str, Dict[str, str]] = {
    'glen_n': dict(title='Glen Flow Exponent', symbol='n', unit=None),
    'A1':     dict(title='Rate Factor (warm branch)',
                   symbol='A_1', unit='MPa^{-n}\\,yr^{-1}'),
    'A2':     dict(title='Rate Factor (cold branch)',
                   symbol='A_2', unit='MPa^{-n}\\,yr^{-1}'),
    'Q1':     dict(title='Activation Energy (warm branch)',
                   symbol='Q_1', unit='J\\,mol^{-1}'),
    'Q2':     dict(title='Activation Energy (cold branch)',
                   symbol='Q_2', unit='J\\,mol^{-1}'),
    'E':      dict(title='Glen Enhancement Factor', symbol='E', unit=None),
    'rho_ice': dict(title='Ice Density', symbol=r'\rho_i', unit='kg\\,m^{-3}'),
    'H':      dict(title='Ice Thickness', symbol='H', unit='m'),
    'd_grain': dict(title='Grain Size', symbol='d', unit='m'),
    'w_water': dict(title='Water Content', symbol='w', unit='\\%'),
    'T_ice':  dict(title='Ice Temperature', symbol='T', unit='{}^\\circ C'),
    'accumulation': dict(title='Surface Mass Balance', symbol=r'\dot{b}',
                        unit='m\\,yr^{-1}'),
}


# ════════════════════════════════════════════════════════════════════════════
# 3. GAZETTEER + NER
# ════════════════════════════════════════════════════════════════════════════
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
                 'water content', 'meltwater content'],
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
             'kg/m3', 'kg m-3', 'g/cm3', 'm', 'km', 'mm', 'cm',
             's^-1', 'yr^-1', '/yr', 'a^-1', 'deg C', 'K'],
}

_CHAR_FOLD = {
    'μ': 'mu', 'µ': 'mu', 'ρ': 'rho', '–': '-', '—': '-',
    '’': "'", '\u00a0': ' ', 'γ': 'gamma', 'σ': 'sigma', 'τ': 'tau',
    'λ': 'lambda', 'θ': 'theta', 'ε': 'epsilon', 'η': 'eta',
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


_NUM_ANY = re.compile(
    r"(?<![a-z0-9.])"
    r"(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"(?:\s*[x×*]\s*10\s*\^?\(?\s*(?P<exp>[+-]?\d+)\s*\)?)?"
)


def _num_value(m: 're.Match') -> float:
    v = float(m.group('val'))
    return v * 10.0 ** int(m.group('exp')) if m.group('exp') else v


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


def heuristic_extract(text_norm: str, text_raw: str, param_key: str,
                      source: str = '') -> List[Dict[str, Any]]:
    aliases = GLACIER_CANON.get(param_key, {}).get('aliases', [])
    patterns = [re.compile(alias_pattern(a)) for a in aliases]
    out: List[Dict[str, Any]] = []
    for p in patterns:
        for m in p.finditer(text_norm):
            window = text_norm[m.end(): m.end() + 100]
            for nm in _NUM_ANY.finditer(window):
                v = _num_value(nm)
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
                break
    return out


# ════════════════════════════════════════════════════════════════════════════
# 4. REGIME TABLES
# ════════════════════════════════════════════════════════════════════════════
N_REGIMES: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ('low', 'cold', 'isotropic'): dict(
        low=1.0, high=2.0, bench=1.5,
        desc='Low-stress divide, cold ice: diffusion creep → n ≈ 1'),
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
        desc='Paterson & Budd 1982 clean-ice warm branch (T ≥ −10 °C)'),
    ('warm', 'dust_rich'): dict(
        low=8.0e13, high=2.5e14, bench=1.5e14,
        desc='Dust/impurity-laden warm ice'),
    ('warm', 'marine'): dict(
        low=5.0e13, high=1.2e14, bench=8.0e13,
        desc='Marine ice with brine networks'),
    ('warm', 'temperate'): dict(
        low=5.0e14, high=2.0e15, bench=1.0e15,
        desc='Temperate (pressure-melting point): water-assisted'),
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
        desc='Single-maximum fabric (vertical c-axis)'),
    ('single_max', 'warm'): dict(
        low=3.0, high=8.0, bench=5.0,
        desc='Warm strong fabric: maximum ice-stream enhancement'),
    ('multi_max', 'cold'): dict(
        low=1.5, high=3.0, bench=2.0,
        desc='Multi-maximum fabric: moderate enhancement'),
    ('shear_margin', 'warm'): dict(
        low=5.0, high=10.0, bench=7.0,
        desc='Shear-margin fabric with water: extreme enhancement'),
}

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


# ════════════════════════════════════════════════════════════════════════════
# 5. REGIME CLASSIFIERS
# ════════════════════════════════════════════════════════════════════════════
def _norm_stress(s: Optional[str]) -> Optional[str]:
    s = (s or '').lower()
    if any(k in s for k in ('low', 'divide', 'stagnant')): return 'low'
    if any(k in s for k in ('intermediate', 'mid', 'transition')):
        return 'intermediate'
    if any(k in s for k in ('high', 'stream', 'shear margin', 'outlet',
                            'surge')): return 'high'
    return None


def _norm_thermal(t: Optional[str]) -> Optional[str]:
    t = (t or '').lower()
    if any(k in t for k in ('warm', 'temperate', 'melt', 'wet')): return 'warm'
    if any(k in t for k in ('cold', 'polar', 'sub-freezing', 'dry')): return 'cold'
    return None


def _norm_fabric(f: Optional[str]) -> Optional[str]:
    f = (f or '').lower()
    if 'single' in f and 'max' in f: return 'single_max'
    if 'multi' in f and 'max' in f: return 'multi_max'
    if 'shear' in f and 'margin' in f: return 'shear_margin'
    if 'iso' in f or 'random' in f: return 'isotropic'
    return None


def _norm_impurity(i: Optional[str]) -> Optional[str]:
    i = (i or '').lower()
    if 'dust' in i or 'mineral' in i or 'ash' in i: return 'dust_rich'
    if 'marine' in i or 'brine' in i: return 'marine'
    if 'temperate' in i or 'water' in i: return 'temperate'
    if 'clean' in i or 'pure' in i: return 'clean'
    return None


def _norm_moisture(m: Optional[str]) -> Optional[str]:
    m = (m or '').lower()
    if 'dry' in m: return 'dry'
    if 'moist' in m or 'wet' in m: return 'moist'
    if 'marine' in m or 'brine' in m: return 'marine'
    return None


def classify_n_regime(stress, thermal, fabric) -> Dict[str, Any]:
    sk, tk, fk = _norm_stress(stress), _norm_thermal(thermal), _norm_fabric(fabric)
    regime = N_REGIMES.get((sk, tk, fk)) if (sk and tk and fk) else None
    if regime is None:
        regime = N_REGIMES[('high', 'cold', 'isotropic')]
        return dict(regime='unclassified → fallback (high,cold,isotropic)',
                    low=regime['low'], high=regime['high'],
                    bench=regime['bench'], inferred=regime['bench'],
                    source='fallback')
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({sk},{tk},{fk})')


def classify_a1_regime(thermal, impurity) -> Dict[str, Any]:
    bk = _norm_thermal(thermal) or 'warm'
    ik = _norm_impurity(impurity) or 'clean'
    regime = A1_REGIMES.get((bk, ik), A1_REGIMES[('warm', 'clean')])
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({bk},{ik})')


def classify_q1_regime(thermal, moisture) -> Dict[str, Any]:
    bk = _norm_thermal(thermal) or 'warm'
    mk = _norm_moisture(moisture) or 'dry'
    regime = Q1_REGIMES.get((bk, mk), Q1_REGIMES[('warm', 'dry')])
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({bk},{mk})')


def classify_e_regime(fabric, thermal) -> Dict[str, Any]:
    fk = _norm_fabric(fabric) or 'isotropic'
    bk = _norm_thermal(thermal) or 'cold'
    regime = E_REGIMES.get((fk, bk), E_REGIMES[('isotropic', 'cold')])
    return dict(regime=regime['desc'],
                low=regime['low'], high=regime['high'],
                bench=regime['bench'], inferred=regime['bench'],
                source=f'classifier({fk},{bk})')


def apply_n_to_e_coupling(n_regime, e_regime) -> Dict[str, Any]:
    n_center = n_regime['inferred']
    for (lo, hi), info in N_TO_E_COUPLING:
        if lo <= n_center <= hi:
            e_lo, e_hi = info['E_band']
            new = dict(e_regime)
            new['low']  = max(e_regime['low'],  e_lo)
            new['high'] = min(e_regime['high'], e_hi)
            if new['high'] < new['low']:
                new['low'], new['high'] = e_lo, e_hi
            new['inferred'] = 0.5 * (new['low'] + new['high'])
            new['coupling'] = info['reason']
            return new
    return e_regime


# ════════════════════════════════════════════════════════════════════════════
# 6. PROVENANCE TAXONOMY
# ════════════════════════════════════════════════════════════════════════════
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
COARSE_PROVENANCE_KEYS = tuple(PROVENANCE_GROUPS.keys())

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

INCUMBENT_ROUTES: Dict[str, Tuple[str, ...]] = {
    'glen_n':  ('deterministic',),
    'A1':      ('llm_grounded',),
    'Q1':      ('deterministic',),
    'E':       ('deterministic',),
    'rho_ice': ('deterministic',),
}

_DERIVED_HINTS: Tuple[str, ...] = (
    'physics_inferred', 'derived', 'arrhenius', 'cross_relation',
    'two_temperature_inversion', 'solver_law', 'model_inversion',
    'consensus', 'n_to_e_coupling',
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


def _legend_key(prov: Any, granularity: str = 'coarse') -> str:
    fine = _norm_provenance(prov)
    if granularity == 'fine':
        return fine
    return _FINE_TO_GROUP.get(fine, 'deterministic')


def _is_context(param: str, prov: Any) -> bool:
    bucket = _FINE_TO_GROUP.get(_norm_provenance(prov), 'deterministic')
    pinned = INCUMBENT_ROUTES.get(param, ())
    return bool(pinned) and bucket not in pinned


def _stamp_provenance(ext: Dict[str, Any], fine_key: str) -> None:
    if fine_key not in FINE_PROVENANCE_KEYS:
        raise ValueError(f'not a fine key: {fine_key!r}')
    ext['_provenance'] = fine_key


# ════════════════════════════════════════════════════════════════════════════
# 7. CANDIDATE DATACLASS — FIXED (no alias!)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class GlacierCandidate:
    """Unified candidate dataclass — includes score, moe_breakdown,
    provenance, context, theory, reasoning, dominant_factor.  No aliasing."""
    param: str
    value: float
    raw_value: float
    raw_unit: str
    score: float
    confidence: float
    method: str
    material: str = 'ice'
    temp_k: Optional[float] = None
    strain_rate: Optional[float] = None
    source_file: str = ''
    source_title: str = ''
    evidence: str = ''
    reasoning: str = ''
    provenance: str = ''
    theory: str = ''
    dominant_factor: str = ''
    context: bool = False
    clamped: bool = False
    moe_breakdown: Dict[str, float] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# 8. GATEKEEPER (unit conversion + plausibility + conf cap)
# ════════════════════════════════════════════════════════════════════════════
def gatekeep(items, param_key, provenance='regex_ner',
             conf_cap=None, n_for_conversion: float = 3.0,
             material: str = 'ice', temp_k: Optional[float] = None,
             strain_rate: Optional[float] = None):
    canon = GLACIER_CANON.get(param_key, {})
    lo, hi = canon.get('plausible', (float('-inf'), float('inf')))

    if conf_cap is None:
        if provenance in ('llm_prior', 'regime_prior'):
            conf_cap = 0.5
        elif provenance == 'physics_inferred':
            conf_cap = 0.75
        else:
            conf_cap = 1.0

    reject_non_positive = param_key in _POSITIVE_LOWER_BOUND
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
            continue
        if not (lo <= v <= hi):
            continue

        key = round(v, 8)
        if key in seen:
            continue
        seen.add(key)

        out.append(GlacierCandidate(
            param=param_key,
            value=v,
            raw_value=float(raw_v),
            raw_unit=raw_u,
            score=float(it.get('score', 0.0)),
            confidence=min(float(it.get('confidence', 1.0)), conf_cap),
            method=str(it.get('method', '')),
            material=material,
            temp_k=temp_k,
            strain_rate=strain_rate,
            source_file=str(it.get('source', '')),
            source_title=str(it.get('property_label', '')),
            evidence=str(it.get('evidence_span') or it.get('evidence') or ''),
            reasoning=str(it.get('reasoning') or ''),
            provenance=provenance,
            theory=str(it.get('theory') or ''),
            dominant_factor=str(it.get('dominant_factor') or ''),
            context=bool(it.get('_context', False)),
            clamped=False,
        ))
    return sorted(out, key=lambda c: -c.confidence)


# ════════════════════════════════════════════════════════════════════════════
# 9. OLLAMA CLIENT
# ════════════════════════════════════════════════════════════════════════════
class OllamaClient:
    def __init__(self, url='http://localhost:11434', model='qwen2.5:7b',
                 timeout=120.0):
        self.url = url.rstrip('/')
        self.model = model
        self.timeout = timeout

    @staticmethod
    def is_available(url='http://localhost:11434') -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        try:
            r = _requests.get(f'{url.rstrip("/")}/api/tags', timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def list_models(url='http://localhost:11434') -> List[str]:
        if not REQUESTS_AVAILABLE:
            return []
        try:
            r = _requests.get(f'{url.rstrip("/")}/api/tags', timeout=3.0)
            if r.status_code == 200:
                return sorted(m.get('name', '')
                              for m in r.json().get('models', []))
        except Exception:
            pass
        return []

    def generate(self, prompt: str, system: Optional[str] = None,
                 timeout: Optional[float] = None) -> Optional[str]:
        if not REQUESTS_AVAILABLE:
            return None
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'format': 'json',
            'options': {'temperature': 0.1, 'top_p': 0.9, 'num_predict': 4096},
        }
        if system:
            payload['system'] = system
        try:
            r = _requests.post(f'{self.url}/api/generate', json=payload,
                               timeout=timeout or self.timeout)
            r.raise_for_status()
            return r.json().get('response', '')
        except Exception as e:
            logger.warning('Ollama failed: %s', e)
            return None


def _parse_llm_json(raw: str) -> list:
    if not raw:
        return []
    txt = re.sub(r'<think>.*?</think>', '', str(raw), flags=re.S)
    txt = txt.replace('```json', '').replace('```', '')
    starts = [i for i in (txt.find('['), txt.find('{')) if i >= 0]
    if not starts:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(txt[min(starts):])
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return []


# ════════════════════════════════════════════════════════════════════════════
# 10. HYBRID RETRIEVER
# ════════════════════════════════════════════════════════════════════════════
def _record_text(rec: dict) -> str:
    if isinstance(rec, dict):
        parts = []
        for k in ('Title', 'title', 'Abstract', 'abstract',
                  'Full Text', 'full_text', 'text', 'content'):
            if k in rec:
                parts.append(str(rec[k]))
        if not parts:
            parts.append(json.dumps(rec, default=str))
        return ' '.join(parts)
    return str(rec)


class HybridRetriever:
    def __init__(self, corpus: List[Dict[str, Any]], use_dense=True):
        self.records = []
        for i, doc in enumerate(corpus):
            src = doc.get('_source') or f'rec[{i}]'
            txt = _record_text(doc)
            self.records.append({
                'id': f'{src}[{i}]',
                'source': src,
                'text': txt,
                'text_norm': norm_text(txt),
                'data': doc,
            })
        self.ids = [r['id'] for r in self.records]
        self._txt = {r['id']: r['text_norm'] for r in self.records}
        self.dense = False
        self.model = None
        self.index = None
        if use_dense and FAISS_AVAILABLE and SBERT_AVAILABLE and self.records:
            try:
                self.model = _SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2', device='cpu')
                emb = self.model.encode(
                    list(self._txt.values()),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32).astype(np.float32)
                self.index = _faiss.IndexFlatIP(emb.shape[1])
                self.index.add(np.ascontiguousarray(emb, dtype='float32'))
                self.dense = True
            except Exception as e:
                logger.warning('Dense channel disabled: %s', e)

    def _score(self, text: str, param_key: str, material: str) -> float:
        score = 0.0
        aliases = GLACIER_CANON.get(param_key, {}).get('aliases', [])
        for a in aliases:
            if a in text:
                score += 3.0
        # material bonus
        if material.lower() in text:
            score += 2.0
        # property hint bonus
        for hint in ('glen', 'arrhenius', 'creep', 'rheology', 'fabric',
                     'ice core', 'torsion'):
            if hint in text:
                score += 0.5
        return score

    def search(self, param_key: str, material='ice', k=6) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        lex = []
        for rid in self.ids:
            s = self._score(self._txt[rid], param_key, material)
            if s > 0:
                lex.append((-s, rid))
        lex_sorted = [rid for _, rid in sorted(lex)]
        rank_lists = [lex_sorted]
        if self.dense:
            q = f'{material} {param_key} ' + ' '.join(
                GLACIER_CANON.get(param_key, {}).get('aliases', [])[:4])
            qv = self.model.encode([q],
                                   normalize_embeddings=True).astype(np.float32)
            _, I = self.index.search(np.ascontiguousarray(qv, 'float32'),
                                     min(10, len(self.ids)))
            rank_lists.append([self.ids[i] for i in I[0]])
        # RRF fusion
        agg = {}
        for lst in rank_lists:
            for r, rid in enumerate(lst):
                agg[rid] = agg.get(rid, 0.0) + 1.0 / (60 + r + 1)
        ranked = [rid for rid, _ in sorted(agg.items(), key=lambda kv: -kv[1])]
        id2rec = {r['id']: r for r in self.records}
        return [id2rec[rid] for rid in ranked if rid in id2rec][:k]


# ════════════════════════════════════════════════════════════════════════════
# 11. LATENT MoE SCORER (10 experts, sum = 1.00)
# ════════════════════════════════════════════════════════════════════════════
class GlacierLatentMoEScorer:
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

    @staticmethod
    def _material_expert(ext_mat: str, target: str) -> float:
        if not ext_mat:
            return 0.4
        a, b = ext_mat.lower().strip(), target.lower().strip()
        if a == b: return 1.0
        if 'ice' in a and 'ice' in b: return 0.9
        if a in b or b in a: return 0.85
        return 0.25

    def _thermal_expert(self, ext_temp, target_temp) -> float:
        if ext_temp is None: return 0.5
        try:
            diff = float(ext_temp) - float(target_temp)
        except (TypeError, ValueError):
            return 0.5
        return float(np.exp(-(diff ** 2) / (2.0 * self.thermal_sigma ** 2)))

    @staticmethod
    def _strain_expert(ext_rate, target_rate) -> float:
        if ext_rate is None or target_rate is None:
            return 0.5
        try:
            r = float(ext_rate)
            if r <= 0 or target_rate <= 0: return 0.5
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
            'explicit': 0.9, 'llm_inferred': 0.5,
            'prior inference': 0.35, 'llm_prior': 0.35,
            'default_fallback': 0.3,
            'arrhenius_cross_relation':  0.60,
            'two_temperature_inversion': 0.65,
            'n_to_e_coupling':           0.55,
            'physics_regime_inference':  0.60,
            'theory_regime_inference':   0.60,
            'regex_ner':                 0.55,
        }.get((method or '').lower(), 0.5)

    @staticmethod
    def _reasoning_expert(reasoning: str, method: str) -> float:
        method_l = (method or '').lower()
        cog = ('llm_inferred', 'heuristic', 'default_fallback',
               'prior inference', 'llm_prior',
               'arrhenius_cross_relation', 'two_temperature_inversion',
               'n_to_e_coupling', 'physics_regime_inference',
               'theory_regime_inference')
        if method_l not in cog:
            return 0.5
        if not reasoning:
            return 0.2
        n = reasoning.count('Step ') + reasoning.count('\n')
        formula = any(k in reasoning for k in
                      ('Arrhenius', 'Q₁', 'A(T)', 'MPa', 'yr', 'kJ',
                       'Glen', 'fabric', 'stress', 'thermal'))
        base = min(0.4 + 0.05 * n, 0.8)
        if formula:
            base = min(base + 0.15, 0.85)
        return float(base)

    @staticmethod
    def _stress_expert(v: float, stress, thermal, fabric) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v): return 0.0
        r = classify_n_regime(stress, thermal, fabric)
        lo, hi, ctr = r['low'], r['high'], r['inferred']
        if lo <= v <= hi:
            sigma = max((hi - lo) / 4.0, 0.25)
            return float(np.exp(-(v - ctr) ** 2 / (2.0 * sigma ** 2)))
        dist = min(abs(v - lo), abs(v - hi))
        return float(np.exp(-dist / 1.0))

    @staticmethod
    def _thermal_branch_expert(v: float, target, candidate) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v): return 0.0
        tb = _norm_thermal(target) or 'warm'
        cb = _norm_thermal(candidate) if candidate else tb
        tag = 1.0 if tb == cb else 0.05
        lo = A1_REGIMES[('warm', 'clean')]['low']
        hi = A1_REGIMES[('warm', 'temperate')]['high']
        if lo <= v <= hi:
            regime = 1.0
        else:
            dist = min(abs(np.log10(v) - np.log10(lo)),
                       abs(np.log10(v) - np.log10(hi)))
            regime = float(np.exp(-dist / 1.5))
        return tag * regime

    @staticmethod
    def _fabric_expert(v: float, fabric, thermal) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v): return 0.0
        r = classify_e_regime(fabric, thermal)
        lo, hi, ctr = r['low'], r['high'], r['inferred']
        if lo <= v <= hi:
            sigma = max((hi - lo) / 4.0, 0.2)
            return float(np.exp(-(v - ctr) ** 2 / (2.0 * sigma ** 2)))
        dist = min(abs(v - lo), abs(v - hi))
        return float(np.exp(-dist / 1.0))

    @staticmethod
    def _density_expert(prov_fine: str) -> float:
        if prov_fine in ('llm_extract', 'llm_reasoned', 'regex_ner'):
            return 1.0
        if prov_fine == 'physics_inferred':
            return 0.7
        if prov_fine == 'regime_prior':
            return 0.4
        return 0.2

    def score(self, extractions, target_material='ice',
              target_temp=-3.0, target_strain_rate=0.1, top_k=8,
              stress_regime=None, thermal_state=None, fabric=None):
        buckets: Dict[str, List[GlacierCandidate]] = {
            p: [] for p in GLACIER_CANON
        }
        for ext in extractions:
            p = ext.get('param')
            if p not in buckets:
                continue
            try:
                v_conv, _ = GlacierUnitConverter.convert(
                    ext['value'], ext.get('unit', ''), p,
                    n=float(ext.get('n_for_conversion', 3.0)))
            except Exception:
                continue
            if not np.isfinite(v_conv):
                continue
            lo, hi = GLACIER_CANON[p]['plausible']
            v_clamped = float(np.clip(v_conv, lo, hi))
            was_clamped = (v_clamped != v_conv)

            s_mat  = self._material_expert(ext.get('material', 'ice'),
                                           target_material)
            s_temp = self._thermal_expert(ext.get('temp'), target_temp)
            s_str  = self._strain_expert(ext.get('strain_rate'),
                                         target_strain_rate)
            s_meth = self._method_expert(ext.get('method', 'unknown'))
            s_conf = float(ext.get('confidence', 0.5) or 0.5)
            s_reas = self._reasoning_expert(ext.get('reasoning', ''),
                                            ext.get('method', 'unknown'))
            s_stress = (self._stress_expert(v_clamped, stress_regime,
                                            thermal_state, fabric)
                        if p == 'glen_n' else 0.0)
            s_thermal = (self._thermal_branch_expert(
                v_clamped, thermal_state,
                ext.get('thermal_branch') or thermal_state)
                if p in ('A1', 'A2', 'Q1', 'Q2') else 0.0)
            s_fab = (self._fabric_expert(v_clamped, fabric, thermal_state)
                     if p == 'E' else 0.0)
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
            total = float(sum(breakdown.values()))

            buckets[p].append(GlacierCandidate(
                param=p,
                value=v_clamped,
                raw_value=float(ext['value']),
                raw_unit=str(ext.get('unit', '')),
                score=total,
                confidence=s_conf,
                method=str(ext.get('method', 'unknown')),
                material=str(ext.get('material', '')),
                temp_k=ext.get('temp'),
                strain_rate=ext.get('strain_rate'),
                source_file=str(ext.get('_source_file', '')),
                source_title=str(ext.get('_source_title', '')),
                evidence=str(ext.get('evidence', '')),
                reasoning=str(ext.get('reasoning', '')),
                provenance=str(ext.get('_provenance', '')),
                theory=str(ext.get('theory', '')),
                dominant_factor=str(ext.get('dominant_factor', '')),
                context=bool(ext.get('_context', False)),
                clamped=was_clamped,
                moe_breakdown=breakdown,
            ))

        for p in buckets:
            buckets[p].sort(key=lambda c: c.score, reverse=True)
            buckets[p] = buckets[p][:top_k]
        return buckets


# ════════════════════════════════════════════════════════════════════════════
# 12. RECOMMENDER — 3-tier cascade
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class GlacierRecommendationBundle:
    candidates: Dict[str, List[GlacierCandidate]]
    regime_summary: Dict[str, Dict[str, Any]]
    retrieval_backend: str
    llm_used: bool

    def best(self, param: str) -> Optional[GlacierCandidate]:
        lst = self.candidates.get(param, [])
        if not lst:
            return None
        ranked = [c for c in lst if not c.context]
        if not ranked:
            ranked = list(lst)
        return max(ranked, key=lambda c: c.score)

    def coverage(self) -> int:
        return sum(1 for p in GLACIER_CANON if self.candidates.get(p))

    def coverage_five(self) -> int:
        targets = ('glen_n', 'A1', 'Q1', 'E', 'rho_ice')
        return sum(1 for p in targets if self.candidates.get(p))


def _build_extraction_prompt(param_key: str, material: str,
                             records: List[Dict[str, Any]],
                             max_chars: int = 1200) -> str:
    canon = GLACIER_CANON.get(param_key, {})
    aliases = ', '.join(f'"{a}"' for a in canon.get('aliases', [])[:8])
    ev = '\n\n'.join(
        f'### EVIDENCE {i} (source={r["source"]})\n'
        f'{r["text"][:max_chars]}' for i, r in enumerate(records))
    return f"""TASK: Extract the glacier parameter `{param_key}` of {material}
from the EVIDENCE records below.  It may be labelled: {aliases}.

STRICT RULES
1. Report ONLY numbers that appear VERBATIM in an EVIDENCE record.
2. Preserve the unit exactly as written (Pa^-3 s^-1, kJ/mol, MPa^-3 yr^-1, ...).
3. Convert nothing — the pipeline handles units.
4. Output a JSON array ONLY.  Schema:
   [{{"value": <number>, "unit": "<string>",
      "property_label": "<verbatim label>",
      "method": "<experiment|torsion test|ice core|review|...>",
      "evidence": "<verbatim quote <=15 words>",
      "confidence": <0.0-1.0>}}]
5. If nothing qualifies, answer [].

{ev}"""


def _build_prior_prompt(param_key: str, material: str,
                        stress: Optional[str], thermal: Optional[str],
                        fabric: Optional[str],
                        records: List[Dict[str, Any]]) -> str:
    canon = GLACIER_CANON.get(param_key, {})
    lo, hi = canon.get('plausible', (float('-inf'), float('inf')))
    return f"""TASK: PRIOR-INFERENCE mode. No verbatim value found for
`{param_key}` of {material}. Reason step-by-step from established glaciology.

Context: stress_regime={stress}, thermal={thermal}, fabric={fabric}.

1. Use textbook values (Paterson & Budd 1982, Cuffey & Paterson 2010).
2. Stay strictly inside [{lo:g}, {hi:g}] {canon.get('unit', '')}.
3. Confidence MUST NOT exceed 0.5.
4. Output JSON array ONLY:
   [{{"value": <number>, "unit": "{canon.get('unit', '')}",
      "property_label": "{param_key}", "method": "prior inference",
      "inference_basis": "<one line>",
      "reasoning": "<Step 1..Step 3>",
      "confidence": <0.0-0.5>}}]

EVIDENCE (context only — do NOT quote as verbatim hits):
{chr(10).join(f"### {r['source']}" + chr(10) + r['text'][:500]
              for r in records[:4]) or "(none)"}"""


def recommend_param_values(param_key: str,
                           retriever: HybridRetriever,
                           ollama: Optional[OllamaClient],
                           material: str = 'ice',
                           stress_regime: Optional[str] = None,
                           thermal_state: Optional[str] = None,
                           fabric: Optional[str] = None,
                           n_hint: float = 3.0,
                           k: int = 6,
                           cascade_mode: str = 'union',
                           ) -> Tuple[List[GlacierCandidate], List[Dict[str, Any]]]:
    records = retriever.search(param_key, material, k=k)
    cands: List[GlacierCandidate] = []
    diag = {'param': param_key, 'records': len(records),
            'tier': 'none', 'cascade_mode': cascade_mode}

    # ── Tier 1 — grounded LLM extraction ────────────────────────────────
    t1 = len(cands)
    if ollama and records:
        prompt = _build_extraction_prompt(param_key, material, records)
        raw = ollama.generate(prompt)
        items = _parse_llm_json(raw) if raw else []
        for it in items:
            if isinstance(it, dict):
                _stamp_provenance(it, 'llm_extract')
                it['_context'] = _is_context(param_key, 'llm_extract')
                it['n_for_conversion'] = n_hint
        new = gatekeep(items, param_key, provenance='llm_extract',
                       n_for_conversion=n_hint, material=material)
        cands.extend(new)
    t1_got = len(cands) - t1

    # ── Tier 2 — deterministic regex NER ────────────────────────────────
    t2 = len(cands)
    if records and (cascade_mode == 'union' or not cands):
        hits = []
        for r in records:
            h = heuristic_extract(r['text_norm'], r['text'], param_key,
                                  source=r['source'])
            for hh in h:
                _stamp_provenance(hh, 'regex_ner')
                hh['_context'] = _is_context(param_key, 'regex_ner')
                hh['n_for_conversion'] = n_hint
            hits.extend(h)
        new = gatekeep(hits, param_key, provenance='regex_ner',
                       n_for_conversion=n_hint, material=material)
        cands.extend(new)
    t2_got = len(cands) - t2

    # ── Tier 3 — prior inference ────────────────────────────────────────
    t3 = len(cands)
    if ollama and (cascade_mode == 'union' or not cands):
        prompt = _build_prior_prompt(param_key, material,
                                     stress_regime, thermal_state, fabric,
                                     records)
        raw = ollama.generate(prompt)
        items = _parse_llm_json(raw) if raw else []
        for it in items:
            if isinstance(it, dict):
                _stamp_provenance(it, 'llm_prior')
                it['_context'] = _is_context(param_key, 'llm_prior')
                it['n_for_conversion'] = n_hint
        new = gatekeep(items, param_key, provenance='llm_prior',
                       conf_cap=0.5, n_for_conversion=n_hint,
                       material=material)
        cands.extend(new)
    t3_got = len(cands) - t3

    diag['tier'] = ('llm' if t1_got else
                    'heuristic' if t2_got else
                    'llm_prior' if t3_got else 'none')
    diag['n_llm'] = t1_got
    diag['n_heuristic'] = t2_got
    diag['n_prior'] = t3_got
    return cands, [diag]


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
    use_llm: bool = True,
    ollama_model: str = 'qwen2.5:7b',
    cascade_mode: str = 'union',
) -> GlacierRecommendationBundle:
    """Full glacier recommender with 3-tier cascade + regime classifiers."""

    # 1. Load corpus
    corpus: List[Dict[str, Any]] = []
    if os.path.isdir(corpus_folder):
        for fn in sorted(os.listdir(corpus_folder)):
            if not fn.lower().endswith('.json'):
                continue
            fpath = os.path.join(corpus_folder, fn)
            try:
                with open(fpath, encoding='utf-8-sig') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning('corpus load %s: %s', fn, e)
                continue
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                if isinstance(row, dict):
                    row['_source'] = fn
                    corpus.append(row)

    # 2. Regime classifiers
    n_reg = classify_n_regime(stress_regime, thermal_state, fabric)
    a1_reg = classify_a1_regime(thermal_state, impurity)
    q1_reg = classify_q1_regime(thermal_state, moisture)
    e_reg = classify_e_regime(fabric, thermal_state)
    e_reg = apply_n_to_e_coupling(n_reg, e_reg)

    # 3. Retriever + LLM client
    retriever = HybridRetriever(corpus, use_dense=True)
    backend = ('hybrid(lexical+faiss)' if retriever.dense
               else 'hybrid(lexical-only)')
    ollama = None
    if use_llm and OllamaClient.is_available():
        ollama = OllamaClient(model=ollama_model)
    llm_used = ollama is not None

    # 4. Per-parameter cascade
    candidates: Dict[str, List[GlacierCandidate]] = {p: [] for p in GLACIER_CANON}
    diagnostics: List[Dict[str, Any]] = []
    n_hint = n_reg['inferred']

    for p in GLACIER_CANON:
        cands, diags = recommend_param_values(
            p, retriever, ollama, material,
            stress_regime, thermal_state, fabric,
            n_hint=n_hint, cascade_mode=cascade_mode)
        candidates[p].extend(cands)
        diagnostics.extend(diags)

    # 5. Regime-prior fallback candidates (guaranteed coverage)
    regime_map = {
        'glen_n': (n_reg, 'combined'),
        'A1':     (a1_reg, 'thermal'),
        'Q1':     (q1_reg, 'thermal'),
        'E':      (e_reg, 'fabric'),
    }
    for p, (reg, dom) in regime_map.items():
        reasoning = (
            f'Step 1: stress={stress_regime}, thermal={thermal_state}, '
            f'fabric={fabric}, impurity={impurity}, moisture={moisture}.\n'
            f'Step 2: classifier → {reg["regime"]}.\n'
            f'Step 3: log-center → {reg["inferred"]:.4g}.')
        if reg.get('coupling'):
            reasoning += f'\nStep 4 (n→E coupling): {reg["coupling"]}.'
        cand = GlacierCandidate(
            param=p,
            value=reg['inferred'],
            raw_value=reg['inferred'],
            raw_unit=GLACIER_CANON[p]['unit'] or 'dimensionless',
            score=0.35,
            confidence=0.35,
            method='physics_regime_inference',
            material=material,
            temp_k=temp_c + 273.15,
            strain_rate=strain_rate,
            source_file=reg['source'],
            source_title=f'{p} regime classifier',
            evidence=reg['regime'],
            reasoning=reasoning,
            provenance='regime_prior',
            dominant_factor=dom,
            context=_is_context(p, 'regime_prior'),
        )
        candidates[p].append(cand)

    # 6. LatentMoE scoring pass
    scorer = GlacierLatentMoEScorer()
    flat = []
    for p, lst in candidates.items():
        for c in lst:
            flat.append({
                'param': p,
                'value': c.raw_value,
                'unit': c.raw_unit,
                'material': c.material,
                'temp': c.temp_k,
                'strain_rate': c.strain_rate,
                'method': c.method,
                'confidence': c.confidence,
                'reasoning': c.reasoning,
                'evidence': c.evidence,
                'theory': c.theory,
                'dominant_factor': c.dominant_factor,
                '_source_file': c.source_file,
                '_source_title': c.source_title,
                '_provenance': c.provenance,
                '_context': c.context,
                'n_for_conversion': n_hint,
            })
    scored = scorer.score(flat, target_material=material,
                          target_temp=temp_c,
                          target_strain_rate=strain_rate,
                          stress_regime=stress_regime,
                          thermal_state=thermal_state,
                          fabric=fabric)

    for p, lst in scored.items():
        candidates[p] = lst

    return GlacierRecommendationBundle(
        candidates=candidates,
        regime_summary={'glen_n': n_reg, 'A1': a1_reg,
                        'Q1': q1_reg, 'E': e_reg},
        retrieval_backend=backend,
        llm_used=llm_used,
    )


# ════════════════════════════════════════════════════════════════════════════
# 13. SIF GENERATOR — INTACT from the original script, now override-aware
# ════════════════════════════════════════════════════════════════════════════
def generate_sif(p: dict) -> str:
    """Generates the Elmer SIF content based on the provided parameters
    dictionary.  AI-recommended overrides from st.session_state are
    injected BEFORE the f-string is evaluated."""
    overrides = st.session_state.get('glacier_overrides', {})
    p = dict(p)
    p.update(overrides)

    return f"""!echo on
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
  Active Solvers(1) =  2
End

Initial Condition 1
  Velocity 1 = 0.0
  Velocity 2 = 0.0
  Velocity 3 = 0.0
  Pressure = 0.0
  Depth = Real 0.0
End

Solver 1
  Exec Solver = "Never"
  Equation = "HeightDepth"
  Procedure = "StructuredProjectToPlane" "StructuredProjectToPlane"
  Active Coordinate = Integer 3
  Operator 1 = depth
  Operator 2 = height
  Dot Product Tolerance = Real 0.9
End

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

Body Force 1
  Name = "BodyForce1"
  Heat Source = 1
  Flow BodyForce 1 = Real 0.0
  Flow BodyForce 2 = Real 0.0
  Flow BodyForce 3 = Real ${p['gravity_formula']}
End

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


# ════════════════════════════════════════════════════════════════════════════
# 14. RECOMMENDER UI — with live regime previews + adoption flow
# ════════════════════════════════════════════════════════════════════════════
def render_recommender_tab() -> None:
    st.subheader('🤖 Glacier Physics-Regime Recommender v1.1.0')
    st.caption(
        'Extracts parameters from a JSON metadatabase, converts SI '
        '(Pa⁻ⁿ s⁻¹, kJ/mol) to Elmer/Ice mixed units (MPa⁻ⁿ yr⁻¹, J/mol), '
        'and scores via a 10-expert LatentMoE.  The n → E cross-coupling '
        'narrows E\'s band when dislocation creep is selected.')

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stress = st.selectbox('Stress regime',
                              ['low', 'intermediate', 'high'],
                              index=2, key='gr_stress')
    with c2:
        thermal = st.selectbox('Thermal state', ['cold', 'warm'],
                               index=0, key='gr_thermal')
    with c3:
        fabric = st.selectbox('Fabric',
                              ['isotropic', 'single_max',
                               'multi_max', 'shear_margin'],
                              index=0, key='gr_fabric')
    with c4:
        impurity = st.selectbox('Impurity',
                                ['clean', 'dust_rich', 'marine', 'temperate'],
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

    st.caption(f'🧭 **n** — {n_prev["regime"]}  ·  '
               f'band `[{n_prev["low"]:.2f}, {n_prev["high"]:.2f}]`  ·  '
               f'center `{n_prev["inferred"]:.2f}`')
    st.caption(f'⚡ **A₁** — {a1_prev["regime"]}  ·  '
               f'band `[{a1_prev["low"]:.2e}, {a1_prev["high"]:.2e}]` '
               f'MPa⁻ⁿ yr⁻¹')
    st.caption(f'🔥 **Q₁** — {q1_prev["regime"]}  ·  '
               f'band `[{q1_prev["low"]/1000:.1f}, '
               f'{q1_prev["high"]/1000:.1f}]` kJ/mol')
    st.caption(f'🔀 **E** — {e_prev["regime"]}  ·  '
               f'band `[{e_prev["low"]:.2f}, {e_prev["high"]:.2f}]`')
    if e_prev.get('coupling'):
        st.caption(f'   📐 n→E coupling: {e_prev["coupling"]}')

    with st.expander('📂 Corpus + LLM settings', expanded=False):
        corpus_folder = st.text_input(
            'Glacier metadatabase folder',
            value='glacier_json_metadatabase', key='gr_corpus')
        use_llm = st.checkbox('Use Ollama LLM (Tier 1 + Tier 3)',
                              value=True, key='gr_use_llm')
        ollama_model = st.text_input('Ollama model', value='qwen2.5:7b',
                                     key='gr_model')
        ollama_url = st.text_input('Ollama URL',
                                   value='http://localhost:11434',
                                   key='gr_url')
        cascade_mode = st.radio('Cascade mode',
                                ['union', 'fallback'],
                                index=0, key='gr_cascade',
                                horizontal=True,
                                help='union: run all tiers, tag '
                                     'non-incumbent routes as context. '
                                     'fallback: first success only.')

    if st.button('🔍 Run Recommender', type='primary',
                 key='gr_run'):
        with st.spinner('Retrieving + scoring candidates…'):
            bundle = recommend_glacier_params(
                stress_regime=stress, thermal_state=thermal,
                fabric=fabric, impurity=impurity, moisture=moisture,
                corpus_folder=corpus_folder,
                use_llm=use_llm, ollama_model=ollama_model,
                cascade_mode=cascade_mode)
        st.session_state['glacier_bundle'] = bundle
        st.success(
            f'✅ {bundle.coverage()}/{len(GLACIER_CANON)} params, '
            f'5-target coverage {bundle.coverage_five()}/5 · '
            f'backend `{bundle.retrieval_backend}` · '
            f'LLM `{"yes" if bundle.llm_used else "no"}`')

    bundle: Optional[GlacierRecommendationBundle] = \
        st.session_state.get('glacier_bundle')
    if bundle is None:
        return

    st.markdown('### Recommended values')
    targets = ['glen_n', 'A1', 'Q1', 'E', 'rho_ice']
    cols = st.columns(len(targets))
    adopted: Dict[str, float] = {}
    for i, p in enumerate(targets):
        best = bundle.best(p)
        with cols[i]:
            meta = GLACIER_PARAM_META[p]
            if best:
                unit = meta.get('unit') or ''
                unit_clean = re.sub(r'\\[a-z]+', ' ', unit).strip()
                val_str = (f'{best.value:.3e} {unit_clean}'
                           if p in ('A1', 'A2', 'Q1', 'Q2')
                           else f'{best.value:.4g} {unit_clean}')
                st.metric(meta['title'][:24], val_str,
                          delta=f'score {best.score:.2f} · '
                                f'{best.provenance or best.method}')
                adopted[p] = best.value
            else:
                st.metric(meta['title'][:24], '—')

    # Alternatives
    with st.expander('🔎 All candidate routes', expanded=False):
        for p in targets:
            rows = []
            for c in bundle.candidates.get(p, []):
                rows.append({
                    'value': f'{c.value:.4g}',
                    'unit': c.raw_unit or GLACIER_CANON[p].get('unit', ''),
                    'score': round(c.score, 3),
                    'conf': round(c.confidence, 3),
                    'provenance': c.provenance or c.method,
                    'method': c.method,
                    'context': '🩶' if c.context else '',
                    'evidence': c.evidence[:80],
                })
            if rows:
                st.markdown(f'**{GLACIER_PARAM_META[p]["title"]}**')
                st.dataframe(pd.DataFrame(rows),
                             use_container_width=True, hide_index=True)

    if st.button('✅ Adopt for ElmerSolver SIF', type='primary',
                 key='gr_adopt'):
        sif_overrides: Dict[str, Any] = {}
        if 'glen_n' in adopted:
            sif_overrides['glen_exponent'] = adopted['glen_n']
        if 'A1' in adopted:
            sif_overrides['rate_factor_1'] = adopted['A1']
        if 'Q1' in adopted:
            sif_overrides['activation_energy_1'] = adopted['Q1']
        if 'E' in adopted:
            sif_overrides['glen_enhancement_factor'] = adopted['E']
        if 'rho_ice' in adopted:
            sif_overrides['density_formula'] = \
                GlacierUnitConverter.density_formula(adopted['rho_ice'])
        st.session_state['glacier_overrides'] = sif_overrides
        # clear widget cache so number_inputs pick up new defaults
        for k in list(st.session_state.keys()):
            if k.startswith('mat_') or k.startswith('sif_'):
                st.session_state.pop(k, None)
        st.success(
            f'✅ Stored {len(sif_overrides)} Elmer/Ice overrides. '
            f'They will be injected into the SIF on the next run. '
            f'Open the **🧊 Material Properties** tab to verify.')

    # Unit conversion demo
    with st.expander('📏 Unit conversion examples', expanded=False):
        st.markdown(
            '**Elmer/Ice mixed system:** Length = m, Stress = MPa, '
            'Time = yr.  A literature `A = 3.5 × 10⁻²⁵ Pa⁻³ s⁻¹` '
            'becomes `~0.011 MPa⁻³ yr⁻¹`.')
        examples = [
            ('A1', 3.5e-25, 'Pa^-3 s^-1', 3.0),
            ('A1', 1.258e13, 'MPa^-3 yr^-1', 3.0),
            ('Q1', 60.0, 'kJ mol^-1', 3.0),
            ('rho_ice', 910.0, 'kg m^-3', 3.0),
            ('T_ice', 250.0, 'K', 3.0),
        ]
        rows = []
        for p, raw, u, n in examples:
            conv, tgt = GlacierUnitConverter.convert(raw, u, p, n=n)
            rows.append({
                'param': p, 'raw': f'{raw:.4g}', 'raw unit': u,
                'converted': f'{conv:.4g}', 'target unit': tgt,
            })
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# 15. STREAMLIT APP
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title='ElmerSolver Launcher + AI Recommender',
                   layout='wide', initial_sidebar_state='expanded')
st.title('🏔️ Himalayan Glacier 3D — ElmerSolver Launcher + AI Recommender')

if 'glacier_overrides' not in st.session_state:
    st.session_state['glacier_overrides'] = {}

# ── Sidebar: execution settings ──────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Execution Settings')
    sif_name = st.text_input('SIF filename',
                             value='himalayan_glacier3d.sif',
                             key='sif_filename')
    work_dir = st.text_input(
        'Working directory',
        value=str(pathlib.Path.cwd()),
        help='Directory where the .sif is written and ElmerSolver is launched',
        key='sif_work_dir')
    elmer_cmd = st.text_input('ElmerSolver command', value='ElmerSolver',
                              key='sif_elmer_cmd')
    st.markdown('---')
    ovr_active = st.session_state.get('glacier_overrides', {})
    if ovr_active:
        st.success(f'🧠 AI overrides active: '
                   f'{len(ovr_active)} parameter(s)')
        with st.expander('Show overrides', expanded=False):
            for k, v in ovr_active.items():
                st.code(f'{k} = {v}')
        if st.button('🗑️ Clear AI overrides'):
            st.session_state['glacier_overrides'] = {}
            st.rerun()
    else:
        st.info('No AI overrides yet. Run the Recommender tab to populate '
                'them.')

# ── Main tabs ─────────────────────────────────────────────────────────────
tab_recommender, tab_sim, tab_solvers, tab_material, tab_boundaries = st.tabs([
    '🤖 AI Recommender (Physics-Regime)',
    '📊 Simulation & Output',
    '⚙️ Solvers Configuration',
    '🧊 Material Properties',
    '🌍 Body Forces & Boundaries',
])

# ── Tab 1: AI Recommender ────────────────────────────────────────────────
with tab_recommender:
    render_recommender_tab()

# ── Tab 2: Simulation & Output ───────────────────────────────────────────
with tab_sim:
    st.subheader('Simulation & Output Settings')
    col1, col2 = st.columns(2)
    with col1:
        mesh_db = st.text_input('Mesh DB Name', value='himalayan_glacier3d',
                                key='sif_mesh_db')
        output_file = st.text_input(
            'Output File', value='Stokes_ELA5000_3D_diagnostic.result',
            key='sif_output_file')
    with col2:
        post_file = st.text_input(
            'Post File', value='Stokes_ELA5000_3D_diagnostic.vtu',
            key='sif_post_file')
        steady_state_max_iter = st.number_input(
            'Steady State Max Iterations', min_value=1, value=1,
            key='sif_ss_max_iter')
    output_intervals = st.number_input(
        'Output Intervals', min_value=1, value=1, key='sif_out_interval')

# ── Tab 3: Solvers Configuration ─────────────────────────────────────────
with tab_solvers:
    st.subheader('Solvers Configuration (Solver 2: Navier-Stokes)')
    col1, col2 = st.columns(2)
    with col1:
        solver2_linear_max_iter = st.number_input(
            'Linear System Max Iterations', min_value=1, value=5000,
            key='sif_s2_lin_max')
        solver2_linear_tol = st.number_input(
            'Linear System Convergence Tolerance', value=1.0e-6,
            format='%.1e', key='sif_s2_lin_tol')
        solver2_steady_tol = st.number_input(
            'Steady State Convergence Tolerance', value=1.0e-5,
            format='%.1e', key='sif_s2_ss_tol')
    with col2:
        solver2_nonlinear_max_iter = st.number_input(
            'Nonlinear System Max Iterations', min_value=1, value=50,
            key='sif_s2_nl_max')
        solver2_nonlinear_tol = st.number_input(
            'Nonlinear System Convergence Tolerance', value=1.0e-4,
            format='%.1e', key='sif_s2_nl_tol')

# ── Tab 4: Material Properties (auto-populated from overrides) ───────────
with tab_material:
    st.subheader('Material Properties (Material 1)')
    _ovr = st.session_state.get('glacier_overrides', {})
    if _ovr:
        st.info('🧠 AI-recommended values have been injected into the '
                'defaults below.  Edit any field to override them for this '
                'session only (the override dict is unaffected).')
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.text_input('Material Name', value='ice',
                                      key='mat_name')
        density_formula = st.text_input(
            'Density Formula (evaluated after `$`)',
            value=_ovr.get('density_formula',
                           '910.0*1.0E-06*(31556926.0)^(-2.0)'),
            key='mat_density')
        viscosity = st.number_input('Viscosity', value=1.0, step=0.1,
                                    key='mat_viscosity')
        glen_exponent = st.number_input(
            'Glen Exponent', value=float(_ovr.get('glen_exponent', 3.0)),
            step=0.1, key='mat_glen_n')
        critical_shear_rate = st.number_input(
            'Critical Shear Rate', value=1.0e-10, format='%.1e',
            key='mat_csr')
        rate_factor_1 = st.number_input(
            'Rate Factor 1', value=float(_ovr.get('rate_factor_1', 1.258e13)),
            format='%.3e', key='mat_rf1')
    with col2:
        rate_factor_2 = st.number_input(
            'Rate Factor 2', value=6.046e28, format='%.3e', key='mat_rf2')
        activation_energy_1 = st.number_input(
            'Activation Energy 1 (J/mol)',
            value=float(_ovr.get('activation_energy_1', 60000.0)),
            step=1000.0, key='mat_ae1')
        activation_energy_2 = st.number_input(
            'Activation Energy 2 (J/mol)', value=139000.0, step=1000.0,
            key='mat_ae2')
        glen_enhancement_factor = st.number_input(
            'Glen Enhancement Factor',
            value=float(_ovr.get('glen_enhancement_factor', 1.0)),
            step=0.1, key='mat_enh')
        limit_temperature = st.number_input(
            'Limit Temperature (°C)', value=-10.0, step=1.0,
            key='mat_lim_temp')
        constant_temperature = st.number_input(
            'Constant Temperature (°C)', value=-3.0, step=1.0,
            key='mat_const_temp')

# ── Tab 5: Body Forces & Boundaries ──────────────────────────────────────
with tab_boundaries:
    st.subheader('Body Forces & Boundary Conditions')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Body Force 1**')
        gravity_formula = st.text_input(
            'Flow BodyForce 3 Formula (evaluated after `$`)',
            value='-9.81 * (31556926.0)^(2.0)', key='sif_gravity')
        st.markdown('**Boundary Targets**')
        bedrock_target = st.number_input(
            'Bedrock Target Boundaries', min_value=1, value=2,
            key='sif_bedrock')
        sides_target = st.number_input(
            'Lateral Walls Target Boundaries', min_value=1, value=1,
            key='sif_sides')
    with col2:
        surface_target = st.number_input(
            'Surface Target Boundaries', min_value=1, value=3,
            key='sif_surface')

# ── Assemble parameters dictionary ───────────────────────────────────────
params = {
    'mesh_db': mesh_db,
    'output_file': output_file,
    'post_file': post_file,
    'steady_state_max_iter': steady_state_max_iter,
    'output_intervals': output_intervals,
    'solver2_linear_max_iter': solver2_linear_max_iter,
    'solver2_linear_tol': solver2_linear_tol,
    'solver2_steady_tol': solver2_steady_tol,
    'solver2_nonlinear_max_iter': solver2_nonlinear_max_iter,
    'solver2_nonlinear_tol': solver2_nonlinear_tol,
    'material_name': material_name,
    'density_formula': density_formula,
    'viscosity': viscosity,
    'glen_exponent': glen_exponent,
    'critical_shear_rate': critical_shear_rate,
    'rate_factor_1': rate_factor_1,
    'rate_factor_2': rate_factor_2,
    'activation_energy_1': activation_energy_1,
    'activation_energy_2': activation_energy_2,
    'glen_enhancement_factor': glen_enhancement_factor,
    'limit_temperature': limit_temperature,
    'constant_temperature': constant_temperature,
    'gravity_formula': gravity_formula,
    'bedrock_target': bedrock_target,
    'sides_target': sides_target,
    'surface_target': surface_target,
}

# ── Dynamic SIF content (overrides applied inside) ───────────────────────
SIF_CONTENT = generate_sif(params)

with st.expander('📄 View Generated SIF file', expanded=False):
    st.code(SIF_CONTENT, language='plaintext')

# ── Main action area ─────────────────────────────────────────────────────
st.markdown('---')
col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button('🚀 Run ElmerSolver', type='primary')
with col2:
    st.markdown(
        'Click the button to write the SIF file and launch '
        '`ElmerSolver <filename.sif>` in the working directory.')

# ── Write SIF + execute ElmerSolver — INTACT ─────────────────────────────
if run_btn:
    work_path = pathlib.Path(work_dir)
    sif_path = work_path / sif_name

    try:
        sif_path.write_text(SIF_CONTENT)
        st.success(f'SIF written to `{sif_path}`')
    except Exception as e:
        st.error(f'Failed to write SIF: {e}')
        st.stop()

    cmd = [elmer_cmd, sif_name]
    st.info(f"Running: `{' '.join(cmd)}`  in  `{work_dir}`")

    output_area = st.empty()
    collected_lines: List[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            collected_lines.append(line)
            output_area.code(''.join(collected_lines),
                             language='plaintext')
        proc.wait()
        exit_code = proc.returncode
    except FileNotFoundError:
        st.error(
            f'`{elmer_cmd}` not found. Make sure ElmerSolver is on your '
            'PATH or provide the full absolute path in the sidebar.')
        st.stop()
    except Exception as e:
        st.error(f'Subprocess failed: {e}')
        st.stop()

    if exit_code == 0:
        st.success('✅ ElmerSolver finished successfully (exit code 0)')
    else:
        st.error(f'❌ ElmerSolver exited with code {exit_code}')

    st.download_button(
        '⬇️ Download SIF file',
        data=SIF_CONTENT,
        file_name=sif_name,
        mime='text/plain',
    )
