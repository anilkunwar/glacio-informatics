# --------------------------------------------------------------
# Activation Energy (Q) in Glacier Ice — HARVESTER
# Domain pivot from nanotwinned-Cu ρ₀ → Glen n → rate factor A → activation
# energy Q. Q is the exponential argument of the Arrhenius rate factor:
#     A(T) = A₀ · exp(-Q / (R·T))
# and in the Elmer/Ice SIF it appears as TWO values (Q1 cold ice, Q2 warm
# ice) whose magnitudes differ by roughly a factor of two (60 vs 139 kJ/mol
# in Paterson & Budd 1982, 60 vs 115 kJ/mol in Cuffey & Paterson 2010).
#
# THE TRAP: the SIF expects J/mol (hence 60e3), but 95% of glaciology
# literature publishes kJ/mol (hence 60). Copying "60" into a SIF that
# expects J/mol makes the glacier barely flow. This campaign therefore
# prioritises MAGNITUDE-AND-UNIT RESOLUTION and CREEP-MECHANISM tagging.
#
# ✅ ACTIVATION_ENERGY_TERMS vocabulary
# ✅ DOMAIN_TERMS = Q ∪ rate factor ∪ Glen ∪ ice physics ∪ glacier
# ✅ analyze_activation_energy_relevance() with back-compat aliases
# ✅ resolve_activation_energy_units() — J/mol / kJ/mol / eV normaliser
# ✅ classify_creep_mechanism() — Q1 (dislocation) vs Q2 (grain-boundary)
# ✅ extract_candidate_activation_energy() — magnitude + unit + regime aware
# ✅ Sibling extractors retained: rate factor A, Glen n, enhancement factor E
# ✅ topic column: 'activation_energy_ice' (this campaign),
#    'rate_factor_ice', 'glen_exponent_ice', 'enhancement_factor_ice' (siblings)
# ✅ query_arxiv(topic=…) is an explicit cache-keyed argument
# ✅ Optional combined-view export (topic=None, all four extractors live)
# ✅ Full text stored in SQLite universe DB is single source of truth
# ✅ Coverage warnings; backfill; BLOB-score repair retained
# ✅ Scopus-format CSV: 22 official + 19 appended columns, exact order, BOM
# --------------------------------------------------------------

# Standard library imports
import os
import re
import struct
import sqlite3
import json
import io
import zipfile
import logging
import time
import datetime
from datetime import datetime
import tempfile
import hashlib
import gc
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Third-party scientific and data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PDF handling
import fitz  # PyMuPDF

# Web and API clients
import requests
import arxiv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Concurrency and system utilities
import concurrent.futures
import psutil

# Machine Learning and NLP
from transformers import AutoTokenizer, AutoModel
import torch

# Retry and robustness decorators
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential

# Numerical acceleration
from numba import njit

# Streamlit for UI
import streamlit as st


# ========================= ENVIRONMENT DETECTION =========================
def is_streamlit_cloud():
    if os.getenv("HOME") == "/home/appuser":
        return True
    if "streamlitapp.com" in os.getenv("HOSTNAME", ""):
        return True
    if os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true":
        return True
    return False


IS_CLOUD = is_streamlit_cloud()


# ========================= CAMPAIGN IDENTIFIER =========================
# The metadata DB carries a 'topic' column so multiple campaigns share one
# SQLite file without contaminating each other's CSV exports.
# Current campaign: activation energy Q for polycrystalline glacier ice.
CURRENT_TOPIC = "activation_energy_ice"

# Known sibling campaigns (for the combined-view toggle)
SIBLING_TOPICS = ("rate_factor_ice", "glen_exponent_ice", "enhancement_factor_ice")


# ========================= OPTIMAL QUERY FOR ACTIVATION ENERGY =========================
# Q appears in three flavours of glaciology paper:
#   1. Field/lab creep experiments reporting Q1 and Q2 (Paterson & Budd lineage)
#   2. Ice-sheet modelling papers where Q1 / Q2 are model parameters
#   3. MD / dislocation-dynamics papers on the mechanism of creep (Q for
#      dislocation glide vs. grain-boundary sliding)
# The RHS of the AND keeps the harvest glaciological — "activation energy"
# alone is a chemistry term and would drag in the entire Arrhenius corpus
# from catalysis to polymer degradation.
ACTIVATION_ENERGY_QUERY = (
    'abs:("activation energy" OR "activation enthalpy" OR "energy barrier" '
    'OR "creep activation" OR "Arrhenius" OR "dislocation creep" '
    'OR "grain boundary sliding" OR "diffusion creep") '
    'AND abs:(glacier OR "ice sheet" OR "ice stream" OR "ice shelf" '
    'OR "polar ice" OR "ice dynamics" OR "polycrystalline ice")'
)

ACTIVATION_ENERGY_CATEGORIES = [
    'physics.geo-ph',   # Geophysics (primary home for glaciology)
    'cond-mat.mtrl-sci',# Materials Science (dislocation creep in ice Ih)
    'physics.comp-ph',  # Computational Physics (Elmer/Ice, MD simulations)
]

ACTIVATION_ENERGY_START_YEAR = 1970  # Q1/Q2 split dates to Paterson & Budd 1982
ACTIVATION_ENERGY_END_YEAR   = 2024
ACTIVATION_ENERGY_MAX_RESULTS = 50


# ========================= PAGE CONFIGURATION (MUST BE FIRST) =========================
if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Activation Energy (Q) Harvester — Glacier Ice Rheology",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_config = {
        "title": "Activation Energy (Q) Harvester — Glacier Ice Rheology",
        "layout": "wide"
    }
    st.session_state.page_config_set = True


# =================================================================
# -------------------------- DOMAIN-SPECIFIC TERM DEFINITIONS --------------------------
# Campaign target: the activation energy Q (J/mol) in the Arrhenius form of
# the Glen flow law. The Elmer/Ice SIF parameterises Q as a two-branch
# function:
#     Q1 (cold ice,  T < Limit Temperature)  ~  60 kJ/mol  (60e3 J/mol)
#     Q2 (warm ice,  T > Limit Temperature)  ~ 139 kJ/mol (139e3 J/mol)
# The physical interpretation: at low T, creep is rate-limited by
# dislocation glide; near the melting point, grain-boundary sliding takes
# over with a different (often lower) activation energy. The magnitude
# shift is a signature, not a coincidence.


# Activation-energy vocabulary — the new primary target vocabulary
ACTIVATION_ENERGY_TERMS = {
    'activation energy', 'activation enthalpy', 'energy barrier',
    'creep activation', 'q =', 'ea =', 'q1', 'q2',
    'dislocation creep', 'grain boundary sliding', 'diffusion creep',
    'rate controlling', 'temperature dependence',
    'activation energy for creep', 'creep activation energy',
    'activation energy q', 'activation energy of creep',
    'cold ice activation', 'warm ice activation',
    'energy of activation', 'arrhenius activation energy',
}


# Rate-factor / Arrhenius vocabulary — retained as a sibling vocabulary
RATE_FACTOR_TERMS = {
    'rate factor', 'pre-exponential factor', 'pre-exponential', 'preexponential',
    'arrhenius factor', 'arrhenius relation', 'arrhenius law',
    'flow law parameter a', 'flow law parameter', 'creep parameter',
    'creep constant', 'arrhenius', 'activation energy',
    'temperature limit', 'limit temperature',
    'rate factor 1', 'rate factor 2', 'rate constant',
    'temperature dependence', 'temperature dependent',
    'cold ice', 'temperate ice', 'warm ice',
    'exponential prefactor', 'exponential factor',
    'activation energy for creep', 'creep activation energy',
}


# Glen flow law / rheology terms
GLEN_RHEOLOGY_TERMS = {
    'glen flow law', 'glens flow law', "glen's flow law", 'glen exponent',
    'glen n', 'flow law exponent', 'creep exponent', 'power law exponent',
    'non-newtonian', 'power law rheology', 'ice rheology',
    'constitutive law', 'constitutive relation', 'viscosity model',
    'strain rate exponent', 'n=3', 'n = 3', 'exponent n',
    'creep parameter', 'flow law parameter', 'strain exponent',
}


# Ice-physics terms
ICE_PHYSICS_TERMS = {
    'strain rate sensitivity', 'strain-rate', 'deviatoric stress',
    'second invariant', 'effective strain rate', 'effective stress',
    'shear stress', 'basal drag', 'internal deformation', 'creep',
    'thermomechanical', 'thermo-mechanical', 'viscous', 'viscosity',
    'activation energy', 'arrhenius', 'temperature dependence',
    'stress exponent', 'secondary creep', 'tertiary creep',
}


# Glacier / ice-sheet vocabulary
GLACIER_TERMS = {
    'glacier', 'ice sheet', 'ice stream', 'ice shelf', 'polycrystalline ice',
    'glacier dynamics', 'ice flow model', 'elmer/ice', 'land ice',
    'cryosphere', 'glaciology', 'mass balance', 'sliding law',
    'antarctica', 'greenland', 'ice cap', 'ice divide', 'ice dynamics',
    'shallow ice approximation', 'full stokes', 'ice thickness',
}


# Base material identifiers (name kept for session-state compatibility)
PVDF_TERMS = {'glacier', 'ice', 'glaciology'}


# Union used everywhere the previous code wrote
# (FRICTION_STRESS_TERMS | MATERIAL_TERMS). Refreshed to include Q terms.
DOMAIN_TERMS = (ACTIVATION_ENERGY_TERMS | RATE_FACTOR_TERMS |
                GLEN_RHEOLOGY_TERMS | ICE_PHYSICS_TERMS | GLACIER_TERMS)


# Legacy aliases so older call sites still resolve
SRS_TERMS = ICE_PHYSICS_TERMS
FRICTION_STRESS_TERMS = GLEN_RHEOLOGY_TERMS
MATERIAL_TERMS = GLACIER_TERMS


# NOTE: DB column names (dopant_present, beta_phase_present, pvdf_present) are deliberately
# preserved so DB schema, store_paper_metadata, and get_paper_info require zero schema changes.
# Semantically for THIS campaign:
#   dopant_present      → activation-energy terms present
#   beta_phase_present  → rate-factor / Arrhenius terms present
#   pvdf_present        → glacier / ice-sheet terms present


# -------------------------- DIRECTORY AND DATABASE CONFIGURATION --------------------------
# Shared DB files host all campaigns. The 'topic' column separates them:
#   'sigma0'                 → legacy friction/lattice-stress rows
#   'srs_ntcu'               → legacy strain-rate-sensitivity rows
#   'rho0_ntcu'              → legacy initial-dislocation-density rows
#   'glen_exponent_ice'      → Glen n rows (sibling)
#   'rate_factor_ice'        → Arrhenius rate factor A rows (sibling)
#   'enhancement_factor_ice' → enhancement factor E rows (sibling)
#   'activation_energy_ice'  → activation energy Q rows (this campaign)


if IS_CLOUD:
    DB_DIR = "/tmp"
    st.info("🌐 Running on Streamlit Cloud: Using temporary storage")
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "ice_rheology_data")
    os.makedirs(DB_DIR, exist_ok=True)


METADATA_DB    = os.path.join(DB_DIR, "ice_rheology_metadata.db")
UNIVERSE_DB    = os.path.join(DB_DIR, "ice_rheology_universe.db")
PDF_STORAGE_DB = os.path.join(DB_DIR, "ice_rheology_pdfs.db")

TEMP_DIR = os.path.join(DB_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

log_file = os.path.join(DB_DIR, "ice_rheology_query.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# -------------------------- SESSION STATE DEFAULTS --------------------------


DEFAULT_STATE = {
    "log_buffer": [],
    "processing": False,
    "search_results": None,
    "relevant_papers": None,
    "downloaded_pdfs": {},
    "zip_buffer": None,
    "processing_time": 0.0,
    "db_stats": {},
    "search_session_id": None,
    "temp_files": [],
    "selected_papers": set(),
    "batch_download_index": 0,
}


for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# -------------------------- LOGGING UTILITY --------------------------


def update_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(entry)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(message)


# -------------------------- SCIBERT LOADING (CACHED) --------------------------


@st.cache_resource
def load_scibert():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    update_log("SciBERT model and tokenizer loaded from cache")
    return tokenizer, model


# -------------------------- EMBEDDING AND SIMILARITY UTILITIES --------------------------


def get_embedding(text: str) -> np.ndarray:
    tokenizer, model = load_scibert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denominator = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return np.dot(a, b) / denominator


# -------------------------- TEXT NORMALIZATION (for extractors) --------------------------


def _normalize_flattened_text(t: str) -> str:
    """
    PyMuPDF flattens superscripts, Unicode minus signs and multiplication
    symbols when extracting text. Normalize the common offenders so regexes
    targeting 's-1', 'x10-4', 'b3', 'nm3', 'A3', 'm-2', 'n≈3', '10⁻²⁵',
    'kJ mol⁻¹' match what the PDF actually produces.
    """
    for a, b in {'−': '-', '–': '-', '⁻': '-', '×': 'x', '≈': '=',
                 '³': '3', '²': '2', '¹': '1', 'Å': 'A', 'Å': 'A',
                 '·': '*', '⋅': '*'}.items():
        t = t.replace(a, b)
    return t


# -------------------------- DOMAIN-SPECIFIC TEXT ANALYSIS --------------------------


def analyze_activation_energy_relevance(text: str) -> Dict[str, Any]:
    """
    Analyze text for presence of activation-energy, rate-factor, and glacier
    terms. Keeps the DB-flag schema so store/get/display paths need zero
    changes; only the semantics shift.
    """
    text_lower = text.lower()
    has_q       = any(term in text_lower for term in ACTIVATION_ENERGY_TERMS)
    has_rate    = any(term in text_lower for term in RATE_FACTOR_TERMS)
    has_glacier = any(term in text_lower for term in GLACIER_TERMS)
    return {
        "dopant_present":     bool(has_q),           # now = activation-energy terms present
        "beta_phase_present": bool(has_rate),        # now = rate-factor / Arrhenius terms present
        "pvdf_present":       bool(has_glacier),     # now = glacier / ice-sheet terms present
    }


# Backwards-compatible aliases so older call sites keep working across campaigns
analyze_rate_factor_relevance  = analyze_activation_energy_relevance
analyze_glen_n_relevance       = analyze_activation_energy_relevance
analyze_rho0_relevance         = analyze_activation_energy_relevance
analyze_srs_relevance          = analyze_activation_energy_relevance
analyze_sigma0_relevance       = analyze_activation_energy_relevance
analyze_dopant_beta_relevance  = analyze_activation_energy_relevance


# -------------------------- DATABASE MANAGER CLASS --------------------------


class DatabaseManager:
    """
    Manages three SQLite databases:
    - Metadata (paper info, scores, topic)
    - Full-text (extracted text, FTS5 index)
    - PDF storage (BLOBs, deduplicated by hash)
    """

    def __init__(self):
        self.metadata_db = METADATA_DB
        self.universe_db = UNIVERSE_DB
        self.pdf_db = PDF_STORAGE_DB
        self.init_databases()
        update_log("Database manager initialized")

    def init_databases(self):
        # ------------------ Metadata Database ------------------
        conn = sqlite3.connect(self.metadata_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            arxiv_id TEXT UNIQUE,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            categories TEXT,
            abstract TEXT,
            pdf_url TEXT,
            published_date TEXT,
            updated_date TEXT,
            doi TEXT,
            relevance_score REAL,
            matched_terms TEXT,
            download_status TEXT,
            pdf_stored BOOLEAN DEFAULT 0,
            fulltext_stored BOOLEAN DEFAULT 0,
            pdf_size INTEGER,
            download_time TIMESTAMP,
            enhanced_relevance_score REAL,
            dopant_present BOOLEAN,
            beta_phase_present BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS search_sessions (
            session_id TEXT PRIMARY KEY,
            query TEXT,
            categories TEXT,
            start_year INTEGER,
            end_year INTEGER,
            max_results INTEGER,
            threshold REAL,
            total_found INTEGER,
            relevant_found INTEGER,
            downloaded_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # Migration: add journal_ref column if it doesn't exist (idempotent)
        try:
            c.execute("ALTER TABLE papers ADD COLUMN journal_ref TEXT")
        except sqlite3.OperationalError:
            pass

        # Migration: add topic column if it doesn't exist (idempotent).
        # Legacy rows default to 'sigma0' so they stay out of newer campaigns.
        try:
            c.execute("ALTER TABLE papers ADD COLUMN topic TEXT DEFAULT 'sigma0'")
        except sqlite3.OperationalError:
            pass

        c.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_score ON papers(relevance_score)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_score ON papers(enhanced_relevance_score)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_status ON papers(download_status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_topic ON papers(topic)")
        conn.commit()
        conn.close()

        # ------------------ Full-text Database ------------------
        conn = sqlite3.connect(self.universe_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS papers_fulltext (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            full_text TEXT,
            text_hash TEXT UNIQUE,
            word_count INTEGER,
            page_count INTEGER,
            extraction_status TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS extracted_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            entity_type TEXT,
            entity_text TEXT,
            context TEXT,
            page_number INTEGER,
            confidence REAL,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers_fulltext(paper_id)
        )""")
        c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
            USING fts5(paper_id, title, abstract, full_text, tokenize='porter')""")
        conn.commit()
        conn.close()

        # ------------------ PDF Storage Database ------------------
        conn = sqlite3.connect(self.pdf_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_storage (
            paper_id TEXT PRIMARY KEY,
            pdf_data BLOB NOT NULL,
            pdf_hash TEXT UNIQUE,
            original_url TEXT,
            file_size INTEGER,
            page_count INTEGER,
            compression_method TEXT DEFAULT 'none',
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS pdf_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            chunk_index INTEGER,
            chunk_data BLOB,
            chunk_hash TEXT,
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES pdf_storage(paper_id),
            UNIQUE(paper_id, chunk_index)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pdf_hash ON pdf_storage(pdf_hash)")
        conn.commit()
        conn.close()

    def get_db_stats(self) -> Dict[str, Any]:
        stats = {}
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers")
            stats['total_papers'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM papers WHERE pdf_stored = 1")
            stats['pdfs_stored'] = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM papers WHERE fulltext_stored = 1")
            stats['fulltext_stored'] = c.fetchone()[0]
            c.execute("SELECT COUNT(DISTINCT year) FROM papers")
            stats['years_covered'] = c.fetchone()[0]
            try:
                c.execute("SELECT topic, COUNT(*) FROM papers GROUP BY topic")
                stats['by_topic'] = {row[0] or 'sigma0': row[1] for row in c.fetchall()}
            except sqlite3.OperationalError:
                stats['by_topic'] = {}
            conn.close()

            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers_fulltext")
            stats['fulltext_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(word_count) FROM papers_fulltext")
            stats['total_words'] = c.fetchone()[0] or 0
            conn.close()

            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM pdf_storage")
            stats['pdf_storage_count'] = c.fetchone()[0]
            c.execute("SELECT SUM(file_size) FROM pdf_storage")
            total_bytes = c.fetchone()[0] or 0
            stats['total_pdf_size_mb'] = round(total_bytes / (1024 * 1024), 2)
            conn.close()
        except Exception as e:
            update_log(f"Error getting DB stats: {e}")
        return stats

    def store_paper_metadata(self, paper: Dict[str, Any]) -> bool:
        """
        Insert or update paper metadata.
        Preserves existing download state so a fresh search does not wipe
        previously-downloaded papers. Writes 'topic' for campaign separation.
        """
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()

            row = c.execute(
                "SELECT download_status, pdf_stored, fulltext_stored, pdf_size, "
                "download_time, enhanced_relevance_score FROM papers WHERE id=?",
                (paper.get('id'),)
            ).fetchone()
            if row and row[1]:
                paper = {
                    **paper,
                    'download_status': row[0],
                    'pdf_stored': row[1],
                    'fulltext_stored': row[2],
                    'pdf_size': row[3],
                    'download_time': row[4],
                    'enhanced_relevance_score':
                        row[5] if row[5] is not None else paper.get('enhanced_relevance_score', 0.0),
                }

            c.execute("""INSERT OR REPLACE INTO papers
                (id, arxiv_id, title, authors, year, categories, abstract,
                pdf_url, published_date, updated_date, doi, journal_ref,
                relevance_score, matched_terms, download_status, pdf_stored,
                fulltext_stored, pdf_size, download_time, enhanced_relevance_score,
                dopant_present, beta_phase_present, topic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper.get('id'), paper.get('arxiv_id'), paper.get('title'),
                    paper.get('authors'), paper.get('year'), paper.get('categories'),
                    paper.get('abstract'), paper.get('pdf_url'), paper.get('published_date'),
                    paper.get('updated_date'), paper.get('doi'), paper.get('journal_ref'),
                    paper.get('relevance_score'), paper.get('matched_terms'),
                    paper.get('download_status'), paper.get('pdf_stored', 0),
                    paper.get('fulltext_stored', 0), paper.get('pdf_size', 0),
                    paper.get('download_time'), paper.get('enhanced_relevance_score'),
                    paper.get('dopant_present'), paper.get('beta_phase_present'),
                    paper.get('topic', 'sigma0')
                ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            update_log(f"Failed to store metadata for {paper.get('id')}: {e}")
            return False

    def store_pdf_data(self, paper_id: str, pdf_bytes: bytes, pdf_url: str) -> bool:
        """
        Store PDF BLOB in the PDF DB, then update the metadata DB flag.
        Two connections because the 'papers' table lives in metadata_db.
        """
        try:
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            file_size = len(pdf_bytes)
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(doc)
                doc.close()
            except Exception:
                page_count = 0

            # ---- PDF storage DB ----
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT paper_id FROM pdf_storage WHERE pdf_hash = ?", (pdf_hash,))
            existing = c.fetchone()
            if existing:
                update_log(f"PDF already exists for {paper_id}")
            else:
                c.execute("""INSERT OR REPLACE INTO pdf_storage
                    (paper_id, pdf_data, pdf_hash, original_url, file_size, page_count)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (paper_id, sqlite3.Binary(pdf_bytes), pdf_hash, pdf_url,
                     file_size, page_count))
            conn.commit()
            conn.close()

            # ---- Metadata DB (separate connection, correct DB) ----
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("UPDATE papers SET pdf_stored = 1, pdf_size = ? WHERE id = ?",
                      (file_size, paper_id))
            conn.commit()
            conn.close()

            update_log(f"Stored PDF for {paper_id} ({file_size/1024:.1f} KB, {page_count} pages)")
            return True
        except Exception as e:
            update_log(f"Failed to store PDF for {paper_id}: {e}")
            return False

    def store_fulltext(self, paper_id: str, title: str, abstract: str,
                       full_text: str, page_count: int = 0) -> bool:
        try:
            text_hash = hashlib.md5(full_text.encode()).hexdigest()
            word_count = len(full_text.split())
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO papers_fulltext
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text, text_hash, word_count, page_count))
            c.execute("""INSERT OR REPLACE INTO papers_fts
                (paper_id, title, abstract, full_text)
                VALUES (?, ?, ?, ?)""",
                (paper_id, title, abstract, full_text))
            conn.commit()
            conn.close()
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("UPDATE papers SET fulltext_stored = 1 WHERE id = ?", (paper_id,))
            conn.commit()
            conn.close()
            update_log(f"Stored full text for {paper_id} ({word_count} words)")
            return True
        except Exception as e:
            update_log(f"Failed to store full text for {paper_id}: {e}")
            return False

    def get_pdf(self, paper_id: str) -> Optional[bytes]:
        try:
            conn = sqlite3.connect(self.pdf_db)
            c = conn.cursor()
            c.execute("SELECT pdf_data FROM pdf_storage WHERE paper_id = ?", (paper_id,))
            result = c.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            update_log(f"Failed to retrieve PDF for {paper_id}: {e}")
            return None

    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.metadata_db)
            c = conn.cursor()
            c.execute("""SELECT title, authors, year, abstract, pdf_url,
                relevance_score, enhanced_relevance_score, dopant_present,
                beta_phase_present, download_status, pdf_stored, fulltext_stored,
                journal_ref
                FROM papers WHERE id = ?""", (paper_id,))
            meta = c.fetchone()
            conn.close()
            if not meta:
                return None
            pdf_bytes = self.get_pdf(paper_id)
            conn = sqlite3.connect(self.universe_db)
            c = conn.cursor()
            c.execute("SELECT word_count FROM papers_fulltext WHERE paper_id = ?", (paper_id,))
            fulltext_info = c.fetchone()
            conn.close()
            return {
                'title': meta[0], 'authors': meta[1], 'year': meta[2],
                'abstract': meta[3], 'pdf_url': meta[4],
                'relevance_score': meta[5], 'enhanced_relevance_score': meta[6],
                'dopant_present': meta[7], 'beta_phase_present': meta[8],
                'download_status': meta[9], 'has_pdf': meta[10],
                'has_fulltext': meta[11], 'journal_ref': meta[12],
                'pdf_bytes': pdf_bytes,
                'word_count': fulltext_info[0] if fulltext_info else 0
            }
        except Exception as e:
            update_log(f"Failed to get paper info for {paper_id}: {e}")
            return None

    def create_zip_from_db(self, paper_ids: List[str]) -> io.BytesIO:
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for paper_id in paper_ids:
                    pdf_data = self.get_pdf(paper_id)
                    if pdf_data:
                        info = self.get_paper_info(paper_id)
                        if info:
                            title = re.sub(r'[^\w\s-]', '', info['title'])[:100]
                            authors = info['authors'].split(',')[0][:50] if info['authors'] else 'unknown'
                            filename = f"{paper_id}_{authors}_{info['year']}_{title}.pdf"
                            filename = re.sub(r'\s+', '_', filename)
                        else:
                            filename = f"{paper_id}.pdf"
                        zip_file.writestr(filename, pdf_data)
            zip_buffer.seek(0)
            update_log(f"Created ZIP with {len(paper_ids)} PDFs")
        except Exception as e:
            update_log(f"Failed to create ZIP: {e}")
        return zip_buffer

    def export_metadata(self, format: str = "csv",
                        topic: Optional[str] = None) -> io.BytesIO:
        """
        Raw metadata export (CSV/JSON/Excel). Optionally filter by topic so
        a Q export doesn't include legacy σ₀ / SRS / ρ₀ / Glen / rate-factor rows.
        """
        try:
            conn = sqlite3.connect(self.metadata_db)
            sql = "SELECT * FROM papers"
            params: Tuple = ()
            if topic:
                sql += " WHERE topic = ?"
                params = (topic,)

            if format.lower() == "csv":
                df = pd.read_sql_query(sql, conn, params=params)
                output = io.BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)
            elif format.lower() == "json":
                df = pd.read_sql_query(sql, conn, params=params)
                output = io.BytesIO()
                df.to_json(output, orient="records", indent=2)
                output.seek(0)
            elif format.lower() == "excel":
                df = pd.read_sql_query(sql, conn, params=params)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Papers')
                output.seek(0)
            else:
                output = io.BytesIO()
            conn.close()
            return output
        except Exception as e:
            update_log(f"Export failed: {e}")
            return io.BytesIO()


db_manager = DatabaseManager()


# -------------------------- PDF DOWNLOAD AND TEXT EXTRACTION --------------------------


def download_pdf_bytes(pdf_url: str) -> Optional[bytes]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download():
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=3))
        response = session.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    try:
        pdf_bytes = _download()
        if len(pdf_bytes) < 1024:
            raise ValueError("PDF file too small")
        return pdf_bytes
    except Exception as e:
        update_log(f"Download failed for {pdf_url}: {e}")
        return None


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(min(50, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1000000]
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def handle_paper_download(paper: Dict[str, Any], manual_download: bool = False) -> Dict[str, Any]:
    paper_id = paper['id']
    if manual_download and paper_id in st.session_state.downloaded_pdfs:
        update_log(f"PDF for {paper_id} already in session")
        return paper
    try:
        update_log(f"Downloading PDF for {paper_id}...")
        pdf_bytes = download_pdf_bytes(paper['pdf_url'])
        if pdf_bytes is None:
            paper['download_status'] = "Failed to download"
            return paper
        full_text = extract_text_from_bytes(pdf_bytes)
        pdf_stored = db_manager.store_pdf_data(paper_id, pdf_bytes, paper['pdf_url'])
        if not full_text.startswith("Error"):
            analysis = analyze_activation_energy_relevance(full_text)
            paper['dopant_present']     = analysis['dopant_present']
            paper['beta_phase_present'] = analysis['beta_phase_present']
            if 'query_emb' in st.session_state:
                full_emb = get_embedding(full_text)
                sim1 = cosine_sim(st.session_state.query_emb, full_emb)
                sim2 = cosine_sim(st.session_state.key_emb, full_emb)
                paper['enhanced_relevance_score'] = round((0.7 * sim1 + 0.3 * sim2) * 100, 2)
            else:
                paper['enhanced_relevance_score'] = 0.0
            text_stored = db_manager.store_fulltext(paper_id, paper['title'],
                                                    paper['abstract'], full_text)
        else:
            text_stored = False
        paper['pdf_stored'] = 1 if pdf_stored else 0
        paper['fulltext_stored'] = 1 if text_stored else 0
        paper['pdf_size'] = len(pdf_bytes)
        paper['download_time'] = datetime.now().isoformat()
        paper['download_status'] = "Successfully downloaded and stored"
        st.session_state.downloaded_pdfs[paper_id] = {
            'pdf_bytes': pdf_bytes,
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year']
        }
        db_manager.store_paper_metadata(paper)
        update_log(f"✅ Successfully processed {paper_id}")
    except Exception as e:
        paper['download_status'] = f"Failed: {str(e)[:100]}"
        update_log(f"❌ Failed to process {paper_id}: {e}")
    return paper


@st.cache_data(ttl=3600)
def query_arxiv(query: str, categories: List[str], max_results: int,
                start_year: int, end_year: int,
                topic: str = "activation_energy_ice") -> List[Dict[str, Any]]:
    """
    Query arXiv for the given search string. The 'topic' argument is part of
    the cache key, so switching campaigns never serves stale cached results
    from another topic. Each returned paper dict carries its 'topic'.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=min(max_results * 2, 500),
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    results = []
    for result in client.results(search):
        if not (any(cat in result.categories for cat in categories) and
                start_year <= result.published.year <= end_year):
            continue

        paper = {
            'id': result.entry_id.split('/')[-1],
            'arxiv_id': result.entry_id,
            'title': result.title,
            'authors': ', '.join(a.name for a in result.authors),
            'year': result.published.year,
            'categories': ', '.join(result.categories),
            'abstract': result.summary,
            'pdf_url': result.pdf_url,
            'published_date': result.published.isoformat(),
            'updated_date': result.updated.isoformat() if result.updated else result.published.isoformat(),
            'doi': result.doi if hasattr(result, 'doi') and result.doi else None,
            'journal_ref': getattr(result, 'journal_ref', None),
            'relevance_score': 0.0,
            'matched_terms': '',
            'download_status': 'Pending',
            'pdf_stored': 0,
            'fulltext_stored': 0,
            'pdf_size': 0,
            'download_time': None,
            'enhanced_relevance_score': 0.0,
            'dopant_present': False,
            'beta_phase_present': False,
            'topic': topic,
        }

        text_lower = (paper['title'] + ' ' + paper['abstract']).lower()
        paper['matched_terms'] = '; '.join(sorted(
            t for t in DOMAIN_TERMS if t in text_lower
        ))

        results.append(paper)
        if len(results) >= max_results:
            break
    return results[:max_results]


# -------------------------- SCOPUS-FORMAT EXPORT (Activation Energy) --------------------------


SCOPUS_COLUMNS = [
    "Authors", "Author full names", "Author(s) ID", "Title", "Year",
    "Source title", "Volume", "Issue", "Art. No.", "Page start", "Page end",
    "Cited by", "DOI", "Link", "Abstract", "Author Keywords", "Index Keywords",
    "Document Type", "Publication Stage", "Open Access", "Source", "EID",
]


# Activation energy leads (primary campaign); rate factor, Glen n, and
# enhancement factor trail (siblings). Two dedicated regime columns carry
# the Q1/Q2 creep-mechanism classification.
EXTRA_COLUMNS = [
    "Full Text", "arXiv ID", "Categories", "PDF URL", "Journal Ref",
    "Relevance Score", "Enhanced Relevance Score", "Matched Terms",
    "Candidate Activation Energy",
    "Candidate Activation Energy Q1 (J/mol)",
    "Candidate Activation Energy Q2 (J/mol)",
    "Candidate Activation Energy Regime",
    "Candidate Rate Factor",
    "Candidate Glen n",
    "Candidate Enhancement Factor",
    "Download Status", "PDF Stored", "Full Text Stored",
    "PDF Size (KB)", "Word Count", "Topic",
]


def format_scopus_authors_short(authors_str: str) -> str:
    """'Junwan Li, Jifa Mei' -> 'Li J.; Mei J.'   (Scopus 'Authors' style)"""
    out = []
    for name in [a.strip() for a in (authors_str or '').split(',') if a.strip()]:
        parts = name.split()
        if len(parts) == 1:
            out.append(parts[0])
        else:
            surname = parts[-1]
            initials = ''.join(p[0].upper() for p in parts[:-1] if p)
            out.append(f"{surname} {initials}")
    return '; '.join(out)


def format_scopus_authors_full(authors_str: str) -> str:
    """'Junwan Li' -> 'Li, Junwan'   (Scopus 'Author full names' style)"""
    out = []
    for name in [a.strip() for a in (authors_str or '').split(',') if a.strip()]:
        parts = name.split()
        if len(parts) >= 2:
            out.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
        else:
            out.append(name)
    return '; '.join(out)


def parse_journal_ref(jr: str) -> Dict[str, str]:
    """Best-effort parse of arXiv journal_ref, e.g.
    'J. Glaciol. 55 (2009) 319-322'  or  'Ann. Glaciol. 60, 174104 (2009)'."""
    out = {"src": "", "vol": "", "pg_start": "", "pg_end": "", "yr": ""}
    if not jr:
        return out
    s = jr.strip()
    ym = re.search(r'(19|20)\d{2}', s)
    if ym:
        out["yr"] = ym.group(0)
    vm = re.match(r'^(?P<src>[A-Za-z][^,;]*?)\s+(?P<vol>\d+)\s*[,(]', s)
    if vm:
        out["src"] = vm.group('src').strip()
        out["vol"] = vm.group('vol')
    pm = re.search(r'(?P<a>\d+)[-–](?P<b>\d+)', s)
    if pm:
        out["pg_start"], out["pg_end"] = pm.group('a'), pm.group('b')
    else:
        am = re.search(r'[,\s](\d{4,6})(?:\s*\(\d{4}\))?$', s)
        if am:
            out["pg_start"] = am.group(1)
    return out


# --------------------------------------------------------------
# CREEP-MECHANISM CLASSIFIER — Q1 (dislocation) vs Q2 (grain-boundary)
# --------------------------------------------------------------
def classify_creep_mechanism(window_text: str) -> str:
    """
    Classify the creep mechanism associated with an activation-energy hit.

    The two-branch Arrhenius parameterisation of the Glen flow law comes
    from a change in the rate-controlling mechanism:
        Q1 — cold ice,  T < -10 °C — dislocation glide / creep
                                    (higher activation energy ~ 60 kJ/mol)
        Q2 — warm ice,  T > -10 °C — grain-boundary sliding / diffusion
                                    (~139 kJ/mol in Paterson & Budd, or
                                     ~115 kJ/mol in Cuffey & Paterson)

    This function inspects the local window for mechanism keywords and
    returns a short tag suitable for appending to the extractor output
    and for a dedicated CSV column.
    """
    t = (window_text or "").lower()

    # Warm-ice / Q2 indicators
    warm_cues = [
        'grain boundary', 'grain-boundary', 'sliding', 'temperate',
        't > -10', 't>-10', 'above -10', 'above-10', 'basal',
        'melting point', 'pressure melting', 'q2', 'q 2',
        'warm ice', 'warm-ice',
    ]
    # Cold-ice / Q1 indicators
    cold_cues = [
        'dislocation', 'creep', 'cold ice', 'cold-ice',
        't < -10', 't<-10', 'below -10', 'below-10',
        'q1', 'q 1', 'glide', 'polar', 'interior',
    ]

    has_warm = any(c in t for c in warm_cues)
    has_cold = any(c in t for c in cold_cues)

    if has_warm and not has_cold:
        return "WARM_ICE (Q2 / Grain Boundary)"
    if has_cold and not has_warm:
        return "COLD_ICE (Q1 / Dislocation)"
    if has_warm and has_cold:
        return "BOTH_REGIMES_NEARBY"
    return "UNSPECIFIED_REGIME"


def _classify_creep_regime_for_row(full_text: str) -> str:
    """
    Convenience wrapper: apply classify_creep_mechanism to the whole text so
    the CSV carries a single summary tag per row. A paper that discusses
    both branches returns BOTH_REGIMES_NEARBY — which is itself useful
    information (the paper is talking about the split explicitly).
    """
    return classify_creep_mechanism(full_text)


# --------------------------------------------------------------
# UNIT & MAGNITUDE RESOLVER — the core safety net for Q
# --------------------------------------------------------------
def resolve_activation_energy_units(value: float, unit_str: str) -> Tuple[float, str]:
    """
    Attempts to standardise an extracted Q value to J/mol (the Elmer/Ice SIF
    standard). Returns (standardised_value, warning_string).

    The trap this function is designed to catch:
        Paper says "Q = 60 kJ mol⁻¹"     → user pastes 60 into a SIF
        SIF expects J/mol                → glacier barely flows
    Correct answer: 6.0 × 10⁴ J/mol, i.e. 60e3.

    Cases handled:
      - Explicit kJ/mol / kJ mol-1  → multiply by 1000
      - Explicit J/mol  / J mol-1   → pass through
      - Explicit eV (MD papers)     → multiply by 96485 (per mole of molecules)
      - No unit found               → infer from magnitude
                                        (10–200  → assume kJ/mol)
                                        (1e4–2e5 → assume J/mol)
    """
    us = (unit_str or "").lower().strip()

    # Case 1: explicit kJ/mol (most common in glaciology literature)
    if 'kj' in us:
        return value * 1000.0, "✅ KJ_TO_J_CONVERTED"

    # Case 2: explicit J/mol (already SIF-ready)
    if 'j' in us and 'kj' not in us:
        return value, "✅ J_MOL"

    # Case 3: eV (rare in ice, but occurs in MD / ab-initio papers)
    if 'ev' in us:
        return value * 96485.0, "⚠️ EV_TO_J_CONVERTED"

    # Case 4: no units found — infer from magnitude (the biggest trap)
    if 10.0 < value < 200.0:
        # A bare 60, 115, 139 is definitively kJ/mol
        return value * 1000.0, "⚠️ ASSUMED_KJ/mol_MULTIPLIED_BY_1000"
    if 10_000.0 < value < 200_000.0:
        # A bare 60000, 115000, 139000 is J/mol
        return value, "⚠️ ASSUMED_J/mol"
    return value, "❌ UNKNOWN_UNIT_MAGNITUDE"


# --------------------------------------------------------------
# ACTIVATION-ENERGY EXTRACTOR — unit + magnitude + regime aware
# --------------------------------------------------------------
def extract_candidate_activation_energy(full_text: str, max_hits: int = 15) -> str:
    """
    Candidate Activation Energy (Q) snippets — UNIT-AND-MAGNITUDE AWARE.

    Extracts Q, resolves units to J/mol (the Elmer/Ice SIF standard),
    classifies the creep regime (Q1 dislocation vs Q2 grain-boundary), and
    emits a warning tag for anything that needed conversion or inference.

    Typical hits to fish for:
      - 'activation energy of 60 kJ mol⁻¹ for dislocation creep'
      - 'Q1 = 60 kJ/mol (T < -10 °C), Q2 = 139 kJ/mol (T > -10 °C)'
      - 'we take Q = 6.0 × 10⁴ J mol⁻¹'
      - 'activation energy 0.53 eV'                       (MD paper)
      - 'the creep activation energy was found to be 60'  (bare number)

    Pointer system for NER, not a final extraction.
    """
    if not full_text or full_text.startswith("Error"):
        return ""

    t = _normalize_flattened_text(full_text)

    # Keywords
    kw = re.compile(
        r'activation\s+energy|activation\s+enthalpy|energy\s+barrier|'
        r'creep\s+activation|temperature\s+of\s+creep|'
        r'\bq\s*[=:~]\s*\d|\bq\s*1\b|\bq\s*2\b|\bq1\b|\bq2\b|'
        r'\bea\s*[=:~]\s*\d|'
        r'dislocation\s+creep|grain\s+boundary\s+sliding|diffusion\s+creep|'
        r'rate\s+controlling|'
        r'arrhenius\s+(?:relation|law|factor)',
        re.IGNORECASE)

    # Number pattern: 60, 139, 115.5, 6.0e4, 5.3e4, 0.53
    num_pattern = re.compile(
        r'\d+(?:\.\d+)?(?:\s*[xX]\s*10\s*[-+]?\s*\d+|(?:\s*[eE]\s*[-+]?\s*\d+))?',
        re.IGNORECASE)

    # Unit pattern: kJ/mol, kJ mol-1, kJmol-1, J/mol, J mol−1, eV
    # Covers the flattened forms 'kJmol-1', 'kJ mol - 1', 'kJ/mole'
    unit_pattern = re.compile(
        r'(?:kJ|J|eV)\s*(?:/\s*mol(?:e|es)?|mol(?:e|es)?\s*[-]?\s*1?|'
        r'per\s+mol(?:e)?|mol\s*-\s*1)?',
        re.IGNORECASE)

    hits, seen = [], set()
    for m in kw.finditer(t):
        window = re.sub(r'\s+', ' ',
                        t[max(0, m.start() - 120): m.end() + 250]).strip()

        vals = num_pattern.findall(window)
        units = unit_pattern.findall(window)

        if vals and window[:60] not in seen:
            seen.add(window[:60])

            # Parse the first numeric value — handle x10N and eN forms
            raw_num_str = vals[0].strip()
            try:
                # Normalise '3 x 10 4' and '3x104' and '3e4' to a float
                norm = re.sub(r'\s+', '', raw_num_str)
                norm = re.sub(r'[xX]10([-+]?\d+)', r'e\1', norm)
                raw_val = float(norm)
            except Exception:
                try:
                    raw_val = float(re.findall(r'\d+(?:\.\d+)?', raw_num_str)[0])
                except Exception:
                    continue

            raw_unit = units[0] if units else "unknown"

            # Resolve to J/mol (SIF standard)
            std_val, unit_warning = resolve_activation_energy_units(raw_val, raw_unit)

            # Classify creep mechanism (Q1 vs Q2)
            regime = classify_creep_mechanism(window)

            hit_str = (
                f"{window[:180]}  "
                f"[→ Raw: {raw_val} {raw_unit} | SIF_Standard: {std_val:.3e} J/mol "
                f"| Regime: {regime} | {unit_warning}]"
            )
            hits.append(hit_str)

        if len(hits) >= max_hits:
            break

    return " || ".join(hits)


def extract_q1_q2_scalars(full_text: str) -> Tuple[str, str]:
    """
    Best-effort single-scalar extraction of Q1 and Q2 (in J/mol) from the
    whole document. Returns two strings suitable for the dedicated CSV
    columns "Candidate Activation Energy Q1 (J/mol)" and "… Q2 (J/mol)".

    Strategy: scan for `Q1 = <num> <unit>`, `Q2 = <num> <unit>`, and the
    common prose forms `Q1 is …` / `Q2 is …` / `Q1 of …` / `Q2 of …`.
    Where the label is absent, fall back to the first cold-ice and warm-ice
    hits found anywhere in the document.
    """
    if not full_text or full_text.startswith("Error"):
        return "", ""

    t = _normalize_flattened_text(full_text)

    # Explicit Q1 = <num><unit> / Q2 = <num><unit>
    label_re = re.compile(
        r'\bq\s*1\s*[=:~]?\s*(\d+(?:\.\d+)?(?:\s*[xX]\s*10\s*[-+]?\s*\d+|'
        r'(?:\s*[eE]\s*[-+]?\s*\d+))?)\s*'
        r'(kJ\s*(?:/\s*mol|mol\s*-?\s*1?|per\s+mol)?|'
        r'J\s*(?:/\s*mol|mol\s*-?\s*1?|per\s+mol)?|eV)?',
        re.IGNORECASE)
    label_re_2 = re.compile(
        r'\bq\s*2\s*[=:~]?\s*(\d+(?:\.\d+)?(?:\s*[xX]\s*10\s*[-+]?\s*\d+|'
        r'(?:\s*[eE]\s*[-+]?\s*\d+))?)\s*'
        r'(kJ\s*(?:/\s*mol|mol\s*-?\s*1?|per\s+mol)?|'
        r'J\s*(?:/\s*mol|mol\s*-?\s*1?|per\s+mol)?|eV)?',
        re.IGNORECASE)

    def _to_jmol(num_str: str, unit_str: str) -> Optional[str]:
        if not num_str:
            return None
        try:
            norm = re.sub(r'\s+', '', num_str)
            norm = re.sub(r'[xX]10([-+]?\d+)', r'e\1', norm)
            v = float(norm)
        except Exception:
            return None
        j, _warn = resolve_activation_energy_units(v, unit_str or "")
        return f"{j:.3e}"

    m1 = label_re.search(t)
    m2 = label_re_2.search(t)

    q1 = _to_jmol(m1.group(1), m1.group(2)) if m1 else None
    q2 = _to_jmol(m2.group(1), m2.group(2)) if m2 else None

    # Fallback: scan the extractor output for the first COLD_ICE and WARM_ICE hits
    if q1 is None or q2 is None:
        blob = extract_candidate_activation_energy(full_text, max_hits=30)
        for hit in blob.split(" || "):
            if "COLD_ICE" in hit and q1 is None:
                mm = re.search(r'SIF_Standard:\s*([\d.eE+-]+)\s*J/mol', hit)
                if mm:
                    q1 = mm.group(1)
            if "WARM_ICE" in hit and q2 is None:
                mm = re.search(r'SIF_Standard:\s*([\d.eE+-]+)\s*J/mol', hit)
                if mm:
                    q2 = mm.group(1)
            if q1 and q2:
                break

    return (q1 or ""), (q2 or "")


# --------------------------------------------------------------
# SIBLING EXTRACTORS — rate factor A, Glen n, enhancement factor E
# --------------------------------------------------------------
def extract_candidate_rate_factor(full_text: str, max_hits: int = 15) -> str:
    """
    Candidate Rate Factor (A) snippets — UNIT-AWARE.

    The rate factor A is a scientific-notation quantity whose units are
    dangerous to ignore:
        1.258e13      a⁻¹ Pa⁻³      (Elmer/Ice SIF convention)
        3.5e-25       s⁻¹ MPa⁻³     (common in glaciology papers)
        2.4e-24       s⁻¹ Pa⁻³      (Paterson & Budd, 0 °C)

    Pointer system for NER, not a final extraction.
    """
    if not full_text or full_text.startswith("Error"):
        return ""

    t = _normalize_flattened_text(full_text)

    kw = re.compile(
        r'rate\s+factor|pre[- ]?exponential(?:\s+factor)?|arrhenius\s+factor|'
        r'flow\s+law\s+parameter\s+a|creep\s+parameter(?:\s+a)?|'
        r'arrhenius\s+(?:relation|law)|'
        r'\bA\s*[=:~]\s*\d|rate\s+constant',
        re.IGNORECASE)

    sci_not = re.compile(
        r'\d+(?:\.\d+)?\s*(?:[eE]|[xX]\s*10|10)\s*[-+]?\s*\d+'
        r'|\d+(?:\.\d+)?\s*[xX]\s*10\s*[-+]?\s*\d+',
        re.IGNORECASE)

    unit_pattern = re.compile(
        r'(?:M?Pa|kPa|GPa)\s*[-]?\s*\d*\s*(?:a|s|yr|sec|year)s?\s*[-]?\s*\d*'
        r'|(?:a|s|yr|sec|year)s?\s*[-]?\s*\d*\s*(?:M?Pa|kPa|GPa)\s*[-]?\s*\d*'
        r'|\b(?:a|s|yr|sec|year)s?\s*[-]\s*1',
        re.IGNORECASE)

    hits, seen = [], set()
    for m in kw.finditer(t):
        window = re.sub(r'\s+', ' ',
                        t[max(0, m.start() - 120): m.end() + 250]).strip()

        vals  = sci_not.findall(window)
        units = unit_pattern.findall(window)

        if vals and window[:60] not in seen:
            seen.add(window[:60])

            unit_warning = ""
            unit_str = ""
            if units:
                unit_str = units[0].strip()
                low = unit_str.lower()
                if 'mpa' in low or 'kpa' in low or 'gpa' in low:
                    unit_warning += " ⚠️ PREFIXED_PRESSURE_UNIT (SIF needs Pa, not MPa/kPa/GPa)"
                if re.search(r'\bs\b|\bsec\b', low) and 'yr' not in low and ' a ' not in f' {low} ':
                    unit_warning += " ⚠️ SECONDS_UNIT (check whether SIF needs a/yr)"

            regime = classify_creep_mechanism(window)

            val_str = vals[0].strip()
            unit_display = unit_str if unit_str else "NO_UNIT"
            hit_str = (f"{window[:200]}  "
                       f"[→ {val_str} {unit_display} | {regime}{unit_warning}]")
            hits.append(hit_str)

        if len(hits) >= max_hits:
            break

    return " || ".join(hits)


def extract_candidate_glen_n(full_text: str, max_hits: int = 15) -> str:
    """
    Candidate Glen Exponent (n) snippets.

    Looks for keyword instances of the Glen exponent near a plausible n value.
    Pointer system for NER, not a final extraction.
    """
    if not full_text or full_text.startswith("Error"):
        return ""
    t = _normalize_flattened_text(full_text)
    kw = re.compile(
        r'glen\s+exponent|glens?\s+flow\s+law|glen\'?s\s+law|'
        r'flow\s+law\s+exponent|creep\s+exponent|power\s+law\s+exponent|'
        r'exponent\s+n|stress\s+exponent|strain\s+rate\s+exponent|'
        r'constitutive\s+(?:law|relation)|ice\s+rheology|'
        r'flow\s+law\s+parameter|creep\s+parameter|rheological\s+exponent',
        re.IGNORECASE)
    num_unit = re.compile(
        r'\bn\s*[=:~≈]\s*[1-5](?:\.\d+)?'
        r'|exponent\s+(?:of\s+)?[1-5](?:\.\d+)?'
        r'|\bn\s+(?:is|was|equal(?:s)?\s+to|taken\s+as)\s+[1-5](?:\.\d+)?'
        r'|\bn\s+(?:between|ranging\s+from|in\s+the\s+range)\s+[1-5](?:\.\d+)?\s*(?:and|to|-)\s*[1-5](?:\.\d+)?',
        re.IGNORECASE)
    hits, seen = [], set()
    for m in kw.finditer(t):
        window = re.sub(r'\s+', ' ',
                        t[max(0, m.start() - 100):m.end() + 160]).strip()
        vals = num_unit.findall(window)
        if vals and window[:60] not in seen:
            seen.add(window[:60])
            hits.append(f"{window[:200]}  [→ {', '.join(v.strip() for v in vals[:3])}]")
        if len(hits) >= max_hits:
            break
    return " || ".join(hits)


def extract_candidate_enhancement_factor(full_text: str, max_hits: int = 15) -> str:
    """
    Candidate enhancement-factor E snippets.

    The enhancement factor E multiplies A to account for anisotropic ice
    fabrics, impurities, or preferred slip.

    Pointer system for NER, not a final extraction.
    """
    if not full_text or full_text.startswith("Error"):
        return ""
    t = _normalize_flattened_text(full_text)
    kw = re.compile(
        r'enhancement\s+factor|enhancement\s+parameter|'
        r'anisotropy\s+factor|fabric\s+enhancement|'
        r'enhancement\s+of\s+the\s+flow\s+law|softening\s+factor|'
        r'hardening\s+factor',
        re.IGNORECASE)
    num_unit = re.compile(
        r'\bE\s*[=:~≈]\s*\d+(?:\.\d+)?'
        r'|enhancement\s+(?:factor|parameter)\s+(?:of\s+)?\d+(?:\.\d+)?'
        r'|\b\d+(?:\.\d+)?\s*(?:x|times)\s+(?:the\s+)?(?:reference|standard)\s+flow'
        r'|\b\d+(?:\.\d+)?-?(?:fold|times)\s+enhancement',
        re.IGNORECASE)
    hits, seen = [], set()
    for m in kw.finditer(t):
        window = re.sub(r'\s+', ' ', t[max(0, m.start() - 80): m.end() + 160]).strip()
        vals = num_unit.findall(window)
        if vals and window[:60] not in seen:
            seen.add(window[:60])
            hits.append(f"{window[:180]}  [→ {', '.join(v.strip() for v in vals[:3])}]")
        if len(hits) >= max_hits:
            break
    return " || ".join(hits)


# Legacy aliases (kept for back-compat with imported call sites)
extract_candidate_rho0   = extract_candidate_activation_energy  # primary now
extract_candidate_srs    = extract_candidate_rate_factor
extract_candidate_sigma0 = extract_candidate_glen_n


def export_scopus_format(include_fulltext: bool = True,
                          fulltext_char_limit: Optional[int] = None,
                          only_with_fulltext: bool = False,
                          require_metadata_flag: bool = False,
                          topic: Optional[str] = None) -> io.BytesIO:
    """
    Build a Scopus-compatible CSV.

    Parameters
    ----------
    include_fulltext : bool
        Emit the appended "Full Text" column from papers_fulltext.
    fulltext_char_limit : int | None
        Truncate each paper's full text to this many chars (None = no limit).
    only_with_fulltext : bool
        Skip rows that have no full_text payload in the universe DB.
    require_metadata_flag : bool
        Skip rows where metadata.fulltext_stored is not 1.
    topic : str | None
        Restrict export to a single campaign (e.g. 'activation_energy_ice').
        None = all campaigns (combined view; all four extractors live).
    """
    try:
        conn = sqlite3.connect(db_manager.metadata_db)
        sql = "SELECT * FROM papers"
        params: Tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        df_meta = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        if df_meta.empty:
            return io.BytesIO()

        conn = sqlite3.connect(db_manager.universe_db)
        df_text = pd.read_sql_query(
            "SELECT paper_id, full_text, word_count FROM papers_fulltext", conn)
        conn.close()
        text_map = {r['paper_id']: (r['full_text'], r['word_count'])
                    for _, r in df_text.iterrows()}

        rows = []
        for _, r in df_meta.iterrows():
            full_text, word_count = text_map.get(r['id'], ("", 0))

            if require_metadata_flag and not bool(r.get('fulltext_stored')):
                continue
            if only_with_fulltext and not full_text:
                continue

            ft = (full_text or "")[:fulltext_char_limit] if fulltext_char_limit else (full_text or "")

            jr = parse_journal_ref(r.get('journal_ref'))
            arxiv_short = str(r.get('arxiv_id', '')).split('/abs/')[-1] or str(r.get('id', ''))
            abs_link = f"https://arxiv.org/abs/{arxiv_short}"

            corpus = ((r.get('abstract') or '') + ' ' + (full_text or '')).lower()
            index_kw = '; '.join(sorted(t for t in DOMAIN_TERMS if t in corpus))

            # Per-row creep-regime summary and Q1/Q2 scalars
            regime_summary = _classify_creep_regime_for_row(full_text or "")
            q1_scalar, q2_scalar = extract_q1_q2_scalars(full_text or "")

            row = {
                # ---------- exact Scopus columns, exact order ----------
                "Authors": format_scopus_authors_short(r.get('authors') or ""),
                "Author full names": format_scopus_authors_full(r.get('authors') or ""),
                "Author(s) ID": "",
                "Title": r.get('title') or "",
                "Year": jr["yr"] or r.get('year') or "",
                "Source title": jr["src"] or "arXiv",
                "Volume": jr["vol"],
                "Issue": "",
                "Art. No.": jr["pg_start"] if jr["pg_start"] and not jr["pg_end"] else "",
                "Page start": jr["pg_start"] if jr["pg_end"] else "",
                "Page end": jr["pg_end"],
                # Semantic Scholar Cited-by intentionally disabled:
                # one HTTPS call per row is not acceptable by default.
                "Cited by": "",
                "DOI": r.get('doi') or "",
                "Link": abs_link,
                "Abstract": r.get('abstract') or "",
                "Author Keywords": "",
                "Index Keywords": index_kw,
                "Document Type": "Article",
                "Publication Stage": "Final",
                "Open Access": "All Open Access",
                "Source": "arXiv",
                "EID": f"arxiv-{arxiv_short}",
                # ---------- appended extra columns ----------
                "Full Text": ft if include_fulltext else "",
                "arXiv ID": arxiv_short,
                "Categories": r.get('categories') or "",
                "PDF URL": r.get('pdf_url') or "",
                "Journal Ref": r.get('journal_ref') or "",
                "Relevance Score": r.get('relevance_score') or 0.0,
                "Enhanced Relevance Score": r.get('enhanced_relevance_score') or 0.0,
                "Matched Terms": r.get('matched_terms') or "",
                "Candidate Activation Energy":
                    extract_candidate_activation_energy(full_text),
                "Candidate Activation Energy Q1 (J/mol)": q1_scalar,
                "Candidate Activation Energy Q2 (J/mol)": q2_scalar,
                "Candidate Activation Energy Regime": regime_summary,
                "Candidate Rate Factor": extract_candidate_rate_factor(full_text),
                "Candidate Glen n": extract_candidate_glen_n(full_text),
                "Candidate Enhancement Factor":
                    extract_candidate_enhancement_factor(full_text),
                "Download Status": r.get('download_status') or "",
                "PDF Stored": r.get('pdf_stored') or 0,
                "Full Text Stored": r.get('fulltext_stored') or 0,
                "PDF Size (KB)": round((r.get('pdf_size') or 0) / 1024, 1),
                "Word Count": word_count or 0,
                "Topic": r.get('topic') or "sigma0",
            }
            rows.append(row)

        df_out = pd.DataFrame(rows, columns=SCOPUS_COLUMNS + EXTRA_COLUMNS)
        df_out.index = range(1, len(df_out) + 1)          # Scopus-style row-number column
        output = io.BytesIO()
        df_out.to_csv(output, index=True, encoding='utf-8-sig')  # BOM = exact Scopus encoding
        output.seek(0)
        update_log(f"Scopus-format CSV exported: {len(df_out)} rows "
                   f"(topic={topic or '(all)'}, only_with_fulltext={only_with_fulltext}, "
                   f"require_metadata_flag={require_metadata_flag})")
        return output
    except Exception as e:
        update_log(f"Scopus-format export failed: {e}")
        return io.BytesIO()


# -------------------------- MAINTENANCE HELPERS --------------------------


def get_missing_fulltext_ids(topic: Optional[str] = None) -> List[str]:
    """Return IDs of metadata rows (optionally a single topic) that have a
    pdf_url but no stored full text."""
    try:
        conn = sqlite3.connect(db_manager.metadata_db)
        c = conn.cursor()
        sql = ("SELECT id FROM papers "
               "WHERE (fulltext_stored = 0 OR fulltext_stored IS NULL) "
               "AND pdf_url IS NOT NULL")
        params: Tuple = ()
        if topic:
            sql += " AND topic = ?"
            params = (topic,)
        rows = c.execute(sql, params).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        update_log(f"get_missing_fulltext_ids failed: {e}")
        return []


def get_fulltext_coverage(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Aggregate coverage counters used to decide whether the CSV will carry text.
    'with_ft_universe_rows' is the number that actually backs the Full Text column.
    'topic=None' gives whole-DB coverage (used by the combined-view toggle).
    """
    where = ""
    params: Tuple = ()
    if topic:
        where = " WHERE topic = ?"
        params = (topic,)

    conn = sqlite3.connect(db_manager.metadata_db)
    total = conn.execute(f"SELECT COUNT(*) FROM papers{where}", params).fetchone()[0]
    with_pdf = conn.execute(
        f"SELECT COUNT(*) FROM papers{where}{' AND' if where else ' WHERE'} pdf_stored = 1",
        params).fetchone()[0]
    with_ft_meta = conn.execute(
        f"SELECT COUNT(*) FROM papers{where}{' AND' if where else ' WHERE'} fulltext_stored = 1",
        params).fetchone()[0]
    ids = [row[0] for row in conn.execute(
        f"SELECT id FROM papers{where}", params).fetchall()]
    conn.close()

    conn = sqlite3.connect(db_manager.universe_db)
    if ids:
        CHUNK = 500
        with_ft_db = 0
        total_words = 0
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            qmarks = ",".join("?" * len(chunk))
            with_ft_db += conn.execute(
                f"SELECT COUNT(*) FROM papers_fulltext WHERE paper_id IN ({qmarks})",
                tuple(chunk)).fetchone()[0]
            total_words += conn.execute(
                f"SELECT COALESCE(SUM(word_count),0) FROM papers_fulltext "
                f"WHERE paper_id IN ({qmarks})", tuple(chunk)).fetchone()[0]
    else:
        with_ft_db = 0
        total_words = 0
    conn.close()

    pct = (100.0 * with_ft_db / total) if total else 0.0
    return {
        "total": total,
        "with_pdf": with_pdf,
        "with_ft_metadata_flag": with_ft_meta,
        "with_ft_universe_rows": with_ft_db,
        "total_words": int(total_words or 0),
        "coverage_pct": round(pct, 1),
        "missing_count": max(0, total - with_ft_db),
        "topic": topic or "(all)",
    }


def backfill_missing_fulltexts(limit: Optional[int] = None,
                               silent: bool = False,
                               topic: Optional[str] = None) -> int:
    """
    Download + extract + store full texts for every metadata row (optionally
    restricted to a topic) that has a pdf_url but no stored full text.
    Idempotent via pdf_hash dedup.
    """
    conn = sqlite3.connect(db_manager.metadata_db)
    q = ("SELECT * FROM papers "
         "WHERE (fulltext_stored = 0 OR fulltext_stored IS NULL) "
         "AND pdf_url IS NOT NULL")
    params: Tuple = ()
    if topic:
        q += " AND topic = ?"
        params = (topic,)
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    if df.empty:
        update_log("Backfill: nothing to do — all rows already have full text")
        return 0

    papers = df.to_dict("records")
    total = len(papers)
    progress = st.progress(0) if not silent else None
    status = st.empty() if not silent else None
    ok = 0
    for i, paper in enumerate(papers, 1):
        if not silent and status is not None:
            status.text(f"⬇️ {i}/{total}: {str(paper.get('title',''))[:60]}…")
        try:
            updated = handle_paper_download(paper, manual_download=True)
            if updated.get('fulltext_stored'):
                ok += 1
        except Exception as e:
            update_log(f"Backfill error for {paper.get('id')}: {e}")
        if not silent and progress is not None:
            progress.progress(i / total)
    if not silent:
        if progress is not None:
            progress.empty()
        if status is not None:
            status.empty()
    update_log(f"Backfill complete: {ok}/{total} papers now carry full text")
    return ok


def repair_relevance_scores() -> int:
    """
    Fix legacy rows where relevance_score was stored as a 4-byte float32 BLOB
    (an earlier version wrote raw numpy floats). Returns count fixed.
    """
    conn = sqlite3.connect(db_manager.metadata_db)
    c = conn.cursor()
    fixed = 0
    for pid, val in c.execute("SELECT id, relevance_score FROM papers").fetchall():
        if isinstance(val, (bytes, bytearray)) and len(val) == 4:
            score = struct.unpack("<f", bytes(val))[0]
            c.execute("UPDATE papers SET relevance_score=? WHERE id=?",
                      (round(float(score), 2), pid))
            fixed += 1
    conn.commit()
    conn.close()
    if fixed:
        update_log(f"Repaired {fixed} BLOB relevance_score rows")
    return fixed


def coverage_warning_message(topic: Optional[str] = None) -> Optional[str]:
    """
    Return the canonical warning string when only some rows carry full text.
    None when coverage is complete or when there is nothing to warn about.
    """
    cov = get_fulltext_coverage(topic=topic)
    total = cov["total"]
    with_full = cov["with_ft_universe_rows"]
    if total == 0:
        return None
    if with_full == 0:
        return (f"⚠️ 0/{total} papers in this campaign have full text extracted. "
                f"The Full Text column will be empty. Use 'Download Enable in "
                f"Batch' or 'Backfill Missing Full Texts' first.")
    if with_full < total:
        return (f"⚠️ Only {with_full}/{total} papers have full text extracted. "
                f"Use 'Download Enable in Batch' or 'Backfill Missing Full Texts' "
                f"to process the remaining {total - with_full}.")
    return None


# -------------------------- USER INTERFACE HELPERS --------------------------


def show_logs(key_suffix: str):
    if st.session_state.log_buffer:
        with st.expander("📋 Processing Logs", expanded=False):
            st.text_area("Logs", "\n".join(st.session_state.log_buffer[-20:]),
                         height=150, key=f"log_display_{key_suffix}")


def create_dashboard():
    stats = db_manager.get_db_stats()
    st.subheader("📊 Database Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Papers", stats.get('total_papers', 0))
    with col2: st.metric("PDFs Stored", stats.get('pdfs_stored', 0))
    with col3: st.metric("Full Text Papers", stats.get('fulltext_stored', 0))
    with col4: st.metric("Total Size", f"{stats.get('total_pdf_size_mb', 0):.1f} MB")
    if stats.get('total_papers', 0) > 0:
        pdf_coverage = (stats.get('pdfs_stored', 0) / stats.get('total_papers', 1)) * 100
        st.progress(pdf_coverage / 100, text=f"PDF Coverage: {pdf_coverage:.1f}%")
    by_topic = stats.get('by_topic') or {}
    if by_topic:
        st.caption("Rows per campaign: " +
                   " · ".join(f"**{k}**={v}" for k, v in by_topic.items()))


# -------------------------- MAIN APPLICATION LAYOUT --------------------------


st.title("🔥 Activation Energy (Q) Harvester — Glacier Ice Rheology")
st.markdown("""
**Magnitude-and-unit-aware tool for searching, downloading, and analysing the
activation energy *Q* in the Arrhenius form of the Glen flow law — with
**unit resolution to J/mol**, **magnitude inference**, and **Q1/Q2 creep-
mechanism tagging** for the Elmer/Ice SIF parameterisation.**
Features:
- **Smart Search**: Query arXiv with relevance scoring
- **PDF Storage**: Store PDFs in SQLite databases with deduplication
- **Full-Text Extraction**: SQLite `universe_db` is the single source of truth for CSV export
- **Unit Resolver**: kJ/mol → J/mol conversion, eV → J/mol conversion, magnitude inference when units are missing
- **Creep-Mechanism Classifier**: tags hits as `Q1 (Dislocation)` or `Q2 (Grain Boundary)`
- **Dedicated Q1/Q2 Columns**: best-effort scalar extraction in J/mol for the SIF
- **Sibling Extractors**: rate factor A, Glen n, and enhancement factor E are harvested alongside
- **Scopus-Format CSV Export**: 22-column Scopus layout + Full Text + candidate snippets
- **Campaign Separation**: `topic` column keeps Q results distinct from siblings
- **Combined View**: optionally merge all campaigns into one NER-ready CSV
- **Coverage Diagnostics**: live partial-coverage warnings and one-click backfill
""")


if IS_CLOUD:
    st.warning("""
⚠️ **Running on Streamlit Cloud**:
- PDF downloads are manual (click individual buttons)
- Use 'Download Enable in Batch' or 'Backfill Missing Full Texts' before exporting CSV
- Data is stored temporarily (may be cleared between sessions)
    """)


show_logs("top")


# -------------------------- SIDEBAR CONFIGURATION --------------------------


with st.sidebar:
    st.header("🔍 Search Configuration")

    # Optimal activation-energy query. The RHS of the AND forces the paper
    # into glaciology proper — "activation energy" alone is a chemistry term
    # and would pull in catalysis, polymer degradation, and battery research.
    default_query = ACTIVATION_ENERGY_QUERY
    query = st.text_area("Search Query", value=default_query, height=160)

    default_cats = ACTIVATION_ENERGY_CATEGORIES
    categories = st.multiselect("Categories", default_cats, default=default_cats)

    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", 1970, current_year,
                                     ACTIVATION_ENERGY_START_YEAR)
    with col2:
        end_year = st.number_input("End Year", start_year, current_year,
                                   min(ACTIVATION_ENERGY_END_YEAR, current_year))

    max_results = st.slider("Maximum Results", 1, 500, ACTIVATION_ENERGY_MAX_RESULTS)
    relevance_threshold = st.slider("Relevance Threshold (%)", 0, 100, 30)

    st.subheader("💾 Storage Options")
    auto_download = st.checkbox("Auto-download PDFs", value=not IS_CLOUD, disabled=IS_CLOUD)

    st.subheader("📤 Export Options")
    export_formats = st.multiselect(
        "Select export formats",
        ["ZIP Archive", "CSV", "JSON", "Excel", "Database Backup"],
        default=["ZIP Archive", "CSV"]
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_btn = st.button("🔍 Search arXiv", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key in ("page_config_set", "page_config", "log_buffer"):
                    continue
                if key in DEFAULT_STATE:
                    st.session_state[key] = DEFAULT_STATE[key]
                else:
                    del st.session_state[key]
            st.rerun()

    st.subheader("🔎 Search Database")
    st.caption("Tip: FTS search runs over stored full texts once they exist. "
               'Try `"activation energy" AND glacier`, '
               '`"grain boundary" AND creep`, '
               '`"dislocation" AND "activation energy"`.')
    db_query = st.text_input(
        "Search stored papers",
        placeholder="e.g., activation energy dislocation creep glacier"
    )
    if st.button("Search in Database", use_container_width=True):
        if db_query:
            try:
                conn = sqlite3.connect(db_manager.universe_db)
                c = conn.cursor()
                c.execute("""SELECT paper_id, title,
                             snippet(papers_fts, 2, '<b>', '</b>', '...', 30)
                             FROM papers_fts WHERE papers_fts MATCH ? LIMIT 10""",
                          (db_query,))
                results = c.fetchall()
                conn.close()
            except sqlite3.OperationalError as e:
                results = []
                st.error(f"FTS query error: {e}. Try quoting terms or using AND/OR between them.")
            if results:
                st.success(f"Found {len(results)} papers")
                for paper_id, title, snippet in results:
                    with st.expander(f"{title[:80]}..."):
                        st.write(f"**ID:** {paper_id}")
                        st.write(f"**Snippet:** {snippet}")
            else:
                st.warning("No results found")


create_dashboard()


# -------------------------- SEARCH AND PROCESSING LOGIC --------------------------


if search_btn:
    if not query.strip():
        st.error("Please enter a search query")
        st.stop()
    if not categories:
        st.error("Please select at least one category")
        st.stop()

    st.session_state.processing = True
    start_time = time.time()
    load_scibert()  # Preload model

    with st.spinner("🔍 Searching arXiv..."):
        papers = query_arxiv(query, categories, max_results, start_year, end_year,
                             topic=CURRENT_TOPIC)

    if not papers:
        st.warning("No papers found matching your criteria")
        st.session_state.processing = False
        st.stop()

    query_emb = get_embedding(query)
    # Q embedding key terms — activation-energy / creep-mechanism vocab plus
    # glacier context. Pulls papers that actually report a Q1 or Q2 to the
    # top of the ranking.
    key_terms = ACTIVATION_ENERGY_TERMS | GLACIER_TERMS | {
        'activation energy', 'activation enthalpy', 'arrhenius',
        'q1', 'q2', 'dislocation creep', 'grain boundary sliding',
        'diffusion creep', 'rate controlling', 'paterson', 'budd',
        'cuffey', 'elmer/ice', 'temperature dependence',
        'cold ice', 'warm ice', 'temperate ice', 'limit temperature',
    }
    key_query = " ".join(key_terms)
    key_emb = get_embedding(key_query)
    st.session_state.query_emb = query_emb
    st.session_state.key_emb = key_emb

    with st.spinner("🧠 Computing SciBERT scores..."):
        for paper in papers:
            text = paper['title'] + " " + paper['abstract']
            paper_emb = get_embedding(text)
            sim1 = cosine_sim(query_emb, paper_emb)
            sim2 = cosine_sim(key_emb, paper_emb)
            paper['relevance_score'] = round((0.7 * sim1 + 0.3 * sim2) * 100, 2)
            db_manager.store_paper_metadata(paper)  # preserves prior download state
        papers.sort(key=lambda x: x['relevance_score'], reverse=True)

    relevant_papers = [p for p in papers if p['relevance_score'] >= relevance_threshold]
    if not relevant_papers:
        st.warning(f"No papers above {relevance_threshold}% relevance threshold")
        st.session_state.processing = False
        st.stop()

    st.success(f"Found **{len(relevant_papers)}** relevant papers "
               f"(topic: `{CURRENT_TOPIC}`)")

    # Attention heatmap
    st.subheader("🗺️ Query Attention Heatmap from SciBERT")
    tokenizer, model = load_scibert()
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    att = outputs.attentions[-1].mean(dim=1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(att, cmap='viridis')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    fig.colorbar(im)
    st.pyplot(fig)

    if auto_download and not IS_CLOUD:
        progress_bar = st.progress(0)
        status_text = st.empty()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(handle_paper_download, paper): i
                       for i, paper in enumerate(relevant_papers)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                paper = future.result()
                relevant_papers[idx] = paper
                completed += 1
                progress_bar.progress(completed / len(relevant_papers))
                status_text.text(f"Processed {completed}/{len(relevant_papers)} papers")
        progress_bar.empty()
        status_text.empty()

    st.session_state.relevant_papers = relevant_papers
    st.session_state.processing_time = time.time() - start_time
    update_log(f"Search completed in {st.session_state.processing_time:.1f} seconds "
               f"(topic={CURRENT_TOPIC})")


# -------------------------- MAINTENANCE & REPAIR --------------------------


st.divider()
st.subheader("🔧 Maintenance & Repair")
st.caption(f"Counters below are scoped to the current campaign "
           f"(topic = `{CURRENT_TOPIC}`).")

cov = get_fulltext_coverage(topic=CURRENT_TOPIC)
cov_cols = st.columns(6)
with cov_cols[0]: st.metric("Papers in Campaign", cov["total"])
with cov_cols[1]: st.metric("With PDF Stored", cov["with_pdf"])
with cov_cols[2]: st.metric("Full Text Flag", cov["with_ft_metadata_flag"])
with cov_cols[3]: st.metric("Full Text Rows", cov["with_ft_universe_rows"])
with cov_cols[4]: st.metric("Coverage", f"{cov['coverage_pct']:.1f}%")
with cov_cols[5]: st.metric("Total Words", f"{cov['total_words']:,}")

_warn = coverage_warning_message(topic=CURRENT_TOPIC)
if _warn:
    st.warning(_warn)
elif cov["total"] > 0:
    st.success("✅ Full text data is present for every row in this campaign — "
               "the Scopus CSV will carry it.")

bf_col1, bf_col2, bf_col3 = st.columns(3)
with bf_col1:
    if st.button("⬇️ Backfill Missing Full Texts", use_container_width=True,
                 key="btn_backfill"):
        n = backfill_missing_fulltexts(topic=CURRENT_TOPIC)
        st.success(f"Processed {n} paper(s) — re-export the Scopus CSV now.")
        st.rerun()
with bf_col2:
    if st.button("🔧 Repair BLOB Relevance Scores", use_container_width=True,
                 key="btn_repair"):
        fixed = repair_relevance_scores()
        st.success(f"Repaired {fixed} BLOB relevance score row(s).")
with bf_col3:
    if st.button("🔄 Refresh Coverage Numbers", use_container_width=True,
                 key="btn_refresh_cov"):
        st.rerun()

_missing_ids = get_missing_fulltext_ids(topic=CURRENT_TOPIC)
if _missing_ids:
    with st.expander(f"🔎 {len(_missing_ids)} paper(s) in this campaign missing full text",
                     expanded=False):
        st.caption("These rows have a pdf_url but no extracted full text "
                   "in the universe DB. Backfill or batch-download to populate.")
        for pid in _missing_ids[:50]:
            st.code(pid)
        if len(_missing_ids) > 50:
            st.caption(f"…and {len(_missing_ids) - 50} more.")


# -------------------------- RESULTS DISPLAY AND BATCH DOWNLOAD --------------------------


if st.session_state.get('relevant_papers'):
    papers = st.session_state.relevant_papers

    if "selected_papers" not in st.session_state:
        st.session_state.selected_papers = set()
    if "batch_download_index" not in st.session_state:
        st.session_state.batch_download_index = 0

    available_paper_ids = {p['id'] for p in papers
                           if p.get('pdf_stored') or p['id'] in st.session_state.downloaded_pdfs}

    st.subheader(f"📄 Search Results ({len(papers)} papers · topic `{CURRENT_TOPIC}`)")

    col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1, 1, 1.5, 1.5])
    with col_sel1:
        if st.button("✓ Select All (with PDFs)"):
            st.session_state.selected_papers = available_paper_ids.copy()
        st.session_state.selected_papers = st.session_state.selected_papers & available_paper_ids
    with col_sel2:
        if st.button("✗ Deselect All"):
            st.session_state.selected_papers.clear()
    with col_sel3:
        BATCH_SIZE = 50
        total = len(papers)
        start_idx = st.session_state.batch_download_index
        end_idx = min(start_idx + BATCH_SIZE, total)
        remaining = total - start_idx

        if remaining > 0:
            label = f"📥 Download Enable in Batch ({start_idx + 1}–{end_idx})"
            if st.button(label):
                with st.spinner("⬇️ Processing batch..."):
                    progress = st.progress(0)
                    status = st.empty()
                    for i in range(start_idx, end_idx):
                        paper = papers[i]
                        status.text(f"Downloading {i+1}/{end_idx}: {paper['title'][:50]}...")
                        progress.progress((i - start_idx + 1) / (end_idx - start_idx))
                        if not (paper.get('pdf_stored') or paper['id'] in st.session_state.downloaded_pdfs):
                            updated = handle_paper_download(paper, manual_download=True)
                            papers[i] = updated
                    st.session_state.relevant_papers = papers
                    st.session_state.batch_download_index = end_idx
                    status.success(f"✅ Batch complete! Processed up to {end_idx}.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.success("✅ All papers download-enabled!")
    with col_sel4:
        st.write(f"**Selected:** {len(st.session_state.selected_papers)} / {len(available_paper_ids)} available")

    if st.session_state.batch_download_index < len(papers):
        st.info(f"✅ Processed {st.session_state.batch_download_index} of {len(papers)} papers. "
                f"Click 'Download Enable in Batch' to continue.")
    else:
        st.success("🎉 All papers are download-enabled! Use 'Select All (with PDFs)' to choose all.")

    # Display each paper
    for i, paper in enumerate(papers):
        enhanced = paper.get('enhanced_relevance_score', 0)
        q_flag = "🟢" if paper.get('dopant_present') else "⚪"
        rate_flag = "🔵" if paper.get('beta_phase_present') else "⚪"
        can_select = paper['id'] in available_paper_ids
        is_selected = paper['id'] in st.session_state.selected_papers

        with st.expander(
            f"**{paper['title']}** ({paper['year']}) - "
            f"Basic: {paper['relevance_score']}% | "
            f"Enhanced: {enhanced:.1f}% "
            f"Q:{q_flag} Rate:{rate_flag}",
            expanded=i < 2
        ):
            col_check, col_info, col_actions = st.columns([0.5, 2.5, 1])

            with col_check:
                if can_select:
                    new_selected = st.checkbox(
                        "",
                        value=is_selected,
                        key=f"select_{paper['id']}_{i}",
                        label_visibility="collapsed"
                    )
                    if new_selected and paper['id'] not in st.session_state.selected_papers:
                        st.session_state.selected_papers.add(paper['id'])
                    elif not new_selected and paper['id'] in st.session_state.selected_papers:
                        st.session_state.selected_papers.discard(paper['id'])
                else:
                    st.empty()

            with col_info:
                st.write(f"**Authors:** {paper['authors']}")
                st.write(f"**Categories:** {paper['categories']}")
                st.write(f"**Matched Terms:** {paper['matched_terms']}")
                st.write(f"**Status:** {paper['download_status']}")
                show_abstract = st.toggle("Show Abstract",
                                          key=f"toggle_abstract_{paper['id']}_{i}")
                if show_abstract:
                    st.markdown(f"> {paper['abstract']}")

            with col_actions:
                if can_select:
                    if paper['id'] in st.session_state.downloaded_pdfs:
                        pdf_bytes = st.session_state.downloaded_pdfs[paper['id']]['pdf_bytes']
                    else:
                        pdf_bytes = db_manager.get_pdf(paper['id'])
                    if pdf_bytes:
                        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                        filename = f"{paper['id']}_{safe_title}.pdf".replace(' ', '_')
                        st.download_button(
                            label="📥 Download",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"dl_{paper['id']}_{i}",
                            use_container_width=True
                        )
                else:
                    if st.button("⬇️ Download Now", key=f"manual_{paper['id']}_{i}",
                                 use_container_width=True):
                        with st.spinner("Downloading..."):
                            updated_paper = handle_paper_download(paper, manual_download=True)
                            papers[i] = updated_paper
                            st.session_state.relevant_papers = papers
                            st.rerun()
                st.markdown(f"[🌐 arXiv Page]({paper['pdf_url'].replace('/pdf/', '/abs/')})")
                st.markdown(f"[📄 Direct PDF]({paper['pdf_url']})")

    # Bulk export section
    st.subheader("📤 Export & Bulk Download")
    export_cols = st.columns(5)
    paper_ids = [p['id'] for p in papers
                 if p.get('pdf_stored') or p['id'] in st.session_state.downloaded_pdfs]
    selected_ids = list(st.session_state.selected_papers)

    # ZIP
    if "ZIP Archive" in export_formats:
        with export_cols[0]:
            scope = st.radio("ZIP Scope", ["All Available", "Selected Only"],
                             key="zip_scope", horizontal=True)
            ids = selected_ids if (scope == "Selected Only" and selected_ids) else paper_ids
            if ids:
                if st.button("📦 Create ZIP", use_container_width=True):
                    with st.spinner(f"Creating ZIP with {len(ids)} PDFs..."):
                        buffer = db_manager.create_zip_from_db(ids)
                        st.session_state.zip_buffer = buffer
                        st.success(f"ZIP created with {len(ids)} PDFs")
                if st.session_state.zip_buffer:
                    st.download_button(
                        "⬇️ Download ZIP",
                        st.session_state.zip_buffer.getvalue(),
                        f"{CURRENT_TOPIC}_papers.zip",
                        "application/zip",
                        use_container_width=True
                    )
            else:
                st.caption("No PDFs to ZIP")

    # Bulk individual download
    with export_cols[1]:
        if selected_ids:
            st.write("**📥 Bulk Download Selected**")
            for pid in selected_ids:
                paper = next((p for p in papers if p['id'] == pid), None)
                if paper:
                    if pid in st.session_state.downloaded_pdfs:
                        data = st.session_state.downloaded_pdfs[pid]['pdf_bytes']
                    else:
                        data = db_manager.get_pdf(pid)
                    if data:
                        title = re.sub(r'[^\w\s-]', '', paper['title'])[:50]
                        fname = f"{pid}_{title}.pdf".replace(' ', '_')
                        st.download_button(
                            f"📄 {paper['title'][:30]}...",
                            data,
                            fname,
                            "application/pdf",
                            key=f"bulk_{pid}",
                            use_container_width=True
                        )
        else:
            st.caption("Select papers above")

    # ------------------------------------------------------------------
    # CSV — Scopus layout + Full Text, scoped to current campaign, with
    # an optional combined-view toggle across all campaigns.
    # ------------------------------------------------------------------
    if "CSV" in export_formats:
        with export_cols[2]:
            include_ft = st.checkbox("Include Full Text column", value=True,
                                     key="csv_fulltext")
            limit = st.number_input("Full-text char limit (0 = no limit)",
                                    0, 200000, 50000, key="csv_ft_limit")
            only_ft = st.checkbox("Only rows with full text", value=False,
                                  key="csv_only_ft",
                                  help="Skip rows whose papers_fulltext entry is "
                                       "empty. Recommended when coverage is partial.")
            require_flag = st.checkbox("Require fulltext_stored flag = 1",
                                       value=False, key="csv_require_flag",
                                       help="Belt-and-braces filter: export only rows "
                                            "whose metadata column fulltext_stored is 1.")
            combined = st.checkbox(
                "Combined view (all campaigns)",
                value=False, key="csv_combined",
                help="Ignore topic filter and export every row in the shared DB. "
                     "Useful for the eventual NER pass, since Q, rate-factor, "
                     "Glen-n, and enhancement-factor columns reinforce each other."
            )
            auto_extract = st.checkbox(
                "Auto-extract missing before export",
                value=False, key="csv_auto_extract",
                help="Downloads + extracts full text for every row in this campaign "
                     "that lacks it, then exports. SLOW on large sets — 1 HTTP call "
                     "per missing paper. Not recommended for >100 rows."
            )

            export_topic = None if combined else CURRENT_TOPIC
            cov_now = get_fulltext_coverage(topic=export_topic)
            total_rows = cov_now["total"]
            with_full = cov_now["with_ft_universe_rows"]
            scope_label = "all campaigns" if combined else f"topic `{CURRENT_TOPIC}`"
            if total_rows > 0:
                st.caption(f"📊 {scope_label} — full text available for "
                           f"**{with_full}/{total_rows}** row(s) "
                           f"({cov_now['coverage_pct']:.1f}%).")
            else:
                st.caption(f"No rows yet for {scope_label} — run a search first.")

            if auto_extract and with_full < total_rows:
                st.warning(
                    f"⚠️ {total_rows - with_full} row(s) in scope still lack "
                    f"full text. Click below to extract them before exporting."
                )
                if st.button("⬇️ Extract missing now",
                             key="btn_extract_now",
                             use_container_width=True):
                    n = backfill_missing_fulltexts(topic=export_topic)
                    st.success(f"Extracted {n} new paper(s). Re-export the Scopus CSV.")
                    st.rerun()

            buf = export_scopus_format(
                include_fulltext=include_ft,
                fulltext_char_limit=(limit or None),
                only_with_fulltext=only_ft,
                require_metadata_flag=require_flag,
                topic=export_topic,     # ← None = combined view
            )

            if buf.getbuffer().nbytes > 0:
                fname = ("ice_combined_metadatabase.csv" if combined
                         else f"{CURRENT_TOPIC}_metadatabase.csv")
                st.download_button(
                    "📊 Scopus-format CSV", buf.getvalue(),
                    fname, "text/csv",
                    use_container_width=True)

                if include_ft and not only_ft and total_rows > 0 and with_full < total_rows:
                    st.warning(
                        f"⚠️ Only {with_full}/{total_rows} papers have full text "
                        f"extracted. The Full Text column will be blank for the "
                        f"remaining {total_rows - with_full} row(s). Use 'Download "
                        f"Enable in Batch' or 'Backfill Missing Full Texts'."
                    )
                elif include_ft and only_ft and with_full < total_rows:
                    st.info(
                        f"ℹ️ 'Only rows with full text' is enabled — the CSV will "
                        f"contain {with_full} of {total_rows} rows."
                    )
            else:
                st.caption("No rows matched the export filters yet.")

    # JSON — scoped to current campaign
    if "JSON" in export_formats:
        with export_cols[3]:
            buf = db_manager.export_metadata("json", topic=CURRENT_TOPIC)
            if buf.getbuffer().nbytes > 0:
                st.download_button("📄 JSON Export", buf.getvalue(),
                                   f"{CURRENT_TOPIC}_metadata.json", "application/json",
                                   use_container_width=True)

    # Excel — scoped to current campaign
    if "Excel" in export_formats:
        with export_cols[4]:
            buf = db_manager.export_metadata("excel", topic=CURRENT_TOPIC)
            if buf.getbuffer().nbytes > 0:
                st.download_button(
                    "📈 Excel Export", buf.getvalue(), f"{CURRENT_TOPIC}_metadata.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)

    # Raw database files
    with st.expander("🗃️ Databases"):
        for label, path in [
            ("Metadata DB", db_manager.metadata_db),
            ("Fulltext DB", db_manager.universe_db),
            ("PDF Storage DB", db_manager.pdf_db)
        ]:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    st.download_button(
                        label=label,
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime="application/octet-stream",
                        use_container_width=True
                    )


# -------------------------- DATABASE MANAGEMENT SECTION --------------------------


st.divider()
st.subheader("🗄️ Database Management")

col_stats, col_clean = st.columns(2)
with col_stats:
    if st.button("🔄 Refresh Statistics", use_container_width=True):
        st.session_state.db_stats = db_manager.get_db_stats()
        st.rerun()
with col_clean:
    if st.button("🧹 Clean Temporary Files", use_container_width=True):
        temp_dir = os.path.join(DB_DIR, "temp")
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except OSError:
                    pass
        st.session_state.temp_files = []
        st.success("Temporary files cleaned")

if st.session_state.db_stats:
    with st.expander("📊 Detailed Statistics"):
        st.json(st.session_state.db_stats)


# -------------------------- FOOTER --------------------------


st.divider()
st.caption(f"""
**Activation Energy (Q) Harvester — Glacier Ice Rheology** |
Campaign: `{CURRENT_TOPIC}` |
Running on {'☁️ Streamlit Cloud' if IS_CLOUD else '💻 Local'} |
Data Directory: `{DB_DIR}` |
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

show_logs("bottom")


# -------------------------- CLEANUP HOOK --------------------------


import atexit


def cleanup():
    temp_dir = os.path.join(DB_DIR, "temp")
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except OSError:
                pass


atexit.register(cleanup)


# -------------------------- END OF FILE --------------------------
# Activation-energy harvester — arXiv query targets Q in glacier / ice-sheet
# rheology:
#   abs:("activation energy" OR "activation enthalpy" OR "energy barrier"
#        OR "creep activation" OR "Arrhenius" OR "dislocation creep"
#        OR "grain boundary sliding" OR "diffusion creep")
#   AND abs:(glacier OR "ice sheet" OR "ice stream" OR "ice shelf"
#            OR "polar ice" OR "ice dynamics" OR "polycrystalline ice")
# restricted to physics.geo-ph / cond-mat.mtrl-sci / physics.comp-ph, which
# is the firewall that keeps catalysis and battery-chemistry papers out of
# the harvest. Each row is labelled topic='activation_energy_ice'; sibling
# campaigns ('rate_factor_ice', 'glen_exponent_ice', 'enhancement_factor_ice')
# sit in the same shared DB and can be exported separately or merged via the
# combined-view toggle.
#
# What makes THIS campaign different from the rate-factor campaign:
#   - The extractor is MAGNITUDE-AND-UNIT AWARE. It captures the numeric
#     value, resolves the unit to J/mol (kJ/mol × 1000, eV × 96485, or
#     inferred from magnitude when the unit is missing), and emits an
#     emoji-tagged warning: ✅ KJ_TO_J_CONVERTED, ✅ J_MOL, ⚠️ EV_TO_J_CONVERTED,
#     ⚠️ ASSUMED_KJ/mol_MULTIPLIED_BY_1000, ⚠️ ASSUMED_J/mol,
#     ❌ UNKNOWN_UNIT_MAGNITUDE.
#   - Every hit carries a CREEP-MECHANISM tag ("Q1 / Dislocation" vs
#     "Q2 / Grain Boundary"), which corresponds exactly to the two SIF
#     branches split at the Limit Temperature.
#   - Two dedicated CSV columns — "Candidate Activation Energy Q1 (J/mol)"
#     and "… Q2 (J/mol)" — carry best-effort single-scalar J/mol values
#     ready to paste into the Elmer/Ice SIF.
#
# Honest expectations:
#   - The canonical values of Q (Paterson & Budd 1982, Cuffey & Paterson
#     2010) live in J. Glaciol. / Ann. Glaciol. / The Physics of Glaciers —
#     not arXiv. What arXiv DOES have is modelling papers (Elmer/Ice, PISM,
#     ISSM, Úa, STREAMICE) and MD / dislocation-dynamics papers where Q is
#     an explicit parameter with units, which the extractor above is
#     designed to catch with the magnitude-and-unit safety net.
#   - The four quantities are physically one story:
#         ε̇ = A₀ · exp(-Q / (R·T)) · E · τⁿ
#     with n the Glen exponent (sibling campaign), A₀ exp(-Q / RT) the
#     Arrhenius rate factor A (sibling campaign), E an enhancement factor
#     for anisotropic fabric (sibling campaign), and Q the activation
#     energy (this campaign). All four candidate columns coexist in each
#     export row; the combined-view toggle lets them reinforce each other
#     in a single NER-ready CSV.
#   - Elmer/Ice SIF files take Q via the two keywords
#         Activation Energy 1 = Real 60e3   ; cold branch, T < Limit Temperature
#         Activation Energy 2 = Real 139e3  ; warm branch, T > Limit Temperature
#         Limit Temperature   = Real -10.0
#     A direct follow-up is to parse the SIF's three keywords and compare
#     them against what modellers actually publish — with the magnitude-
#     and-unit-aware extractor already flagging any paper whose units need
#     conversion (kJ/mol → J/mol) before entering the SIF.
# --------------------------------------------------------------
