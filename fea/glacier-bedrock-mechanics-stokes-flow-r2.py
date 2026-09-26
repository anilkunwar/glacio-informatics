#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Elmer FEM .sif Generator for Glacier Ice Flow (Stokes) with Bedrock Mechanics
================================================================================

OVERVIEW:
This application generates complete, ready-to-run Elmer FEM input files (.sif)
and DEM lookup tables (.dat) for 2D/3D glacier Stokes-flow simulations with
bedrock and surface DEM mapping.

KEY FEATURES:
1. FULL GLACIER STOKES FLOW:
   - StructuredMeshMapper for DEM → mesh mapping
   - StructuredProjectToPlane for flow-depth/height postprocessing
   - Navier-Stokes solver configured for Stokes (creeping) flow

2. GLEN'S FLOW LAW MATERIAL:
   - Rate factors (Paterson values), activation energies
   - Glen exponent, enhancement factor, critical shear rate
   - Arrhenius temperature-dependent viscosity

3. DEM DATA MANAGEMENT:
   - Editable bedrock and surface elevation tables
   - Upload custom .dat files or use built-in defaults
   - Auto-included in generated .sif via "include" directives

4. BOUNDARY CONDITIONS:
   - Bedrock: no-slip + DEM, optional Weertman sliding law
   - Lateral sides: no normal velocity
   - Surface: stress-free + DEM
   - Additional user-defined BCs

5. ROBUST ARCHITECTURE:
   - Session state persistence across reruns
   - One-click ZIP bundling
   - Defensive multiselects

AUTHOR: Glacier Mechanics Research
DATE: 2024
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import zipfile
import io
import math
import os

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Elmer Glacier Stokes Generator",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stCodeBlock { background-color: #f8f9fa; border-radius: 0.5rem; padding: 1rem; }
    .stTextArea textarea { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85em; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #007bff; }
    .download-section { background: #e8f4fd; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #b6d4fe; }
    .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0.25rem; }
    .success-box { background: #d4edda; border-left: 4px solid #28a745; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0.25rem; }
    .info-box { background: #d1ecf1; border-left: 4px solid #17a2c8; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0.25rem; }
    h1, h2, h3 { color: #2c3e50; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #007bff; }
</style>
""", unsafe_allow_html=True)

st.title("🏔️ Elmer FEM Generator for Glacier Stokes Flow with Bedrock Mechanics")
st.markdown("""
**Generate complete `.sif` input files + DEM `.dat` tables for glacier ice-flow simulations.**  
_Full Stokes solver with Glen's flow law, Arrhenius viscosity, DEM-mapped bedrock/surface boundaries._
""")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def uk(section: str, var: str, suffix: str = "") -> str:
    return f"{section}_{var}_{suffix}".strip("_")

def safe_multiselect(label, options, default=None, key=None, **kwargs):
    if default is None:
        default = []
    valid_defaults = [d for d in default if d in options]
    if len(valid_defaults) < len(default):
        invalid = set(default) - set(options)
        st.warning(f"⚠️ Removed invalid defaults for '{label}': {invalid}")
    return st.multiselect(label, options, default=valid_defaults, key=key, **kwargs)

# ==============================================================================
# DEFAULT DEM DATA (from provided files)
# ==============================================================================
DEFAULT_BEDROCK_DATA = [
    (0, 0), (45.9, 3.672), (89.88, 7.1904), (132.82, 10.646),
    (174.95, 14.232), (216.4, 18.087), (257.22, 22.264),
    (297.46, 26.791), (337.15, 31.69), (376.31, 36.975),
    (414.96, 42.66), (453.11, 48.762), (490.76, 55.295),
    (527.92, 62.276), (564.58, 69.724), (600.73, 77.657),
    (636.38, 86.098), (671.53, 95.025), (706.18, 104.35),
    (740.37, 113.99), (774.11, 123.87), (807.42, 133.91),
    (840.33, 144.06), (872.85, 154.25), (905, 164.42),
    (936.8, 174.51), (968.27, 184.48), (999.43, 194.24),
    (1030.3, 203.76), (1060.9, 212.95), (1091.2, 221.75),
    (1121.2, 230.07), (1151.1, 237.83), (1180.6, 244.91),
    (1210, 251.18), (1239.2, 256.55), (1268, 261.05),
    (1296.6, 264.72), (1324.8, 267.69), (1352.7, 270.08),
    (1380.2, 272.03), (1407.3, 273.68), (1434, 275.18),
    (1460.3, 276.64), (1486.2, 278.17), (1511.8, 279.84),
    (1537, 281.71), (1561.9, 283.82), (1586.4, 286.19),
    (1610.6, 288.83), (1634.4, 291.74), (1658, 294.92),
    (1681.3, 298.36), (1704.2, 302.05), (1726.9, 305.95),
    (1749.3, 310.03), (1771.5, 314.25), (1793.4, 318.6),
    (1815.1, 323.04), (1836.6, 327.57), (1857.8, 332.18),
    (1878.9, 336.84), (1899.7, 341.55), (1920.3, 346.3),
    (1940.7, 351.09), (1961, 355.91), (1981, 360.74),
    (2000.8, 365.6), (2020.5, 370.47), (2039.9, 375.34),
    (2059.2, 380.22), (2078.3, 385.1), (2097.2, 389.97),
    (2115.9, 394.84), (2134.5, 399.7), (2152.8, 404.55),
    (2171, 409.38), (2189, 414.19), (2206.8, 418.99),
    (2224.4, 423.76), (2241.9, 428.5), (2259.1, 433.22),
    (2276.2, 437.9), (2293, 442.54), (2309.7, 447.14),
    (2326.1, 451.7), (2342.3, 456.21), (2358.4, 460.66),
    (2374.1, 465.04), (2389.7, 469.35), (2404.9, 473.59),
    (2419.9, 477.74), (2434.5, 481.81), (2448.8, 485.78),
    (2462.7, 489.63), (2476, 493.34), (2488.8, 496.88),
    (2500, 500)
]

DEFAULT_SURFACE_DATA = [
    (0, 4.8975), (45.9, 8.5695), (89.88, 12.088), (132.82, 15.543),
    (174.95, 19.13), (216.4, 22.984), (257.22, 27.161),
    (297.46, 31.689), (337.15, 36.588), (376.31, 41.872),
    (414.96, 47.558), (453.11, 53.66), (490.76, 60.193),
    (527.92, 67.174), (564.58, 74.622), (600.73, 82.555),
    (636.38, 90.996), (671.53, 99.923), (706.18, 109.25),
    (740.37, 118.89), (774.11, 128.76), (807.42, 138.81),
    (840.33, 148.96), (872.85, 159.15), (905, 169.32),
    (936.8, 179.41), (968.27, 189.37), (999.43, 199.14),
    (1030.3, 208.66), (1060.9, 217.85), (1091.2, 226.65),
    (1121.2, 251.7), (1151.1, 268.04), (1180.6, 279.24),
    (1210, 290.86), (1239.2, 299.67), (1268, 308.65),
    (1296.6, 315.68), (1324.8, 322.74), (1352.7, 328.35),
    (1380.2, 333.96), (1407.3, 338.51), (1434, 343.1),
    (1460.3, 346.97), (1486.2, 350.91), (1511.8, 354.36),
    (1537, 357.93), (1561.9, 361.18), (1586.4, 364.57),
    (1610.6, 367.75), (1634.4, 371.07), (1658, 374.28),
    (1681.3, 377.65), (1704.2, 380.96), (1726.9, 384.43),
    (1749.3, 387.89), (1771.5, 391.49), (1793.4, 395.11),
    (1815.1, 398.87), (1836.6, 402.65), (1857.8, 406.56),
    (1878.9, 410.5), (1899.7, 414.52), (1920.3, 418.55),
    (1940.7, 422.64), (1961, 426.71), (1981, 430.81),
    (2000.8, 434.88), (2020.5, 438.96), (2039.9, 443.02),
    (2059.2, 447.07), (2078.3, 451.1), (2097.2, 455.14),
    (2115.9, 459.16), (2134.5, 463.18), (2152.8, 467.19),
    (2171, 471.2), (2189, 475.18), (2206.8, 479.16),
    (2224.4, 483.11), (2241.9, 487.05), (2259.1, 490.95),
    (2276.2, 494.83), (2293, 498.66), (2309.7, 502.45),
    (2326.1, 506.17), (2342.3, 509.84), (2358.4, 513.42),
    (2374.1, 516.92), (2389.7, 520.31), (2404.9, 523.6),
    (2419.9, 526.78), (2434.5, 529.8), (2448.8, 532.61),
    (2462.7, 535.03), (2476, 536.88), (2488.8, 537.94),
    (2500, 538.45)
]

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "generated_content" not in st.session_state:
    st.session_state.generated_content = {}
if "generation_timestamp" not in st.session_state:
    st.session_state.generation_timestamp = None
if "table_data_bedrock" not in st.session_state:
    st.session_state.table_data_bedrock = None
if "table_data_surface" not in st.session_state:
    st.session_state.table_data_surface = None

# ==============================================================================
# SIDEBAR: GLOBAL SETTINGS
# ==============================================================================
st.sidebar.header("⚙️ Global Settings")

project_name = st.sidebar.text_input(
    "Project Name", value="Glacier_Stokes", key=uk("global", "project"),
    help="Base name for generated files."
)

author = st.sidebar.text_input(
    "Author", value="Researcher", key=uk("global", "author")
)

date_str = datetime.now().strftime("%Y-%m-%d")

st.sidebar.subheader("📁 Output Settings")
sif_filename = st.sidebar.text_input(
    ".sif Filename", value=f"{project_name.lower()}.sif", key=uk("out", "sif")
)
mesh_db_dir = st.sidebar.text_input(
    "Mesh DB Directory", value=".", key=uk("out", "meshdbdir"),
    help='Directory containing mesh partition files (e.g. ".")'
)
mesh_db_name = st.sidebar.text_input(
    "Mesh DB Name", value="testglacier", key=uk("out", "meshdbname"),
    help='Mesh directory name (referenced in Header → Mesh DB)'
)
results_dir = st.sidebar.text_input(
    "Results Directory", value="", key=uk("out", "resdir"),
    help='Results directory (empty string = current dir in SIF)'
)

# Sidebar quick access
if st.session_state.generated_content:
    st.sidebar.success("✅ Files generated and cached!")
else:
    st.sidebar.info("🔄 Configure parameters and click 'Generate All Files'")

# ==============================================================================
# TABS
# ==============================================================================
tab_sim, tab_material, tab_solvers, tab_dem, tab_bc, tab_generate = st.tabs([
    "🖥️ Simulation",
    "🧊 Ice Material (Glen's Law)",
    "🔧 Solvers Configuration",
    "🗺️ DEM Data (Bedrock & Surface)",
    "🔗 Boundary Conditions",
    "📥 Generate & Download"
])

# ==============================================================================
# TAB 1: SIMULATION SETTINGS
# ==============================================================================
with tab_sim:
    st.header("🖥️ Simulation Configuration")
    st.markdown('<div class="info-box">Configure the overall simulation parameters, coordinate system, and output settings for the glacier Stokes-flow problem.</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("Coordinate & Type")
        coord_system = st.selectbox(
            "Coordinate System",
            ["Cartesian 2D", "Cartesian 3D", "Axi Symmetric", "Cylindrical"],
            index=0, key=uk("sim", "coordsys"),
            help="Glacier simulations are typically Cartesian 2D (flowline) or 3D."
        )

        n_coords = 3 if "3D" in coord_system else 2
        coord_mapping_default = "1 2 3" if n_coords == 3 else "1 2 3"
        coord_mapping = st.text_input(
            "Coordinate Mapping", value=coord_mapping_default, key=uk("sim", "coordmap"),
            help="Mapping of Elmer coordinates to physical coordinates."
        )

        sim_type = st.selectbox(
            "Simulation Type",
            ["Steady", "Transient"],
            index=0, key=uk("sim", "simtype"),
            help="Steady = diagnostic; Transient = prognostic evolution."
        )

        if sim_type == "Steady":
            steady_max_iter = st.number_input(
                "Steady State Max Iterations", value=1, min_value=1, key=uk("sim", "steadymaxiter"),
                help="Number of steady-state iterations (1 for purely diagnostic Stokes)."
            )
        else:
            steady_max_iter = 1
            dt = st.number_input("Time Step Δt [yr]", value=1.0, step=0.1, format="%.2f", key=uk("sim", "dt"))
            t_end = st.number_input("End Time [yr]", value=100.0, step=1.0, format="%.1f", key=uk("sim", "tend"))
            n_steps = st.number_input("Number of Steps", value=100, min_value=1, key=uk("sim", "nsteps"))

        max_output_level = st.slider(
            "Max Output Level", min_value=1, max_value=10, value=4, key=uk("sim", "maxout"),
            help="Controls verbosity of Elmer output. Higher = more debug info."
        )

    with col_s2:
        st.subheader("Output Files")
        output_intervals = st.number_input(
            "Output Intervals", value=1, min_value=1, key=uk("sim", "outint"),
            help="Write results every N iterations/steps."
        )

        ela_value = st.number_input(
            "ELA [m]", value=400.0, step=10.0, format="%.0f", key=uk("sim", "ela"),
            help="Equilibrium Line Altitude — used in naming convention."
        )

        diag_or_progn = st.selectbox(
            "Run Mode Label", ["diagnostic", "prognostic"], index=0, key=uk("sim", "runmode"),
            help="Label used in output filename for clarity."
        )

        output_file = st.text_input(
            "Output File",
            value=f"Stokes_ELA{int(ela_value)}_{diag_or_progn}.result",
            key=uk("sim", "outfile")
        )
        post_file = st.text_input(
            "Post File (.vtu)",
            value=f"Stokes_ELA_{int(ela_value)}_{diag_or_progn}.vtu",
            key=uk("sim", "postfile"),
            help="VTK unstructured grid file for ParaView visualization."
        )

        init_dirichlet = st.checkbox(
            "Initialize Dirichlet Conditions", value=False, key=uk("sim", "initdirichlet"),
            help="Usually False for glacier problems with DEM-mapped boundaries."
        )

    st.divider()
    st.subheader("📋 Simulation Summary")
    summary_data = {
        "Parameter": ["Coordinate System", "Simulation Type", "ELA", "Output File", "Post File"],
        "Value": [coord_system, sim_type, f"{ela_value} m", output_file, post_file]
    }
    st.table(pd.DataFrame(summary_data))

# ==============================================================================
# TAB 2: ICE MATERIAL (GLEN'S FLOW LAW)
# ==============================================================================
with tab_material:
    st.header("🧊 Ice Material Properties — Glen's Flow Law")
    st.markdown('<div class="info-box"><strong>Glen\'s flow law:</strong> ε̇ = Aτⁿ⁻¹τ, where A is the rate factor (Arrhenius), n is the Glen exponent. The viscosity is η = 1/(2A) τ<sup>1−n</sup>. We use the m-yr-MPa unit system: 1 yr = 31556926.0 s.</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Density & Units")
        st.markdown("**Unit System**: m-yr-MPa (standard for glaciology)")

        ice_density_si = st.number_input(
            "Ice Density ρ [kg/m³]", value=910.0, step=1.0, key=uk("mat", "density_si"),
            help="Standard ice density in SI units. Converted to m-yr-MPa internally."
        )

        sec_per_yr = 31556926.0
        density_mpa = ice_density_si * 1.0e-6 * sec_per_yr**(-2.0)
        st.info(f"**Density in m-yr-MPa**: {density_mpa:.6e}")

        gravity = st.number_input(
            "Gravitational Acceleration g [m/s²]", value=9.81, step=0.01, format="%.2f",
            key=uk("mat", "gravity"),
            help="Standard gravitational acceleration."
        )
        gravity_mpa = -gravity * sec_per_yr**2.0
        st.info(f"**Gravity BodyForce (y) in m-yr-MPa**: {gravity_mpa:.6e}")

    with col_m2:
        st.subheader("Glen's Flow Law Parameters")
        glen_n = st.number_input(
            "Glen Exponent n", value=3.0, step=0.5, min_value=1.0,
            key=uk("mat", "glenn"),
            help="Typically n=3 for ice. Higher values → more nonlinear."
        )
        enhancement_factor = st.number_input(
            "Glen Enhancement Factor", value=1.0, step=0.1, min_value=0.1,
            key=uk("mat", "enhancement"),
            help="EISMINT enhancement factor. Typically 1.0–8.0."
        )
        crit_shear = st.number_input(
            "Critical Shear Rate [1/yr]", value=1.0e-10, format="%.1e",
            key=uk("mat", "critshear"),
            help="Regularization parameter to avoid infinite viscosity at zero stress."
        )

    st.divider()
    st.subheader("Arrhenius Rate Factors (Paterson–Hooke)")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Rate Factors** (in MPa⁻³·a⁻¹)")
        rate_factor_1 = st.number_input(
            "Rate Factor A₁ (T < T_lim)", value=1.258e13, format="%.3e",
            key=uk("mat", "rf1"),
            help="Paterson rate factor for temperate/cold transition (T ≥ -10°C)."
        )
        rate_factor_2 = st.number_input(
            "Rate Factor A₂ (T ≥ T_lim)", value=6.046e28, format="%.3e",
            key=uk("mat", "rf2"),
            help="Paterson rate factor for cold ice (T < -10°C)."
        )

    with col_a2:
        st.markdown("**Activation Energies** (in J/mol = SI)")
        activation_energy_1 = st.number_input(
            "Activation Energy Q₁ [J/mol]", value=60e3, format="%.0e",
            key=uk("mat", "ae1"),
            help="Activation energy for T ≥ -10°C regime."
        )
        activation_energy_2 = st.number_input(
            "Activation Energy Q₂ [J/mol]", value=139e3, format="%.0e",
            key=uk("mat", "ae2"),
            help="Activation energy for T < -10°C regime."
        )

    st.divider()
    st.subheader("Temperature Configuration")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        limit_temp = st.number_input(
            "Limit Temperature [°C]", value=-10.0, step=0.5,
            key=uk("mat", "limtemp"),
            help="Temperature to switch between the two Arrhenius regimes."
        )
    with col_t2:
        use_temp_var = st.checkbox(
            "Use temperature field variable?", value=False, key=uk("mat", "usetempvar"),
            help="If True, uses 'Temp Homologous' from TemperateIceSolver. If False, uses constant temperature below."
        )
        if not use_temp_var:
            const_temp = st.number_input(
                "Constant Temperature [°C]", value=-3.0, step=0.5,
                key=uk("mat", "consttemp"),
                help="Constant temperature used when no temperature variable is solved."
            )
        else:
            const_temp = -3.0  # unused but defined

    st.divider()
    st.markdown("### 📋 Material Summary (as it will appear in .sif)")
    mat_summary = f"""Material 1
  Name = "ice"
  Density = Real {density_mpa:.10e}
  Viscosity Model = String "Glen"
  Viscosity = Real 1.0
  Glen Exponent = Real {glen_n}
  Critical Shear Rate = Real {crit_shear}
  Rate Factor 1 = Real {rate_factor_1:.3e}
  Rate Factor 2 = Real {rate_factor_2:.3e}
  Activation Energy 1 = Real {activation_energy_1}
  Activation Energy 2 = Real {activation_energy_2}
  Glen Enhancement Factor = Real {enhancement_factor}
  Limit Temperature = Real {limit_temp}
  Constant Temperature = Real {const_temp}
End"""
    st.code(mat_summary, language="fortran")

# ==============================================================================
# TAB 3: SOLVERS CONFIGURATION
# ==============================================================================
with tab_solvers:
    st.header("🔧 Solver Configuration")
    st.markdown("Configure the three solvers: **StructuredMeshMapper**, **StructuredProjectToPlane**, and **Navier-Stokes (Stokes flow)**.")

    # ---- Solver 1: StructuredMeshMapper ----
    with st.expander("📐 Solver 1: StructuredMeshMapper (DEM → Mesh)", expanded=True):
        st.markdown("""
        Maps DEM elevation data onto the originally rectangular mesh. 
        This runs **before** the simulation to deform the mesh into the glacier geometry.
        """)
        col_s1a, col_s1b = st.columns(2)
        with col_s1a:
            s1_active_coord = st.selectbox(
                "Active Coordinate", [1, 2, 3], index=1,
                key=uk("s1", "activecoord"),
                help="Coordinate direction of mesh update (2=y for 2D flowline)."
            )
            s1_exec = st.selectbox(
                "Exec Solver", ["before Simulation", "always", "after Simulation", "never"],
                index=0, key=uk("s1", "exec"),
                help="'before Simulation' runs mesh mapping once at start."
            )
        with col_s1b:
            s1_mesh_vel_first_zero = st.checkbox(
                "Mesh Velocity First Zero", value=True,
                key=uk("s1", "meshvelfirstzero"),
                help="Set first timestep mesh velocity to zero (avoids unrealistic initial velocity)."
            )
            s1_dot_tol = st.number_input(
                "Dot Product Tolerance", value=0.01, format="%.4f",
                key=uk("s1", "dottol"),
                help="Accuracy for vector projections during mesh mapping."
            )

    # ---- Solver 2: StructuredProjectToPlane ----
    with st.expander("📏 Solver 2: StructuredProjectToPlane (Depth/Height)", expanded=True):
        st.markdown("""
        Postprocessing solver that computes **flow depth** (distance from bedrock) 
        and **height** (distance from surface) by projecting onto the horizontal plane.
        """)
        col_s2a, col_s2b = st.columns(2)
        with col_s2a:
            s2_active_coord = st.selectbox(
                "Active Coordinate", [1, 2, 3], index=1,
                key=uk("s2", "activecoord"),
                help="Projection direction (2=y for vertical projection in 2D)."
            )
        with col_s2b:
            s2_compute_depth = st.checkbox("Compute Depth", value=True, key=uk("s2", "depth"))
            s2_compute_height = st.checkbox("Compute Height", value=True, key=uk("s2", "height"))

    # ---- Solver 3: Navier-Stokes (Stokes) ----
    with st.expander("🌊 Solver 3: Navier-Stokes — Stokes Flow Solver", expanded=True):
        st.markdown("""
        The central solver: full Stokes equations for glacier flow. 
        Configured for nonlinear Glen's law viscosity with Newton or Picard iteration.
        """)
        st.subheader("Linear System")
        col_ls1, col_ls2 = st.columns(2)
        with col_ls1:
            ls_solver = st.selectbox(
                "Linear System Solver", ["Direct", "Iterative"],
                index=0, key=uk("s3", "lssolver"),
                help="Direct (UMFPACK) for small problems; Iterative for large 3D."
            )
            if ls_solver == "Direct":
                ls_direct_method = st.selectbox(
                    "Direct Method", ["UMFPACK", "MUMPS", "Pardiso"],
                    index=0, key=uk("s3", "lsdirect")
                )
            else:
                ls_iter_method = st.selectbox(
                    "Iterative Method", ["GCR", "BiCGStab", "BiCGStabL", "CG", "GMRES"],
                    index=0, key=uk("s3", "lsiter"),
                    help="GCR and BiCGStab are common for Stokes."
                )
            ls_max_iter = st.number_input(
                "Max Iterations", value=5000, min_value=10,
                key=uk("s3", "lsmaxiter")
            )
            ls_conv_tol = st.number_input(
                "Convergence Tolerance", value=1.0e-6, format="%.1e",
                key=uk("s3", "lsconvtol")
            )
        with col_ls2:
            ls_abort = st.checkbox(
                "Abort Not Converged", value=False, key=uk("s3", "lsabort"),
                help="If False, continues even if linear solver doesn't converge."
            )
            if ls_solver == "Iterative":
                ls_precond = st.selectbox(
                    "Preconditioning", ["ILU0", "ILU1", "ILU2", "None"],
                    index=1, key=uk("s3", "lsprecond")
                )
            else:
                ls_precond = "ILU1"  # unused for direct
            ls_residual_output = st.number_input(
                "Residual Output Interval", value=1, min_value=0,
                key=uk("s3", "lsresout"),
                help="Print residual every N iterations (0=off)."
            )
            optimize_bw = st.checkbox(
                "Optimize Bandwidth", value=True, key=uk("s3", "optbw"),
                help="Renumbers nodes to reduce matrix bandwidth."
            )

        st.subheader("Stabilization")
        stab_method = st.selectbox(
            "Stabilization Method", ["Stabilized", "P2/P1", "Bubbles"],
            index=0, key=uk("s3", "stabmethod"),
            help="Stabilized = PSPG/SUPG. P2/P1 = mixed finite element. Bubbles = mini-element."
        )
        flow_model = st.selectbox(
            "Flow Model", ["Stokes", "Full"],
            index=0, key=uk("s3", "flowmodel"),
            help="Stokes = creeping flow (no inertia). Full = Navier-Stokes with inertia."
        )

        st.subheader("Nonlinear System (Picard → Newton)")
        col_nl1, col_nl2 = st.columns(2)
        with col_nl1:
            nl_conv_tol = st.number_input(
                "Nonlinear Convergence Tolerance", value=1.0e-4, format="%.1e",
                key=uk("s3", "nlconvtol"),
                help="Relative change in solution for convergence."
            )
            nl_max_iter = st.number_input(
                "Nonlinear Max Iterations", value=50, min_value=1,
                key=uk("s3", "nlmaxiter"),
                help="Maximum Picard/Newton iterations."
            )
            nl_conv_measure = st.selectbox(
                "Convergence Measure", ["Solution", "Residual"],
                index=0, key=uk("s3", "nlconvmeasure")
            )
        with col_nl2:
            newton_after_iter = st.number_input(
                "Newton After Iterations", value=3, min_value=0,
                key=uk("s3", "newtonafter"),
                help="Switch from Picard to Newton after this many iterations."
            )
            newton_after_tol = st.number_input(
                "Newton After Tolerance", value=1.0e-1, format="%.1e",
                key=uk("s3", "newtonaftertol"),
                help="Switch to Newton when residual drops below this."
            )
            nl_relax = st.number_input(
                "Relaxation Factor (0=off)", value=0.0, min_value=0.0, max_value=1.0,
                format="%.2f", key=uk("s3", "nlrelax"),
                help="Under-relaxation for Picard iteration. 0.0 = no relaxation."
            )

        st.subheader("Steady State")
        ss_conv_tol = st.number_input(
            "Steady State Convergence Tolerance", value=1.0e-5, format="%.1e",
            key=uk("s3", "ssconvtol"),
            help="Tolerance for steady-state convergence check."
        )

        s3_exec_solver = st.selectbox(
            "Exec Solver", ["always", "before Simulation", "never"],
            index=0, key=uk("s3", "exec"),
            help="'always' = normal execution. 'never' = disable this solver."
        )

# ==============================================================================
# TAB 4: DEM DATA (BEDROCK & SURFACE)
# ==============================================================================
with tab_dem:
    st.header("🗺️ DEM Elevation Data")
    st.markdown("""
    Provide bedrock and surface elevation profiles as two-column data: **x [m]  z [m]**.  
    These are included in the `.sif` boundary conditions via `include "filename.dat"`.
    """)

    dem_choice = st.selectbox(
        "Select DEM to Edit", ["Bedrock", "Surface"],
        key=uk("dem", "choice")
    )

    if dem_choice == "Bedrock":
        session_key = "table_data_bedrock"
        default_data = DEFAULT_BEDROCK_DATA
        fname_default = f"steady_ELA{int(ela_value)}_bedrock.dat"
        col_name = "Bedrock_Elevation_m"
    else:
        session_key = "table_data_surface"
        default_data = DEFAULT_SURFACE_DATA
        fname_default = f"steady_ELA{int(ela_value)}_surface.dat"
        col_name = "Surface_Elevation_m"

    if st.session_state[session_key] is None:
        st.session_state[session_key] = pd.DataFrame(
            default_data, columns=["X_m", col_name]
        )

    st.subheader(f"📄 {dem_choice} DEM Data")

    col_dem1, col_dem2 = st.columns([3, 1])
    with col_dem1:
        edited_df = st.data_editor(
            st.session_state[session_key],
            num_rows="dynamic",
            key=uk("dem", f"editor_{dem_choice}"),
            column_config={
                "X_m": st.column_config.NumberColumn("x [m]", format="%.2f"),
                col_name: st.column_config.NumberColumn("z [m]", format="%.4f"),
            },
            hide_index=True
        )
        st.session_state[session_key] = edited_df.copy()

    with col_dem2:
        st.markdown("**Statistics**")
        st.metric("Points", len(edited_df))
        if len(edited_df) > 0:
            st.metric("x range", f"{edited_df['X_m'].min():.1f} – {edited_df['X_m'].max():.1f}")
            st.metric("z range", f"{edited_df[col_name].min():.2f} – {edited_df[col_name].max():.2f}")

    st.divider()

    # Upload custom DEM file
    with st.expander("📤 Upload Custom DEM File", expanded=False):
        uploaded_dem = st.file_uploader(
            f"Upload {dem_choice} .dat file (two columns: x z)",
            type=["dat", "txt", "csv"],
            key=uk("dem", f"upload_{dem_choice}")
        )
        if uploaded_dem is not None:
            try:
                content = uploaded_dem.read().decode("utf-8")
                rows = []
                for line in content.strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        rows.append((float(parts[0]), float(parts[1])))
                if rows:
                    new_df = pd.DataFrame(rows, columns=["X_m", col_name])
                    st.session_state[session_key] = new_df
                    st.success(f"Loaded {len(rows)} points from uploaded file!")
                    st.rerun()
                else:
                    st.error("No valid data rows found in uploaded file.")
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    # Download current DEM
    dem_fname = st.text_input("DEM Filename", value=fname_default, key=uk("dem", f"fname_{dem_choice}"))

    # Generate .dat content (Elmer format: two columns, space-separated, no header)
    dem_content = "\n".join(
        f"{row['X_m']:.4f} {row[col_name]:.4f}"
        for _, row in edited_df.iterrows()
    )

    st.download_button(
        label=f"⬇️ Download {dem_fname}",
        data=dem_content,
        file_name=dem_fname,
        mime="text/plain",
        key=uk("dem", f"dl_{dem_choice}")
    )

    # Preview plot
    st.subheader("📈 DEM Preview")
    if len(edited_df) > 1:
        chart_data = pd.DataFrame({
            "x [m]": edited_df["X_m"],
            "z [m]": edited_df[col_name]
        })
        st.line_chart(chart_data.set_index("x [m]"))

# ==============================================================================
# TAB 5: BOUNDARY CONDITIONS
# ==============================================================================
with tab_bc:
    st.header("🔗 Boundary Conditions")
    st.markdown("""
    Define boundary conditions for the glacier domain. The typical setup is:
    - **BC 1 (Bedrock)**: No-slip + DEM mapping via `Bottom Surface`
    - **BC 2 (Sides)**: No normal flow at lateral boundaries
    - **BC 3 (Surface)**: Stress-free + DEM mapping via `Top Surface`
    """)

    # ---- Bedrock BC ----
    with st.expander("🪨 BC 1: Bedrock (Bottom)", expanded=True):
        st.markdown("No-slip condition at the glacier bed. The bedrock elevation is mapped from the DEM file.")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            bc1_name = st.text_input("Name", value="bedrock", key=uk("bc1", "name"))
            bc1_target = st.number_input("Target Boundaries", value=1, min_value=1, step=1, key=uk("bc1", "target"))
            bc1_bedrock_fname = st.text_input(
                "Bedrock DEM File", value=f"steady_ELA{int(ela_value)}_bedrock.dat",
                key=uk("bc1", "demfile"),
                help="Filename included for Bottom Surface elevation."
            )
        with col_b2:
            bc1_v1 = st.number_input("Velocity 1 (x)", value=0.0, key=uk("bc1", "v1"))
            bc1_v2 = st.number_input("Velocity 2 (y)", value=0.0, key=uk("bc1", "v2"))
            bc1_sliding = st.checkbox(
                "Enable Sliding Law (Weertman)", value=False, key=uk("bc1", "sliding"),
                help="Replace no-slip with Weertman sliding: τ_b = C · |u_b|^(m-1) · u_b"
            )

        if bc1_sliding:
            st.markdown("#### Weertman Sliding Parameters")
            col_sl1, col_sl2 = st.columns(2)
            with col_sl1:
                sliding_c = st.number_input(
                    "Sliding Coefficient C", value=1.0e-2, format="%.1e",
                    key=uk("bc1", "slidingc"),
                    help="Weertman sliding coefficient in m-yr-MPa units."
                )
                sliding_m = st.number_input(
                    "Sliding Exponent m", value=1.0, step=0.5,
                    key=uk("bc1", "slidingm"),
                    help="Weertman exponent (m=1 = linear sliding, m=3 = typical nonlinear)."
                )
            with col_sl2:
                st.markdown("""
                **Sliding Law**: τ_b = C · |u_b|^(m−1) · u_b
                
                In Elmer, this is typically implemented via:
                - `Sliding Coefficient = Real C`
                - `Sliding Exponent = Real m`
                - Remove fixed Velocity BC
                """)

    # ---- Sides BC ----
    with st.expander("↔️ BC 2: Lateral Sides", expanded=True):
        st.markdown("No normal flow at the left and right domain boundaries.")
        col_b2a, col_b2b = st.columns(2)
        with col_b2a:
            bc2_name = st.text_input("Name", value="sides", key=uk("bc2", "name"))
            bc2_targets = st.text_input(
                "Target Boundaries (space-separated)", value="3 4",
                key=uk("bc2", "targets"),
                help="E.g., '3 4' for left and right boundaries."
            )
        with col_b2b:
            bc2_v1 = st.number_input("Velocity 1 (x)", value=0.0, key=uk("bc2", "v1"))
            bc2_v2_free = st.checkbox(
                "Velocity 2 (y) free (stress-free)", value=True, key=uk("bc2", "v2free"),
                help="If checked, no BC is applied for velocity 2 (natural BC = zero normal stress)."
            )
            if not bc2_v2_free:
                bc2_v2 = st.number_input("Velocity 2 (y)", value=0.0, key=uk("bc2", "v2"))
            else:
                bc2_v2 = None

    # ---- Surface BC ----
    with st.expander("🏔️ BC 3: Glacier Surface (Top)", expanded=True):
        st.markdown("Stress-free surface. The surface elevation is mapped from the DEM file via `Top Surface`.")
        col_b3a, col_b3b = st.columns(2)
        with col_b3a:
            bc3_name = st.text_input("Name", value="surface", key=uk("bc3", "name"))
            bc3_target = st.number_input("Target Boundaries", value=2, min_value=1, step=1, key=uk("bc3", "target"))
            bc3_surface_fname = st.text_input(
                "Surface DEM File", value=f"steady_ELA{int(ela_value)}_surface.dat",
                key=uk("bc3", "demfile"),
                help="Filename included for Top Surface elevation."
            )
        with col_b3b:
            bc3_stress_free = st.checkbox(
                "Stress-Free (Natural BC)", value=True, key=uk("bc3", "stressfree"),
                help="If checked, no explicit traction BC is applied (natural BC = zero stress)."
            )
            bc3_depth_zero = st.checkbox(
                "Set Depth = 0 at Surface", value=False, key=uk("bc3", "depthzero"),
                help="Optionally set Depth variable to zero at the surface."
            )

    # ---- Additional Custom BCs ----
    with st.expander("➕ Additional Custom Boundary Conditions", expanded=False):
        n_custom_bc = st.number_input("Number of Additional BCs", value=0, min_value=0, max_value=10, key=uk("bc", "ncustom"))
        custom_bcs = []
        for i in range(n_custom_bc):
            st.markdown(f"**Custom BC {i+1}**")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                cname = st.text_input(f"Name", value=f"custom_bc_{i+1}", key=uk("bc", f"custom_name_{i}"))
                ctarget = st.number_input(f"Target Boundaries", value=i+4, key=uk("bc", f"custom_target_{i}"))
            with col_c2:
                ctype = st.selectbox(f"Type", ["No-slip", "Stress-free", "Custom"], index=0, key=uk("bc", f"custom_type_{i}"))
                ccustom = st.text_area(f"Custom SIF lines", value="", height=100, key=uk("bc", f"custom_lines_{i}"))
            custom_bcs.append({"name": cname, "target": ctarget, "type": ctype, "custom": ccustom})

# ==============================================================================
# SIF GENERATION FUNCTION
# ==============================================================================
def generate_sif(params: dict) -> str:
    """Generate the complete .sif file content as a string."""

    p = params  # shorthand

    lines = []

    # ---- Header ----
    lines.append("!echo on")
    lines.append("Header")
    lines.append("  !CHECK KEYWORDS Warn")
    lines.append(f'  Mesh DB "." "{p["mesh_db_name"]}"')
    lines.append('  Include Path ""')
    lines.append(f'  Results Directory "{p["results_dir"]}"')
    lines.append("End")
    lines.append("")

    # ---- Simulation ----
    lines.append("Simulation")
    lines.append(f'  Max Output Level = {p["max_output_level"]}')
    lines.append(f'  Coordinate System = "{p["coord_system"]}"')
    mapping_parts = p["coord_mapping"].split()
    n_map = len(mapping_parts)
    lines.append(f'  Coordinate Mapping({n_map}) = {" ".join(mapping_parts)}')
    lines.append(f'  Simulation Type = "{p["sim_type"]}"')

    if p["sim_type"] == "Steady":
        lines.append(f'  Steady State Max Iterations = {p["steady_max_iter"]}')
    else:
        lines.append(f'  Timestep Intervals = {p["n_steps"]}')
        lines.append(f'  Timestep Sizes = {p["dt"]}')
        lines.append(f'  Output Intervals = 1')

    lines.append(f'  Output Intervals = {p["output_intervals"]}')
    lines.append(f'  Output File = "{p["output_file"]}"')
    lines.append(f'  Post File = "{p["post_file"]}"')
    if not p["init_dirichlet"]:
        lines.append("  Initialize Dirichlet Conditions = Logical False")
    lines.append("End")
    lines.append("")

    # ---- Constants ----
    lines.append("Constants")
    lines.append("  Stefan Boltzmann = 5.67e-08")
    lines.append("End")
    lines.append("")

    # ---- Body 1 ----
    lines.append("Body 1")
    lines.append('  Name = "Glacier"')
    lines.append("  Body Force = 1")
    lines.append("  Equation = 1")
    lines.append("  Material = 1")
    lines.append("  Initial Condition = 1")
    lines.append("End")
    lines.append("")

    # ---- Equation 1 ----
    lines.append("Equation 1")
    lines.append('  Name = "Equation1"')
    lines.append('  Convection = "computed"')
    lines.append('  Flow Solution Name = String "Flow Solution"')
    lines.append("  Active Solvers(3) = 1 2 3")
    lines.append("End")
    lines.append("")

    # ---- Initial Condition 1 ----
    lines.append("Initial Condition 1")
    lines.append("  Velocity 1 = 0.0")
    lines.append("  Velocity 2 = 0.0")
    lines.append("  Pressure = 0.0")
    lines.append("  Depth = Real 0.0")
    lines.append("End")
    lines.append("")

    # ---- Solver 1: StructuredMeshMapper ----
    lines.append("! maps DEM's at the very beginning")
    lines.append("! to originally rectangular mesh")
    lines.append("! see Top and Bottom Surface in BC's")
    lines.append("Solver 1")
    lines.append(f'  Exec Solver = "{p["s1_exec"]}"')
    lines.append('  Equation = "MapCoordinate"')
    lines.append('  Procedure = "StructuredMeshMapper" "StructuredMeshMapper"')
    lines.append(f'  Active Coordinate = Integer {p["s1_active_coord"]}')
    lines.append('  Mesh Velocity Variable = String "Mesh Velocity 2"')
    if p["s1_mesh_vel_first_zero"]:
        lines.append("  Mesh Velocity First Zero = Logical True")
    lines.append(f'  Dot Product Tolerance = Real {p["s1_dot_tol"]}')
    lines.append("End")
    lines.append("")

    # ---- Solver 2: StructuredProjectToPlane ----
    lines.append("! Flow Depth still for postprocessing, only,")
    lines.append("! now replaced by structured version")
    lines.append("Solver 2")
    lines.append('  Equation = "HeightDepth"')
    lines.append('  Procedure = "StructuredProjectToPlane" "StructuredProjectToPlane"')
    lines.append(f'  Active Coordinate = Integer {p["s2_active_coord"]}')
    if p["s2_compute_depth"]:
        lines.append("  Operator 1 = depth")
    if p["s2_compute_height"]:
        lines.append("  Operator 2 = height")
    lines.append("End")
    lines.append("")

    # ---- Solver 3: Navier-Stokes ----
    lines.append("! the central part of the problem: the Stokes solver")
    lines.append("Solver 3")
    if p["s3_exec_solver"] == "never":
        lines.append('  Exec Solver = "Never"')
    else:
        lines.append(f'  Exec Solver = "{p["s3_exec_solver"]}"')
    lines.append('  Equation = "Navier-Stokes"')
    if p["optimize_bw"]:
        lines.append("  Optimize Bandwidth = Logical True")

    # Linear system
    if p["ls_solver"] == "Direct":
        lines.append("  ! direct solver")
        lines.append("  Linear System Solver = Direct")
        lines.append(f'  Linear System Direct Method = "{p["ls_direct_method"]}"')
    else:
        lines.append("  ! iterative solver")
        lines.append("  Linear System Solver = Iterative")
        lines.append(f'  Linear System Iterative Method = "{p["ls_iter_method"]}"')

    lines.append(f"  Linear System Max Iterations = {p['ls_max_iter']}")
    lines.append(f"  Linear System Convergence Tolerance = {p['ls_conv_tol']:.1E}")
    if not p["ls_abort"]:
        lines.append("  Linear System Abort Not Converged = False")

    if p["ls_solver"] == "Iterative":
        lines.append(f'  Linear System Preconditioning = "{p["ls_precond"]}"')

    lines.append(f"  Linear System Residual Output = {p['ls_residual_output']}")
    lines.append(f"  Flow Model = {p['flow_model']}")
    lines.append("")
    lines.append(f"  Steady State Convergence Tolerance = {p['ss_conv_tol']:.1E}")
    lines.append(f'  Stabilization Method = {p["stab_method"]}')
    lines.append("")
    lines.append(f"  Nonlinear System Convergence Tolerance = {p['nl_conv_tol']:.1E}")
    lines.append(f'  Nonlinear System Convergence Measure = {p["nl_conv_measure"]}')
    lines.append(f"  Nonlinear System Max Iterations = {p['nl_max_iter']}")
    lines.append(f"  Nonlinear System Newton After Iterations = {p['newton_after_iter']}")
    lines.append(f"  Nonlinear System Newton After Tolerance = {p['newton_after_tol']:.1E}")
    lines.append('  Exported Variable 1 = -dofs 3 "Mesh Velocity"')

    if p["nl_relax"] > 0.0:
        lines.append(f"  Nonlinear System Relaxation Factor = {p['nl_relax']}")

    lines.append("End")
    lines.append("")

    # ---- Material 1 ----
    lines.append("! we use m-yr-MPa system 1 yr = 31556926.0 sec")
    lines.append("Material 1")
    lines.append('  Name = "ice"')
    lines.append(f"  Density = Real {p['density_mpa']:.10e}")
    lines.append("  !----------------")
    lines.append("  ! viscosity stuff")
    lines.append("  !----------------")
    lines.append('  Viscosity Model = String "Glen"')
    lines.append("  ! Viscosity has to be set to a dummy value")
    lines.append("  ! to avoid warning output from Elmer")
    lines.append("  Viscosity = Real 1.0")
    lines.append(f"  Glen Exponent = Real {p['glen_n']}")
    lines.append(f"  Critical Shear Rate = Real {p['crit_shear']}")
    lines.append("  ! Rate Factors (Paterson value in MPa^-3a^-1)")
    lines.append(f"  Rate Factor 1 = Real {p['rate_factor_1']:.3e}")
    lines.append(f"  Rate Factor 2 = Real {p['rate_factor_2']:.3e}")
    lines.append("  ! these are in SI units - no problem, as long as")
    lines.append("  ! the gas constant also is")
    lines.append(f"  Activation Energy 1 = Real {p['activation_energy_1']}")
    lines.append(f"  Activation Energy 2 = Real {p['activation_energy_2']}")
    lines.append(f"  Glen Enhancement Factor = Real {p['enhancement_factor']}")
    lines.append("  ! the variable taken to evaluate the Arrhenius law")
    lines.append("  ! in general this should be the temperature relative")
    lines.append("  ! to pressure melting point.")
    if p["use_temp_var"]:
        lines.append('  Temperature Field Variable = String "Temp Homologous"')
    else:
        lines.append("  !  Temperature Field Variable = String \"Temp Homologous\"")
    lines.append(f"  ! the temperature to switch between the two regimes in the flow law")
    lines.append(f"  Limit Temperature = Real {p['limit_temp']}")
    if not p["use_temp_var"]:
        lines.append("  ! In case there is no temperature variable (which here is the case)")
        lines.append(f"  Constant Temperature = Real {p['const_temp']}")
    lines.append("End")
    lines.append("")

    # ---- Body Force 1 ----
    lines.append("Body Force 1")
    lines.append('  Name = "BodyForce1"')
    lines.append("  Heat Source = 1")
    lines.append(f"  Flow BodyForce 1 = Real 0.0")
    lines.append(f"  Flow BodyForce 2 = Real {p['gravity_mpa']:.10e}  !MPa - a - m")
    lines.append("End")
    lines.append("")

    # ---- Boundary Condition 1: Bedrock ----
    lines.append("Boundary Condition 1")
    lines.append(f'  Name = "{p["bc1_name"]}"')
    lines.append(f"  Target Boundaries = {p['bc1_target']}")
    lines.append("! include the bedrock DEM, which has two columns")
    lines.append("  Bottom Surface = Variable Coordinate 1")
    lines.append("  Real cubic")
    lines.append(f'     include  "{p["bc1_bedrock_fname"]}"')
    lines.append("  End")

    if not p["bc1_sliding"]:
        lines.append("  Velocity 1 = Real 0.0e0")
        lines.append("  Velocity 2 = Real 0.0e0")
    else:
        lines.append(f"  Sliding Coefficient = Real {p['sliding_c']:.3e}")
        lines.append(f"  Sliding Exponent = Real {p['sliding_m']}")
    lines.append("End")
    lines.append("")

    # ---- Boundary Condition 2: Sides ----
    targets = p["bc2_targets"].split()
    n_targets = len(targets)
    lines.append("Boundary Condition 2")
    lines.append(f'  Name = "{p["bc2_name"]}"')
    if n_targets == 1:
        lines.append(f"  Target Boundaries = {targets[0]}")
    else:
        lines.append(f"  Target Boundaries({n_targets}) = {' '.join(targets)}  ! combine left and right boundary")
    lines.append(f"  Velocity 1 = Real {p['bc2_v1']:.1e}")
    if p["bc2_v2"] is not None:
        lines.append(f"  Velocity 2 = Real {p['bc2_v2']:.1e}")
    lines.append("End")
    lines.append("")

    # ---- Boundary Condition 3: Surface ----
    lines.append("Boundary Condition 3")
    lines.append(f'  Name = "{p["bc3_name"]}"')
    lines.append(f"  Target Boundaries = {p['bc3_target']}")
    lines.append("! include the surface DEM, which has two columns")
    lines.append("  Top Surface = Variable Coordinate 1")
    lines.append("  Real cubic")
    lines.append(f'     include  "{p["bc3_surface_fname"]}"')
    lines.append("  End")
    if p["bc3_depth_zero"]:
        lines.append("  Depth = Real 0.0")
    lines.append("End")
    lines.append("")

    # ---- Custom BCs ----
    for i, cbc in enumerate(p.get("custom_bcs", [])):
        lines.append(f"Boundary Condition {4+i}")
        lines.append(f'  Name = "{cbc["name"]}"')
        lines.append(f'  Target Boundaries = {cbc["target"]}')
        if cbc["type"] == "No-slip":
            lines.append("  Velocity 1 = Real 0.0")
            lines.append("  Velocity 2 = Real 0.0")
        elif cbc["type"] == "Stress-free":
            lines.append("  ! Natural BC (stress-free) — no explicit lines needed")
        if cbc["custom"].strip():
            for custom_line in cbc["custom"].strip().split('\n'):
                lines.append(f"  {custom_line}")
        lines.append("End")
        lines.append("")

    return "\n".join(lines)


# ==============================================================================
# TAB 6: GENERATE & DOWNLOAD
# ==============================================================================
with tab_generate:
    st.header("📥 Generate & Download Files")
    st.markdown("Generate the complete `.sif` file and DEM `.dat` files, then download individually or as a ZIP bundle.")

    # Collect all parameters
    params = {
        "mesh_db_name": mesh_db_name,
        "results_dir": results_dir,
        "max_output_level": max_output_level,
        "coord_system": coord_system,
        "coord_mapping": coord_mapping,
        "sim_type": sim_type,
        "steady_max_iter": steady_max_iter if sim_type == "Steady" else 1,
        "dt": dt if sim_type == "Transient" else 0,
        "t_end": t_end if sim_type == "Transient" else 0,
        "n_steps": n_steps if sim_type == "Transient" else 0,
        "output_intervals": output_intervals,
        "output_file": output_file,
        "post_file": post_file,
        "init_dirichlet": init_dirichlet,
        # Material
        "density_mpa": density_mpa,
        "gravity_mpa": gravity_mpa,
        "glen_n": glen_n,
        "enhancement_factor": enhancement_factor,
        "crit_shear": crit_shear,
        "rate_factor_1": rate_factor_1,
        "rate_factor_2": rate_factor_2,
        "activation_energy_1": activation_energy_1,
        "activation_energy_2": activation_energy_2,
        "limit_temp": limit_temp,
        "use_temp_var": use_temp_var,
        "const_temp": const_temp,
        # Solver 1
        "s1_active_coord": s1_active_coord,
        "s1_exec": s1_exec,
        "s1_mesh_vel_first_zero": s1_mesh_vel_first_zero,
        "s1_dot_tol": s1_dot_tol,
        # Solver 2
        "s2_active_coord": s2_active_coord,
        "s2_compute_depth": s2_compute_depth,
        "s2_compute_height": s2_compute_height,
        # Solver 3
        "s3_exec_solver": s3_exec_solver,
        "optimize_bw": optimize_bw,
        "ls_solver": ls_solver,
        "ls_direct_method": ls_direct_method if ls_solver == "Direct" else "UMFPACK",
        "ls_iter_method": ls_iter_method if ls_solver == "Iterative" else "GCR",
        "ls_max_iter": ls_max_iter,
        "ls_conv_tol": ls_conv_tol,
        "ls_abort": ls_abort,
        "ls_precond": ls_precond,
        "ls_residual_output": ls_residual_output,
        "flow_model": flow_model,
        "ss_conv_tol": ss_conv_tol,
        "stab_method": stab_method,
        "nl_conv_tol": nl_conv_tol,
        "nl_conv_measure": nl_conv_measure,
        "nl_max_iter": nl_max_iter,
        "newton_after_iter": newton_after_iter,
        "newton_after_tol": newton_after_tol,
        "nl_relax": nl_relax,
        # BC 1
        "bc1_name": bc1_name,
        "bc1_target": bc1_target,
        "bc1_bedrock_fname": bc1_bedrock_fname,
        "bc1_sliding": bc1_sliding,
        "sliding_c": sliding_c if bc1_sliding else 1e-2,
        "sliding_m": sliding_m if bc1_sliding else 1.0,
        # BC 2
        "bc2_name": bc2_name,
        "bc2_targets": bc2_targets,
        "bc2_v1": bc2_v1,
        "bc2_v2": bc2_v2 if not bc2_v2_free else None,
        # BC 3
        "bc3_name": bc3_name,
        "bc3_target": bc3_target,
        "bc3_surface_fname": bc3_surface_fname,
        "bc3_depth_zero": bc3_depth_zero,
        # Custom BCs
        "custom_bcs": custom_bcs,
    }

    # Add transient params if needed
    if sim_type == "Transient":
        params["dt"] = dt
        params["t_end"] = t_end
        params["n_steps"] = n_steps

    # Generate button
    if st.button("🚀 Generate All Files", type="primary", key="generate_btn"):
        with st.spinner("Generating .sif and .dat files..."):
            # Generate SIF
            sif_content = generate_sif(params)

            # Generate bedrock .dat
            bedrock_df = st.session_state.table_data_bedrock
            if bedrock_df is not None:
                bedrock_content = "\n".join(
                    f"{row['X_m']:.4f} {row['Bedrock_Elevation_m']:.4f}"
                    for _, row in bedrock_df.iterrows()
                )
            else:
                bedrock_content = "0 0\n2500 500"

            # Generate surface .dat
            surface_df = st.session_state.table_data_surface
            if surface_df is not None:
                surface_content = "\n".join(
                    f"{row['X_m']:.4f} {row['Surface_Elevation_m']:.4f}"
                    for _, row in surface_df.iterrows()
                )
            else:
                surface_content = "0 5\n2500 540"

            # Store in session state
            st.session_state.generated_content = {
                sif_filename: sif_content,
                bc1_bedrock_fname: bedrock_content,
                bc3_surface_fname: surface_content,
            }
            st.session_state.generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.success("✅ All files generated successfully!")

    # ---- Display and Download ----
    if st.session_state.generated_content:
        st.markdown(f'<div class="download-section"><strong>Generated at:</strong> {st.session_state.generation_timestamp}</div>', unsafe_allow_html=True)

        for fname, content in st.session_state.generated_content.items():
            st.subheader(f"📄 {fname}")

            if fname.endswith(".sif"):
                st.code(content, language="fortran")
            else:
                with st.expander(f"View {fname}", expanded=False):
                    st.code(content, language="plaintext")

            st.download_button(
                label=f"⬇️ Download {fname}",
                data=content,
                file_name=fname,
                mime="text/plain",
                key=uk("gen", f"dl_{fname.replace('.', '_')}")
            )
            st.divider()

        # ---- ZIP Bundle ----
        st.subheader("📦 Download All as ZIP")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname, content in st.session_state.generated_content.items():
                zf.writestr(fname, content)

        zip_filename = f"{project_name}_bundle.zip"
        st.download_button(
            label=f"📦 Download {zip_filename}",
            data=zip_buffer.getvalue(),
            file_name=zip_filename,
            mime="application/zip",
            key="zip_download"
        )

        # ---- File sizes ----
        st.subheader("📊 File Sizes")
        size_data = []
        total = 0
        for fname, content in st.session_state.generated_content.items():
            size_bytes = len(content.encode('utf-8'))
            total += size_bytes
            if size_bytes > 1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            else:
                size_str = f"{size_bytes} B"
            size_data.append({"File": fname, "Size": size_str, "Lines": content.count('\n') + 1})

        size_data.append({"File": "TOTAL", "Size": f"{total/1024:.1f} KB", "Lines": sum(d["Lines"] for d in size_data)})
        st.table(pd.DataFrame(size_data))
    else:
        st.info("Click **Generate All Files** above to create the .sif and .dat files.")
