#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Elmer FEM .sif Generator for Glacier Modeling - PARAMETERIZED EDITION
- ANALYTICAL GEOMETRY: Linear bed and surface slopes replacing external DEMs.
- USER-DEFINABLE PARAMETERS: All key geometry and material parameters at the top.
- SELF-CONTAINED: Uses MATC expressions for boundary profiles.
- COMPLETE DOWNLOAD: ZIP bundling + persistent session state.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import io
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Elmer Glacier Generator",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stCodeBlock { background-color: #f8f9fa; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    .download-section { background: #e8f4fd; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .info-box { background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0.25rem; }
</style>
""", unsafe_allow_html=True)

st.title("🏔️ Elmer FEM Generator for Glacier Modeling")
st.markdown("""
**Generate complete, parameterized `.sif` input files for 2D steady-state glacier flow simulations.**  
_Uses analytical geometry (linear slopes) and MATC expressions, eliminating the need for external DEM files._
""")

# ====================== HELPER: UNIQUE KEY GENERATOR ======================
def uk(section: str, var: str, suffix: str = "") -> str:
    """Generate unique key for Streamlit widgets to prevent state collisions"""
    return f"{section}_{var}_{suffix}".strip("_")

# ====================== SESSION STATE INITIALIZATION ======================
if "generated_content" not in st.session_state:
    st.session_state.generated_content = {}
if "generation_timestamp" not in st.session_state:
    st.session_state.generation_timestamp = None

# ====================== SIDEBAR: GLOBAL SETTINGS ======================
st.sidebar.header("⚙️ Global Settings")
project_name = st.sidebar.text_input("Project Name", value="Glacier_Stokes_2D", key=uk("global", "project"))
author = st.sidebar.text_input("Author", value="Your Name", key=uk("global", "author"))
date_str = datetime.now().strftime("%Y-%m-%d")

st.sidebar.subheader("📁 Output Files")
sif_filename = st.sidebar.text_input(".sif Filename", value=f"{project_name.lower()}.sif", key=uk("out", "sif"))
mesh_name = st.sidebar.text_input("Mesh Database Name", value="testglacier", key=uk("out", "mesh"))

# ====================== TABS NAVIGATION ======================
tab_geom, tab_mat, tab_phys, tab_bc, tab_gen = st.tabs([
    "🏔️ Geometry & Domain",
    "🧊 Material Properties",
    "⚙️ Solvers & Physics",
    "🔗 Boundary Conditions",
    "📥 Generate Files"
])

# ====================== TAB 1: GEOMETRY & DOMAIN ======================
with tab_geom:
    st.header("🏔️ Glacier Geometry Parameters")
    st.markdown('<div class="info-box">💡 Define the analytical linear profiles for the glacier bed and surface. All units are in meters [m] or dimensionless slopes.</div>', unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Domain & Bedrock")
        L = st.number_input("Domain Length (L) [m]", value=10000.0, step=100.0, key=uk("geom", "L"))
        H_bed = st.number_input("Bed Elevation at x=0 (H_bed) [m]", value=-500.0, step=10.0, key=uk("geom", "H_bed"))
        bed_slope = st.number_input("Bed Slope (dy/dx)", value=0.01, step=0.001, format="%.4f", key=uk("geom", "bed_slope"))
        
    with col_g2:
        st.subheader("Surface Profile")
        surface_elev = st.number_input("Surface Elevation at x=0 [m]", value=100.0, step=10.0, key=uk("geom", "surface_elev"))
        surface_slope = st.number_input("Surface Slope (dy/dx)", value=0.05, step=0.001, format="%.4f", key=uk("geom", "surface_slope"))
        
        use_angle = st.checkbox("Define surface slope by angle instead", value=False, key=uk("geom", "use_angle"))
        if use_angle:
            angle = st.number_input("Surface Angle [degrees]", value=2.86, step=0.1, key=uk("geom", "angle"))
            surface_slope = np.tan(np.radians(angle))
            st.info(f"Calculated surface slope: {surface_slope:.6f}")

    # Preview Plot
    st.subheader("📈 Geometry Preview")
    x = np.linspace(0, L, 100)
    y_bed = H_bed + bed_slope * x
    y_surf = surface_elev + surface_slope * x
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, y_bed, y_surf, color='lightblue', alpha=0.6, label='Ice')
    ax.plot(x, y_bed, color='saddlebrown', linewidth=2, label='Bedrock')
    ax.plot(x, y_surf, color='blue', linewidth=2, label='Surface')
    ax.set_xlabel("Distance x [m]")
    ax.set_ylabel("Elevation y [m]")
    ax.set_title("Glacier Cross-Section Profile")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('auto')
    st.pyplot(fig)

# ====================== TAB 2: MATERIAL PROPERTIES ======================
with tab_mat:
    st.header("🧊 Ice Material Properties")
    st.markdown("Parameters for Glen's flow law and ice density.")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Basic Properties")
        density = st.number_input("Density [kg/m³]", value=910.0, step=10.0, key=uk("mat", "density"))
        glen_exponent = st.number_input("Glen Exponent (n)", value=3.0, step=1.0, key=uk("mat", "glen_n"))
        enhancement = st.number_input("Glen Enhancement Factor (E)", value=1.0, step=0.1, key=uk("mat", "enhancement"))
        crit_shear = st.number_input("Critical Shear Rate [1/s]", value=1.0e-10, format="%.1e", key=uk("mat", "crit_shear"))
        
    with col_m2:
        st.subheader("Temperature & Rate Factors")
        const_temp = st.number_input("Constant Temperature [°C]", value=-3.0, step=1.0, key=uk("mat", "const_temp"))
        limit_temp = st.number_input("Limit Temperature [°C]", value=-10.0, step=1.0, key=uk("mat", "limit_temp"))
        
        st.markdown("Rate Factors (A) and Activation Energies (Q):")
        rate_factor_1 = st.number_input("Rate Factor 1 (T > Limit) [Pa⁻³ s⁻¹]", value=1.258e13, format="%.3e", key=uk("mat", "rf1"))
        act_energy_1 = st.number_input("Activation Energy 1 (T > Limit) [J/mol]", value=60e3, step=1e3, key=uk("mat", "ae1"))
        rate_factor_2 = st.number_input("Rate Factor 2 (T <= Limit) [Pa⁻³ s⁻¹]", value=6.046e28, format="%.3e", key=uk("mat", "rf2"))
        act_energy_2 = st.number_input("Activation Energy 2 (T <= Limit) [J/mol]", value=139e3, step=1e3, key=uk("mat", "ae2"))

# ====================== TAB 3: SOLVERS & PHYSICS ======================
with tab_phys:
    st.header("⚙️ Solvers & Physics Settings")
    
    st.subheader("Simulation Control")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        max_output_level = st.number_input("Max Output Level", value=4, min_value=1, max_value=5, key=uk("phys", "max_out"))
        steady_max_iter = st.number_input("Steady State Max Iterations", value=1, min_value=1, key=uk("phys", "steady_iter"))
        output_intervals = st.number_input("Output Intervals", value=1, min_value=1, key=uk("phys", "out_int"))
        
    with col_p2:
        results_dir = st.text_input("Results Directory", value="", key=uk("phys", "res_dir"))
        output_file = st.text_input("Output File", value="Stokes_diagnostic.result", key=uk("phys", "out_file"))
        post_file = st.text_input("Post File", value="Stokes_diagnostic.vtu", key=uk("phys", "post_file"))

    st.subheader("Stokes Solver Parameters")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        linear_max_iter = st.number_input("Linear System Max Iterations", value=5000, key=uk("solver", "lin_max_iter"))
        lin_conv_tol = st.number_input("Linear Convergence Tolerance", value=1.0e-6, format="%.1e", key=uk("solver", "lin_tol"))
        nonlin_max_iter = st.number_input("Nonlinear Max Iterations", value=50, key=uk("solver", "nonlin_max_iter"))
        nonlin_conv_tol = st.number_input("Nonlinear Convergence Tolerance", value=1.0e-4, format="%.1e", key=uk("solver", "nonlin_tol"))
        
    with col_s2:
        steady_conv_tol = st.number_input("Steady State Convergence Tolerance", value=1.0e-5, format="%.1e", key=uk("solver", "steady_tol"))
        newton_after_iter = st.number_input("Newton After Iterations", value=3, key=uk("solver", "newton_iter"))
        newton_after_tol = st.number_input("Newton After Tolerance", value=1.0e-1, format="%.1e", key=uk("solver", "newton_tol"))
        gravity = st.number_input("Gravity Magnitude [m/s²]", value=9.81, step=0.01, key=uk("solver", "gravity"))

# ====================== TAB 4: BOUNDARY CONDITIONS ======================
with tab_bc:
    st.header("🔗 Boundary Conditions")
    st.markdown("Specify the target boundary indices from your mesh for each physical boundary.")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.subheader("Bedrock (No-Slip)")
        bc_bed = st.text_input("Target Boundary (e.g., '1' or '1 2')", value="1", key=uk("bc", "bed"))
    with col_b2:
        st.subheader("Sides (No-Slip X)")
        bc_sides = st.text_input("Target Boundaries (e.g., '3 4')", value="3 4", key=uk("bc", "sides"))
    with col_b3:
        st.subheader("Surface (Free)")
        bc_surface = st.text_input("Target Boundary (e.g., '2')", value="2", key=uk("bc", "surface"))

# ====================== TAB 5: GENERATE FILES ======================
with tab_gen:
    st.header("📥 Generate Elmer Input File (.sif)")
    
    if st.session_state.generated_content:
        st.success(f"✅ File generated at {st.session_state.generation_timestamp}!")
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        
        gc = st.session_state.generated_content
        st.download_button(
            label="📄 Download .sif File",
            data=gc['sif_content'],
            file_name=gc['sif_filename'],
            mime="text/plain",
            key="dl_sif_persistent"
        )
        
        # ZIP download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(gc['sif_filename'], gc['sif_content'])
            readme = f"""Glacier Modeling Elmer Project: {project_name}
Generated on: {date_str}
Author: {author}

Files included:
- {gc['sif_filename']}: Main Elmer input file.

Instructions:
1. Ensure your mesh files (e.g., {mesh_name}.mesh, {mesh_name}.nodes, {mesh_name}.elements) are in the same directory.
2. Run ElmerGrid to generate the mesh if you haven't already:
   ElmerGrid 1 2 {mesh_name} -autoclean
3. Run the simulation:
   ElmerSolver {gc['sif_filename']}
4. View results in ElmerPost or ParaView using {gc['post_file']}.
"""
            zip_file.writestr("README.txt", readme)
            
        st.download_button(
            label="📦 Download Project as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{project_name}_elmer_project.zip",
            mime="application/zip",
            key="dl_zip_persistent"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Generate .sif File", type="primary", use_container_width=True):
        
        # Handle unit scaling (MPa-a-m vs Standard SI)
        use_scaled_units = st.checkbox("Use original MPa-a-m unit scaling (as in reference)", value=True, key=uk("gen", "scaled_units"))
        
        if use_scaled_units:
            dens_str = f"${density}*1.0E-06*(31556926.0)^(-2.0)"
            gravity_str = f"${-gravity} * (31556926.0)^(2.0)"
        else:
            dens_str = f"${density}"
            gravity_str = f"${-gravity}"

        sif_content = f"""!echo on
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "{mesh_name}"
  Include Path ""
  Results Directory "{results_dir}"
End

Simulation
  Max Output Level = {max_output_level}
  Coordinate System = "Cartesian 2D"
  Coordinate Mapping(3) = 1 2 3
  Simulation Type = "Steady"
  Steady State Max Iterations = {steady_max_iter}
  Output Intervals = {output_intervals}
  Output File = "{output_file}"
  Post File = "{post_file}"
  Initialize Dirichlet Conditions = Logical False

  ! ==============================================================
  ! USER PARAMETERS – change these to match your glacier geometry
  ! ==============================================================
  $L = {L:.1f}                 ! Domain length in x-direction [m]
  $H_bed = {H_bed:.1f}              ! Bed elevation at x=0 [m] (negative)
  $bed_slope = {bed_slope:.6f}            ! Bed slope (dy/dx)
  $surface_elev = {surface_elev:.1f}        ! Surface elevation at x=0 [m]
  $surface_slope = {surface_slope:.6f}        ! Surface slope (dy/dx)
  ! ==============================================================

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
  Active Solvers(3) = 1 2 3
End

Initial Condition 1
  Velocity 1 = 0.0
  Velocity 2 = 0.0
  Pressure = 0.0
  Depth = Real 0.0
End

! --------------------------------------------------------------
! Solver 1: Mesh mapping to follow the prescribed bed and surface
! --------------------------------------------------------------
Solver 1
  Exec Solver = "before Simulation"
  Equation = "MapCoordinate"
  Procedure = "StructuredMeshMapper" "StructuredMeshMapper"
  Active Coordinate = Integer 2
  Mesh Velocity Variable = String "Mesh Velocity 2"
  Mesh Velocity First Zero = Logical True
  Dot Product Tolerance = Real 0.01
End

Solver 2
  Equation = "HeightDepth"
  Procedure = "StructuredProjectToPlane" "StructuredProjectToPlane"
  Active Coordinate = Integer 2
  Operator 1 = depth
  Operator 2 = height
End

! --------------------------------------------------------------
! Solver 3: Stokes flow
! --------------------------------------------------------------
Solver 3
  Equation = "Navier-Stokes"
  Optimize Bandwidth = Logical True
  Linear System Solver = Direct
  Linear System Direct Method = "UMFPACK"
  Linear System Max Iterations = {linear_max_iter}
  Linear System Convergence Tolerance = {lin_conv_tol:.1e}
  Linear System Abort Not Converged = False
  Linear System Preconditioning = "ILU1"
  Linear System Residual Output = 1
  Flow Model = Stokes
  Steady State Convergence Tolerance = {steady_conv_tol:.1e}
  Stabilization Method = Stabilized
  Nonlinear System Convergence Tolerance = {nonlin_conv_tol:.1e}
  Nonlinear System Convergence Measure = Solution
  Nonlinear System Max Iterations = {nonlin_max_iter}
  Nonlinear System Newton After Iterations = {newton_after_iter}
  Nonlinear System Newton After Tolerance = {newton_after_tol:.1e}
  Exported Variable 1 = -dofs 3 "Mesh Velocity"
End

! --------------------------------------------------------------
! Material (ice)
! --------------------------------------------------------------
Material 1
  Name = "ice"
  Density = Real {dens_str}
  Viscosity Model = String "Glen"
  Viscosity = Real 1.0
  Glen Exponent = Real {glen_exponent}
  Critical Shear Rate = Real {crit_shear:.1e}
  Rate Factor 1 = Real {rate_factor_1:.3e}
  Rate Factor 2 = Real {rate_factor_2:.3e}
  Activation Energy 1 = Real {act_energy_1:.1f}
  Activation Energy 2 = Real {act_energy_2:.1f}
  Glen Enhancement Factor = Real {enhancement}
  Limit Temperature = Real {limit_temp}
  Constant Temperature = Real {const_temp}
End

Body Force 1
  Name = "BodyForce1"
  Heat Source = 1
  Flow BodyForce 1 = Real 0.0
  Flow BodyForce 2 = Real {gravity_str}   ! Gravity in y-direction
End

! --------------------------------------------------------------
! BOUNDARY CONDITIONS with MATC expressions
! --------------------------------------------------------------
Boundary Condition 1
  Name = "bedrock"
  Target Boundaries = {bc_bed}
  ! Bedrock profile computed from user parameters
  Bottom Surface = Variable Coordinate 1
    Real MATC "$H_bed + $bed_slope * tx"
  End
  Velocity 1 = Real 0.0
  Velocity 2 = Real 0.0
End

Boundary Condition 2
  Name = "sides"
  Target Boundaries = {bc_sides}
  Velocity 1 = Real 0.0
End

Boundary Condition 3
  Name = "surface"
  Target Boundaries = {bc_surface}
  ! Surface profile computed from user parameters
  Top Surface = Variable Coordinate 1
    Real MATC "$surface_elev + $surface_slope * tx"
  End
End
"""
        
        st.session_state.generated_content = {
            'sif_content': sif_content,
            'sif_filename': sif_filename,
            'post_file': post_file,
            'mesh_name': mesh_name,
            'project_name': project_name
        }
        st.session_state.generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **💡 Pro Tips for Glacier Modeling:**
    - 📐 **Mesh Generation**: Ensure your 2D mesh has boundary markers matching the "Target Boundaries" specified above (e.g., 1 for bed, 2 for surface, 3 & 4 for sides).
    - 🧊 **Glen's Flow Law**: The rate factors and activation energies are set to standard ice values. Adjust the "Enhancement Factor" (E) to match soft/hard ice conditions (typically 1.0 to 3.0).
    - 📉 **MATC Expressions**: The `Bottom Surface` and `Top Surface` use Elmer's `MATC` interpreter. `tx` represents the local x-coordinate.
    - ⚖️ **Unit Scaling**: The default generation uses the MPa-a-m (MegaPascal-annum-meter) scaling from the reference. Uncheck the box to use standard SI units (kg, m, s, Pa).
    """)
