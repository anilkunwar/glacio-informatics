import streamlit as st
import subprocess
import pathlib
import os

# ──────────────────────────────────────────────
# 1. SIF content embedded as a Python string
# ──────────────────────────────────────────────
SIF_CONTENT = r"""!echo on
Header
  !CHECK KEYWORDS Warn
  Mesh DB "." "himalayan_glacier3d"
  Include Path ""
  Results Directory ""
End

Simulation
  Max Output Level = 4
  Coordinate System = "Cartesian 3D"
  Coordinate Mapping(3) = 1 2 3
  Simulation Type = "Steady"
  Steady State Max Iterations = 1
  Output Intervals = 1
  Output File = "Stokes_ELA5000_3D_diagnostic.result"
  Post File = "Stokes_ELA5000_3D_diagnostic.vtu"
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
  ! UPDATED: Only 2 solvers now. Mapper is gone.
  Active Solvers(1) =  2
End

Initial Condition 1
  Velocity 1 = 0.0
  Velocity 2 = 0.0
  Velocity 3 = 0.0
  Pressure = 0.0
  Depth = Real 0.0
End

! ==============================================================================
! Solver 1: Flow Depth for postprocessing (Renumbered from Solver 2)
! CRITICAL FIX: Increased tolerance to accept the pre-deformed Y-variation
! ==============================================================================
Solver 1
  Exec Solver = "Never"
  Equation = "HeightDepth"
  Procedure = "StructuredProjectToPlane" "StructuredProjectToPlane"
  Active Coordinate = Integer 3
  Operator 1 = depth
  Operator 2 = height
  Dot Product Tolerance = Real 0.9
End

! ==============================================================================
! Solver 2: The central 3D Stokes solver (Renumbered from Solver 3)
! ==============================================================================
Solver 2
  Equation = "Navier-Stokes"
  Optimize Bandwidth = Logical True
  Linear System Solver = Direct
  Linear System Direct Method = "UMFPACK"
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0E-06
  Linear System Abort Not Converged = False
  Linear System Preconditioning = "ILU1"
  Linear System Residual Output = 1
  Flow Model = Stokes
  Steady State Convergence Tolerance = 1.0E-05
  Stabilization Method = Stabilized
  Nonlinear System Convergence Tolerance = 1.0E-04
  Nonlinear System Convergence Measure = Solution
  Nonlinear System Max Iterations = 50
  Nonlinear System Newton After Iterations = 3
  Nonlinear System Newton After Tolerance = 1.0E-01
  Exported Variable 1 = -dofs 3 "Mesh Velocity"
End

! ==============================================================================
! Material & Body Forces
! ==============================================================================
Material 1
  Name = "ice"
  Density = Real $910.0*1.0E-06*(31556926.0)^(-2.0)
  Viscosity Model = String "Glen"
  Viscosity = Real 1.0
  Glen Exponent = Real 3.0
  Critical Shear Rate = Real 1.0e-10
  Rate Factor 1 = Real 1.258e13
  Rate Factor 2 = Real 6.046e28
  Activation Energy 1 = Real 60e3
  Activation Energy 2 = Real 139e3
  Glen Enhancement Factor = Real 1.0
  Limit Temperature = Real -10.0
  Constant Temperature = Real -3.0
End

Body Force 1
  Name = "BodyForce1"
  Heat Source = 1
  Flow BodyForce 1 = Real 0.0
  Flow BodyForce 2 = Real 0.0
  Flow BodyForce 3 = Real $-9.81 * (31556926.0)^(2.0)
End

! ==============================================================================
! Boundary Conditions
! CRITICAL FIX: "include" and "Surface" lines REMOVED so Elmer uses your
! pre-deformed mesh.nodes file exactly as it is.
! ==============================================================================

! Bedrock (Bottom face) - Boundary Group 2
Boundary Condition 1
  Name = "bedrock"
  Target Boundaries = 2
  Velocity 1 = Real 0.0e0
  Velocity 2 = Real 0.0e0
  Velocity 3 = Real 0.0e0
End

! Lateral Walls (All 4 side walls) - Boundary Group 1
Boundary Condition 2
  Name = "sides"
  Target Boundaries = 1
  Velocity 1 = Real 0.0e0
  Velocity 2 = Real 0.0e0
End

! Surface (Top face) - Boundary Group 3
Boundary Condition 3
  Name = "surface"
  Target Boundaries = 3
End
"""

# ──────────────────────────────────────────────
# 2. Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="ElmerSolver Launcher", layout="wide")
st.title("🏔️ Himalayan Glacier 3D — ElmerSolver Launcher")

# --- sidebar: configurable paths ---
with st.sidebar:
    st.header("Settings")
    sif_name = st.text_input("SIF filename", value="himalayan_glacier3d.sif")
    work_dir  = st.text_input(
        "Working directory",
        value=str(pathlib.Path.cwd()),
        help="Directory where the .sif is written and ElmerSolver is launched",
    )
    elmer_cmd = st.text_input("ElmerSolver command", value="ElmerSolver")

# --- show the embedded SIF in an expander ---
with st.expander("📄 View embedded SIF file", expanded=False):
    st.code(SIF_CONTENT, language="plaintext")

# --- main action area ---
col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🚀 Run ElmerSolver", type="primary")
with col2:
    st.markdown(
        "Click the button to write the SIF file and launch "
        "`ElmerSolver <filename.sif>` in the working directory."
    )

# ──────────────────────────────────────────────
# 3. Write SIF + execute ElmerSolver
# ──────────────────────────────────────────────
if run_btn:
    work_path = pathlib.Path(work_dir)
    sif_path  = work_path / sif_name

    # --- write the SIF ---
    try:
        sif_path.write_text(SIF_CONTENT)
        st.success(f"SIF written to `{sif_path}`")
    except Exception as e:
        st.error(f"Failed to write SIF: {e}")
        st.stop()

    # --- run ElmerSolver, stream stdout/stderr live ---
    cmd = [elmer_cmd, sif_name]
    st.info(f"Running: `{' '.join(cmd)}`  in  `{work_dir}`")

    output_area = st.empty()
    collected_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,          # line-buffered
        )

        for line in proc.stdout:
            collected_lines.append(line)
            # refresh the code block every few lines to avoid flicker
            output_area.code("".join(collected_lines), language="plaintext")

        proc.wait()
        exit_code = proc.returncode

    except FileNotFoundError:
        st.error(
            f"`{elmer_cmd}` not found. Make sure ElmerSolver is on your PATH "
            "or provide the full path in the sidebar."
        )
        st.stop()

    # --- final status ---
    if exit_code == 0:
        st.success(f"✅ ElmerSolver finished successfully (exit code 0)")
    else:
        st.error(f"❌ ElmerSolver exited with code {exit_code}")

    # --- offer a download button for the SIF ---
    st.download_button(
        "⬇️ Download SIF file",
        data=SIF_CONTENT,
        file_name=sif_name,
        mime="text/plain",
    )
