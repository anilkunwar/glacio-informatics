import streamlit as st
import subprocess
import pathlib

# ──────────────────────────────────────────────
# 1. SIF Generator Function
# ──────────────────────────────────────────────
def generate_sif(p: dict) -> str:
    """Generates the Elmer SIF content based on the provided parameters dictionary."""
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

! ==============================================================================
! Material & Body Forces
! ==============================================================================
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

! ==============================================================================
! Boundary Conditions
! CRITICAL FIX: "include" and "Surface" lines REMOVED so Elmer uses your
! pre-deformed mesh.nodes file exactly as it is.
! ==============================================================================

! Bedrock (Bottom face) - Boundary Group {p['bedrock_target']}
Boundary Condition 1
  Name = "bedrock"
  Target Boundaries = {p['bedrock_target']}
  Velocity 1 = Real 0.0e0
  Velocity 2 = Real 0.0e0
  Velocity 3 = Real 0.0e0
End

! Lateral Walls (All 4 side walls) - Boundary Group {p['sides_target']}
Boundary Condition 2
  Name = "sides"
  Target Boundaries = {p['sides_target']}
  Velocity 1 = Real 0.0e0
  Velocity 2 = Real 0.0e0
End

! Surface (Top face) - Boundary Group {p['surface_target']}
Boundary Condition 3
  Name = "surface"
  Target Boundaries = {p['surface_target']}
End
"""

# ──────────────────────────────────────────────
# 2. Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="ElmerSolver Launcher", layout="wide")
st.title("🏔️ Himalayan Glacier 3D — ElmerSolver Launcher")

# --- Sidebar: Execution Settings ---
with st.sidebar:
    st.header("⚙️ Execution Settings")
    sif_name = st.text_input("SIF filename", value="himalayan_glacier3d.sif")
    work_dir = st.text_input(
        "Working directory",
        value=str(pathlib.Path.cwd()),
        help="Directory where the .sif is written and ElmerSolver is launched",
    )
    elmer_cmd = st.text_input("ElmerSolver command", value="ElmerSolver")
    
    st.markdown("---")
    st.info("Adjust the parameters in the main tabs to customize the generated SIF file. Refresh the page to reset to defaults.")

# --- Main Page: Parameter Tabs ---
tab_sim, tab_solvers, tab_material, tab_boundaries = st.tabs([
    "📊 Simulation & Output", 
    "⚙️ Solvers Configuration", 
    "🧊 Material Properties", 
    "🌍 Body Forces & Boundaries"
])

with tab_sim:
    st.subheader("Simulation & Output Settings")
    col1, col2 = st.columns(2)
    with col1:
        mesh_db = st.text_input("Mesh DB Name", value="himalayan_glacier3d")
        output_file = st.text_input("Output File", value="Stokes_ELA5000_3D_diagnostic.result")
    with col2:
        post_file = st.text_input("Post File", value="Stokes_ELA5000_3D_diagnostic.vtu")
        steady_state_max_iter = st.number_input("Steady State Max Iterations", min_value=1, value=1)
    output_intervals = st.number_input("Output Intervals", min_value=1, value=1)

with tab_solvers:
    st.subheader("Solvers Configuration (Solver 2: Navier-Stokes)")
    col1, col2 = st.columns(2)
    with col1:
        solver2_linear_max_iter = st.number_input("Linear System Max Iterations", min_value=1, value=5000)
        solver2_linear_tol = st.number_input("Linear System Convergence Tolerance", value=1.0e-6, format="%.1e")
        solver2_steady_tol = st.number_input("Steady State Convergence Tolerance", value=1.0e-5, format="%.1e")
    with col2:
        solver2_nonlinear_max_iter = st.number_input("Nonlinear System Max Iterations", min_value=1, value=50)
        solver2_nonlinear_tol = st.number_input("Nonlinear System Convergence Tolerance", value=1.0e-4, format="%.1e")

with tab_material:
    st.subheader("Material Properties (Material 1)")
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.text_input("Material Name", value="ice")
        density_formula = st.text_input("Density Formula (evaluated after '$')", value="910.0*1.0E-06*(31556926.0)^(-2.0)")
        viscosity = st.number_input("Viscosity", value=1.0, step=0.1)
        glen_exponent = st.number_input("Glen Exponent", value=3.0, step=0.1)
        critical_shear_rate = st.number_input("Critical Shear Rate", value=1.0e-10, format="%.1e")
        rate_factor_1 = st.number_input("Rate Factor 1", value=1.258e13, format="%.3e")
    with col2:
        rate_factor_2 = st.number_input("Rate Factor 2", value=6.046e28, format="%.3e")
        activation_energy_1 = st.number_input("Activation Energy 1 (J/mol)", value=60000.0, step=1000.0)
        activation_energy_2 = st.number_input("Activation Energy 2 (J/mol)", value=139000.0, step=1000.0)
        glen_enhancement_factor = st.number_input("Glen Enhancement Factor", value=1.0, step=0.1)
        limit_temperature = st.number_input("Limit Temperature (°C)", value=-10.0, step=1.0)
        constant_temperature = st.number_input("Constant Temperature (°C)", value=-3.0, step=1.0)

with tab_boundaries:
    st.subheader("Body Forces & Boundary Conditions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Body Force 1**")
        gravity_formula = st.text_input("Flow BodyForce 3 Formula (evaluated after '$')", value="-9.81 * (31556926.0)^(2.0)")
        st.markdown("**Boundary Targets**")
        bedrock_target = st.number_input("Bedrock Target Boundaries", min_value=1, value=2)
        sides_target = st.number_input("Lateral Walls Target Boundaries", min_value=1, value=1)
    with col2:
        surface_target = st.number_input("Surface Target Boundaries", min_value=1, value=3)
        st.markdown("<br>", unsafe_allow_html=True) # Visual spacer

# --- Assemble Parameters Dictionary ---
params = {
    "mesh_db": mesh_db,
    "output_file": output_file,
    "post_file": post_file,
    "steady_state_max_iter": steady_state_max_iter,
    "output_intervals": output_intervals,
    "solver2_linear_max_iter": solver2_linear_max_iter,
    "solver2_linear_tol": solver2_linear_tol,
    "solver2_steady_tol": solver2_steady_tol,
    "solver2_nonlinear_max_iter": solver2_nonlinear_max_iter,
    "solver2_nonlinear_tol": solver2_nonlinear_tol,
    "material_name": material_name,
    "density_formula": density_formula,
    "viscosity": viscosity,
    "glen_exponent": glen_exponent,
    "critical_shear_rate": critical_shear_rate,
    "rate_factor_1": rate_factor_1,
    "rate_factor_2": rate_factor_2,
    "activation_energy_1": activation_energy_1,
    "activation_energy_2": activation_energy_2,
    "glen_enhancement_factor": glen_enhancement_factor,
    "limit_temperature": limit_temperature,
    "constant_temperature": constant_temperature,
    "gravity_formula": gravity_formula,
    "bedrock_target": bedrock_target,
    "sides_target": sides_target,
    "surface_target": surface_target,
}

# --- Dynamic SIF Content ---
SIF_CONTENT = generate_sif(params)

# --- Show the embedded SIF in an expander ---
with st.expander("📄 View Generated SIF file", expanded=False):
    st.code(SIF_CONTENT, language="plaintext")

# --- Main Action Area ---
st.markdown("---")
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
    collected_lines = []

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
            # refresh the code block dynamically to show live output
            output_area.code("".join(collected_lines), language="plaintext")

        proc.wait()
        exit_code = proc.returncode

    except FileNotFoundError:
        st.error(
            f"`{elmer_cmd}` not found. Make sure ElmerSolver is on your PATH "
            "or provide the full absolute path in the sidebar."
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
