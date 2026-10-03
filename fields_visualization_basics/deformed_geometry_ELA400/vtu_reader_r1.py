import streamlit as st
import os
import glob
import numpy as np
import plotly.graph_objects as go
import meshio
import warnings

warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Elmer Glacier Viewer",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "deformed_geometry")
COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens', 'Jet', 'Rainbow']

# =============================================
# LAZY LOADING FUNCTIONS (Memory Efficient)
# =============================================
@st.cache_data
def get_vtu_files(directory: str, prefix: str):
    """Finds VTU/PVTU files without loading their contents into memory."""
    if not os.path.isdir(directory):
        return []
    
    pvtu_files = sorted(glob.glob(os.path.join(directory, "**", f"{prefix}*.pvtu"), recursive=True))
    if pvtu_files:
        return pvtu_files
    
    return sorted(glob.glob(os.path.join(directory, "**", f"{prefix}*.vtu"), recursive=True))

@st.cache_data
def load_single_timestep(vtu_file: str):
    """Loads ONLY a single VTU file on demand, saving massive amounts of RAM."""
    try:
        mesh = meshio.read(vtu_file)
    except Exception as e:
        st.error(f"Failed to read {vtu_file}: {e}")
        return None

    points = mesh.points.astype(np.float32)
    
    triangles = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data.astype(np.int32)
            break
            
    return {
        "points": points,
        "triangles": triangles,
        "point_data": mesh.point_data
    }

# =============================================
# MAIN APP
# =============================================
def main():
    st.title("🏔️ Elmer Glacier 3D Diagnostic Viewer")
    st.caption("✅ Memory-Efficient Lazy Loading | Powered by meshio & Plotly")

    st.sidebar.header("⚙️ Configuration")
    data_dir = st.sidebar.text_input("Results Directory", value=DEFAULT_DATA_DIR)
    prefix = st.sidebar.text_input("File Prefix", value="Stokes_ELA400_3D_diagnostic")

    # 1. Get file list (very fast, near-zero memory)
    vtu_files = get_vtu_files(data_dir, prefix)

    if not vtu_files:
        st.error(f"No `.vtu`/`.pvtu` files found in `{data_dir}` matching prefix `{prefix}`.")
        st.info("Ensure the data folder exists and contains the simulation output.")
        return

    st.success(f"✅ Found {len(vtu_files)} timestep file(s)!")

    with st.expander("📁 Files detected"):
        st.write([os.path.basename(f) for f in vtu_files])

    # 2. Load ONLY the first timestep to discover available fields
    with st.spinner("Inspecting file metadata..."):
        first_mesh = load_single_timestep(vtu_files[0])
    
    if first_mesh is None or not first_mesh["point_data"]:
        st.error("The loaded file contains no point-data fields. Nothing to visualize.")
        return

    available_fields = list(first_mesh["point_data"].keys())
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Visualization Controls")

    default_field = "Velocity" if "Velocity" in available_fields else available_fields[0]
    field = st.sidebar.selectbox("Select Field", available_fields, index=available_fields.index(default_field))

    if len(vtu_files) > 1:
        timestep_idx = st.sidebar.slider("Timestep", 0, len(vtu_files) - 1, 0)
    else:
        timestep_idx = 0
        st.sidebar.info("Only 1 timestep found — slider disabled.")

    colormap = st.sidebar.selectbox("Colormap", COLORMAPS, index=0)
    z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 50.0, 10.0, 1.0,
                               help="Glaciers are thin; exaggerate Z to see thickness.")
    
    max_points = st.sidebar.number_input("Max Points (Decimation)",
                                         min_value=10000, max_value=2000000,
                                         value=150000, step=10000)

    # 3. LAZY LOAD the selected timestep (Only consumes memory for ONE timestep)
    with st.spinner(f"Loading timestep {timestep_idx + 1}..."):
        current_mesh = load_single_timestep(vtu_files[timestep_idx])

    if current_mesh is None:
        st.error("Failed to load the selected timestep.")
        return

    pts = current_mesh["points"].copy()
    pts[:, 2] *= z_exag
    triangles = current_mesh["triangles"]
    
    raw = current_mesh["point_data"].get(field)
    if raw is None:
        st.warning(f"Field '{field}' not found in this timestep.")
        return

    # Process field data dynamically
    if raw.ndim == 1:
        values = np.where(np.isnan(raw), 0, raw.astype(np.float32))
        label = field
    else:
        magnitude = np.linalg.norm(raw, axis=1)
        values = np.where(np.isnan(magnitude), 0, magnitude.astype(np.float32))
        label = f"{field} (Magnitude)"

    valid_mask = ~np.isnan(values)
    plot_pts = pts[valid_mask]
    plot_vals = values[valid_mask]

    # Decimation logic
    if triangles is None and len(plot_pts) > max_points:
        indices = np.random.choice(len(plot_pts), max_points, replace=False)
        render_pts = plot_pts[indices]
        render_vals = plot_vals[indices]
        st.sidebar.warning(f"⚠️ Decimated to {max_points} points for browser performance.")
    else:
        if triangles is not None and len(pts) > max_points:
            st.sidebar.info("ℹ️ Explicit mesh detected. Decimation disabled to preserve surface topology.")
        render_pts = pts
        render_vals = values

    st.markdown(f"### 📈 {label} at Timestep {timestep_idx + 1}")

    cmin = float(np.nanmin(render_vals))
    cmax = float(np.nanmax(render_vals))

    col_a, col_b = st.columns(2)
    with col_a:
        auto_scale = st.checkbox("Auto Color Scale", value=True)
    with col_b:
        if not auto_scale:
            cmin = st.number_input("Min Limit", value=cmin, format="%.3e")
            cmax = st.number_input("Max Limit", value=cmax, format="%.3e")
        else:
            cmin, cmax = None, None

    fig = go.Figure()

    if triangles is not None:
        fig.add_trace(go.Mesh3d(
            x=render_pts[:, 0], y=render_pts[:, 1], z=render_pts[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            intensity=render_vals,
            colorscale=colormap,
            intensitymode='vertex',
            cmin=cmin, cmax=cmax,
            opacity=0.9,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
            hovertemplate=f'<b>{label}:</b> %{{intensity:.3e}}<br><b>X:</b> %{{x:.2f}}<br><b>Y:</b> %{{y:.2f}}<br><b>Z:</b> %{{z:.2f}}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Mesh3d(
            x=render_pts[:, 0], y=render_pts[:, 1], z=render_pts[:, 2],
            alphahull=5,
            intensity=render_vals,
            colorscale=colormap,
            intensitymode='vertex',
            cmin=cmin, cmax=cmax,
            opacity=0.85,
            lighting=dict(ambient=0.8, diffuse=0.6, specular=0.5, roughness=0.5),
            hovertemplate=f'<b>{label}:</b> %{{intensity:.3e}}<br><b>X:</b> %{{x:.2f}}<br><b>Y:</b> %{{y:.2f}}<br><b>Z:</b> %{{z:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.6)),
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m, exaggerated)")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Field Statistics"):
        clean = render_vals[~np.isnan(render_vals)]
        if len(clean) > 0:
            st.write({
                "Min": f"{float(np.nanmin(clean)):.3e}",
                "Max": f"{float(np.nanmax(clean)):.3e}",
                "Mean": f"{float(np.nanmean(clean)):.3e}",
                "Std Dev": f"{float(np.nanstd(clean)):.3e}"
            })
        else:
            st.write("No valid data points.")

if __name__ == "__main__":
    main()
