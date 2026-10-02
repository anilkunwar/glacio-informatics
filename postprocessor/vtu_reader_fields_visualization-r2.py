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
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "himalayan_glacier3d")

COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens', 'Jet', 'Rainbow']

# =============================================
# LOAD DATA
# =============================================
@st.cache_data
def load_elmer_vtu_data(directory: str, prefix: str = "Stokes_ELA400_3D_diagnostic"):
    """Loads Elmer VTU/PVTU files from the specified directory."""
    if not os.path.isdir(directory):
        return None

    # FIX #1: recursive discovery, .pvtu takes priority over .vtu pieces.
    # Rationale (from diagnostic note):
    #   - .pvtu masters live in subfolders (e.g. results/) for parallel runs.
    #   - A stray serial .vtu used to shadow several .pvtu files.
    #   - Parallel piece files (prefix_t0001_0.vtu, _1.vtu, ...) used to be
    #     globbed as separate "timesteps".
    pvtu_files = sorted(glob.glob(os.path.join(directory, "**", f"{prefix}*.pvtu"), recursive=True))
    if pvtu_files:
        vtu_files = pvtu_files
    else:
        vtu_files = sorted(glob.glob(os.path.join(directory, "**", f"{prefix}*.vtu"), recursive=True))

    if not vtu_files:
        return None

    try:
        mesh0 = meshio.read(vtu_files[0])
    except Exception as e:
        st.error(f"Failed to read VTU file: {e}")
        return None

    points = mesh0.points.astype(np.float32)
    n_pts = len(points)

    # Extract surface triangles if they exist
    triangles = None
    for cell_block in mesh0.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data.astype(np.int32)
            break

    fields = {}
    field_info = {}

    for key, arr in mesh0.point_data.items():
        arr = arr.astype(np.float32)
        if arr.ndim == 1:
            field_info[key] = "scalar"
            fields[key] = np.full((len(vtu_files), n_pts), np.nan, dtype=np.float32)
            fields[key][0] = arr
        else:
            field_info[key] = "vector"
            fields[key] = np.full((len(vtu_files), n_pts, arr.shape[1]), np.nan, dtype=np.float32)
            fields[key][0] = arr

    # Load remaining timesteps with a progress bar
    progress_text = "Loading timesteps..."
    my_bar = st.progress(0, text=progress_text)

    for t in range(1, len(vtu_files)):
        try:
            mesh = meshio.read(vtu_files[t])
            for key in field_info.keys():
                if key in mesh.point_data:
                    fields[key][t] = mesh.point_data[key].astype(np.float32)
        except Exception:
            pass
        my_bar.progress((t + 1) / len(vtu_files), text=f"{progress_text} ({t+1}/{len(vtu_files)})")

    my_bar.empty()

    return {
        "vtu_files": vtu_files,
        "n_timesteps": len(vtu_files),
        "points": points,
        "triangles": triangles,
        "fields": fields,
        "field_info": field_info
    }

# =============================================
# MAIN APP
# =============================================
def main():
    st.title("🏔️ Elmer Glacier 3D Diagnostic Viewer")
    st.caption("✅ Plotly Mesh3d with alphahull fallback | Powered by meshio")

    st.sidebar.header("⚙️ Configuration")

    data_dir = st.sidebar.text_input("Results Directory", value=DEFAULT_DATA_DIR)
    prefix = st.sidebar.text_input("File Prefix", value="Stokes_ELA400_3D_diagnostic")

    with st.spinner("Loading simulation data..."):
        data = load_elmer_vtu_data(data_dir, prefix)

    if data is None:
        st.error(f"No `.vtu`/`.pvtu` files found in `{data_dir}` matching prefix `{prefix}`.")
        st.info("Ensure the data folder exists in your repository and is tracked by Git (or Git LFS).")
        return

    st.success(f"✅ Loaded {data['n_timesteps']} timestep(s) successfully!")

    # FIX #4: show exactly what got matched, so silent glob failures are visible.
    with st.expander("📁 Files loaded"):
        st.write([os.path.basename(f) for f in data['vtu_files']])

    # FIX #3: bail out cleanly if the file has no point_data.
    # st.selectbox([]) raises on some Streamlit versions and is useless anyway.
    available_fields = list(data['field_info'].keys())
    if not available_fields:
        st.error("The loaded file contains no point-data fields. Nothing to visualize.")
        return

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Visualization Controls")

    default_field = "Velocity" if "Velocity" in available_fields else available_fields[0]
    field = st.sidebar.selectbox(
        "Select Field",
        available_fields,
        index=available_fields.index(default_field),
    )

    # FIX #2: st.slider(min=0, max=0) raises StreamlitInvalidMinMaxError.
    # n_timesteps == 1 is the *expected* case for a diagnostic solve.
    if data['n_timesteps'] > 1:
        timestep = st.sidebar.slider("Timestep", 0, data['n_timesteps'] - 1, 0)
    else:
        timestep = 0
        st.sidebar.info("Only 1 timestep found — slider disabled.")

    colormap = st.sidebar.selectbox("Colormap", COLORMAPS, index=0)
    z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 50.0, 10.0, 1.0,
                               help="Glaciers are thin; exaggerate Z to see thickness.")

    max_points = st.sidebar.number_input("Max Points (Decimation)",
                                         min_value=10000, max_value=1000000,
                                         value=150000, step=10000)

    pts = data['points'].copy()
    pts[:, 2] *= z_exag

    kind = data['field_info'][field]
    raw = data['fields'][field][timestep]

    if kind == "scalar":
        values = np.where(np.isnan(raw), 0, raw)
        label = field
    else:
        magnitude = np.linalg.norm(raw, axis=1)
        values = np.where(np.isnan(magnitude), 0, magnitude)
        label = f"{field} (Magnitude)"

    valid_mask = ~np.isnan(values)
    plot_pts = pts[valid_mask]
    plot_vals = values[valid_mask]

    if len(plot_pts) > max_points and data['triangles'] is None:
        indices = np.random.choice(len(plot_pts), max_points, replace=False)
        render_pts = plot_pts[indices]
        render_vals = plot_vals[indices]
        st.sidebar.warning(f"⚠️ Decimated to {max_points} points for browser performance.")
    elif data['triangles'] is not None and len(pts) > max_points:
        st.sidebar.info("ℹ️ Explicit mesh detected. Decimation disabled to preserve surface topology.")
        render_pts = pts
        render_vals = values
    else:
        render_pts = pts
        render_vals = values

    st.markdown(f"### 📈 {label} at Timestep {timestep + 1}")

    cmin = float(np.min(render_vals))
    cmax = float(np.max(render_vals))

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

    if data['triangles'] is not None:
        st.info("Rendering explicit surface mesh.")
        fig.add_trace(go.Mesh3d(
            x=render_pts[:, 0], y=render_pts[:, 1], z=render_pts[:, 2],
            i=data['triangles'][:, 0], j=data['triangles'][:, 1], k=data['triangles'][:, 2],
            intensity=render_vals,
            colorscale=colormap,
            intensitymode='vertex',
            cmin=cmin, cmax=cmax,
            opacity=0.9,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
            hovertemplate=f'<b>{label}:</b> %{{intensity:.3e}}<br><b>X:</b> %{{x:.2f}}<br><b>Y:</b> %{{y:.2f}}<br><b>Z:</b> %{{z:.2f}}<extra></extra>'
        ))
    else:
        st.info("No explicit surface triangles found. Rendering surface using Plotly alphahull.")
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
                "Min": f"{float(clean.min()):.3e}",
                "Max": f"{float(clean.max()):.3e}",
                "Mean": f"{float(clean.mean()):.3e}",
                "Std Dev": f"{float(clean.std()):.3e}"
            })
        else:
            st.write("No valid data points.")

if __name__ == "__main__":
    main()
