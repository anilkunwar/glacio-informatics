import streamlit as st
import os
import glob
import zipfile
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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
# PATH CONFIGURATION (Cloud-Safe)
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_NAME = "himalayan_glacier3d"
DATA_DIR = os.path.join(SCRIPT_DIR, DEFAULT_DIR_NAME)

COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens', 'Jet', 'Rainbow']

# =============================================
# DATA EXTRACTION & LOADING
# =============================================
@st.cache_data
def ensure_data_dir(directory: str) -> str:
    """Self-extract <directory>.zip on first load if the folder is missing."""
    if os.path.isdir(directory) and os.listdir(directory):
        return directory

    base = os.path.basename(directory)
    # Check root and 'data/' subfolder for the zip
    zip_paths = [
        os.path.join(SCRIPT_DIR, f"{base}.zip"),
        os.path.join(SCRIPT_DIR, "data", f"{base}.zip")
    ]

    for zpath in zip_paths:
        if not os.path.isfile(zpath):
            continue
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                tops = {n.split("/")[0] for n in names}
                if tops == {base}:          # zip already contains the folder
                    zf.extractall(SCRIPT_DIR)
                else:                       # bare files -> extract into the folder
                    os.makedirs(directory, exist_ok=True)
                    zf.extractall(directory)
            return directory
        except Exception as e:
            st.error(f"Failed to extract {zpath}: {e}")

    return directory

@st.cache_data
def load_glacier_data(directory: str, prefix: str):
    """Loads Elmer VTU/PVTU files. Handles serial, parallel, and subfolders."""
    if not os.path.isdir(directory):
        return None

    # Discover files. Prefer .pvtu (parallel master) if present, else .vtu.
    pvtu_files = sorted(set(
        glob.glob(os.path.join(directory, "**", f"{prefix}*.pvtu"), recursive=True)
    ))
    if pvtu_files:
        vtu_files = pvtu_files
    else:
        vtu_files = sorted(set(
            glob.glob(os.path.join(directory, "**", f"{prefix}*.vtu"), recursive=True)
        ))

    if not vtu_files:
        return None

    try:
        mesh0 = meshio.read(vtu_files[0])
    except Exception as e:
        st.error(f"Failed to read base VTU: {e}")
        return None

    points = mesh0.points.astype(np.float32)
    n_pts = len(points)

    triangles = None
    for cell_block in mesh0.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data.astype(np.int32)
            break

    fields, field_info = {}, {}
    for key, arr in mesh0.point_data.items():
        arr = arr.astype(np.float32)
        if arr.ndim == 1:
            field_info[key] = "scalar"
            fields[key] = np.full((len(vtu_files), n_pts), np.nan, dtype=np.float32)
            fields[key][0] = arr
        elif arr.ndim == 2:
            field_info[key] = "vector"
            fields[key] = np.full((len(vtu_files), n_pts, arr.shape[1]), np.nan, dtype=np.float32)
            fields[key][0] = arr
        else:
            # Tensor or higher-rank: flatten trailing dims
            flat = arr.reshape(arr.shape[0], -1)
            field_info[key] = f"tensor{arr.shape[1:]}"
            fields[key] = np.full((len(vtu_files), n_pts, flat.shape[1]), np.nan, dtype=np.float32)
            fields[key][0] = flat

    for t in range(1, len(vtu_files)):
        try:
            mesh = meshio.read(vtu_files[t])
            if mesh.points.shape[0] != n_pts:
                st.warning(
                    f"Timestep {t} has {mesh.points.shape[0]} points "
                    f"(expected {n_pts}); skipping."
                )
                continue
            for key in field_info.keys():
                if key in mesh.point_data:
                    arr = mesh.point_data[key].astype(np.float32)
                    if arr.ndim > 2:
                        arr = arr.reshape(arr.shape[0], -1)
                    fields[key][t] = arr
        except Exception as e:
            st.warning(f"Failed to read timestep {t} ({os.path.basename(vtu_files[t])}): {e}")

    return {
        "vtu_files": vtu_files, "n_timesteps": len(vtu_files), "points": points,
        "triangles": triangles, "has_surface": triangles is not None,
        "field_info": field_info, "fields": fields
    }

# =============================================
# STREAMLIT APP
# =============================================
def main():
    st.title("🏔️ Elmer Glacier 3D Diagnostic Viewer")
    st.caption("✅ Plotly (with Alphahull Fallback) + Matplotlib | Elmer FEM Output")

    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ Configuration")
    prefix = st.sidebar.text_input("File Prefix", value="Stokes_ELA400_3D_diagnostic")

    # Manual cache clear button
    if st.sidebar.button("🔄 Clear Cache & Reload"):
        load_glacier_data.clear()
        ensure_data_dir.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Rendering Controls")
    z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 100.0, 10.0, 1.0,
                               help="Exaggerate Z to see ice thickness.")
    max_points = st.sidebar.number_input("Max Points (Decimation)", 10000, 1000000, 150000, 10000)

    # --- Auto-Load Data ---
    data_dir = ensure_data_dir(DATA_DIR)
    with st.spinner("Loading glacier mesh data..."):
        data = load_glacier_data(data_dir, prefix)

    if data is None:
        st.error(f"No `.vtu`/`.pvtu` files found matching prefix `{prefix}`.")
        with st.expander("🔍 Debug Info", expanded=True):
            st.write(f"Looking in: `{data_dir}`")
            if not os.path.isdir(data_dir):
                st.write("❌ Directory does not exist. Ensure your data is committed "
                         "to GitHub or drop a `.zip` in the repo root.")
        return

    st.success(f"✅ Loaded {data['n_timesteps']} timestep(s) from {len(data['vtu_files'])} files.")

    # Debug: list files that were loaded
    with st.expander("📁 Files loaded"):
        st.write([os.path.basename(f) for f in data["vtu_files"]])

    # --- Guard: no fields at all ---
    available_fields = list(data['field_info'].keys())
    if not available_fields:
        st.error("No point-data fields found in the VTU/PVTU file(s).")
        return

    # --- Main Controls ---
    col1, col2, col3 = st.columns(3)
    with col1:
        default_field = "Velocity" if "Velocity" in available_fields else available_fields[0]
        field = st.selectbox(
            "Select Field", available_fields,
            index=available_fields.index(default_field)
        )
    with col2:
        if data['n_timesteps'] > 1:
            timestep = st.slider("Timestep", 0, data['n_timesteps'] - 1, 0)
        else:
            timestep = 0
            st.info("Only 1 timestep found — slider disabled.")
    with col3:
        colormap = st.selectbox("Colormap", COLORMAPS, index=0)

    # --- Data Processing ---
    pts = data['points'].copy()
    pts[:, 2] *= z_exag  # Apply Z-exaggeration

    kind = data['field_info'][field]
    raw = data['fields'][field][timestep]

    # Compute validity BEFORE substituting NaNs so we can mask correctly.
    if kind == "scalar":
        values = raw
        label = field
        valid_mask = ~np.isnan(values)
    else:
        magnitude = np.linalg.norm(raw, axis=1)
        values = magnitude
        label = f"{field} (Magnitude)"
        valid_mask = ~np.isnan(values)

    plot_pts = pts[valid_mask]
    plot_vals = values[valid_mask]

    # --- Triangle handling and decimation ---
    # If we have explicit triangles, we must remap them to the masked point list
    # and must NOT blindly decimate points (that would invalidate triangle indices).
    triangles = None
    if data['has_surface'] and data['triangles'] is not None:
        raw_tris = data['triangles']
        # Drop triangles that reference any invalid (NaN) vertex
        tri_keep = valid_mask[raw_tris].all(axis=1)
        raw_tris = raw_tris[tri_keep]
        # Remap original indices -> masked indices
        remap = np.full(len(valid_mask), -1, dtype=np.int64)
        remap[valid_mask] = np.arange(int(valid_mask.sum()))
        triangles = remap[raw_tris]

        if len(plot_pts) > max_points:
            st.info(
                f"Decimation disabled because explicit triangles are present "
                f"({len(plot_pts):,} points). Plotting full mesh."
            )
    else:
        # No triangles — safe to decimate freely (alphahull will be used).
        if len(plot_pts) > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(plot_pts), max_points, replace=False)
            plot_pts = plot_pts[idx]
            plot_vals = plot_vals[idx]

    if len(plot_vals) == 0:
        st.warning("No valid (non-NaN) values to plot for this field/timestep.")
        return

    cmin, cmax = float(np.min(plot_vals)), float(np.max(plot_vals))
    auto_scale = st.checkbox("Auto Color Scale", value=True)
    if not auto_scale:
        c1, c2 = st.columns(2)
        cmin = c1.number_input("Min Limit", value=cmin, format="%.3e")
        cmax = c2.number_input("Max Limit", value=cmax, format="%.3e")

    # =============================================
    # PLOTLY 3D (PRIMARY)
    # =============================================
    st.subheader(f"📈 {label} at Timestep {timestep + 1}")
    fig = go.Figure()

    if triangles is not None:
        fig.add_trace(go.Mesh3d(
            x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            intensity=plot_vals, colorscale=colormap, intensitymode='vertex',
            cmin=cmin, cmax=cmax, opacity=0.9,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
            hovertemplate=(
                f'<b>{label}:</b> %{{intensity:.3e}}<br>'
                'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            )
        ))
    else:
        # Alphahull fallback. Browser-safe cap.
        if len(plot_pts) > 50000:
            st.warning("⚠️ Auto-decimating to 50k points for Alphahull surface "
                       "generation to prevent browser crash.")
            rng = np.random.default_rng(0)
            idx = rng.choice(len(plot_pts), 50000, replace=False)
            plot_pts = plot_pts[idx]
            plot_vals = plot_vals[idx]
            # Recompute color limits for the decimated set
            cmin, cmax = float(np.min(plot_vals)), float(np.max(plot_vals))

        fig.add_trace(go.Mesh3d(
            x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2],
            alphahull=5,  # Generates surface mathematically!
            intensity=plot_vals, colorscale=colormap, intensitymode='vertex',
            cmin=cmin, cmax=cmax, opacity=0.9,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
            hovertemplate=(
                f'<b>{label}:</b> %{{intensity:.3e}}<br>'
                'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            )
        ))

    fig.update_layout(
        height=700, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.6)),
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m, exaggerated)")
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # =============================================
    # MATPLOTLIB FALLBACK / EXPORT VIEW
    # =============================================
    with st.expander("🖼️ 2D Top-Down Projection (Matplotlib)"):
        fig2, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(plot_pts[:, 0], plot_pts[:, 1], c=plot_vals,
                        s=2, cmap=colormap.lower())
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{label} (Top-Down)")
        plt.colorbar(sc, ax=ax, label=label)
        st.pyplot(fig2)
        plt.close(fig2)  # prevent memory leak across reruns

    # =============================================
    # STATISTICS
    # =============================================
    with st.expander("📊 Field Statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", f"{np.min(plot_vals):.3e}")
        c2.metric("Max", f"{np.max(plot_vals):.3e}")
        c3.metric("Mean", f"{np.mean(plot_vals):.3e}")
        c4.metric("Std Dev", f"{np.std(plot_vals):.3e}")

if __name__ == "__main__":
    main()
