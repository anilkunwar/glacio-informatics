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
# CONSTANTS & HELPERS
# =============================================
COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens', 'Jet', 'Rainbow']

@st.cache_data
def load_elmer_vtu_data(directory: str, prefix: str = "Stokes_ELA400_3D_diagnostic"):
    """
    Loads Elmer VTU files from the specified directory.
    Returns a dictionary with mesh data and field information.
    """
    search_pattern = os.path.join(directory, f"{prefix}*.vtu")
    vtu_files = sorted(glob.glob(search_pattern))
    
    if not vtu_files:
        # Fallback: try any .vtu in the directory
        vtu_files = sorted(glob.glob(os.path.join(directory, "*.vtu")))
        
    if not vtu_files:
        return None

    # Load the first file to get mesh structure and field names
    try:
        mesh0 = meshio.read(vtu_files[0])
    except Exception as e:
        st.error(f"Failed to read VTU file: {e}")
        return None

    points = mesh0.points.astype(np.float32)
    n_pts = len(points)
    
    # Look for surface triangles first. Elmer 3D usually outputs tetrahedra.
    triangles = None
    for cell_block in mesh0.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data.astype(np.int32)
            break
    
    has_surface = triangles is not None

    # Extract fields from the first timestep
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

    # Load remaining timesteps (if it's a time-series)
    for t in range(1, len(vtu_files)):
        try:
            mesh = meshio.read(vtu_files[t])
            for key in field_info.keys():
                if key in mesh.point_data:
                    fields[key][t] = mesh.point_data[key].astype(np.float32)
        except Exception as e:
            st.warning(f"Could not load timestep {t}: {e}")

    return {
        "vtu_files": vtu_files,
        "n_timesteps": len(vtu_files),
        "points": points,
        "triangles": triangles,
        "has_surface": has_surface,
        "field_info": field_info,
        "fields": fields
    }

# =============================================
# MAIN APP
# =============================================
def main():
    st.markdown("<h1 style='text-align: center;'>🏔️ Elmer Glacier 3D Diagnostic Viewer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Visualize Stokes flow, velocity, pressure, and depth from Elmer FEM output.</p>", unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Defaults pulled directly from your .sif file
    default_dir = "himalayan_glacier3d"
    data_dir = st.sidebar.text_input("Results Directory", value=default_dir)
    prefix = st.sidebar.text_input("File Prefix", value="Stokes_ELA400_3D_diagnostic")
    
    if st.sidebar.button("🔄 Load Data", type="primary"):
        st.session_state.data = load_elmer_vtu_data(data_dir, prefix)
        st.session_state.loaded = True

    if 'loaded' not in st.session_state or not st.session_state.loaded:
        st.info("👈 Please configure the directory and click **Load Data** in the sidebar.")
        return

    data = st.session_state.data
    if data is None:
        st.error(f"No `.vtu` files found in `{data_dir}` matching prefix `{prefix}`.")
        st.info("💡 Tip: Ensure your Elmer `.sif` has `Output Format = vtu` in the `ResultOutputSolver` block.")
        return

    st.success(f"✅ Loaded {data['n_timesteps']} timestep(s) successfully!")

    # -----------------------------------------
    # CONTROLS
    # -----------------------------------------
    st.markdown("### 🎛️ Visualization Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        available_fields = list(data['field_info'].keys())
        default_field = "Velocity" if "Velocity" in available_fields else (available_fields[0] if available_fields else None)
        field = st.selectbox("Select Field", available_fields, index=available_fields.index(default_field) if default_field else 0)
    
    with col2:
        timestep = st.slider("Timestep", 0, data['n_timesteps'] - 1, 0)
    
    with col3:
        colormap = st.selectbox("Colormap", COLORMAPS, index=0)
        
    with col4:
        z_exag = st.slider("Z Exaggeration", 1.0, 100.0, 10.0, 1.0, help="Glaciers are thin; exaggerate Z to see thickness.")

    # Performance decimation
    max_points = st.sidebar.number_input("Max Points (Decimation)", min_value=10000, max_value=1000000, value=150000, step=10000)
    point_size = st.sidebar.slider("Point Size (if cloud)", 1, 15, 3)

    # -----------------------------------------
    # DATA PROCESSING
    # -----------------------------------------
    pts = data['points'].copy()
    pts[:, 2] *= z_exag  # Apply Z exaggeration
    
    kind = data['field_info'][field]
    raw = data['fields'][field][timestep]
    
    if kind == "scalar":
        values = np.where(np.isnan(raw), 0, raw)
        label = field
    else:
        magnitude = np.linalg.norm(raw, axis=1)
        values = np.where(np.isnan(magnitude), 0, magnitude)
        label = f"{field} (Magnitude)"

    # Clean NaNs for plotting
    valid_mask = ~np.isnan(values)
    plot_pts = pts[valid_mask]
    plot_vals = values[valid_mask]

    # Decimate for performance if needed
    if len(plot_pts) > max_points:
        indices = np.random.choice(len(plot_pts), max_points, replace=False)
        plot_pts = plot_pts[indices]
        plot_vals = plot_vals[indices]
        st.sidebar.warning(f"⚠️ Decimated to {max_points} points for browser performance.")

    # -----------------------------------------
    # PLOTTING
    # -----------------------------------------
    st.markdown(f"### 📈 {label} at Timestep {timestep + 1}")
    
    cmin = float(np.min(plot_vals))
    cmax = float(np.max(plot_vals))
    
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
    
    if data['has_surface']:
        st.info("Rendering surface mesh.")
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=data['triangles'][:, 0], j=data['triangles'][:, 1], k=data['triangles'][:, 2],
            intensity=values,
            colorscale=colormap,
            intensitymode='vertex',
            cmin=cmin, cmax=cmax,
            opacity=0.9,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
            hovertemplate=f'<b>{label}:</b> %{{intensity:.3e}}<br><b>X:</b> %{{x:.2f}}<br><b>Y:</b> %{{y:.2f}}<br><b>Z:</b> %{{z:.2f}}<extra></extra>'
        ))
    else:
        st.info("Rendering as 3D Point Cloud (Volume tetrahedra detected).")
        fig.add_trace(go.Scatter3d(
            x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=plot_vals,
                colorscale=colormap,
                cmin=cmin, cmax=cmax,
                opacity=0.85,
                line=dict(width=0)
            ),
            hovertemplate=f'<b>{label}:</b> %{{marker.color:.3e}}<br><b>X:</b> %{{x:.2f}}<br><b>Y:</b> %{{y:.2f}}<br><b>Z:</b> %{{z:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="data", 
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.6)), # Angled top-down view ideal for glaciers
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m, exaggerated)")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # STATISTICS
    # -----------------------------------------
    st.markdown("### 📊 Field Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min", f"{np.min(plot_vals):.3e}")
    col2.metric("Max", f"{np.max(plot_vals):.3e}")
    col3.metric("Mean", f"{np.mean(plot_vals):.3e}")
    col4.metric("Std Dev", f"{np.std(plot_vals):.3e}")

if __name__ == "__main__":
    main()
