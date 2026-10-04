import streamlit as st
import os
import glob
import tempfile
import zipfile
import numpy as np
import plotly.graph_objects as go
import meshio
import warnings
import shutil

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
# MESH PROCESSING
# =============================================
def extract_surface_triangles(mesh):
    """Extracts outer surface triangles from volume meshes (hex/tet) for Plotly."""
    for cell_block in mesh.cells:
        ctype = cell_block.type
        data = cell_block.data
        
        if ctype == "triangle":
            return data.astype(np.int32)
            
        if ctype == "tetra" and data.shape[1] == 4:
            face_defs = [(0,1,2), (0,1,3), (1,2,3), (0,2,3)]
            face_dict = {}
            for tet in data:
                for f in face_defs:
                    ordered = tuple(tet[list(f)])
                    sorted_f = tuple(sorted(ordered))
                    if sorted_f in face_dict: face_dict[sorted_f]['count'] += 1
                    else: face_dict[sorted_f] = {'count': 1, 'ordered': ordered}
            return np.array([v['ordered'] for v in face_dict.values() if v['count'] == 1], dtype=np.int32)
            
        if ctype == "hexahedron" and data.shape[1] == 8:
            face_defs = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
            face_dict = {}
            for hx in data:
                for f in face_defs:
                    ordered = tuple(hx[list(f)])
                    sorted_f = tuple(sorted(ordered))
                    if sorted_f in face_dict: face_dict[sorted_f]['count'] += 1
                    else: face_dict[sorted_f] = {'count': 1, 'ordered': ordered}
            tris = []
            for v in face_dict.values():
                if v['count'] == 1:
                    f = v['ordered']
                    tris.append((f[0], f[1], f[2]))
                    tris.append((f[0], f[2], f[3]))
            return np.array(tris, dtype=np.int32)
    return None

# =============================================
# HELPER: FIELD LABEL WITH UNITS
# =============================================
def get_field_label(field_name):
    """Appends physical units based on field name."""
    name_lower = field_name.lower()
    if 'velocity' in name_lower or 'vel' in name_lower:
        return f"{field_name} (m/yr)"
    elif 'pressure' in name_lower or 'pres' in name_lower:
        return f"{field_name} (MPa)"
    return field_name

# =============================================
# LOAD DATA
# =============================================
@st.cache_data
def load_elmer_vtu_data(directory: str, prefix: str = "Stokes_ELA5000_3D_diagnostic_t0001"):
    if not os.path.isdir(directory):
        return None

    # If prefix is empty (from uploads), this becomes "**/*.vtu" which matches all
    pvtu_files = sorted(glob.glob(os.path.join(directory, "**", f"{prefix}*.pvtu"), recursive=True))
    vtu_files = pvtu_files if pvtu_files else sorted(glob.glob(os.path.join(directory, "**", f"{prefix}*.vtu"), recursive=True))

    if not vtu_files:
        return None

    try:
        mesh0 = meshio.read(vtu_files[0])
    except Exception as e:
        st.error(f"Failed to read VTU file: {e}")
        return None

    points = mesh0.points.astype(np.float32)
    n_pts = len(points)

    triangles = extract_surface_triangles(mesh0)

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
    st.caption("✅ Hexahedron/Tetra Surface Extraction | Powered by meshio & Plotly")

    st.sidebar.header("⚙️ Configuration")
    source_choice = st.sidebar.radio("Data Source", ["Default Directory", "Upload Files"])
    
    temp_dir = None
    try:
        if source_choice == "Default Directory":
            data_dir = st.sidebar.text_input("Results Directory", value=DEFAULT_DATA_DIR)
            prefix = st.sidebar.text_input("File Prefix", value="Stokes_ELA5000_3D_diagnostic_t0001")
        else:
            uploaded_files = st.sidebar.file_uploader(
                "Upload .vtu, .pvtu, or .zip", 
                accept_multiple_files=True, 
                type=["vtu", "pvtu", "zip"]
            )
            if not uploaded_files:
                st.info("👈 Please upload files to begin.")
                st.stop()
            
            with st.spinner("Preparing uploaded files..."):
                temp_dir = tempfile.mkdtemp()
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith(".zip"):
                        zip_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                    else:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                data_dir = temp_dir
                prefix = "" # Match all files in the temp directory

        with st.spinner("Loading simulation data..."):
            data = load_elmer_vtu_data(data_dir, prefix)

        if data is None:
            st.error(f"No `.vtu`/`.pvtu` files found matching the criteria.")
            return

        st.success(f"✅ Loaded {data['n_timesteps']} timestep(s) successfully!")

        with st.expander("📁 Files loaded"):
            st.write([os.path.basename(f) for f in data['vtu_files']])

        available_fields = list(data['field_info'].keys())
        if not available_fields:
            st.error("The loaded file contains no point-data fields. Nothing to visualize.")
            return

        st.sidebar.markdown("---")
        st.sidebar.header("🎛️ Visualization Controls")

        default_field = "Velocity" if "Velocity" in available_fields else available_fields[0]
        field = st.sidebar.selectbox("Select Field", available_fields, index=available_fields.index(default_field))

        if data['n_timesteps'] > 1:
            timestep = st.sidebar.slider("Timestep", 0, data['n_timesteps'] - 1, 0)
        else:
            timestep = 0
            st.sidebar.info("Only 1 timestep found — slider disabled.")

        colormap = st.sidebar.selectbox("Colormap", COLORMAPS, index=0)
        z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 50.0, 10.0, 1.0, help="Glaciers are thin; exaggerate Z.")
        max_points = st.sidebar.number_input("Max Points (Decimation)", min_value=10000, max_value=2000000, value=150000, step=10000)

        st.sidebar.markdown("---")
        st.sidebar.header("🎨 Figure Customization")
        
        opacity = st.sidebar.slider("Mesh Opacity", 0.1, 1.0, 0.9, 0.05)
        
        st.sidebar.subheader("Text & Labels")
        font_size = st.sidebar.slider("General Font Size", 8, 24, 12)
        title_size = st.sidebar.slider("Title Size", 12, 32, 16)
        
        st.sidebar.subheader("Axes & Ticks")
        axis_line_width = st.sidebar.slider("Axis Line Width", 1, 5, 2)
        tick_width = st.sidebar.slider("Tick Width", 1, 5, 2)
        tick_len = st.sidebar.slider("Tick Length", 2, 10, 5)
        
        st.sidebar.subheader("Colorbar")
        cb_thickness = st.sidebar.slider("Colorbar Thickness", 10, 50, 20)
        cb_len = st.sidebar.slider("Colorbar Length", 0.2, 1.0, 0.8)
        cb_title_size = st.sidebar.slider("Colorbar Title Size", 8, 20, 12)
        cb_tick_size = st.sidebar.slider("Colorbar Tick Size", 8, 20, 10)

        pts = data['points'].copy()
        pts[:, 2] *= z_exag

        kind = data['field_info'][field]
        raw = data['fields'][field][timestep]

        # Robust flattening to prevent NumPy broadcasting errors
        if kind == "scalar":
            flat_raw = raw.flatten()
            values = np.where(np.isnan(flat_raw), 0.0, flat_raw)
        else:
            magnitude = np.linalg.norm(raw, axis=-1)
            values = np.where(np.isnan(magnitude), 0.0, magnitude)
            
        label = get_field_label(field)

        valid_mask = ~np.isnan(values)
        valid_mask = np.asarray(valid_mask).flatten()
        
        # Safety check for mask length
        if len(valid_mask) != len(pts):
            st.error("⚠️ Shape mismatch between points and field data.")
            st.stop()
            
        plot_pts = pts[valid_mask]
        plot_vals = values[valid_mask]

        # Decimation logic
        if len(plot_pts) > max_points and data['triangles'] is None:
            indices = np.random.choice(len(plot_pts), max_points, replace=False)
            render_pts = plot_pts[indices]
            render_vals = plot_vals[indices]
            st.sidebar.warning(f"⚠️ Decimated to {max_points} points for browser performance.")
        else:
            if data['triangles'] is not None and len(pts) > max_points:
                st.sidebar.info("ℹ️ Explicit mesh detected. Decimation disabled to preserve topology.")
            render_pts = pts
            render_vals = values

        st.markdown(f"### 📈 {label} at Timestep {timestep + 1}")

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

        mesh_kwargs = dict(
            x=render_pts[:, 0], y=render_pts[:, 1], z=render_pts[:, 2],
            intensity=render_vals,
            colorscale=colormap,
            intensitymode='vertex',
            cmin=cmin, cmax=cmax,
            opacity=opacity, # Applied here
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
            hovertemplate=f'<b>{label}:</b> %{{intensity:.3e}}<br><b>X:</b> %{{x:.2f}}<br><b>Y:</b> %{{y:.2f}}<br><b>Z:</b> %{{z:.2f}}<extra></extra>'
        )

        if data['triangles'] is not None:
            st.info("✅ Rendering explicit surface mesh (Hexahedra/Tetra/Triangles).")
            mesh_kwargs.update(dict(
                i=data['triangles'][:, 0], 
                j=data['triangles'][:, 1], 
                k=data['triangles'][:, 2]
            ))
            fig.add_trace(go.Mesh3d(**mesh_kwargs))
        else:
            st.info("ℹ️ No explicit surface found. Rendering surface using Plotly alphahull.")
            mesh_kwargs['alphahull'] = 5
            mesh_kwargs['opacity'] = min(opacity, 0.85) # Alphahull looks better slightly more transparent
            fig.add_trace(go.Mesh3d(**mesh_kwargs))

        fig.update_layout(
            title=f"{label} at Timestep {timestep + 1}",
            title_font=dict(size=title_size),
            font=dict(size=font_size),
            height=700,
            margin=dict(l=0, r=0, t=60, b=0),
            scene=dict(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.6)),
                xaxis=dict(
                    title="X (m)",
                    title_font=dict(size=font_size),
                    tickfont=dict(size=font_size),
                    linewidth=axis_line_width,
                    tickwidth=tick_width,
                    ticklen=tick_len
                ),
                yaxis=dict(
                    title="Y (m)",
                    title_font=dict(size=font_size),
                    tickfont=dict(size=font_size),
                    linewidth=axis_line_width,
                    tickwidth=tick_width,
                    ticklen=tick_len
                ),
                zaxis=dict(
                    title="Z (m, exaggerated)",
                    title_font=dict(size=font_size),
                    tickfont=dict(size=font_size),
                    linewidth=axis_line_width,
                    tickwidth=tick_width,
                    ticklen=tick_len
                )
            ),
            coloraxis_colorbar=dict(
                thickness=cb_thickness,
                len=cb_len,
                title=label,
                title_font=dict(size=cb_title_size),
                tickfont=dict(size=cb_tick_size),
                tickformat=".2e"
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
                
    finally:
        # Clean up temporary directory to prevent disk leaks on Streamlit Cloud
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
