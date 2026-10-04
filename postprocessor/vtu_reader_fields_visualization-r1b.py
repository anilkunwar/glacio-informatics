import streamlit as st
import os
import glob
import tempfile
import zipfile
import numpy as np
import meshio
import plotly.graph_objects as go
import io
import gc
import warnings

warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Elmer Glacier NPZ Toolkit",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# HEXAHEDRON / TETRAHEDRON SURFACE EXTRACTION
# =============================================
def extract_surface_triangles(mesh):
    """Extracts outer surface triangles from volume meshes for Plotly."""
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
# PROCESSING FUNCTION
# =============================================
def process_vtu_to_npz(file_paths):
    """Processes VTU files and returns compressed NPZ bytes."""
    fields_dict = {}
    points = None
    triangles = None
    n_timesteps = len(file_paths)

    for t, filepath in enumerate(file_paths):
        try:
            mesh = meshio.read(filepath)
        except Exception as e:
            st.error(f"Failed to read {filepath}: {e}")
            return None, None
            
        if t == 0:
            points = mesh.points.astype(np.float32)
            n_pts = len(points)
            triangles = extract_surface_triangles(mesh)
            if triangles is None:
                triangles = np.array([], dtype=np.int32)
            
            for key, arr in mesh.point_data.items():
                if np.issubdtype(arr.dtype, np.number):
                    comps = 1 if arr.ndim == 1 else arr.shape[1]
                    shape = (n_timesteps, n_pts, comps)
                    fields_dict[key] = np.full(shape, np.nan, dtype=np.float32)
        
        for key in fields_dict.keys():
            if key in mesh.point_data:
                arr = np.asarray(mesh.point_data[key], dtype=np.float32)
                if arr.ndim == 1:
                    fields_dict[key][t, :, 0] = arr.flatten()
                else:
                    fields_dict[key][t, :, :] = arr
                    
        del mesh
        gc.collect() # Force garbage collection to save memory

    save_dict = {
        "points": points,
        "triangles": triangles,
        "n_timesteps": np.array([n_timesteps])
    }
    for key, arr in fields_dict.items():
        save_dict[f"field_{key}"] = arr

    # Save to in-memory BytesIO buffer to avoid disk I/O
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **save_dict)
    buffer.seek(0)
    return save_dict, buffer.getvalue()

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
# MAIN APP
# =============================================
tab1, tab2 = st.tabs(["🔄 1. Load & Convert to NPZ", "📊 2. Visualize NPZ"])

# =============================================
# TAB 1: LOAD & CONVERT
# =============================================
with tab1:
    st.title("🔄 Load & Convert Elmer Output")
    st.markdown("Load your simulation results, convert them to a highly compressed `.npz` format, and proceed to visualization.")

    source_choice = st.radio("Select Data Source", ["Default Directory", "Upload Files"], horizontal=True)
    
    file_paths = []
    
    if source_choice == "Default Directory":
        default_dir = "himalayan_glacier3d"
        st.info(f"Looking for `.vtu` / `.pvtu` files in: `./{default_dir}`")
        if os.path.isdir(default_dir):
            pvtu_files = sorted(glob.glob(os.path.join(default_dir, "**", "*.pvtu"), recursive=True))
            vtu_files = sorted(glob.glob(os.path.join(default_dir, "**", "*.vtu"), recursive=True))
            file_paths = pvtu_files if pvtu_files else vtu_files
            
            if file_paths:
                st.success(f"✅ Found {len(file_paths)} file(s) in the default directory.")
                with st.expander("Preview Files"):
                    st.write([os.path.basename(f) for f in file_paths[:10]] + (["..."] if len(file_paths) > 10 else []))
            else:
                st.warning(f"⚠️ No `.vtu` or `.pvtu` files found in `./{default_dir}`. Please use the Upload option.")
        else:
            st.warning(f"⚠️ Directory `./{default_dir}` does not exist. Please use the Upload option.")

    else:
        uploaded_files = st.file_uploader(
            "Upload VTU/PVTU files (or a single .zip containing them)",
            accept_multiple_files=True,
            type=["vtu", "pvtu", "zip"]
        )
        if uploaded_files:
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
            
            pvtu_files = sorted(glob.glob(os.path.join(temp_dir, "**", "*.pvtu"), recursive=True))
            vtu_files = sorted(glob.glob(os.path.join(temp_dir, "**", "*.vtu"), recursive=True))
            file_paths = pvtu_files if pvtu_files else vtu_files
            
            if file_paths:
                st.success(f"✅ Extracted and found {len(file_paths)} file(s).")

    # Conversion Trigger
    if file_paths:
        st.markdown("---")
        if st.button("⚙️ Convert to NPZ", type="primary", use_container_width=True):
            with st.spinner("Processing mesh and compressing data... This may take a moment."):
                save_dict, npz_bytes = process_vtu_to_npz(file_paths)
                
                if npz_bytes is not None:
                    # Cache in session state
                    st.session_state['npz_bytes'] = npz_bytes
                    st.session_state['n_timesteps'] = int(save_dict['n_timesteps'][0])
                    st.success("🎉 Conversion successful! Data is cached in memory.")
                    
                    # Download Button
                    st.download_button(
                        label="📥 Download glacier_data.npz",
                        data=npz_bytes,
                        file_name="glacier_data.npz",
                        mime="application/octet-stream"
                    )
                    
                    st.info("👉 **Next Step:** Click on the **'📊 2. Visualize NPZ'** tab above to view your data.")
                else:
                    st.error("Conversion failed. Check the logs for details.")

# =============================================
# TAB 2: VISUALIZATION
# =============================================
with tab2:
    st.title("📊 Elmer NPZ Diagnostic Viewer")
    
    if 'npz_bytes' not in st.session_state:
        st.warning("⚠️ No NPZ data loaded. Please go to **Tab 1** to load and convert your files first.")
        st.stop()

    # Load from session state cache (very fast, no disk read)
    with st.spinner("Loading cached NPZ data..."):
        data = np.load(io.BytesIO(st.session_state['npz_bytes']), allow_pickle=True)
        
    points = data['points']
    triangles = data['triangles'] if len(data['triangles']) > 0 else None
    n_timesteps = int(data['n_timesteps'][0])
    
    fields = {}
    for key in data.files:
        if key.startswith('field_'):
            fields[key[6:]] = data[key]
            
    if not fields:
        st.error("❌ No fields found in the NPZ file.")
        st.stop()

    st.success(f"✅ Loaded {n_timesteps} timestep(s) successfully from cache!")

    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Data Controls")
    
    available_fields = list(fields.keys())
    default_field = "Velocity" if "Velocity" in available_fields else available_fields[0]
    field = st.sidebar.selectbox("Select Field", available_fields, index=available_fields.index(default_field))
    
    if n_timesteps > 1:
        timestep = st.sidebar.slider("Timestep", 0, n_timesteps - 1, 0)
    else:
        timestep = 0
        st.sidebar.info("Only 1 timestep found.")
        
    z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 50.0, 10.0, 1.0, help="Glaciers are thin; exaggerate Z to see thickness.")
    max_points = st.sidebar.number_input("Max Points (Decimation)", min_value=10000, max_value=2000000, value=150000, step=10000)

    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Figure Customization")
    
    # Opacity Control (NEW)
    opacity = st.sidebar.slider("Mesh Opacity", 0.1, 1.0, 0.9, 0.05, help="Adjust the transparency of the 3D mesh.")
    
    # Font & Label Sizes
    st.sidebar.subheader("Text & Labels")
    font_size = st.sidebar.slider("General Font Size", 8, 24, 12)
    title_size = st.sidebar.slider("Title Size", 12, 32, 16)
    
    # Axes & Ticks
    st.sidebar.subheader("Axes & Ticks")
    axis_line_width = st.sidebar.slider("Axis Line Width", 1, 5, 2)
    tick_width = st.sidebar.slider("Tick Width", 1, 5, 2)
    tick_len = st.sidebar.slider("Tick Length", 2, 10, 5)
    
    # Colorbar
    st.sidebar.subheader("Colorbar")
    colormap = st.sidebar.selectbox("Colormap", ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens', 'Jet', 'Rainbow'], index=0)
    cb_thickness = st.sidebar.slider("Colorbar Thickness", 10, 50, 20)
    cb_len = st.sidebar.slider("Colorbar Length", 0.2, 1.0, 0.8)
    cb_title_size = st.sidebar.slider("Colorbar Title Size", 8, 20, 12)
    cb_tick_size = st.sidebar.slider("Colorbar Tick Size", 8, 20, 10)
    
    auto_scale = st.sidebar.checkbox("Auto Color Scale", value=True)

    # --- Data Processing ---
    pts = points.copy()
    pts[:, 2] *= z_exag
    
    raw = np.asarray(fields[field][timestep])
    
    if len(raw.shape) == 1 or raw.shape[-1] == 1:
        kind = "scalar"
        flat_raw = raw.flatten()
        values = np.where(np.isnan(flat_raw), 0.0, flat_raw)
    else:
        kind = "vector"
        magnitude = np.linalg.norm(raw, axis=-1)
        values = np.where(np.isnan(magnitude), 0.0, magnitude)
        
    valid_mask = np.asarray(~np.isnan(values)).flatten()
    if len(valid_mask) != len(pts):
        st.error("⚠️ Shape mismatch between points and field data.")
        st.stop()
        
    plot_pts = pts[valid_mask]
    plot_vals = values[valid_mask]
    
    if triangles is not None and len(points) > max_points:
        st.sidebar.info("ℹ️ Explicit mesh detected. Decimation disabled to preserve surface topology.")
        render_pts = pts
        render_vals = values
        render_tris = triangles
    elif len(plot_pts) > max_points:
        st.sidebar.warning(f"⚠️ Decimated to {max_points} points for browser performance.")
        indices = np.random.choice(len(plot_pts), max_points, replace=False)
        render_pts = plot_pts[indices]
        render_vals = plot_vals[indices]
        render_tris = None
    else:
        render_pts = pts
        render_vals = values
        render_tris = triangles

    cmin = float(np.nanmin(render_vals))
    cmax = float(np.nanmax(render_vals))
    
    if not auto_scale:
        col_a, col_b = st.columns(2)
        with col_a:
            cmin = st.number_input("Min Limit", value=cmin, format="%.3e")
        with col_b:
            cmax = st.number_input("Max Limit", value=cmax, format="%.3e")
    else:
        cmin, cmax = None, None

    # Get formatted label with units
    label = get_field_label(field)

    # --- Plotly Figure Construction ---
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

    if render_tris is not None:
        mesh_kwargs.update(dict(i=render_tris[:, 0], j=render_tris[:, 1], k=render_tris[:, 2]))
        fig.add_trace(go.Mesh3d(**mesh_kwargs))
    else:
        mesh_kwargs['alphahull'] = 5
        mesh_kwargs['opacity'] = min(opacity, 0.85) # Alphahull looks better slightly more transparent
        fig.add_trace(go.Mesh3d(**mesh_kwargs))

    fig.update_layout(
        title=f"{label} at Timestep {timestep + 1}",
        title_font=dict(size=title_size),
        font=dict(size=font_size),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.6)),
            xaxis=dict(title="X (m)", title_font=dict(size=font_size), tickfont=dict(size=font_size), linewidth=axis_line_width, tickwidth=tick_width, ticklen=tick_len),
            yaxis=dict(title="Y (m)", title_font=dict(size=font_size), tickfont=dict(size=font_size), linewidth=axis_line_width, tickwidth=tick_width, ticklen=tick_len),
            zaxis=dict(title="Z (m, exaggerated)", title_font=dict(size=font_size), tickfont=dict(size=font_size), linewidth=axis_line_width, tickwidth=tick_width, ticklen=tick_len)
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
