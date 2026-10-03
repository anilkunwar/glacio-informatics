import streamlit as st
import os
import zipfile
import tempfile
import glob
import numpy as np
import meshio
import plotly.graph_objects as go
import shutil

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Elmer VTU/NPZ Toolkit",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# HEXAHEDRON / TETRAHEDRON SURFACE EXTRACTION
# =============================================
def extract_surface_triangles(mesh):
    """
    Extracts outer surface triangles from volume meshes (hex/tet) for Plotly/NPZ.
    Plotly only supports triangle meshes, so we convert hexahedra/tetrahedra 
    to their boundary triangles by finding faces shared by only one cell.
    """
    for cell_block in mesh.cells:
        ctype = cell_block.type
        data = cell_block.data
        
        # 1. Already a surface mesh
        if ctype == "triangle":
            return data.astype(np.int32)
            
        # 2. Tetrahedral volume mesh (4 nodes per cell)
        if ctype == "tetra" and data.shape[1] == 4:
            face_defs = [(0,1,2), (0,1,3), (1,2,3), (0,2,3)]
            face_dict = {}
            for tet in data:
                for f in face_defs:
                    ordered = tuple(tet[list(f)])
                    sorted_f = tuple(sorted(ordered))
                    if sorted_f in face_dict:
                        face_dict[sorted_f]['count'] += 1
                    else:
                        face_dict[sorted_f] = {'count': 1, 'ordered': ordered}
            return np.array([v['ordered'] for v in face_dict.values() if v['count'] == 1], dtype=np.int32)
            
        # 3. Hexahedral volume mesh (8 nodes per cell)
        if ctype == "hexahedron" and data.shape[1] == 8:
            face_defs = [
                (0,1,2,3), (4,5,6,7), # bottom, top
                (0,1,5,4), (1,2,6,5), # front, right
                (2,3,7,6), (3,0,4,7)  # back, left
            ]
            face_dict = {}
            for hx in data:
                for f in face_defs:
                    ordered = tuple(hx[list(f)])
                    sorted_f = tuple(sorted(ordered))
                    if sorted_f in face_dict:
                        face_dict[sorted_f]['count'] += 1
                    else:
                        face_dict[sorted_f] = {'count': 1, 'ordered': ordered}
            tris = []
            for v in face_dict.values():
                if v['count'] == 1:
                    f = v['ordered']
                    # Split quad face into 2 triangles
                    tris.append((f[0], f[1], f[2]))
                    tris.append((f[0], f[2], f[3]))
            return np.array(tris, dtype=np.int32)
            
    return None

# =============================================
# MAIN APP WITH TABS
# =============================================
tab1, tab2 = st.tabs(["🔄 VTU to NPZ Converter", "📊 NPZ Viewer"])

# =============================================
# TAB 1: CONVERTER
# =============================================
with tab1:
    st.title("⚙️ Elmer VTU/PVTU to NPZ Converter")
    st.markdown("""
    This tool converts heavy, multi-file Elmer outputs into a single, highly compressed `.npz` file.  
    💡 **Pro Tip:** If your total VTU file size is > 150 MB, **run this script locally** on your machine to avoid Streamlit Cloud's RAM limits during conversion.
    """)

    uploaded_files = st.file_uploader(
        "Upload VTU/PVTU files (or a single .zip containing them)",
        accept_multiple_files=True,
        type=["vtu", "pvtu", "zip"]
    )

    if not uploaded_files:
        st.info("👈 Please upload your Elmer output files to begin.")
        st.stop()

    temp_dir = tempfile.mkdtemp()

    try:
        with st.spinner("Preparing files..."):
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

            vtu_files = sorted(glob.glob(os.path.join(temp_dir, "**/*.vtu"), recursive=True))
            pvtu_files = sorted(glob.glob(os.path.join(temp_dir, "**/*.pvtu"), recursive=True))
            
            # Prioritize .pvtu if both exist (parallel output)
            files_to_process = pvtu_files if pvtu_files else vtu_files

            if not files_to_process:
                st.error("❌ No .vtu or .pvtu files found in the upload.")
                st.stop()

            st.success(f"✅ Found {len(files_to_process)} file(s) to process.")

        st.markdown("---")
        st.subheader("Processing Mesh Data...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        fields_dict = {}
        points = None
        triangles = None
        n_timesteps = len(files_to_process)

        for t, filepath in enumerate(files_to_process):
            status_text.text(f"Reading timestep {t + 1}/{n_timesteps}: {os.path.basename(filepath)}")
            
            mesh = meshio.read(filepath)
            
            if t == 0:
                points = mesh.points.astype(np.float32)
                n_pts = len(points)
                
                # Extract surface triangles (supports hex, tet, and native triangles)
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
                        
            # CRITICAL: Delete mesh object to free RAM immediately
            del mesh
            progress_bar.progress((t + 1) / n_timesteps)

        status_text.text("✅ Processing complete! Compressing data...")

        save_dict = {
            "points": points,
            "triangles": triangles,
            "n_timesteps": np.array([n_timesteps])
        }
        
        for key, arr in fields_dict.items():
            save_dict[f"field_{key}"] = arr

        output_filename = "glacier_data.npz"
        output_path = os.path.join(temp_dir, output_filename)
        
        np.savez_compressed(output_path, **save_dict)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        st.success(f"🎉 Successfully created `{output_filename}` ({file_size_mb:.2f} MB)")

        # Ensure allowance of NPZ file downloads
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download glacier_data.npz",
                data=f,
                file_name="glacier_data.npz",
                mime="application/octet-stream"
            )

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# =============================================
# TAB 2: VIEWER
# =============================================
with tab2:
    st.title("📊 Elmer NPZ Diagnostic Viewer")
    st.markdown("Upload the `.npz` file generated above to visualize it with full customization.")

    npz_file = st.file_uploader("Upload `.npz` file", type=["npz"])
    
    if not npz_file:
        st.info("👈 Please upload an `.npz` file to begin visualization.")
        st.stop()

    with st.spinner("Loading NPZ data..."):
        data = np.load(npz_file, allow_pickle=True)
        
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

    st.success(f"✅ Loaded {n_timesteps} timestep(s) successfully!")

    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Data Controls")
    
    available_fields = list(fields.keys())
    default_field = "Velocity" if "Velocity" in available_fields else available_fields[0]
    field = st.sidebar.selectbox("Select Field", available_fields, index=available_fields.index(default_field))
    
    if n_timesteps > 1:
        timestep = st.sidebar.slider("Timestep", 0, n_timesteps - 1, 0)
    else:
        timestep = 0
        st.sidebar.info("Only 1 timestep found — slider disabled.")
        
    z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 50.0, 10.0, 1.0, help="Glaciers are thin; exaggerate Z to see thickness.")
    max_points = st.sidebar.number_input("Max Points (Decimation)", min_value=10000, max_value=2000000, value=150000, step=10000)

    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Figure Customization")
    
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
    
    raw = fields[field][timestep]
    kind = "scalar" if raw.shape[-1] == 1 else "vector"
    
    if kind == "scalar":
        values = np.where(np.isnan(raw), 0, raw.flatten())
        label = field
    else:
        magnitude = np.linalg.norm(raw, axis=1)
        values = np.where(np.isnan(magnitude), 0, magnitude)
        label = f"{field} (Magnitude)"
        
    valid_mask = ~np.isnan(values)
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

    # --- Plotly Figure Construction ---
    fig = go.Figure()
    
    if render_tris is not None:
        fig.add_trace(go.Mesh3d(
            x=render_pts[:, 0], y=render_pts[:, 1], z=render_pts[:, 2],
            i=render_tris[:, 0], j=render_tris[:, 1], k=render_tris[:, 2],
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

    # Apply comprehensive customizations
    fig.update_layout(
        title=f"{label} at Timestep {timestep + 1}",
        title_font=dict(size=title_size),
        font=dict(size=font_size),
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
