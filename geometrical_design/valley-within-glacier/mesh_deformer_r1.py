import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from joblib import Parallel, delayed

# ==========================================
# 1. Page config & Robust Directory Handling
# ==========================================
st.set_page_config(page_title="Glacier Mesh Deformer & Analyzer", layout="wide")

# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths (tries direct relative path first)
DAT_DIR = os.path.join(BASE_DIR, "surface_bedrock")
MESH_DIR = os.path.join(BASE_DIR, "undeformed_geometry")

# Fallback: If the above don't exist, try the nested structure from your GitHub repo
if not os.path.exists(DAT_DIR):
    DAT_DIR = os.path.join(BASE_DIR, "geometrical_design", "valley-within-glacier", "surface_bedrock")
if not os.path.exists(MESH_DIR):
    MESH_DIR = os.path.join(BASE_DIR, "geometrical_design", "valley-within-glacier", "undeformed_geometry")

# Ensure directories exist to prevent crashes
os.makedirs(MESH_DIR, exist_ok=True)
os.makedirs(DAT_DIR, exist_ok=True)

st.title("🏔️ Glacier Mesh Deformer & Analyzer")
st.caption("Loads mesh and profile data, applies Y-direction variations, and exports updated meshes.")

st.info(f"📁 **Mesh directory:** `{MESH_DIR}`\n📁 **Data directory:** `{DAT_DIR}`")

if st.button("🔄 Reload Data"):
    st.cache_data.clear()
    st.rerun()  # Note: st.experimental_rerun() is deprecated in newer Streamlit versions

# ==========================================
# 2. Data Loading & Validation
# ==========================================
@st.cache_data
def load_data():
    dat_files = glob.glob(os.path.join(DAT_DIR, "*.dat"))
    mesh_nodes_file = os.path.join(MESH_DIR, "mesh.nodes")
    
    dat_files_found = len(dat_files)
    mesh_found = os.path.exists(mesh_nodes_file)
    
    return dat_files, mesh_nodes_file, dat_files_found, mesh_found

dat_files, mesh_nodes_file, dat_files_found, mesh_found = load_data()

if dat_files_found == 0:
    st.warning(f"⚠️ No `.dat` files found in `{DAT_DIR}`. Please ensure `steady_ELA400_bedrock.dat` and `steady_ELA400_surface.dat` are in this directory.")
else:
    st.success(f"✅ Found {dat_files_found} `.dat` file(s) in `{DAT_DIR}`.")

if not mesh_found:
    st.warning(f"⚠️ Mesh file not found at `{mesh_nodes_file}`. Please ensure `mesh.nodes` is in this directory.")
else:
    st.success(f"✅ Found `mesh.nodes` in `{MESH_DIR}`.")

# ==========================================
# 3. Main App Logic (Only runs if files are found)
# ==========================================
if dat_files_found > 0 and mesh_found:
    
    @st.cache_data
    def read_mesh(filepath):
        return pd.read_csv(filepath, sep='\s+', header=None, names=['ID', 'Flag', 'X', 'Y', 'Z'])

    @st.cache_data
    def read_dat(filepath):
        return pd.read_csv(filepath, sep='\s+', header=None, names=['X', 'Z'])

    nodes_orig = read_mesh(mesh_nodes_file)
    
    st.sidebar.header("📂 File Selection")
    surface_file = st.sidebar.selectbox("Surface Profile (.dat)", dat_files)
    bedrock_file = st.sidebar.selectbox("Bedrock Profile (.dat)", dat_files)
    
    surface_df = read_dat(surface_file)
    bedrock_df = read_dat(bedrock_file)

    # ==========================================
    # 4. Deformation Logic (with Joblib)
    # ==========================================
    def deform_chunk(chunk, bedrock, surface, profile_type, params):
        x_mesh = chunk['X'].values
        y_mesh = chunk['Y'].values
        
        z_bed_interp = np.interp(x_mesh, bedrock['X'], bedrock['Z'])
        z_surf_interp = np.interp(x_mesh, surface['X'], surface['Z'])
        
        y_center = params.get('y_center', 500.0)
        y_max = chunk['Y'].max() if 'Y' in chunk else 1000.0
        
        if profile_type == "U-Valley (Parabolic)":
            steepness = params.get('steepness', 0.0003)
            y_var = steepness * (y_mesh - y_center)**2
        elif profile_type == "V-Valley (Linear)":
            steepness = params.get('steepness', 0.05)
            y_var = steepness * np.abs(y_mesh - y_center)
        elif profile_type == "Lateral Moraines":
            sigma = params.get('width', 100.0)
            height = params.get('height', 50.0)
            y_var = height * (np.exp(-((y_mesh - 50)**2)/(2*sigma**2)) + np.exp(-((y_mesh - (y_max-50))**2)/(2*sigma**2)))
        elif profile_type == "Asymmetric Valley":
            s_left = params.get('steepness_left', 0.0002)
            s_right = params.get('steepness_right', 0.0005)
            y_var = np.where(y_mesh < y_center, s_left * (y_mesh - y_center)**2, s_right * (y_mesh - y_center)**2)
        else:
            y_var = np.zeros_like(y_mesh)

        if params.get('add_trench', False):
            trench_center = params.get('trench_x', 1500.0)
            trench_width = params.get('trench_w', 200.0)
            trench_depth = params.get('trench_d', 50.0)
            x_var = -trench_depth * np.exp(-((x_mesh - trench_center)**2)/(2*trench_width**2))
            z_bed_interp += x_var
            z_surf_interp += x_var * 0.2 
            
        new_bottom = z_bed_interp + y_var
        new_top = z_surf_interp + y_var
        
        z_max_orig = params.get('z_max_orig', 5.0)
        z_normalized = chunk['Z'] / z_max_orig if z_max_orig > 0 else 0
        
        chunk['Z_new'] = new_bottom + z_normalized * (new_top - new_bottom)
        return chunk

    # ==========================================
    # 5. Sidebar Controls
    # ==========================================
    st.sidebar.header("⛰️ Y-Direction Variation")
    profile_type = st.sidebar.selectbox(
        "Valley Profile Type", 
        ["U-Valley (Parabolic)", "V-Valley (Linear)", "Lateral Moraines", "Asymmetric Valley", "None (Flat Slab)"]
    )

    params = {'z_max_orig': nodes_orig['Z'].max() if 'Z' in nodes_orig else 5.0}

    if "U-Valley" in profile_type or "Asymmetric" in profile_type:
        params['y_center'] = st.sidebar.slider("Valley Center (Y)", 0.0, 1000.0, 500.0)
    if "U-Valley" in profile_type or "V-Valley" in profile_type:
        params['steepness'] = st.sidebar.slider("Wall Steepness", 0.00001, 0.005, 0.0003, format="%.5f")
    if "Asymmetric" in profile_type:
        params['steepness_left'] = st.sidebar.slider("Left Wall Steepness", 0.00001, 0.005, 0.0002, format="%.5f")
        params['steepness_right'] = st.sidebar.slider("Right Wall Steepness", 0.00001, 0.005, 0.0005, format="%.5f")
    if "Moraines" in profile_type:
        params['width'] = st.sidebar.slider("Moraine Width (Sigma)", 10.0, 300.0, 100.0)
        params['height'] = st.sidebar.slider("Moraine Height (m)", 0.0, 200.0, 50.0)

    st.sidebar.header("🕳️ X-Direction Features")
    params['add_trench'] = st.sidebar.checkbox("Add Subglacial Trench (Overdeepening)")
    if params['add_trench']:
        params['trench_x'] = st.sidebar.slider("Trench Center (X)", 0.0, 2500.0, 1800.0)
        params['trench_w'] = st.sidebar.slider("Trench Width", 50.0, 500.0, 200.0)
        params['trench_d'] = st.sidebar.slider("Trench Depth (m)", 0.0, 200.0, 50.0)

    # ==========================================
    # 6. Main Dashboard Tabs
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📊 Original Geometry", "🏔️ Deformed 3D Mesh", "💾 Export Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Undeformed 3D Mesh Nodes")
            fig = px.scatter_3d(nodes_orig, x='X', y='Y', z='Z', color='Z', 
                                opacity=0.6, color_continuous_scale='Bluered_r')
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("1D Flowline Profiles (.dat)")
            fig2d = go.Figure()
            fig2d.add_trace(go.Scatter(x=surface_df['X'], y=surface_df['Z'], name='Surface', line=dict(color='blue')))
            fig2d.add_trace(go.Scatter(x=bedrock_df['X'], y=bedrock_df['Z'], name='Bedrock', line=dict(color='brown')))
            fig2d.update_layout(xaxis_title="Distance X (m)", yaxis_title="Elevation Z (m)")
            st.plotly_chart(fig2d, use_container_width=True)

    with tab2:
        st.subheader("Deformed 3D Mesh with Y-Variations")
        with st.spinner("Deforming mesh using parallel processing..."):
            n_jobs = os.cpu_count() or 4
            chunks = np.array_split(nodes_orig, n_jobs)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(deform_chunk)(chunk.copy(), bedrock_df, surface_df, profile_type, params) 
                for chunk in chunks
            )
            nodes_deformed = pd.concat(results)
            
        fig_def = px.scatter_3d(nodes_deformed, x='X', y='Y', z='Z_new', color='Z_new', 
                                opacity=0.8, color_continuous_scale='Earth')
        fig_def.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_def, use_container_width=True)
        
        st.session_state['nodes_deformed'] = nodes_deformed

    with tab3:
        st.subheader("Export Processed Data")
        if 'nodes_deformed' in st.session_state:
            csv_mesh = st.session_state['nodes_deformed'][['ID', 'Flag', 'X', 'Y', 'Z_new']].to_csv(index=False, sep=' ', header=False, float_format='%.4f')
            st.download_button("⬇️ Download Deformed mesh.nodes", csv_mesh, file_name="mesh.nodes")
