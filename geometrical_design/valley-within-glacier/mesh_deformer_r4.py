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

if st.button("🔄 Reload Data", key="reload_data_btn"):
    st.cache_data.clear()
    st.rerun()

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
        # Use the Python engine + explicit regex separator for robust whitespace parsing.
        # Return a fresh copy so mutation-heavy downstream code can't poison the cache.
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=['ID', 'Flag', 'X', 'Y', 'Z'],
            engine="python",
        )
        return df.copy()

    @st.cache_data
    def read_dat(filepath):
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=['X', 'Z'],
            engine="python",
        )
        return df.copy()

    nodes_orig = read_mesh(mesh_nodes_file)

    st.sidebar.header("📂 File Selection")
    surface_file = st.sidebar.selectbox(
        "Surface Profile (.dat)", dat_files, key="surface_file_select"
    )
    bedrock_file = st.sidebar.selectbox(
        "Bedrock Profile (.dat)", dat_files, key="bedrock_file_select"
    )

    surface_df = read_dat(surface_file)
    bedrock_df = read_dat(bedrock_file)

    # ==========================================
    # 4. Deformation Logic
    # ==========================================
    def deform_chunk(chunk, bedrock, surface, profile_type, params):
        # ---- Defensive hardening ----------------------------------------
        if not isinstance(chunk, pd.DataFrame):
            raise TypeError(f"deform_chunk expected DataFrame, got {type(chunk)}")
        if chunk.empty:
            return chunk
        # -----------------------------------------------------------------

        x_mesh = chunk['X'].to_numpy()
        y_mesh = chunk['Y'].to_numpy()

        z_bed_interp = np.interp(x_mesh, bedrock['X'].to_numpy(), bedrock['Z'].to_numpy())
        z_surf_interp = np.interp(x_mesh, surface['X'].to_numpy(), surface['Z'].to_numpy())

        y_center = params.get('y_center', 500.0)
        y_max = chunk['Y'].max() if 'Y' in chunk else 1000.0

        if profile_type == "U-Valley (Parabolic)":
            steepness = params.get('steepness', 0.0003)
            y_var = steepness * (y_mesh - y_center) ** 2
        elif profile_type == "V-Valley (Linear)":
            steepness = params.get('steepness', 0.05)
            y_var = steepness * np.abs(y_mesh - y_center)
        elif profile_type == "Lateral Moraines":
            sigma = params.get('width', 100.0)
            height = params.get('height', 50.0)
            y_var = height * (
                np.exp(-((y_mesh - 50) ** 2) / (2 * sigma ** 2))
                + np.exp(-((y_mesh - (y_max - 50)) ** 2) / (2 * sigma ** 2))
            )
        elif profile_type == "Asymmetric Valley":
            s_left = params.get('steepness_left', 0.0002)
            s_right = params.get('steepness_right', 0.0005)
            y_var = np.where(
                y_mesh < y_center,
                s_left * (y_mesh - y_center) ** 2,
                s_right * (y_mesh - y_center) ** 2,
            )
        else:
            y_var = np.zeros_like(y_mesh)

        if params.get('add_trench', False):
            trench_center = params.get('trench_x', 1500.0)
            trench_width = params.get('trench_w', 200.0)
            trench_depth = params.get('trench_d', 50.0)
            x_var = -trench_depth * np.exp(-((x_mesh - trench_center) ** 2) / (2 * trench_width ** 2))
            z_bed_interp = z_bed_interp + x_var
            z_surf_interp = z_surf_interp + x_var * 0.2

        new_bottom = z_bed_interp + y_var
        new_top = z_surf_interp + y_var

        # ==========================================
        # CRITICAL SAFEGUARD: Prevent zero-thickness (degenerate) elements
        # ==========================================
        # Elmer's "ElementMetric: Degenerate 2D element" with |dCoord| = 0 arises
        # when a whole column of nodes collapses to a single Z because
        # (new_top - new_bottom) went to zero (or negative). That can happen near
        # interpolation edges, or when the trench/gaussian features push the
        # bedrock up past the interpolated surface. Enforcing a strictly positive
        # minimum thickness guarantees every vertical layer retains a distinct Z.
        min_thickness = 0.1  # 10 cm minimum ice thickness (metres)

        raw_thickness = new_top - new_bottom
        new_top = np.where(
            raw_thickness < min_thickness,
            new_bottom + min_thickness,
            new_top,
        )
        # ==========================================

        z_max_orig = params.get('z_max_orig', 5.0)
        z_normalized = chunk['Z'] / z_max_orig if z_max_orig > 0 else 0

        chunk = chunk.copy()
        chunk['Z_new'] = new_bottom + z_normalized * (new_top - new_bottom)
        return chunk

    # ==========================================
    # 4b. Net Elevation Angle (Z-X slope) Computation
    # ==========================================
    def compute_elevation_angles(surface, bedrock):
        """
        Compute the net elevation (slope) angle theta = arctan(dZ/dX) for both the
        surface and bedrock 1-D flowline profiles.

        np.gradient uses second-order central differences in the interior and
        one-sided (first-order) differences at the endpoints, and it correctly
        handles non-uniform X spacing — which matters here because the .dat
        files are not guaranteed to be evenly sampled.

        Returns
        -------
        results : DataFrame with columns
                  ['X', 'Surface_Angle_deg', 'Bedrock_Angle_deg',
                   'Surface_dZdX', 'Bedrock_dZdX']
        """
        # Sort by X to guarantee the finite-difference stencil is well-posed
        surf = surface.sort_values('X').reset_index(drop=True)
        bed = bedrock.sort_values('X').reset_index(drop=True)

        # ---- Numerical derivative dZ/dX ------------------------------------
        dZ_s_dX = np.gradient(surf['Z'].to_numpy(), surf['X'].to_numpy())
        dZ_b_dX = np.gradient(bed['Z'].to_numpy(), bed['X'].to_numpy())

        # ---- Elevation angle in degrees ------------------------------------
        angle_surface = np.degrees(np.arctan(dZ_s_dX))
        angle_bedrock = np.degrees(np.arctan(dZ_b_dX))

        # Bedrock was possibly resampled at different X; interpolate onto the
        # surface X-grid so the two columns line up row-by-row for export.
        angle_bedrock_on_surf = np.interp(surf['X'].to_numpy(),
                                          bed['X'].to_numpy(),
                                          angle_bedrock)
        dZ_b_dX_on_surf = np.interp(surf['X'].to_numpy(),
                                    bed['X'].to_numpy(),
                                    dZ_b_dX)

        results = pd.DataFrame({
            'X': surf['X'].to_numpy(),
            'Surface_Angle_deg': angle_surface,
            'Bedrock_Angle_deg': angle_bedrock_on_surf,
            'Surface_dZdX': dZ_s_dX,
            'Bedrock_dZdX': dZ_b_dX_on_surf,
        })
        return results

    # ==========================================
    # 5. Sidebar Controls
    # ==========================================
    st.sidebar.header("⛰️ Y-Direction Variation")
    profile_type = st.sidebar.selectbox(
        "Valley Profile Type",
        ["U-Valley (Parabolic)", "V-Valley (Linear)", "Lateral Moraines", "Asymmetric Valley", "None (Flat Slab)"],
        key="profile_type_select",
    )

    params = {'z_max_orig': nodes_orig['Z'].max() if 'Z' in nodes_orig else 5.0}

    if "U-Valley" in profile_type or "Asymmetric" in profile_type:
        params['y_center'] = st.sidebar.slider(
            "Valley Center (Y)", 0.0, 1000.0, 500.0, key="y_center_slider"
        )
    if "U-Valley" in profile_type or "V-Valley" in profile_type:
        params['steepness'] = st.sidebar.slider(
            "Wall Steepness", 0.00001, 0.005, 0.0003, format="%.5f", key="steepness_slider"
        )
    if "Asymmetric" in profile_type:
        params['steepness_left'] = st.sidebar.slider(
            "Left Wall Steepness", 0.00001, 0.005, 0.0002, format="%.5f", key="steepness_left_slider"
        )
        params['steepness_right'] = st.sidebar.slider(
            "Right Wall Steepness", 0.00001, 0.005, 0.0005, format="%.5f", key="steepness_right_slider"
        )
    if "Moraines" in profile_type:
        params['width'] = st.sidebar.slider(
            "Moraine Width (Sigma)", 10.0, 300.0, 100.0, key="moraine_width_slider"
        )
        params['height'] = st.sidebar.slider(
            "Moraine Height (m)", 0.0, 200.0, 50.0, key="moraine_height_slider"
        )

    st.sidebar.header("🕳️ X-Direction Features")
    params['add_trench'] = st.sidebar.checkbox(
        "Add Subglacial Trench (Overdeepening)", key="trench_checkbox"
    )
    if params['add_trench']:
        params['trench_x'] = st.sidebar.slider(
            "Trench Center (X)", 0.0, 2500.0, 1800.0, key="trench_x_slider"
        )
        params['trench_w'] = st.sidebar.slider(
            "Trench Width", 50.0, 500.0, 200.0, key="trench_w_slider"
        )
        params['trench_d'] = st.sidebar.slider(
            "Trench Depth (m)", 0.0, 200.0, 50.0, key="trench_d_slider"
        )

    # Expose the minimum-thickness safeguard in the UI so users can tune it if
    # their solver needs a fatter minimum (e.g. very coarse vertical meshes).
    st.sidebar.header("🛡️ Mesh Robustness")
    params['min_thickness'] = st.sidebar.number_input(
        "Minimum Ice Thickness (m)",
        min_value=0.0,
        max_value=50.0,
        value=0.1,
        step=0.1,
        format="%.2f",
        key="min_thickness_input",
        help=(
            "Guarantees new_top - new_bottom >= this value everywhere. "
            "Prevents 'ElementMetric: Degenerate 2D element' (|dCoord| = 0) "
            "errors in Elmer by never collapsing a vertical node column."
        ),
    )

    # ==========================================
    # 6. Main Dashboard Tabs
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Original Geometry",
        "🏔️ Deformed 3D Mesh",
        "📐 Elevation Angles (Z-X)",
        "💾 Export Data",
    ])

    # --------------------------------------------------------------
    # Tab 1 — Original Geometry
    # --------------------------------------------------------------
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Undeformed 3D Mesh Nodes")
            fig = px.scatter_3d(nodes_orig, x='X', y='Y', z='Z', color='Z',
                                opacity=0.6, color_continuous_scale='Bluered_r')
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.subheader("1D Flowline Profiles (.dat)")
            fig2d = go.Figure()
            fig2d.add_trace(go.Scatter(x=surface_df['X'], y=surface_df['Z'], name='Surface', line=dict(color='blue')))
            fig2d.add_trace(go.Scatter(x=bedrock_df['X'], y=bedrock_df['Z'], name='Bedrock', line=dict(color='brown')))
            fig2d.update_layout(xaxis_title="Distance X (m)", yaxis_title="Elevation Z (m)")
            st.plotly_chart(fig2d, width="stretch")

    # --------------------------------------------------------------
    # Tab 2 — Deformed 3D Mesh
    # --------------------------------------------------------------
    with tab2:
        st.subheader("Deformed 3D Mesh with Y-Variations")
        with st.spinner("Deforming mesh using parallel processing..."):
            # ---- Fixed chunking + parallel call --------------------------------
            # Split *row indices*, not the DataFrame itself. np.array_split on a
            # DataFrame coerces via np.asarray and returns ndarrays, which is
            # exactly what caused `chunk['X']` to raise IndexError in the worker.
            n_jobs = min(os.cpu_count() or 4, 8)   # cap to avoid Cloud memory thrash

            idx_chunks = np.array_split(np.arange(len(nodes_orig)), n_jobs)
            chunks = [
                nodes_orig.iloc[idx].reset_index(drop=True).copy()
                for idx in idx_chunks
            ]

            # Threading backend: NumPy releases the GIL during np.interp, so we
            # still get parallel speedup without loky spawning subprocesses
            # (which is what triggers the unrelated `Popen.kill()` AttributeError
            # on Python 3.12 / Streamlit Cloud).
            results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(deform_chunk)(chunk, bedrock_df, surface_df, profile_type, params)
                for chunk in chunks
            )
            nodes_deformed = pd.concat(results, ignore_index=True)
            # --------------------------------------------------------------------

        # ---- Post-deformation sanity check --------------------------------
        # Quickly verify no vertical column has collapsed. Grouping by (X, Y)
        # and checking the Z-range is the same logic Elmer uses when it flags
        # a degenerate 2D element.
        col_ranges = (
            nodes_deformed.groupby(['X', 'Y'])['Z_new']
            .agg(lambda s: s.max() - s.min())
        )
        collapsed = int((col_ranges < params['min_thickness'] - 1e-9).sum())
        if collapsed > 0:
            st.error(
                f"⚠️ {collapsed} vertical column(s) have Z-range < "
                f"{params['min_thickness']} m. Increase 'Minimum Ice Thickness' "
                "in the sidebar to guarantee Elmer-compatible elements."
            )
        else:
            st.success(
                f"✅ All {len(col_ranges):,} vertical columns have Z-range ≥ "
                f"{params['min_thickness']} m — mesh is Elmer-safe."
            )

        # ---- Small on-page diagnostic ------------------------------------
        thickness_stats = pd.DataFrame({
            'metric': ['min', 'mean', 'max'],
            'thickness (m)': [
                float(col_ranges.min()),
                float(col_ranges.mean()),
                float(col_ranges.max()),
            ],
        })
        with st.expander("📏 Vertical column thickness statistics"):
            st.dataframe(thickness_stats, width="stretch", hide_index=True)

        fig_def = px.scatter_3d(nodes_deformed, x='X', y='Y', z='Z_new', color='Z_new',
                                opacity=0.8, color_continuous_scale='Earth')
        fig_def.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_def, width="stretch")

        st.session_state['nodes_deformed'] = nodes_deformed

    # --------------------------------------------------------------
    # Tab 3 — Net Elevation Angle (Z-X) numerics + plots
    # --------------------------------------------------------------
    # Recompute angle table (Streamlit caches read_dat, so this is cheap)
    angles_df = compute_elevation_angles(surface_df, bedrock_df)
    st.session_state['angles_df'] = angles_df

    with tab3:
        st.subheader("📐 Net Elevation Angle θ = arctan(dZ/dX) along the X-axis")
        st.caption(
            "Numerical derivative computed with second-order central differences in the interior "
            "and one-sided differences at the endpoints (via `np.gradient`), which handles "
            "non-uniform X spacing in the `.dat` files."
        )

        # ---------- Interactive Plotly version of the Matplotlib figure ----------
        fig_ang = go.Figure()
        fig_ang.add_trace(go.Scatter(
            x=angles_df['X'], y=angles_df['Surface_Angle_deg'],
            name='Surface Slope Angle', line=dict(color='royalblue', width=2),
            hovertemplate='X = %{x:.1f} m<br>θ_surface = %{y:.3f}°<extra></extra>',
        ))
        fig_ang.add_trace(go.Scatter(
            x=angles_df['X'], y=angles_df['Bedrock_Angle_deg'],
            name='Bedrock Slope Angle', line=dict(color='saddlebrown', width=2, dash='dash'),
            hovertemplate='X = %{x:.1f} m<br>θ_bedrock = %{y:.3f}°<extra></extra>',
        ))
        fig_ang.add_hline(y=0, line=dict(color='black', width=0.8, dash='dot'))
        fig_ang.update_layout(
            title='Net Elevation Angle (Slope) along the X-axis',
            xaxis_title='Distance X (m)',
            yaxis_title='Slope Angle (Degrees)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=60, b=0),
            hovermode='x unified',
        )
        st.plotly_chart(fig_ang, width="stretch")

        # ---------- Numeric summary ----------
        st.markdown("### 📊 Numeric Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Max Surface Angle", f"{angles_df['Surface_Angle_deg'].max():.3f}°")
        with c2:
            st.metric("Min Surface Angle", f"{angles_df['Surface_Angle_deg'].min():.3f}°")
        with c3:
            st.metric("Max Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].max():.3f}°")
        with c4:
            st.metric("Min Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].min():.3f}°")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Mean Surface Angle", f"{angles_df['Surface_Angle_deg'].mean():.3f}°")
        with c6:
            st.metric("Mean Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].mean():.3f}°")
        with c7:
            st.metric("Std Surface Angle", f"{angles_df['Surface_Angle_deg'].std():.3f}°")
        with c8:
            st.metric("Std Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].std():.3f}°")

        # ---------- Full numeric table (displayed, not just downloadable) ----------
        st.markdown("### 🔢 Full Numeric Table")
        st.dataframe(
            angles_df.style.format({
                'X': '{:.3f}',
                'Surface_Angle_deg': '{:.6f}',
                'Bedrock_Angle_deg': '{:.6f}',
                'Surface_dZdX': '{:.6e}',
                'Bedrock_dZdX': '{:.6e}',
            }),
            width="stretch",
            height=420,
        )

        # ---------- Local CSV download for the angle table ----------
        csv_angles_tab3 = angles_df.to_csv(index=False, float_format='%.8f')
        st.download_button(
            "⬇️ Download elevation_angles.csv",
            csv_angles_tab3,
            file_name="elevation_angles.csv",
            mime="text/csv",
            key="download_angles_tab3",  # <-- unique key prevents StreamlitDuplicateElementId
        )

    # --------------------------------------------------------------
    # Tab 4 — Export Data
    # --------------------------------------------------------------
    with tab4:
        st.subheader("Export Processed Data")
        if 'nodes_deformed' in st.session_state:
            csv_mesh = st.session_state['nodes_deformed'][['ID', 'Flag', 'X', 'Y', 'Z_new']].to_csv(
                index=False, sep=' ', header=False, float_format='%.4f'
            )
            st.download_button(
                "⬇️ Download Deformed mesh.nodes",
                csv_mesh,
                file_name="mesh.nodes",
                key="download_mesh_tab4",  # <-- unique key
            )

        if 'angles_df' in st.session_state:
            st.markdown("---")
            st.markdown("**Elevation-angle table** (also available on the *Elevation Angles* tab)")
            csv_angles_tab4 = st.session_state['angles_df'].to_csv(index=False, float_format='%.8f')
            st.download_button(
                "⬇️ Download elevation_angles.csv",
                csv_angles_tab4,
                file_name="elevation_angles.csv",
                mime="text/csv",
                key="download_angles_tab4",  # <-- unique key (different from tab3)
            )
