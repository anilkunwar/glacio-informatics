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


def _prune(dirnames):
    dirnames[:] = [d for d in dirnames
                   if not d.startswith(".")
                   and d not in ("__pycache__", "node_modules", "venv", "site-packages")]


def _bounded_walk(root, max_depth=5):
    root = os.path.normpath(os.path.abspath(root))
    if not os.path.isdir(root):
        return
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.rstrip(os.sep).count(os.sep) - base >= max_depth:
            dirnames[:] = []          # stop descending deeper
            continue
        _prune(dirnames)
        yield dirpath, filenames


def find_mesh_nodes_file():
    """
    Locate `mesh.nodes`, preferring the CURRENT layout
    (`geometrical_design/himalayan_glacier/...`) over any stale legacy copy.

    Ordering matters: explicit candidates are checked first, then a bounded
    filesystem walk. The walk itself is ordered so the `himalayan_glacier`
    tree is searched before the legacy `valley-within-glacier` tree, in case
    an old `mesh.nodes` is still sitting around in the repo.
    """
    candidates = [
        # New layout: data folders directly under the app folder
        os.path.join(BASE_DIR, "undeformed_geometry", "mesh.nodes"),
        # CURRENT layout: app folder is inside geometrical_design/himalayan_glacier
        os.path.join(BASE_DIR, "geometrical_design", "himalayan_glacier",
                     "undeformed_geometry", "mesh.nodes"),
        # CURRENT layout: from the app folder, step up into geometrical_design
        os.path.join(BASE_DIR, os.pardir, "himalayan_glacier",
                     "undeformed_geometry", "mesh.nodes"),
        # Legacy nested layout (old fallback, kept for compatibility)
        os.path.join(BASE_DIR, "geometrical_design", "valley-within-glacier",
                     "undeformed_geometry", "mesh.nodes"),
        # Legacy layout as a SIBLING of the app folder
        os.path.join(BASE_DIR, os.pardir, "valley-within-glacier",
                     "undeformed_geometry", "mesh.nodes"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c

    # Last resort: bounded walk. Search himalayan_glacier first so that if a
    # stale valley-within-glacier/mesh.nodes is still present, it doesn't win.
    preferred_roots = [
        os.path.join(BASE_DIR, "geometrical_design", "himalayan_glacier"),
        os.path.join(BASE_DIR, os.pardir, "himalayan_glacier"),
        BASE_DIR,
        os.path.join(BASE_DIR, os.pardir),
    ]
    seen = set()
    for root in preferred_roots:
        root = os.path.normpath(root)
        if root in seen:
            continue
        seen.add(root)
        for dirpath, filenames in _bounded_walk(root):
            if "mesh.nodes" in filenames:
                return os.path.join(dirpath, "mesh.nodes")
    return None


def find_dat_dir():
    """
    Locate the directory containing the profile `.dat` files, preferring the
    CURRENT `himalayan_glacier` layout over any stale legacy copy.
    """
    candidates = [
        os.path.join(BASE_DIR, "surface_bedrock"),
        os.path.join(BASE_DIR, "geometrical_design", "himalayan_glacier", "surface_bedrock"),
        os.path.join(BASE_DIR, os.pardir, "himalayan_glacier", "surface_bedrock"),
        os.path.join(BASE_DIR, "geometrical_design", "valley-within-glacier", "surface_bedrock"),
        os.path.join(BASE_DIR, os.pardir, "valley-within-glacier", "surface_bedrock"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.dat")):
            return c

    best_dir, best_n = None, 0
    preferred_roots = [
        os.path.join(BASE_DIR, "geometrical_design", "himalayan_glacier"),
        os.path.join(BASE_DIR, os.pardir, "himalayan_glacier"),
        BASE_DIR,
        os.path.join(BASE_DIR, os.pardir),
    ]
    seen = set()
    for root in preferred_roots:
        root = os.path.normpath(root)
        if root in seen:
            continue
        seen.add(root)
        for dirpath, filenames in _bounded_walk(root):
            n = sum(1 for f in filenames if f.lower().endswith(".dat"))
            if n > best_n:
                best_dir, best_n = dirpath, n
    return best_dir


mesh_nodes_file = find_mesh_nodes_file()
mesh_found = mesh_nodes_file is not None
MESH_DIR = (os.path.dirname(mesh_nodes_file) if mesh_found
            else os.path.join(BASE_DIR, "undeformed_geometry"))
DAT_DIR = find_dat_dir() or os.path.join(BASE_DIR, "surface_bedrock")

# NOTE: os.makedirs() removed — creating empty folders only hides the
# real problem. Delete the empty geometrical_design/ tree it already
# created inside himalayan_glacier if you ran this locally.

st.title("🏔️ Glacier Mesh Deformer & Analyzer")
st.caption("Loads mesh and profile data, applies Y-direction variations, and exports updated meshes.")

st.info(f"📁 **Mesh directory:** `{MESH_DIR}`\n📁 **Data directory:** `{DAT_DIR}`")

if st.button("🔄 Reload Data", key="reload_data_btn"):
    st.cache_data.clear()
    # Also wipe the selectbox memory so the auto-detection runs fresh
    for k in ("surface_file_select", "bedrock_file_select"):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ==========================================
# 2. Data Loading & Validation
# ==========================================
@st.cache_data
def load_data():
    dat_files = sorted(glob.glob(os.path.join(DAT_DIR, "*.dat")))
    mesh_found = mesh_nodes_file is not None and os.path.isfile(mesh_nodes_file)
    return dat_files, mesh_nodes_file, len(dat_files), mesh_found

dat_files, mesh_nodes_file, dat_files_found, mesh_found = load_data()

if dat_files_found == 0:
    st.warning(
        f"⚠️ No `.dat` files found in `{DAT_DIR}`. Please ensure "
        "`steady_ELA5000_bedrock.dat` and `steady_ELA5000_surface.dat` are in this directory."
    )
else:
    st.success(f"✅ Found {dat_files_found} `.dat` file(s) in `{DAT_DIR}`.")
    # Show the actual filenames found so the user can immediately spot a missing file
    st.caption("Files discovered: " + ", ".join(f"`{os.path.basename(f)}`" for f in dat_files))

if not mesh_found:
    st.warning(
        "⚠️ **`mesh.nodes` not found anywhere in the repo** (searched the app folder "
        f"and the repo root above it). On Streamlit Cloud the app only sees files "
        "**committed to GitHub**, so check:\n"
        "1. `mesh.nodes` was actually pushed into the new `himalayan_glacier` folder "
        "(look at the file tree on github.com, not just your local disk).\n"
        "2. Exact spelling **and capitalization** — Linux is case-sensitive, so "
        "`Undeformed_Geometry`/`Mesh.nodes` ≠ `undeformed_geometry`/`mesh.nodes`.\n"
        "3. The folder isn't excluded by `.gitignore`.\n"
        "After fixing, push and press **🔄 Reload Data** (or reboot the app)."
    )
else:
    st.success(f"✅ Found `mesh.nodes` at `{mesh_nodes_file}`.")
    # Warn if the auto-discovery picked up the legacy valley-within-glacier copy
    if "valley-within-glacier" in mesh_nodes_file:
        st.warning(
            "⚠️ The loaded `mesh.nodes` lives in the **legacy** `valley-within-glacier` "
            "folder. If you intended to use the file in `himalayan_glacier/undeformed_geometry/`, "
            "either:\n"
            "1. Delete/rename the stale `geometrical_design/valley-within-glacier/"
            "undeformed_geometry/mesh.nodes` in your repo, or\n"
            "2. Push the new `mesh.nodes` into `himalayan_glacier/undeformed_geometry/` "
            "and press **🔄 Reload Data**."
        )

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

    # ==========================================
    # 3c. Spatial Domain Detection (X, Y, Z extents)
    # ==========================================
    # The mesh coordinates drive every spatial slider and every fallback in
    # the deformation math. Reading them once here (instead of hard-coding
    # 0–1000 / 0–1000 from the old mesh) means the UI and the deform_chunk
    # defaults automatically track whatever mesh.nodes is actually loaded.
    X_MESH_MIN = float(nodes_orig['X'].min())
    X_MESH_MAX = float(nodes_orig['X'].max())
    Y_MESH_MIN = float(nodes_orig['Y'].min())
    Y_MESH_MAX = float(nodes_orig['Y'].max())
    Z_MESH_MAX = float(nodes_orig['Z'].max()) if 'Z' in nodes_orig else 5.0
    Z_MESH_MIN = float(nodes_orig['Z'].min()) if 'Z' in nodes_orig else 0.0

    st.caption(
        f"📐 **Mesh extent** — "
        f"X: `{X_MESH_MIN:.1f}` → `{X_MESH_MAX:.1f}` m · "
        f"Y: `{Y_MESH_MIN:.1f}` → `{Y_MESH_MAX:.1f}` m · "
        f"Z: `{Z_MESH_MIN:.1f}` → `{Z_MESH_MAX:.1f}` m · "
        f"`{len(nodes_orig):,}` nodes"
    )

    # ==========================================
    # 3b. Intuitive .dat file auto-detection
    # ==========================================
    # Two things had to be fixed here:
    #
    #  (1) Streamlit persists widget selections in `st.session_state` keyed by
    #      the widget's `key=`. Even after we changed the code to pass an
    #      explicit `index=`, Streamlit kept restoring the *previously chosen*
    #      value from session memory on every rerun — so the user still saw
    #      `bedrock.dat` in both dropdowns. We therefore explicitly delete the
    #      two selectbox keys from session_state before instantiating the
    #      widgets, which forces Streamlit to honour our new `index=` defaults.
    #
    #  (2) If only one .dat file exists in the folder, both dropdowns are
    #      *forced* to point at it (there is nothing else to pick). The guard
    #      below only kicks in when there are >= 2 files, and nudges the
    #      bedrock default off the surface default so the 1D plot shows two
    #      distinct traces.

    # --- (1) Clear stale widget memory so our new defaults actually take effect
    if "surface_file_select" in st.session_state:
        del st.session_state["surface_file_select"]
    if "bedrock_file_select" in st.session_state:
        del st.session_state["bedrock_file_select"]

    # --- (2) Auto-detect surface and bedrock files by looking at filenames
    surface_candidates = [
        i for i, f in enumerate(dat_files)
        if "surface" in os.path.basename(f).lower()
    ]
    bedrock_candidates = [
        i for i, f in enumerate(dat_files)
        if "bedrock" in os.path.basename(f).lower()
    ]

    surf_idx = surface_candidates[0] if surface_candidates else 0
    bed_idx = bedrock_candidates[0] if bedrock_candidates else min(1, len(dat_files) - 1)

    # --- (3) Strict guard: never let both dropdowns point at the same file
    #         by default (unless there is literally only one .dat file).
    if len(dat_files) > 1 and surf_idx == bed_idx:
        bed_idx = (surf_idx + 1) % len(dat_files)

    st.sidebar.header("📂 File Selection")

    surface_file = st.sidebar.selectbox(
        "Surface Profile (.dat)",
        dat_files,
        index=surf_idx,
        key="surface_file_select",
        help="Auto-detected from the filename (looks for 'surface'). Override if needed.",
    )
    bedrock_file = st.sidebar.selectbox(
        "Bedrock Profile (.dat)",
        dat_files,
        index=bed_idx,
        key="bedrock_file_select",
        help="Auto-detected from the filename (looks for 'bedrock'). Override if needed.",
    )

    st.sidebar.caption(
        f"Surface → `{os.path.basename(surface_file)}`  \n"
        f"Bedrock → `{os.path.basename(bedrock_file)}`"
    )

    if os.path.abspath(surface_file) == os.path.abspath(bedrock_file):
        if len(dat_files) == 1:
            st.sidebar.error(
                "⚠️ Only **one** `.dat` file exists in the data folder, so both "
                "dropdowns point at it. Add a second `.dat` file (e.g. "
                "`steady_ELA5000_surface.dat`) to the `surface_bedrock` folder "
                "and click **🔄 Reload Data**."
            )
        else:
            st.sidebar.warning(
                "⚠️ Surface and bedrock point at the *same* file. "
                "Pick different profiles to see both traces."
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

        # ---- Y-domain: GLOBAL mesh extent, not per-chunk values ----
        # Previously `y_center` defaulted to 500 and `y_max` was read from the
        # current chunk with `chunk['Y'].max()`. That worked only because each
        # parallel chunk happened to span a full Z-layer; it also locked the
        # valley center to the old 1000 m-wide domain. We now pull BOTH the
        # extents and the center from `params`, which is populated from the
        # actual mesh bounds in Section 3c and the sidebar slider in Section 5.
        y_min = params.get('y_mesh_min', 0.0)
        y_max = params.get('y_mesh_max', 1000.0)
        y_center = params.get('y_center', (y_min + y_max) / 2.0)

        if profile_type == "U-Valley (Parabolic)":
            steepness = params.get('steepness', 0.0003)
            y_var = steepness * (y_mesh - y_center) ** 2
        elif profile_type == "V-Valley (Linear)":
            steepness = params.get('steepness', 0.05)
            y_var = steepness * np.abs(y_mesh - y_center)
        elif profile_type == "Lateral Moraines":
            sigma = params.get('width', 100.0)
            height = params.get('height', 50.0)
            # Anchor moraines to the GLOBAL walls, not to the local chunk's max.
            # `edge` keeps the ridges a fixed distance off each wall regardless
            # of how wide the mesh gets (50 m, or 5% of the span — whichever is
            # smaller, so narrow meshes stay sensible too).
            edge = min(50.0, 0.05 * (y_max - y_min))
            y_var = height * (
                np.exp(-((y_mesh - (y_min + edge)) ** 2) / (2 * sigma ** 2))
                + np.exp(-((y_mesh - (y_max - edge)) ** 2) / (2 * sigma ** 2))
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
        min_thickness = float(params.get('min_thickness', 0.1))

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

    # Pull every spatial bound from the actual mesh (Section 3c) so the UI
    # tracks the loaded geometry automatically instead of the old hard-coded
    # 0–1000 window. `z_max_orig` is likewise read from the file.
    params = {
        'z_max_orig': Z_MESH_MAX,
        'x_mesh_min': X_MESH_MIN, 'x_mesh_max': X_MESH_MAX,
        'y_mesh_min': Y_MESH_MIN, 'y_mesh_max': Y_MESH_MAX,
    }

    # Default valley center = geometric midpoint of the mesh in Y.
    default_y_center = float((Y_MESH_MIN + Y_MESH_MAX) / 2.0)

    if "U-Valley" in profile_type or "Asymmetric" in profile_type:
        params['y_center'] = st.sidebar.slider(
            "Valley Center (Y)",
            min_value=float(Y_MESH_MIN),
            max_value=float(Y_MESH_MAX),
            value=default_y_center,          # ← mesh midpoint (e.g. 1000 m)
            step=float(max(1.0, (Y_MESH_MAX - Y_MESH_MIN) / 1000.0)),
            key="y_center_slider",
            help=(
                "Midpoint of the glacier cross-section. Defaults to the mesh's "
                f"Y midpoint ({default_y_center:.1f} m). "
                f"Valid range: {Y_MESH_MIN:.1f} – {Y_MESH_MAX:.1f} m."
            ),
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
        # Trench sliders are now bounded by the actual X domain of the mesh,
        # not the stale 0–2500 m range from the old grid.
        trench_default_x = float(X_MESH_MIN + 0.72 * (X_MESH_MAX - X_MESH_MIN))
        params['trench_x'] = st.sidebar.slider(
            "Trench Center (X)",
            min_value=float(X_MESH_MIN),
            max_value=float(X_MESH_MAX),
            value=trench_default_x,
            key="trench_x_slider",
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
            st.caption(
                f"Surface: `{os.path.basename(surface_file)}` · "
                f"Bedrock: `{os.path.basename(bedrock_file)}`"
            )
            fig2d = go.Figure()
            fig2d.add_trace(go.Scatter(
                x=surface_df['X'], y=surface_df['Z'],
                name=f"Surface ({os.path.basename(surface_file)})",
                line=dict(color='blue'),
            ))
            fig2d.add_trace(go.Scatter(
                x=bedrock_df['X'], y=bedrock_df['Z'],
                name=f"Bedrock ({os.path.basename(bedrock_file)})",
                line=dict(color='brown'),
            ))
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
