import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from joblib import Parallel, delayed

# scipy is optional; if it's missing we fall back to a pivot-based grid
try:
    from scipy.interpolate import griddata as _scipy_griddata
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_griddata = None
    _SCIPY_AVAILABLE = False

# ==========================================
# 1. Page config & Robust Directory Handling
# ==========================================
st.set_page_config(page_title="Glacier Mesh Deformer & Analyzer", layout="wide")

# ------------------------------------------------------------------------------
# Path resolution — single source of truth.
#
# The script lives *inside* `himalayan_glacier`, so:
#
#     BASE_DIR = /mount/src/.../geometrical_design/himalayan_glacier
#
# and the data folders sit alongside the script:
#
#     BASE_DIR/undeformed_geometry/mesh.nodes
#     BASE_DIR/surface_bedrock/*.dat
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_MESH_DIR = os.path.join(BASE_DIR, "undeformed_geometry")
EXPECTED_DAT_DIR = os.path.join(BASE_DIR, "surface_bedrock")
EXPECTED_MESH_FILE = os.path.join(EXPECTED_MESH_DIR, "mesh.nodes")


# ------------------------------------------------------------------------------
# Diagnostic resolvers
#
# These do more than a plain `os.path.isfile` check. If the exact expected path
# is missing, they inspect the directory and try to figure out WHY:
#
#   * Case mismatch         → `Mesh.nodes` instead of `mesh.nodes`
#   * Hidden extension      → `mesh.nodes.txt` (Windows "hide extensions" trap)
#   * Any single-file dir   → almost certainly the intended file with a typo
#   * Multiple candidates   → list them so the user can see what's there
#   * Directory missing     → say so explicitly
#   * Directory empty       → say so explicitly (also catches >100 MB files
#                             that GitHub silently refused to push)
# ------------------------------------------------------------------------------
def diagnose_mesh_file():
    """Locate mesh.nodes, with a helpful diagnostic message when it's missing."""
    # 1. Exact match — the happy path.
    if os.path.isfile(EXPECTED_MESH_FILE):
        return EXPECTED_MESH_FILE, True, ""

    # 2. Directory exists? Then enumerate what's actually inside.
    if os.path.isdir(EXPECTED_MESH_DIR):
        files = os.listdir(EXPECTED_MESH_DIR)
        visible_files = [
            f for f in files
            if not f.startswith(".") and os.path.isfile(os.path.join(EXPECTED_MESH_DIR, f))
        ]

        # Case-insensitive match: `Mesh.nodes`, `MESH.NODES`, ...
        for f in visible_files:
            if f.lower() == "mesh.nodes":
                return (
                    os.path.join(EXPECTED_MESH_DIR, f),
                    True,
                    f"⚠️ Case mismatch — using `{f}` instead of the expected `mesh.nodes`. "
                    "Rename it on GitHub so future runs don't depend on this fallback.",
                )

        # Single file in the folder → almost certainly the intended file with a
        # wrong name (most often `mesh.nodes.txt` from Windows hiding extensions).
        if len(visible_files) == 1:
            only = visible_files[0]
            return (
                os.path.join(EXPECTED_MESH_DIR, only),
                True,
                f"⚠️ Expected `mesh.nodes` but found `{only}` — auto-selecting it. "
                "This is usually the Windows 'hidden .txt extension' trap; rename "
                "the file on GitHub to remove the extra suffix.",
            )

        # More than one file → refuse to guess, list them all.
        if len(visible_files) > 1:
            return (
                None,
                False,
                f"⚠️ `mesh.nodes` not found in `{EXPECTED_MESH_DIR}`.\n\n"
                f"Files present: {', '.join(f'`{f}`' for f in visible_files)}",
            )

        return (
            None,
            False,
            f"⚠️ Directory `{EXPECTED_MESH_DIR}` exists but is empty "
            "(no visible files inside). If you did place `mesh.nodes` there "
            "locally, it likely wasn't pushed — files >100 MB require Git LFS.",
        )

    return (
        None,
        False,
        f"⚠️ Directory `{EXPECTED_MESH_DIR}` does not exist.\n\n"
        f"Expected layout:\n"
        f"```\n{BASE_DIR}/\n"
        f"├── undeformed_geometry/\n"
        f"│   └── mesh.nodes\n"
        f"└── surface_bedrock/\n"
        f"    ├── steady_ELA5000_bedrock.dat\n"
        f"    └── steady_ELA5000_surface.dat\n```",
    )


def diagnose_dat_dir():
    """Return (dir_or_None, found_bool, diagnostic_message)."""
    if os.path.isdir(EXPECTED_DAT_DIR):
        dat_files = sorted(glob.glob(os.path.join(EXPECTED_DAT_DIR, "*.dat")))
        if dat_files:
            return EXPECTED_DAT_DIR, True, ""
        visible_files = [
            f for f in os.listdir(EXPECTED_DAT_DIR)
            if not f.startswith(".") and os.path.isfile(os.path.join(EXPECTED_DAT_DIR, f))
        ]
        if visible_files:
            return (
                EXPECTED_DAT_DIR,
                False,
                f"⚠️ No `.dat` files in `{EXPECTED_DAT_DIR}`.\n\n"
                f"Files present: {', '.join(f'`{f}`' for f in visible_files)}\n\n"
                "Rename to the expected pattern (`*_surface.dat`, `*_bedrock.dat`) "
                "or commit the missing files.",
            )
        return (
            EXPECTED_DAT_DIR,
            False,
            f"⚠️ Directory `{EXPECTED_DAT_DIR}` exists but is empty.",
        )
    return (
        None,
        False,
        f"⚠️ Directory `{EXPECTED_DAT_DIR}` does not exist.",
    )


mesh_nodes_file, mesh_found, mesh_diagnostic_msg = diagnose_mesh_file()
DAT_DIR, dat_dir_found, dat_diagnostic_msg = diagnose_dat_dir()
MESH_DIR = EXPECTED_MESH_DIR

st.title("🏔️ Glacier Mesh Deformer & Analyzer")
st.caption("Loads mesh and profile data, applies Y-direction variations, and exports updated meshes.")

st.info(
    f"📁 **Base directory:** `{BASE_DIR}`  \n"
    f"📁 **Mesh directory:** `{MESH_DIR}`  \n"
    f"📁 **Data directory:** `{DAT_DIR}`"
)

if st.button("🔄 Reload Data", key="reload_data_btn"):
    st.cache_data.clear()
    for k in ("surface_file_select", "bedrock_file_select"):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ==========================================
# 2. Data Loading & Validation
# ==========================================
@st.cache_data
def load_data():
    dat_files = sorted(glob.glob(os.path.join(DAT_DIR, "*.dat"))) if DAT_DIR else []
    return dat_files, len(dat_files)


dat_files, dat_files_found = load_data()

# ---- .dat files -----------------------------------------------------------
if dat_files_found == 0:
    st.warning(
        f"{dat_diagnostic_msg}\n\n"
        "**Checklist for `.dat` files on Streamlit Cloud:**\n"
        "1. File names match exactly (Linux is case-sensitive).\n"
        "2. The files are actually committed and pushed to GitHub — check the "
        "file tree on github.com, not just your local folder.\n"
        "3. The `surface_bedrock` folder isn't listed in `.gitignore`."
    )
else:
    st.success(f"✅ Found {dat_files_found} `.dat` file(s) in `{DAT_DIR}`.")
    st.caption("Files discovered: " + ", ".join(f"`{os.path.basename(f)}`" for f in dat_files))

# ---- mesh.nodes -----------------------------------------------------------
if not mesh_found:
    st.warning(
        f"{mesh_diagnostic_msg}\n\n"
        "**Why this happens on Streamlit Cloud (even if it looks correct locally):**\n"
        "1. **Hidden extensions** — Windows may have silently saved it as "
        "`mesh.nodes.txt`. The diagnostic above reveals the true filename.\n"
        "2. **Case sensitivity** — Linux is strictly case-sensitive. "
        "`Mesh.nodes` ≠ `mesh.nodes`.\n"
        "3. **Git push / file size** — the file exists *locally* but wasn't pushed "
        "to GitHub. Files >100 MB require Git LFS. Check the actual github.com "
        "file tree, not your local folder.\n"
        "4. **`.gitignore`** — ensure `.nodes` files or the folder aren't ignored.\n\n"
        "After fixing, commit, push, and press **🔄 Reload Data**."
    )
else:
    if mesh_diagnostic_msg:
        st.warning(f"✅ Mesh file located — but note: {mesh_diagnostic_msg}")
    else:
        st.success(f"✅ Found `mesh.nodes` at `{mesh_nodes_file}`.")

# ==========================================
# 3. Main App Logic (Only runs if files are found)
# ==========================================
if dat_files_found > 0 and mesh_found:

    @st.cache_data
    def read_mesh(filepath):
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
    if "surface_file_select" in st.session_state:
        del st.session_state["surface_file_select"]
    if "bedrock_file_select" in st.session_state:
        del st.session_state["bedrock_file_select"]

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
        if not isinstance(chunk, pd.DataFrame):
            raise TypeError(f"deform_chunk expected DataFrame, got {type(chunk)}")
        if chunk.empty:
            return chunk

        x_mesh = chunk['X'].to_numpy()
        y_mesh = chunk['Y'].to_numpy()

        z_bed_interp = np.interp(x_mesh, bedrock['X'].to_numpy(), bedrock['Z'].to_numpy())
        z_surf_interp = np.interp(x_mesh, surface['X'].to_numpy(), surface['Z'].to_numpy())

        # ---- Y-domain: GLOBAL mesh extent ----
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

        # CRITICAL SAFEGUARD: prevent zero-thickness (degenerate) elements
        min_thickness = float(params.get('min_thickness', 0.1))
        raw_thickness = new_top - new_bottom
        new_top = np.where(
            raw_thickness < min_thickness,
            new_bottom + min_thickness,
            new_top,
        )

        z_max_orig = params.get('z_max_orig', 5.0)
        z_normalized = chunk['Z'] / z_max_orig if z_max_orig > 0 else 0

        chunk = chunk.copy()
        chunk['Z_new'] = new_bottom + z_normalized * (new_top - new_bottom)
        return chunk

    # ==========================================
    # 4b. Net Elevation Angle (Z-X slope) Computation
    # ==========================================
    def compute_elevation_angles(surface, bedrock):
        surf = surface.sort_values('X').reset_index(drop=True)
        bed = bedrock.sort_values('X').reset_index(drop=True)

        dZ_s_dX = np.gradient(surf['Z'].to_numpy(), surf['X'].to_numpy())
        dZ_b_dX = np.gradient(bed['Z'].to_numpy(), bed['X'].to_numpy())

        angle_surface = np.degrees(np.arctan(dZ_s_dX))
        angle_bedrock = np.degrees(np.arctan(dZ_b_dX))

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
    # 4c. Visualization helpers
    # ==========================================
    # A curated set of Plotly colorscales, grouped by family so users can
    # browse them. All of these are valid Plotly names — the previous
    # "50 colormaps" list included many non-existent entries that would
    # raise at render time. If a name is ever rejected, `safe_plot` falls
    # back to Viridis instead of crashing the whole tab.
    COLORSCALES = [
        # Perceptually uniform / Matplotlib
        'Viridis', 'Cividis', 'Inferno', 'Magma', 'Plasma', 'Turbo',
        'Jet', 'Hot', 'Rainbow', 'Portland', 'Blackbody', 'Electric',
        'Earth', 'Dense', 'Deep', 'Delta', 'Solar', 'Sunset', 'Sunsetdark',
        'Picnic', 'Topo', 'Haline', 'Ice', 'Gray',
        # carto / antic / etc.
        'Aggrnyl', 'Agsunset', 'Algae', 'Amp', 'Armadillo', 'Armyrose',
        'Azure', 'Bluered', 'Blugrn', 'Bluyl', 'Brbg', 'Brwnyl',
        'Bugnyl', 'Burg', 'Burgyl', 'Darkmint', 'Emrld', 'Fall', 'Geyser',
        'Gnbu', 'Grdyl', 'Grnyl', 'Icefire', 'Mint', 'Oryel', 'Peach',
        'Pinkyl', 'Prgn', 'Purd', 'Purp', 'Purpor', 'Redor',
        'Teal', 'Tealgrn', 'Tealrose', 'Temps', 'Tropic',
        'Ylgn', 'Ylgnbu', 'Ylorbr', 'Ylorrd',
        # ColorBrewer sequential
        'Blues', 'Reds', 'Greens', 'Greys', 'Oranges', 'Purples',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        # Diverging
        'RdBu', 'RdGy', 'PRGn', 'PiYG', 'BrBG', 'PuOr',
        'RdYlBu', 'RdYlGn', 'Spectral', 'Balance', 'Curl', 'Vivid',
        # Cyclical
        'Hsv', 'Phase', 'Twilight', 'Mrybm', 'Mygbm',
    ]


    def resolve_colorscale(name, reverse=False):
        """Return (colorscale, was_valid). Falls back to Viridis if unknown."""
        base = name[:-2] if (reverse and name.endswith('_r')) else name
        if reverse and not base.endswith('_r'):
            candidate = base + '_r'
        else:
            candidate = base
        if candidate in COLORSCALES:
            return candidate, True
        # Fallback: try the base name; if that also fails, Viridis.
        if base in COLORSCALES:
            return base, False
        return 'Viridis', False


    def grid_from_xyz(df, x_col, y_col, z_col, reduce='top'):
        """
        Convert a long-form (X, Y, Z) table (multiple Z per X,Y column) into a
        regular grid suitable for `go.Surface`.

        reduce:
          'top'    — max Z at each (X, Y) column (glacier surface)
          'bottom' — min Z at each (X, Y) column (bedrock)
          'mid'    — mean Z at each (X, Y) column (mid-depth slice)

        Returns (X_grid, Y_grid, Z_grid, was_full_grid).
        Falls back to a pivot-table-based grid when scipy is unavailable.
        """
        if reduce == 'top':
            agg = df.groupby(['X', 'Y'])[z_col].max()
        elif reduce == 'bottom':
            agg = df.groupby(['X', 'Y'])[z_col].min()
        else:
            agg = df.groupby(['X', 'Y'])[z_col].mean()

        piv = agg.unstack('Y')  # index=X, columns=Y
        xi = piv.index.to_numpy()
        yi = piv.columns.to_numpy()
        Z = piv.to_numpy()

        # If the grid is complete (no NaNs), a raw pivot is already a valid
        # surface. Return immediately to save time — plotly bilinearly
        # interpolates between grid points when rendering.
        if not np.isnan(Z).any():
            X_grid, Y_grid = np.meshgrid(xi, yi, indexing='ij')
            return X_grid, Y_grid, Z, True

        # Otherwise fill gaps. Prefer scipy cubic for smoothness; fall back to
        # linear then nearest, and finally to the raw NaN-sparse pivot.
        if _SCIPY_AVAILABLE:
            pts = agg.index.to_frame(index=False).to_numpy()
            vals = agg.to_numpy()
            X_grid, Y_grid = np.meshgrid(xi, yi, indexing='ij')
            try:
                Z = _scipy_griddata(pts, vals, (X_grid, Y_grid), method='cubic')
            except Exception:
                Z = _scipy_griddata(pts, vals, (X_grid, Y_grid), method='linear')
            return X_grid, Y_grid, Z, False

        X_grid, Y_grid = np.meshgrid(xi, yi, indexing='ij')
        return X_grid, Y_grid, Z, False


    def add_visualization_controls():
        """Sidebar block that returns a dict of user-selected appearance knobs."""
        st.sidebar.header("🎨 Visualization")

        # --- colorscale ---
        colormap_name = st.sidebar.selectbox(
            "Color Scale",
            options=COLORSCALES,
            index=COLORSCALES.index('Viridis') if 'Viridis' in COLORSCALES else 0,
            key="colormap_select",
            help="Any Plotly built-in colorscale. Reverse with the checkbox below.",
        )
        colormap_reversed = st.sidebar.checkbox(
            "Reverse Colors", value=False, key="colormap_reverse"
        )

        # --- appearance ---
        with st.sidebar.expander("Line & Marker", expanded=False):
            line_width = st.slider("Line Thickness", 1, 10, 3, key="vis_line_width")
            marker_size = st.slider("Marker Size", 2, 15, 6, key="vis_marker_size")
            marker_opacity = st.slider("Marker / Surface Opacity", 0.1, 1.0, 0.85,
                                       key="vis_marker_opacity")

        with st.sidebar.expander("Axes & Grid", expanded=False):
            axes_line_width = st.slider("Axes Line Width", 1, 5, 2, key="vis_axes_lw")
            tick_width = st.slider("Tick Width", 1, 5, 2, key="vis_tick_w")
            tick_length = st.slider("Tick Length", 4, 12, 6, key="vis_tick_len")
            tick_font_size = st.slider("Tick Font Size", 8, 16, 10, key="vis_tick_fs")
            title_font_size = st.slider("Title Font Size", 10, 22, 14, key="vis_title_fs")
            grid_width = st.slider("Grid Line Width", 0, 3, 1, key="vis_grid_w")
            show_grid = st.checkbox("Show Grid", value=True, key="vis_show_grid")

        # --- 3D rendering mode ---
        with st.sidebar.expander("3D Mesh Rendering", expanded=True):
            render_mode = st.radio(
                "Render mode",
                options=["Scatter (nodes)", "Continuous surface"],
                index=1,
                key="vis_render_mode",
                help=(
                    "Scatter shows every node as a point — fast and true to the "
                    "mesh. Surface reconstructs a smooth sheet from the top/bottom "
                    "of each vertical column."
                ),
            )

            if render_mode == "Continuous surface":
                surface_layer = st.radio(
                    "Surface layer",
                    options=["Top (glacier surface)", "Bottom (bedrock)", "Mid-depth"],
                    index=0,
                    key="vis_surface_layer",
                )
                interp_method = st.selectbox(
                    "Gap-fill interpolation",
                    options=["cubic", "linear", "nearest"] if _SCIPY_AVAILABLE else ["(scipy not installed)"],
                    index=0,
                    key="vis_interp_method",
                    help="Only used when the grid has missing cells.",
                )
                show_wireframe = st.checkbox(
                    "Show wireframe overlay", value=False, key="vis_wireframe"
                )
                wireframe_width = st.slider(
                    "Wireframe width", 1, 4, 1, key="vis_wireframe_w"
                )
            else:
                surface_layer = "Top (glacier surface)"
                interp_method = "cubic"
                show_wireframe = False
                wireframe_width = 1

        # --- figure size ---
        with st.sidebar.expander("Figure Size", expanded=False):
            figure_height = st.slider("Figure height (px)", 400, 1400, 700, 50,
                                      key="vis_fig_h")

        return {
            'colormap': colormap_name,
            'colormap_reversed': colormap_reversed,
            'line_width': line_width,
            'marker_size': marker_size,
            'marker_opacity': marker_opacity,
            'axes_line_width': axes_line_width,
            'tick_width': tick_width,
            'tick_length': tick_length,
            'tick_font_size': tick_font_size,
            'title_font_size': title_font_size,
            'grid_width': grid_width,
            'show_grid': show_grid,
            'render_mode': render_mode,
            'surface_layer': surface_layer,
            'interp_method': interp_method,
            'show_wireframe': show_wireframe,
            'wireframe_width': wireframe_width,
            'figure_height': figure_height,
        }


    def make_2d_flowline_fig(surface, bedrock, vis):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=surface['X'], y=surface['Z'],
            name=f"Surface ({os.path.basename(surface_file)})",
            mode='lines+markers',
            line=dict(color='royalblue', width=vis['line_width']),
            marker=dict(size=vis['marker_size'], color='royalblue',
                        opacity=vis['marker_opacity']),
            hovertemplate='X: %{x:.1f} m<br>Z: %{y:.1f} m<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=bedrock['X'], y=bedrock['Z'],
            name=f"Bedrock ({os.path.basename(bedrock_file)})",
            mode='lines+markers',
            line=dict(color='saddlebrown', width=vis['line_width'], dash='dash'),
            marker=dict(size=vis['marker_size'], color='saddlebrown',
                        opacity=vis['marker_opacity']),
            hovertemplate='X: %{x:.1f} m<br>Z: %{y:.1f} m<extra></extra>',
        ))
        axis_cfg = dict(
            showgrid=vis['show_grid'],
            gridwidth=vis['grid_width'],
            gridcolor='lightgray',
            linewidth=vis['axes_line_width'],
            tickwidth=vis['tick_width'],
            ticklen=vis['tick_length'],
            tickfont=dict(size=vis['tick_font_size']),
            title_font=dict(size=vis['title_font_size']),
            showline=True, linecolor='black', mirror=True,
            zeroline=True, zerolinewidth=1, zerolinecolor='gray',
        )
        fig.update_layout(
            title=dict(text='1D Flowline Profiles',
                       font=dict(size=vis['title_font_size'] + 2)),
            xaxis_title='Distance X (m)',
            yaxis_title='Elevation Z (m)',
            xaxis=axis_cfg, yaxis=axis_cfg,
            height=vis['figure_height'],
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1,
                        font=dict(size=vis['tick_font_size'])),
            plot_bgcolor='white', paper_bgcolor='white',
            hovermode='x unified',
            margin=dict(l=60, r=30, t=60, b=50),
        )
        return fig


    def make_angle_fig(angles, vis):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=angles['X'], y=angles['Surface_Angle_deg'],
            name='Surface Slope Angle',
            line=dict(color='royalblue', width=vis['line_width'] + 1),
            hovertemplate='X: %{x:.1f} m<br>θ: %{y:.3f}°<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=angles['X'], y=angles['Bedrock_Angle_deg'],
            name='Bedrock Slope Angle',
            line=dict(color='saddlebrown', width=vis['line_width'], dash='dash'),
            hovertemplate='X: %{x:.1f} m<br>θ: %{y:.3f}°<extra></extra>',
        ))
        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'))
        axis_cfg = dict(
            showgrid=vis['show_grid'],
            gridwidth=vis['grid_width'],
            gridcolor='lightgray',
            linewidth=vis['axes_line_width'],
            tickwidth=vis['tick_width'],
            ticklen=vis['tick_length'],
            tickfont=dict(size=vis['tick_font_size']),
            title_font=dict(size=vis['title_font_size']),
            showline=True, linecolor='black', mirror=True,
        )
        fig.update_layout(
            title=dict(text='Net Elevation Angle (Slope) along X-axis',
                       font=dict(size=vis['title_font_size'] + 2)),
            xaxis_title='Distance X (m)',
            yaxis_title='Slope Angle (Degrees)',
            xaxis=axis_cfg, yaxis=axis_cfg,
            height=vis['figure_height'],
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            plot_bgcolor='white', paper_bgcolor='white',
            hovermode='x unified',
            margin=dict(l=60, r=30, t=60, b=50),
        )
        return fig


    def make_3d_mesh_fig(df, z_col, vis, title):
        """
        3D figure that renders either the raw node cloud (scatter) or a
        reconstructed continuous surface from the top/bottom/mid of each
        vertical column.
        """
        colorscale, _ = resolve_colorscale(
            vis['colormap'], vis['colormap_reversed']
        )

        fig = go.Figure()

        if vis['render_mode'] == "Scatter (nodes)":
            fig.add_trace(go.Scatter3d(
                x=df['X'], y=df['Y'], z=df[z_col],
                mode='markers',
                marker=dict(
                    size=vis['marker_size'],
                    color=df[z_col],
                    colorscale=colorscale,
                    opacity=vis['marker_opacity'],
                    colorbar=dict(
                        title='Elevation (m)',
                        tickfont=dict(size=vis['tick_font_size']),
                        title_font=dict(size=vis['title_font_size']),
                        len=0.75, thickness=20,
                    ),
                ),
                hovertemplate=(
                    'X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Z: %{z:.1f} m<extra></extra>'
                ),
            ))
        else:
            layer = vis['surface_layer']
            reduce = 'top' if layer.startswith('Top') else (
                'bottom' if layer.startswith('Bottom') else 'mid'
            )
            X_grid, Y_grid, Z_grid, was_full = grid_from_xyz(
                df, 'X', 'Y', z_col, reduce=reduce
            )

            fig.add_trace(go.Surface(
                x=X_grid, y=Y_grid, z=Z_grid,
                colorscale=colorscale,
                opacity=vis['marker_opacity'],
                showscale=True,
                colorbar=dict(
                    title='Elevation (m)',
                    tickfont=dict(size=vis['tick_font_size']),
                    title_font=dict(size=vis['title_font_size']),
                    len=0.75, thickness=20,
                ),
                lighting=dict(ambient=0.55, diffuse=0.7,
                              fresnel=0.15, specular=0.15, roughness=0.4),
                lightposition=dict(x=0, y=0, z=1000),
            ))

            if vis['show_wireframe']:
                nx, ny = X_grid.shape
                step_x = max(1, nx // 25)
                step_y = max(1, ny // 25)
                for i in range(0, nx, step_x):
                    fig.add_trace(go.Scatter3d(
                        x=X_grid[i, :], y=Y_grid[i, :], z=Z_grid[i, :],
                        mode='lines',
                        line=dict(color='black', width=vis['wireframe_width']),
                        opacity=0.35, showlegend=False, hoverinfo='skip',
                    ))
                for j in range(0, ny, step_y):
                    fig.add_trace(go.Scatter3d(
                        x=X_grid[:, j], y=Y_grid[:, j], z=Z_grid[:, j],
                        mode='lines',
                        line=dict(color='black', width=vis['wireframe_width']),
                        opacity=0.35, showlegend=False, hoverinfo='skip',
                    ))

            if not was_full:
                st.caption(
                    "ℹ️ Grid had missing cells — gaps were filled with "
                    f"`{vis['interp_method']}` interpolation before rendering."
                )

        fig.update_layout(
            title=dict(text=title, font=dict(size=vis['title_font_size'] + 2)),
            height=vis['figure_height'],
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor='white',
            scene=dict(
                xaxis=dict(
                    title='X (m)',
                    title_font=dict(size=vis['title_font_size']),
                    tickfont=dict(size=vis['tick_font_size']),
                    linewidth=vis['axes_line_width'],
                    showgrid=vis['show_grid'],
                    gridwidth=vis['grid_width'],
                ),
                yaxis=dict(
                    title='Y (m)',
                    title_font=dict(size=vis['title_font_size']),
                    tickfont=dict(size=vis['tick_font_size']),
                    linewidth=vis['axes_line_width'],
                    showgrid=vis['show_grid'],
                    gridwidth=vis['grid_width'],
                ),
                zaxis=dict(
                    title='Elevation Z (m)',
                    title_font=dict(size=vis['title_font_size']),
                    tickfont=dict(size=vis['tick_font_size']),
                    linewidth=vis['axes_line_width'],
                    showgrid=vis['show_grid'],
                    gridwidth=vis['grid_width'],
                ),
                aspectmode='data',
                bgcolor='white',
            ),
        )
        return fig


    # ==========================================
    # 5. Sidebar Controls
    # ==========================================
    st.sidebar.header("⛰️ Y-Direction Variation")
    profile_type = st.sidebar.selectbox(
        "Valley Profile Type",
        ["U-Valley (Parabolic)", "V-Valley (Linear)", "Lateral Moraines",
         "Asymmetric Valley", "None (Flat Slab)"],
        key="profile_type_select",
    )

    params = {
        'z_max_orig': Z_MESH_MAX,
        'x_mesh_min': X_MESH_MIN, 'x_mesh_max': X_MESH_MAX,
        'y_mesh_min': Y_MESH_MIN, 'y_mesh_max': Y_MESH_MAX,
    }

    default_y_center = float((Y_MESH_MIN + Y_MESH_MAX) / 2.0)

    if "U-Valley" in profile_type or "Asymmetric" in profile_type:
        params['y_center'] = st.sidebar.slider(
            "Valley Center (Y)",
            min_value=float(Y_MESH_MIN),
            max_value=float(Y_MESH_MAX),
            value=default_y_center,
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
            "Wall Steepness", 0.00001, 0.005, 0.0003, format="%.5f",
            key="steepness_slider"
        )
    if "Asymmetric" in profile_type:
        params['steepness_left'] = st.sidebar.slider(
            "Left Wall Steepness", 0.00001, 0.005, 0.0002, format="%.5f",
            key="steepness_left_slider"
        )
        params['steepness_right'] = st.sidebar.slider(
            "Right Wall Steepness", 0.00001, 0.005, 0.0005, format="%.5f",
            key="steepness_right_slider"
        )
    if "Moraines" in profile_type:
        params['width'] = st.sidebar.slider(
            "Moraine Width (Sigma)", 10.0, 300.0, 100.0,
            key="moraine_width_slider"
        )
        params['height'] = st.sidebar.slider(
            "Moraine Height (m)", 0.0, 200.0, 50.0,
            key="moraine_height_slider"
        )

    st.sidebar.header("🕳️ X-Direction Features")
    params['add_trench'] = st.sidebar.checkbox(
        "Add Subglacial Trench (Overdeepening)", key="trench_checkbox"
    )
    if params['add_trench']:
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

    st.sidebar.header("🛡️ Mesh Robustness")
    params['min_thickness'] = st.sidebar.number_input(
        "Minimum Ice Thickness (m)",
        min_value=0.0, max_value=50.0, value=0.1, step=0.1, format="%.2f",
        key="min_thickness_input",
        help=(
            "Guarantees new_top - new_bottom >= this value everywhere. "
            "Prevents 'ElementMetric: Degenerate 2D element' (|dCoord| = 0) "
            "errors in Elmer by never collapsing a vertical node column."
        ),
    )

    # Visualization controls live below the physics controls so the sidebar
    # reads top-to-bottom: geometry → deformation → appearance.
    vis_config = add_visualization_controls()

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
            try:
                fig = make_3d_mesh_fig(nodes_orig, 'Z', vis_config,
                                       title='Undeformed 3D Mesh')
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Could not render 3D figure: {e}")
        with col2:
            st.subheader("1D Flowline Profiles (.dat)")
            st.caption(
                f"Surface: `{os.path.basename(surface_file)}` · "
                f"Bedrock: `{os.path.basename(bedrock_file)}`"
            )
            try:
                fig2d = make_2d_flowline_fig(surface_df, bedrock_df, vis_config)
                st.plotly_chart(fig2d, width="stretch")
            except Exception as e:
                st.error(f"Could not render 2D figure: {e}")

    # --------------------------------------------------------------
    # Tab 2 — Deformed 3D Mesh
    # --------------------------------------------------------------
    with tab2:
        st.subheader("Deformed 3D Mesh with Y-Variations")
        with st.spinner("Deforming mesh using parallel processing..."):
            n_jobs = min(os.cpu_count() or 4, 8)
            idx_chunks = np.array_split(np.arange(len(nodes_orig)), n_jobs)
            chunks = [
                nodes_orig.iloc[idx].reset_index(drop=True).copy()
                for idx in idx_chunks
            ]

            results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(deform_chunk)(chunk, bedrock_df, surface_df,
                                      profile_type, params)
                for chunk in chunks
            )
            nodes_deformed = pd.concat(results, ignore_index=True)

        # ---- Post-deformation sanity check --------------------------------
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

        try:
            fig_def = make_3d_mesh_fig(
                nodes_deformed, 'Z_new', vis_config,
                title=f"Deformed 3D Mesh — {profile_type}"
            )
            st.plotly_chart(fig_def, width="stretch")
        except Exception as e:
            st.error(f"Could not render deformed 3D figure: {e}")

        st.session_state['nodes_deformed'] = nodes_deformed

    # --------------------------------------------------------------
    # Tab 3 — Net Elevation Angle (Z-X) numerics + plots
    # --------------------------------------------------------------
    angles_df = compute_elevation_angles(surface_df, bedrock_df)
    st.session_state['angles_df'] = angles_df

    with tab3:
        st.subheader("📐 Net Elevation Angle θ = arctan(dZ/dX) along the X-axis")
        st.caption(
            "Numerical derivative computed with second-order central differences "
            "in the interior and one-sided differences at the endpoints (via "
            "`np.gradient`), which handles non-uniform X spacing in the `.dat` files."
        )

        try:
            fig_ang = make_angle_fig(angles_df, vis_config)
            st.plotly_chart(fig_ang, width="stretch")
        except Exception as e:
            st.error(f"Could not render angle figure: {e}")

        st.markdown("### 📊 Numeric Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Max Surface Angle", f"{angles_df['Surface_Angle_deg'].max():.3f}°")
        with c2: st.metric("Min Surface Angle", f"{angles_df['Surface_Angle_deg'].min():.3f}°")
        with c3: st.metric("Max Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].max():.3f}°")
        with c4: st.metric("Min Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].min():.3f}°")

        c5, c6, c7, c8 = st.columns(4)
        with c5: st.metric("Mean Surface Angle", f"{angles_df['Surface_Angle_deg'].mean():.3f}°")
        with c6: st.metric("Mean Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].mean():.3f}°")
        with c7: st.metric("Std Surface Angle", f"{angles_df['Surface_Angle_deg'].std():.3f}°")
        with c8: st.metric("Std Bedrock Angle", f"{angles_df['Bedrock_Angle_deg'].std():.3f}°")

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

        csv_angles_tab3 = angles_df.to_csv(index=False, float_format='%.8f')
        st.download_button(
            "⬇️ Download elevation_angles.csv",
            csv_angles_tab3,
            file_name="elevation_angles.csv",
            mime="text/csv",
            key="download_angles_tab3",
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
                key="download_mesh_tab4",
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
                key="download_angles_tab4",
            )
