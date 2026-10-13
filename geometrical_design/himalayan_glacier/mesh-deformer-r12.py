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
#     BASE_DIR = .../himalayan_glacier
#     BASE_DIR/undeformed_geometry/mesh.nodes
#     BASE_DIR/surface_bedrock/*.dat
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_MESH_DIR = os.path.join(BASE_DIR, "undeformed_geometry")
EXPECTED_DAT_DIR = os.path.join(BASE_DIR, "surface_bedrock")
EXPECTED_MESH_FILE = os.path.join(EXPECTED_MESH_DIR, "mesh.nodes")


# ------------------------------------------------------------------------------
# Diagnostic resolvers
# ------------------------------------------------------------------------------
def diagnose_mesh_file():
    """Locate mesh.nodes, with a helpful diagnostic message when it's missing."""
    if os.path.isfile(EXPECTED_MESH_FILE):
        return EXPECTED_MESH_FILE, True, ""

    if os.path.isdir(EXPECTED_MESH_DIR):
        files = os.listdir(EXPECTED_MESH_DIR)
        visible_files = [
            f for f in files
            if not f.startswith(".") and os.path.isfile(os.path.join(EXPECTED_MESH_DIR, f))
        ]

        for f in visible_files:
            if f.lower() == "mesh.nodes":
                return (
                    os.path.join(EXPECTED_MESH_DIR, f),
                    True,
                    f"⚠️ Case mismatch — using `{f}` instead of the expected `mesh.nodes`. "
                    "Rename it on GitHub so future runs don't depend on this fallback.",
                )

        if len(visible_files) == 1:
            only = visible_files[0]
            return (
                os.path.join(EXPECTED_MESH_DIR, only),
                True,
                f"⚠️ Expected `mesh.nodes` but found `{only}` — auto-selecting it. "
                "This is usually the Windows 'hidden .txt extension' trap; rename "
                "the file on GitHub to remove the extra suffix.",
            )

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
    for k in (
        "surface_file_select", "bedrock_file_select",
        "surface_df_mod", "bedrock_df_mod",
        "_profile_mod_sources",
        "use_modified_profiles", "use_modified_profiles_cb",
        "surface_point_editor", "bedrock_point_editor",
        "surf_pt_idx", "bed_pt_idx",
    ):
        if k in st.session_state:
            del st.session_state[k]
    # Also drop any per-point slider widget keys.
    for k in list(st.session_state.keys()):
        if k.startswith("surf_z_slider_") or k.startswith("bed_z_slider_"):
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
    # NOTE: we no longer unconditionally delete the selectbox keys. Streamlit
    # restores the widget value from session_state; wiping it on every rerun
    # was silently reverting the user's manual selection to the auto-detected
    # default. The "🔄 Reload Data" button still clears these keys.
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

        # Sort both profiles once (np.interp requires increasing xp).
        surf_s = surface.sort_values('X')
        bed_s = bedrock.sort_values('X')
        z_bed_interp = np.interp(x_mesh, bed_s['X'].to_numpy(), bed_s['Z'].to_numpy())
        z_surf_interp = np.interp(x_mesh, surf_s['X'].to_numpy(), surf_s['Z'].to_numpy())

        y_min = params.get('y_mesh_min', 0.0)
        y_max = params.get('y_mesh_max', 1000.0)
        y_center = params.get('y_center', (y_min + y_max) / 2.0)

        if profile_type == "U-Valley (Parabolic)":
            steepness = params.get('steepness_u', 0.0003)
            y_var = steepness * (y_mesh - y_center) ** 2
        elif profile_type == "V-Valley (Linear)":
            steepness = params.get('steepness_v', 0.05)
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
        surf = surface.sort_values('X').drop_duplicates('X').reset_index(drop=True)
        bed = bedrock.sort_values('X').drop_duplicates('X').reset_index(drop=True)

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

    # ============================================================
    # 4b2. Profile Modification Utilities
    # ============================================================

    def apply_profile_shift(df, shift_m):
        """Shift all Z values by a constant offset."""
        df = df.copy()
        df['Z'] = df['Z'] + shift_m
        return df

    def apply_profile_scale(df, scale_factor, anchor='mean'):
        """Scale Z values around an anchor point (start/mean/absolute)."""
        df = df.copy()
        if anchor == 'start':
            z0 = df['Z'].iloc[0]
            df['Z'] = z0 + (df['Z'] - z0) * scale_factor
        elif anchor == 'mean':
            z_mean = df['Z'].mean()
            df['Z'] = z_mean + (df['Z'] - z_mean) * scale_factor
        else:
            df['Z'] = df['Z'] * scale_factor
        return df

    def apply_profile_smooth(df, window=5):
        """Moving-average smoothing."""
        df = df.copy().sort_values('X').reset_index(drop=True)
        df['Z'] = df['Z'].rolling(window=window, center=True,
                                  min_periods=1).mean()
        return df

    def apply_profile_tilt(df, slope_m_per_km):
        """Apply a linear tilt across the profile."""
        df = df.copy()
        x_km = (df['X'] - df['X'].min()) / 1000.0
        df['Z'] = df['Z'] + slope_m_per_km * x_km
        return df

    def apply_profile_bump(df, center_x, width, amplitude):
        """Add or subtract a Gaussian bump at a chosen location."""
        df = df.copy()
        bump = amplitude * np.exp(-((df['X'] - center_x) ** 2) / (2 * width ** 2))
        df['Z'] = df['Z'] + bump
        return df

    def apply_profile_step(df, step_x, step_height):
        """Add a step function (useful for icefall / riegel features)."""
        df = df.copy()
        df['Z'] = df['Z'] + np.where(df['X'] >= step_x, step_height, 0.0)
        return df

    def apply_custom_function(df, expression):
        """Evaluate a user-supplied Python expression as Z_new = f(x, z_orig).

        The expression has access to:
          x       — numpy array of X distances
          z_orig  — numpy array of original Z values
          np      — numpy module
        """
        df = df.copy()
        x = df['X'].to_numpy()
        z_orig = df['Z'].to_numpy()
        safe_globals = {'np': np, 'x': x, 'z_orig': z_orig}
        new_z = eval(expression, {'__builtins__': {}}, safe_globals)
        new_z = np.asarray(new_z, dtype=float)
        if new_z.shape != z_orig.shape:
            new_z = np.full_like(z_orig, float(new_z))
        df['Z'] = new_z
        return df

    def _sorted_xy(df):
        d = df.sort_values('X').drop_duplicates('X').reset_index(drop=True)
        return d['X'].to_numpy(), d['Z'].to_numpy()

    def enforce_max_depth(surface_df, bedrock_df, max_depth):
        """Hanging-glacier constraint: raise bedrock so depth ≤ max_depth."""
        bedrock_df = bedrock_df.copy()
        sx, sz = _sorted_xy(surface_df)
        bx = bedrock_df['X'].to_numpy()
        surf_z = np.interp(bx, sx, sz)
        depth = surf_z - bedrock_df['Z'].to_numpy()
        mask = depth > max_depth
        bedrock_df.loc[mask, 'Z'] = surf_z[mask] - max_depth
        return bedrock_df

    def enforce_min_depth(surface_df, bedrock_df, min_depth):
        """Prevent ice vanishing: lower bedrock so depth ≥ min_depth."""
        bedrock_df = bedrock_df.copy()
        sx, sz = _sorted_xy(surface_df)
        bx = bedrock_df['X'].to_numpy()
        surf_z = np.interp(bx, sx, sz)
        depth = surf_z - bedrock_df['Z'].to_numpy()
        mask = depth < min_depth
        bedrock_df.loc[mask, 'Z'] = surf_z[mask] - min_depth
        return bedrock_df

    def enforce_bedrock_below_surface(surface_df, bedrock_df, margin=0.5):
        """Hard safeguard: bedrock must never pierce the surface."""
        bedrock_df = bedrock_df.copy()
        sx, sz = _sorted_xy(surface_df)
        bx = bedrock_df['X'].to_numpy()
        surf_z = np.interp(bx, sx, sz)
        mask = bedrock_df['Z'].to_numpy() >= surf_z - margin
        bedrock_df.loc[mask, 'Z'] = surf_z[mask] - margin
        return bedrock_df

    def resample_profile(df, dx=50.0, x_min=None, x_max=None):
        """Resample to uniform X spacing via linear interpolation."""
        d = df.sort_values('X').drop_duplicates('X')
        x_min = x_min if x_min is not None else float(d['X'].min())
        x_max = x_max if x_max is not None else float(d['X'].max())
        x_new = np.arange(x_min, x_max + dx / 2, dx)
        z_new = np.interp(x_new, d['X'].to_numpy(), d['Z'].to_numpy())
        return pd.DataFrame({'X': x_new, 'Z': z_new})

    def profile_to_dat_string(df):
        """Serialise a profile DataFrame to .dat file content."""
        d = df.sort_values('X')
        lines = []
        for _, row in d.iterrows():
            lines.append(f"{row['X']:.4f}    {row['Z']:.4f}")
        return "\n".join(lines) + "\n"

    def regrid_to_common_x(surface_df, bedrock_df):
        """Return (surface, bedrock) resampled onto the bedrock X grid."""
        bx = bedrock_df.sort_values('X').drop_duplicates('X')
        x_common = bx['X'].to_numpy()
        sx, sz = _sorted_xy(surface_df)
        surf_z = np.interp(x_common, sx, sz)
        surface_aligned = pd.DataFrame({'X': x_common, 'Z': surf_z})
        return surface_aligned, bx.reset_index(drop=True).copy()

    def validate_profiles(surface_df, bedrock_df, max_depth=None, min_depth=None):
        """Return a dict of validation flags and statistics."""
        surf, bed = regrid_to_common_x(surface_df, bedrock_df)
        depth = surf['Z'].to_numpy() - bed['Z'].to_numpy()
        flags = {
            'depth_max': float(depth.max()),
            'depth_min': float(depth.min()),
            'depth_mean': float(depth.mean()),
            'depth_std': float(depth.std()),
            'n_above_surface': int((depth < 0).sum()),
            'n_exceeds_max': int((depth > max_depth).sum()) if max_depth is not None else 0,
            'n_below_min': int((depth < min_depth).sum()) if min_depth is not None else 0,
            'n_points': len(depth),
        }
        flags['is_valid'] = (
            flags['n_above_surface'] == 0
            and flags['n_exceeds_max'] == 0
            and flags['n_below_min'] == 0
        )
        return flags

    # ==========================================
    # 4c. Visualization helpers
    # ==========================================
    COLORSCALES = [
        'Viridis', 'Cividis', 'Inferno', 'Magma', 'Plasma', 'Turbo',
        'Jet', 'Hot', 'Rainbow', 'Portland', 'Blackbody', 'Electric',
        'Earth', 'Dense', 'Deep', 'Delta', 'Solar', 'Sunset', 'Sunsetdark',
        'Picnic', 'Topo', 'Haline', 'Ice', 'Gray',
        'Aggrnyl', 'Agsunset', 'Algae', 'Amp', 'Armadillo', 'Armyrose',
        'Azure', 'Bluered', 'Blugrn', 'Bluyl', 'Brbg', 'Brwnyl',
        'Bugnyl', 'Burg', 'Burgyl', 'Darkmint', 'Emrld', 'Fall', 'Geyser',
        'Gnbu', 'Grdyl', 'Grnyl', 'Icefire', 'Mint', 'Oryel', 'Peach',
        'Pinkyl', 'Prgn', 'Purd', 'Purp', 'Purpor', 'Redor',
        'Teal', 'Tealgrn', 'Tealrose', 'Temps', 'Tropic',
        'Ylgn', 'Ylgnbu', 'Ylorbr', 'Ylorrd',
        'Blues', 'Reds', 'Greens', 'Greys', 'Oranges', 'Purples',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'RdBu', 'RdGy', 'PRGn', 'PiYG', 'BrBG', 'PuOr',
        'RdYlBu', 'RdYlGn', 'Spectral', 'Balance', 'Curl', 'Vivid',
        'Hsv', 'Phase', 'Twilight', 'Mrybm', 'Mygbm',
    ]

    def resolve_colorscale(name, reverse=False):
        """Return (colorscale, was_valid). Plotly accepts the `_r` suffix
        for any built-in colorscale, so reversal is just a name append."""
        base = name[:-2] if name.endswith('_r') else name
        candidate = base + '_r' if reverse else base
        valid = base in COLORSCALES
        return candidate, valid

    def grid_from_xyz(df, x_col, y_col, z_col, reduce='top', interp_method='cubic'):
        """Convert a long-form (X, Y, Z) table into a regular grid."""
        if reduce == 'top':
            agg = df.groupby(['X', 'Y'])[z_col].max()
        elif reduce == 'bottom':
            agg = df.groupby(['X', 'Y'])[z_col].min()
        else:
            agg = df.groupby(['X', 'Y'])[z_col].mean()

        piv = agg.unstack('Y')
        xi = piv.index.to_numpy()
        yi = piv.columns.to_numpy()
        Z = piv.to_numpy()

        if not np.isnan(Z).any():
            X_grid, Y_grid = np.meshgrid(xi, yi, indexing='ij')
            return X_grid, Y_grid, Z, True

        if _SCIPY_AVAILABLE:
            pts = agg.index.to_frame(index=False).to_numpy()
            vals = agg.to_numpy()
            X_grid, Y_grid = np.meshgrid(xi, yi, indexing='ij')
            try:
                Z = _scipy_griddata(pts, vals, (X_grid, Y_grid), method=interp_method)
                if np.isnan(Z).any() and interp_method != 'linear':
                    Z = _scipy_griddata(pts, vals, (X_grid, Y_grid), method='linear')
                if np.isnan(Z).any():
                    Z = _scipy_griddata(pts, vals, (X_grid, Y_grid), method='nearest')
            except Exception:
                Z = _scipy_griddata(pts, vals, (X_grid, Y_grid), method='linear')
            return X_grid, Y_grid, Z, False

        filled = piv.interpolate(axis=0, limit_direction='both') \
                    .interpolate(axis=1, limit_direction='both')
        X_grid, Y_grid = np.meshgrid(xi, yi, indexing='ij')
        return X_grid, Y_grid, filled.to_numpy(), False

    def add_visualization_controls():
        """Sidebar block that returns a dict of user-selected appearance knobs."""
        st.sidebar.header("🎨 Visualization")

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

            # ── Z-scale exaggeration (always available for 3D) ──
            z_exaggeration = st.slider(
                "Z-scale exaggeration",
                min_value=0.1, max_value=20.0, value=1.0, step=0.1,
                key="vis_z_exaggeration",
                help=(
                    "Multiply Z values by this factor to exaggerate vertical "
                    "relief. 1.0 = true scale, >1 = stretched vertically, "
                    "<1 = compressed."
                ),
            )

            if render_mode == "Continuous surface":
                surface_layer = st.radio(
                    "Surface layer",
                    options=[
                        "Full geometry (top + bottom)",
                        "Top (glacier surface)",
                        "Bottom (bedrock)",
                        "Mid-depth",
                    ],
                    index=0,          # ← default is now Full geometry
                    key="vis_surface_layer",
                )

                # ── Extra controls for Full geometry mode ──
                if surface_layer.startswith("Full"):
                    show_side_walls = st.checkbox(
                        "Show side walls (close volume)",
                        value=True,
                        key="vis_show_side_walls",
                        help="Render filled side panels connecting top and bottom surfaces.",
                    )
                    bedrock_cscale_name = st.selectbox(
                        "Bedrock colorscale",
                        options=COLORSCALES,
                        index=COLORSCALES.index('Earth') if 'Earth' in COLORSCALES else 0,
                        key="vis_bedrock_cscale",
                        help="Colorscale used for the bottom (bedrock) surface.",
                    )
                    bedrock_cscale_reversed = st.checkbox(
                        "Reverse bedrock colors", value=False,
                        key="vis_bedrock_cscale_rev",
                    )

                    # ── Per-surface opacity & rendering style ──
                    st.markdown("**🧊 Top surface (glacier)**")
                    top_render_style = st.radio(
                        "Top render style",
                        options=[
                            "Transparent solid",
                            "Wireframe only",
                            "Ghost (single color)",
                            "Hidden",
                        ],
                        index=0,
                        key="vis_top_render_style",
                        horizontal=True,
                        help=(
                            "• **Transparent solid** — semi-transparent surface with "
                            "colormap; reduce opacity to see bedrock beneath.\n"
                            "• **Wireframe only** — just mesh lines, no fill; "
                            "bedrock colormap is fully visible.\n"
                            "• **Ghost** — single ice-blue color with adjustable "
                            "transparency; bedrock colormap dominates.\n"
                            "• **Hidden** — top surface not rendered; use to "
                            "inspect bedrock in isolation."
                        ),
                    )
                    top_opacity = st.slider(
                        "Top surface opacity",
                        min_value=0.05, max_value=1.0, value=0.55, step=0.05,
                        key="vis_top_opacity",
                        help=(
                            "Opacity of the glacier surface. Lower values "
                            "reveal the bedrock beneath. Ignored for "
                            "'Wireframe only' and 'Hidden' modes."
                        ),
                    )

                    st.markdown("**🪨 Bottom surface (bedrock)**")
                    bottom_opacity = st.slider(
                        "Bottom surface opacity",
                        min_value=0.05, max_value=1.0, value=0.95, step=0.05,
                        key="vis_bottom_opacity",
                        help="Opacity of the bedrock surface.",
                    )
                else:
                    show_side_walls = False
                    bedrock_cscale_name = 'Earth'
                    bedrock_cscale_reversed = False
                    top_render_style = 'Transparent solid'
                    top_opacity = 0.55
                    bottom_opacity = 0.95

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
                show_side_walls = False
                bedrock_cscale_name = 'Earth'
                bedrock_cscale_reversed = False
                top_render_style = 'Transparent solid'
                top_opacity = 0.55
                bottom_opacity = 0.95

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
            'z_exaggeration': z_exaggeration,
            'show_side_walls': show_side_walls,
            'bedrock_cscale': bedrock_cscale_name,
            'bedrock_cscale_reversed': bedrock_cscale_reversed,
            'top_render_style': top_render_style,
            'top_opacity': top_opacity,
            'bottom_opacity': bottom_opacity,
        }

    def make_2d_flowline_fig(surface, bedrock, vis,
                             surface_label=None, bedrock_label=None):
        fig = go.Figure()
        if surface_label is None:
            surface_label = f"Surface ({os.path.basename(surface_file)})"
        if bedrock_label is None:
            bedrock_label = f"Bedrock ({os.path.basename(bedrock_file)})"
        fig.add_trace(go.Scatter(
            x=surface['X'], y=surface['Z'],
            name=surface_label,
            mode='lines+markers',
            line=dict(color='royalblue', width=vis['line_width']),
            marker=dict(size=vis['marker_size'], color='royalblue',
                        opacity=vis['marker_opacity']),
            hovertemplate='X: %{x:.1f} m<br>Z: %{y:.1f} m<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=bedrock['X'], y=bedrock['Z'],
            name=bedrock_label,
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

    def make_modified_flowline_fig(surface_mod, bedrock_mod, vis,
                                   surface_orig=None, bedrock_orig=None,
                                   show_ice_thickness=True,
                                   show_orig_ghost=False,
                                   show_markers=True,
                                   show_depth_refs=True):
        """Dedicated figure for modified profiles only.

        Row 1 : modified surface + bedrock  (optional faded originals)
        Row 2 : ice thickness = surface − bedrock  (optional)
        """
        from plotly.subplots import make_subplots

        rows = 2 if show_ice_thickness else 1
        row_heights = [0.58, 0.42] if show_ice_thickness else [1.0]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=row_heights,
        )

        lw = vis['line_width']
        ms = vis['marker_size']
        mo = vis['marker_opacity']
        mode_str = 'lines+markers' if show_markers else 'lines'

        # ─────────────────────────────────────────────────────
        #  Faded original reference (ghost)
        # ─────────────────────────────────────────────────────
        if show_orig_ghost and surface_orig is not None:
            fig.add_trace(go.Scatter(
                x=surface_orig['X'], y=surface_orig['Z'],
                name='Original Surface (ref)',
                mode='lines',
                line=dict(color='royalblue', width=1, dash='dot'),
                opacity=0.30,
                hovertemplate='X: %{x:.1f} m<br>Z: %{y:.1f} m (orig)<extra></extra>',
            ), row=1, col=1)

        if show_orig_ghost and bedrock_orig is not None:
            fig.add_trace(go.Scatter(
                x=bedrock_orig['X'], y=bedrock_orig['Z'],
                name='Original Bedrock (ref)',
                mode='lines',
                line=dict(color='saddlebrown', width=1, dash='dot'),
                opacity=0.30,
                hovertemplate='X: %{x:.1f} m<br>Z: %{y:.1f} m (orig)<extra></extra>',
            ), row=1, col=1)

        # ─────────────────────────────────────────────────────
        #  Modified surface
        # ─────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=surface_mod['X'], y=surface_mod['Z'],
            name='Modified Surface',
            mode=mode_str,
            line=dict(color='dodgerblue', width=lw),
            marker=dict(size=ms, color='dodgerblue', opacity=mo),
            hovertemplate='X: %{x:.1f} m<br>Z: %{y:.1f} m<extra></extra>',
        ), row=1, col=1)

        # ─────────────────────────────────────────────────────
        #  Modified bedrock
        # ─────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=bedrock_mod['X'], y=bedrock_mod['Z'],
            name='Modified Bedrock',
            mode=mode_str,
            line=dict(color='sienna', width=lw, dash='dash'),
            marker=dict(size=ms, color='sienna', opacity=mo),
            hovertemplate='X: %{x:.1f} m<br>Z: %{y:.1f} m<extra></extra>',
        ), row=1, col=1)

        # ─────────────────────────────────────────────────────
        #  Depth reference lines on profile panel
        # ─────────────────────────────────────────────────────
        if show_depth_refs:
            z_all = np.concatenate([
                surface_mod['Z'].to_numpy(),
                bedrock_mod['Z'].to_numpy(),
            ])
            z_min_val = float(np.floor(z_all.min() / 500) * 500)
            z_max_val = float(np.ceil(z_all.max() / 500) * 500)
            for z_ref in np.arange(z_min_val, z_max_val + 1, 500):
                fig.add_hline(
                    y=z_ref,
                    line=dict(color='lightgray', width=0.5, dash='dot'),
                    row=1, col=1,
                )

        # ─────────────────────────────────────────────────────
        #  Ice thickness subplot
        # ─────────────────────────────────────────────────────
        if show_ice_thickness:
            surf_s = surface_mod.sort_values('X').drop_duplicates('X')
            bed_s = bedrock_mod.sort_values('X').drop_duplicates('X')

            x_common = bed_s['X'].to_numpy()
            z_surf = np.interp(
                x_common,
                surf_s['X'].to_numpy(),
                surf_s['Z'].to_numpy(),
            )
            z_bed = bed_s['Z'].to_numpy()
            ice_thickness = z_surf - z_bed

            fig.add_trace(go.Scatter(
                x=x_common, y=ice_thickness,
                name='Ice Thickness',
                mode='lines+markers' if show_markers else 'lines',
                line=dict(color='mediumseagreen', width=lw),
                marker=dict(size=max(2, ms - 2), color='mediumseagreen',
                            opacity=mo),
                fill='tozeroy',
                fillcolor='rgba(60, 179, 113, 0.15)',
                hovertemplate='X: %{x:.1f} m<br>H: %{y:.1f} m<extra></extra>',
            ), row=2, col=1)

            # ── Max thickness star marker ──
            i_max = int(np.argmax(ice_thickness))
            fig.add_trace(go.Scatter(
                x=[x_common[i_max]],
                y=[ice_thickness[i_max]],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='star'),
                text=[f'{ice_thickness[i_max]:.1f} m'],
                textposition='top center',
                textfont=dict(size=vis['tick_font_size'], color='red'),
                name=f'Max thickness',
                showlegend=True,
            ), row=2, col=1)

            # ── Thickness reference lines ──
            if show_depth_refs:
                h_max = float(ice_thickness.max())
                for h_ref in range(10, int(h_max) + 1, 10):
                    fig.add_hline(
                        y=h_ref,
                        line=dict(color='lightgray', width=0.5, dash='dot'),
                        row=2, col=1,
                    )

        # ─────────────────────────────────────────────────────
        #  Layout
        # ─────────────────────────────────────────────────────
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
            title=dict(
                text='Modified Flowline Profiles',
                font=dict(size=vis['title_font_size'] + 2),
            ),
            height=vis['figure_height'] * (1.5 if show_ice_thickness else 1.0),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.02,
                xanchor='right', x=1,
                font=dict(size=vis['tick_font_size']),
            ),
            plot_bgcolor='white', paper_bgcolor='white',
            hovermode='x unified',
            margin=dict(l=60, r=30, t=60, b=50),
        )

        fig.update_xaxes(
            title_text='' if show_ice_thickness else 'Distance X (m)',
            **axis_cfg, row=1, col=1,
        )
        fig.update_yaxes(
            title_text='Elevation Z (m)',
            **axis_cfg, row=1, col=1,
        )

        if show_ice_thickness:
            fig.update_xaxes(
                title_text='Distance X (m)',
                **axis_cfg, row=2, col=1,
            )
            fig.update_yaxes(
                title_text='Ice Thickness (m)',
                **axis_cfg, row=2, col=1,
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
        colorscale, _ = resolve_colorscale(
            vis['colormap'], vis['colormap_reversed']
        )

        z_exag = vis.get('z_exaggeration', 1.0)

        # ── Side-wall colorscale (subtle grey-brown) ──
        WALL_COLOR = [
            [0.0, 'rgb(130, 110, 90)'],
            [1.0, 'rgb(160, 140, 120)'],
        ]

        # ── Ghost-mode single-color colorscale (ice blue) ──
        GHOST_ICE_BLUE = [
            [0.0, 'rgb(135, 186, 235)'],
            [1.0, 'rgb(135, 186, 235)'],
        ]

        # ── Contour colorscale for top surface edges ──
        GHOST_CONTOUR = [
            [0.0, 'rgb(70, 130, 210)'],
            [1.0, 'rgb(70, 130, 210)'],
        ]

        top_style = vis.get('top_render_style', 'Transparent solid')
        top_op = vis.get('top_opacity', 0.55)
        bot_op = vis.get('bottom_opacity', 0.95)

        fig = go.Figure()

        # ═══════════════════════════════════════════════════════════
        #  Scatter mode
        # ═══════════════════════════════════════════════════════════
        if vis['render_mode'] == "Scatter (nodes)":
            z_plot = df[z_col] * z_exag
            fig.add_trace(go.Scatter3d(
                x=df['X'], y=df['Y'], z=z_plot,
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
                    'X: %{x:.1f} m<br>Y: %{y:.1f} m<br>'
                    f'Z: %{{z:.1f}} m' +
                    (f'  (×{z_exag:.1f})' if z_exag != 1.0 else '') +
                    '<extra></extra>'
                ),
            ))

        # ═══════════════════════════════════════════════════════════
        #  Continuous surface mode
        # ═══════════════════════════════════════════════════════════
        else:
            layer = vis['surface_layer']

            # ───────────────────────────────────────────────────────
            #  Full geometry: top + bottom + optional side walls
            # ───────────────────────────────────────────────────────
            if layer.startswith('Full'):
                # Compute both surfaces
                X_top, Y_top, Z_top, was_full_top = grid_from_xyz(
                    df, 'X', 'Y', z_col,
                    reduce='top',
                    interp_method=vis['interp_method'],
                )
                X_bot, Y_bot, Z_bot, was_full_bot = grid_from_xyz(
                    df, 'X', 'Y', z_col,
                    reduce='bottom',
                    interp_method=vis['interp_method'],
                )

                # Apply Z exaggeration
                Z_top_plot = Z_top * z_exag
                Z_bot_plot = Z_bot * z_exag

                # Resolve bedrock colorscale
                bed_cscale, _ = resolve_colorscale(
                    vis.get('bedrock_cscale', 'Earth'),
                    vis.get('bedrock_cscale_reversed', False),
                )

                # Decide colorbar positioning
                top_has_colorbar = top_style in (
                    'Transparent solid', 'Ghost (single color)'
                )
                # Ghost still gets a colorbar showing Z range
                top_shows_scale = (top_style == 'Transparent solid')
                # Offset bottom colorbar when top also has one
                bot_cbar_x = 1.15 if top_has_colorbar else 1.0

                # ─────────────────────────────────────────────────
                #  Top surface (glacier) — 4 rendering styles
                # ─────────────────────────────────────────────────
                top_lighting = dict(
                    ambient=0.55, diffuse=0.7,
                    fresnel=0.15, specular=0.15, roughness=0.4,
                )

                if top_style == 'Transparent solid':
                    fig.add_trace(go.Surface(
                        x=X_top, y=Y_top, z=Z_top_plot,
                        colorscale=colorscale,
                        opacity=top_op,
                        showscale=top_shows_scale,
                        colorbar=dict(
                            title='Z surf (m)',
                            tickfont=dict(size=vis['tick_font_size']),
                            title_font=dict(size=vis['title_font_size']),
                            len=0.75, thickness=20,
                            x=1.0,
                        ),
                        name='Glacier Surface',
                        lighting=top_lighting,
                        lightposition=dict(x=0, y=0, z=1000),
                        contours=dict(
                            z=dict(
                                show=True,
                                usecolormap=False,
                                color='rgba(255,255,255,0.25)',
                                highlightcolor='rgba(255,255,255,0.4)',
                                project=dict(z=False),
                            ),
                        ),
                    ))

                elif top_style == 'Ghost (single color)':
                    fig.add_trace(go.Surface(
                        x=X_top, y=Y_top, z=Z_top_plot,
                        colorscale=GHOST_ICE_BLUE,
                        opacity=top_op,
                        showscale=False,
                        name='Glacier Surface (ghost)',
                        lighting=dict(
                            ambient=0.7, diffuse=0.5,
                            fresnel=0.05, specular=0.05, roughness=0.6,
                        ),
                        lightposition=dict(x=0, y=0, z=800),
                        contours=dict(
                            z=dict(
                                show=True,
                                usecolormap=False,
                                color='rgba(40,100,200,0.3)',
                                highlightcolor='rgba(40,100,200,0.5)',
                                project=dict(z=False),
                            ),
                        ),
                    ))

                elif top_style == 'Wireframe only':
                    # No filled Surface trace — just wireframe lines
                    nx_t, ny_t = X_top.shape
                    step_x = max(1, nx_t // 35)
                    step_y = max(1, ny_t // 35)
                    wf_color = 'royalblue'
                    wf_w = 2
                    for i in range(0, nx_t, step_x):
                        fig.add_trace(go.Scatter3d(
                            x=X_top[i, :], y=Y_top[i, :], z=Z_top_plot[i, :],
                            mode='lines',
                            line=dict(color=wf_color, width=wf_w),
                            opacity=0.65,
                            showlegend=(i == 0),
                            name='Glacier wireframe' if i == 0 else '',
                            hoverinfo='skip',
                        ))
                    for j in range(0, ny_t, step_y):
                        fig.add_trace(go.Scatter3d(
                            x=X_top[:, j], y=Y_top[:, j], z=Z_top_plot[:, j],
                            mode='lines',
                            line=dict(color=wf_color, width=wf_w),
                            opacity=0.65,
                            showlegend=False, hoverinfo='skip',
                        ))

                # 'Hidden' → nothing rendered for top surface

                # ─────────────────────────────────────────────────
                #  Bottom surface (bedrock) — always rendered
                # ─────────────────────────────────────────────────
                fig.add_trace(go.Surface(
                    x=X_bot, y=Y_bot, z=Z_bot_plot,
                    colorscale=bed_cscale,
                    opacity=bot_op,
                    showscale=True,
                    colorbar=dict(
                        title='Z bed (m)',
                        tickfont=dict(size=vis['tick_font_size']),
                        title_font=dict(size=vis['title_font_size']),
                        len=0.75, thickness=20,
                        x=bot_cbar_x,
                    ),
                    name='Bedrock',
                    lighting=dict(
                        ambient=0.65, diffuse=0.6,
                        fresnel=0.1, specular=0.1, roughness=0.5,
                    ),
                    lightposition=dict(x=0, y=0, z=500),
                ))

                # ─────────────────────────────────────────────────
                #  Side walls (filled panels closing the volume)
                # ─────────────────────────────────────────────────
                if vis.get('show_side_walls', False):
                    # Align bottom grid to top grid if shapes differ
                    if X_bot.shape == X_top.shape:
                        Z_bot_aligned = Z_bot_plot
                    else:
                        if _SCIPY_AVAILABLE:
                            pts_b = np.column_stack([
                                X_bot.ravel(), Y_bot.ravel()
                            ])
                            vals_b = Z_bot_plot.ravel()
                            pts_t = np.column_stack([
                                X_top.ravel(), Y_top.ravel()
                            ])
                            Z_bot_aligned = _scipy_griddata(
                                pts_b, vals_b, pts_t, method='linear'
                            ).reshape(X_top.shape)
                            nan_mask = np.isnan(Z_bot_aligned)
                            if nan_mask.any():
                                Z_near = _scipy_griddata(
                                    pts_b, vals_b, pts_t, method='nearest'
                                ).reshape(X_top.shape)
                                Z_bot_aligned[nan_mask] = Z_near[nan_mask]
                        else:
                            Z_bot_aligned = Z_bot_plot[
                                :X_top.shape[0], :X_top.shape[1]
                            ] if (Z_bot_plot.shape[0] >= X_top.shape[0] and
                                  Z_bot_plot.shape[1] >= X_top.shape[1]) \
                              else np.full_like(Z_top_plot, np.nanmean(Z_bot_plot))

                    nx, ny = X_top.shape
                    n_t = 8
                    t_arr = np.linspace(0.0, 1.0, n_t)

                    def _safe_fill(arr_1d):
                        a = arr_1d.copy()
                        if not np.isnan(a).any():
                            return a
                        valid = ~np.isnan(a)
                        if valid.any():
                            a[~valid] = np.interp(
                                np.where(~valid)[0],
                                np.where(valid)[0],
                                a[valid],
                            )
                        return a

                    # Front wall (j=0) and back wall (j=ny-1)
                    for j_idx in [0, ny - 1]:
                        x_edge = X_top[:, j_idx]
                        y_val = float(Y_top[0, j_idx])
                        z_top_e = _safe_fill(Z_top_plot[:, j_idx])
                        z_bot_e = _safe_fill(Z_bot_aligned[:, j_idx])

                        X_w, _ = np.meshgrid(x_edge, t_arr, indexing='ij')
                        Y_w = np.full_like(X_w, y_val)
                        Z_w = (np.outer(z_bot_e, np.ones(n_t))
                               + np.outer(z_top_e - z_bot_e, t_arr))

                        fig.add_trace(go.Surface(
                            x=X_w, y=Y_w, z=Z_w,
                            colorscale=WALL_COLOR,
                            opacity=0.70, showscale=False,
                            showlegend=False, hoverinfo='skip',
                        ))

                    # Left wall (i=0) and right wall (i=nx-1)
                    for i_idx in [0, nx - 1]:
                        y_edge = Y_top[i_idx, :]
                        x_val = float(X_top[i_idx, 0])
                        z_top_e = _safe_fill(Z_top_plot[i_idx, :])
                        z_bot_e = _safe_fill(Z_bot_aligned[i_idx, :])

                        Y_w, _ = np.meshgrid(y_edge, t_arr, indexing='ij')
                        X_w = np.full_like(Y_w, x_val)
                        Z_w = (np.outer(z_bot_e, np.ones(n_t))
                               + np.outer(z_top_e - z_bot_e, t_arr))

                        fig.add_trace(go.Surface(
                            x=X_w, y=Y_w, z=Z_w,
                            colorscale=WALL_COLOR,
                            opacity=0.70, showscale=False,
                            showlegend=False, hoverinfo='skip',
                        ))

                # ── Wireframe overlay on top surface (when style is solid/ghost) ──
                if vis['show_wireframe'] and top_style in (
                    'Transparent solid', 'Ghost (single color)'
                ):
                    nx_t, ny_t = X_top.shape
                    step_x = max(1, nx_t // 25)
                    step_y = max(1, ny_t // 25)
                    for i in range(0, nx_t, step_x):
                        fig.add_trace(go.Scatter3d(
                            x=X_top[i, :], y=Y_top[i, :], z=Z_top_plot[i, :],
                            mode='lines',
                            line=dict(color='black',
                                      width=vis['wireframe_width']),
                            opacity=0.35, showlegend=False, hoverinfo='skip',
                        ))
                    for j in range(0, ny_t, step_y):
                        fig.add_trace(go.Scatter3d(
                            x=X_top[:, j], y=Y_top[:, j], z=Z_top_plot[:, j],
                            mode='lines',
                            line=dict(color='black',
                                      width=vis['wireframe_width']),
                            opacity=0.35, showlegend=False, hoverinfo='skip',
                        ))

                if not (was_full_top and was_full_bot):
                    st.caption(
                        "ℹ️ Grid had missing cells — gaps were filled with "
                        f"`{vis['interp_method']}` interpolation before rendering."
                    )

            # ───────────────────────────────────────────────────────
            #  Single layer (original behaviour)
            # ───────────────────────────────────────────────────────
            else:
                reduce = 'top' if layer.startswith('Top') else (
                    'bottom' if layer.startswith('Bottom') else 'mid'
                )
                X_grid, Y_grid, Z_grid, was_full = grid_from_xyz(
                    df, 'X', 'Y', z_col,
                    reduce=reduce,
                    interp_method=vis['interp_method'],
                )

                Z_grid_plot = Z_grid * z_exag

                fig.add_trace(go.Surface(
                    x=X_grid, y=Y_grid, z=Z_grid_plot,
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
                            x=X_grid[i, :], y=Y_grid[i, :], z=Z_grid_plot[i, :],
                            mode='lines',
                            line=dict(color='black',
                                      width=vis['wireframe_width']),
                            opacity=0.35, showlegend=False, hoverinfo='skip',
                        ))
                    for j in range(0, ny, step_y):
                        fig.add_trace(go.Scatter3d(
                            x=X_grid[:, j], y=Y_grid[:, j], z=Z_grid_plot[:, j],
                            mode='lines',
                            line=dict(color='black',
                                      width=vis['wireframe_width']),
                            opacity=0.35, showlegend=False, hoverinfo='skip',
                        ))

                if not was_full:
                    st.caption(
                        "ℹ️ Grid had missing cells — gaps were filled with "
                        f"`{vis['interp_method']}` interpolation before rendering."
                    )

        # ═══════════════════════════════════════════════════════════
        #  Layout (with Z-exaggeration label)
        # ═══════════════════════════════════════════════════════════
        z_axis_title = 'Elevation Z (m)'
        if z_exag != 1.0:
            z_axis_title = f'Elevation Z (m)  [×{z_exag:.1f}]'

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
                    title=z_axis_title,
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

    if any(k in profile_type for k in ("U-Valley", "V-Valley", "Asymmetric")):
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

    if "U-Valley" in profile_type:
        params['steepness_u'] = st.sidebar.slider(
            "U-Valley Wall Steepness (m/m²)",
            0.00001, 0.005, 0.0003, format="%.5f",
            key="steepness_u_slider",
        )
    if "V-Valley" in profile_type:
        params['steepness_v'] = st.sidebar.slider(
            "V-Valley Wall Steepness (m/m)",
            0.001, 0.200, 0.050, format="%.4f",
            key="steepness_v_slider",
            help="Linear rise per metre from the valley centre.",
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

    vis_config = add_visualization_controls()

    # ==========================================
    # 6. Main Dashboard Tabs
    # ==========================================
    tab_edit, tab1, tab_mod, tab2, tab3, tab4 = st.tabs([
        "✏️ Profile Editor",
        "📊 Original Geometry",
        "🔄 Modified Profiles",
        "🏔️ Deformed 3D Mesh",
        "📐 Elevation Angles (Z-X)",
        "💾 Export Data",
    ])

    # ================================================================
    # Tab 0: Profile Editor  (FIXED + ENHANCED)
    # ================================================================
    with tab_edit:
        st.subheader("✏️ Dynamic 1D Flowline Profile Editor")
        st.caption(
            "Modify surface and/or bedrock profiles interactively. "
            "In **Surface only** mode the bedrock is **frozen** — it will "
            "never change regardless of constraints. Switch to **Coupled** "
            "mode if you want the bedrock to auto-adjust for hanging-glacier "
            "depth limits."
        )

        # ---- Seed / sync modified profiles with the current source files ----
        current_sources = (os.path.abspath(surface_file), os.path.abspath(bedrock_file))
        if (
            "surface_df_mod" not in st.session_state
            or "bedrock_df_mod" not in st.session_state
            or st.session_state.get("_profile_mod_sources") != current_sources
        ):
            st.session_state["surface_df_mod"] = surface_df.copy()
            st.session_state["bedrock_df_mod"] = bedrock_df.copy()
            st.session_state["_profile_mod_sources"] = current_sources

        if "use_modified_profiles" not in st.session_state:
            st.session_state["use_modified_profiles"] = True

        # ---- Edit mode selector -------------------------------------
        edit_mode = st.radio(
            "Edit Mode",
            options=[
                "Surface only (fixed bedrock)",
                "Bedrock only (fixed surface)",
                "Both independently",
                "Coupled (depth-constrained — hanging glacier)",
            ],
            index=0,   # Default to Surface-only: bedrock frozen by default.
            key="profile_edit_mode",
            horizontal=True,
            help=(
                "• **Surface only** — modify the surface Z; bedrock is FROZEN.\n"
                "• **Bedrock only** — modify the bedrock Z; surface is FROZEN.\n"
                "• **Both** — edit each profile independently.\n"
                "• **Coupled** — enforce max/min ice depth; editing one "
                "profile auto-adjusts the OTHER profile."
            ),
        )

        # ---- Constraint panel ---------------------------------------
        st.markdown("### 📐 Depth Constraints")

        # ── MASTER TOGGLE ──
        constraints_enabled = st.toggle(
            "Enable depth constraints",
            value=False,
            key="constraints_enabled",
            help=(
                "OFF → Free modification: no depth limits, edits always "
                "proceed, status shown for information only.\n"
                "ON → Depth limits active: choose a max allowable depth "
                "and whether violations warn or auto-correct."
            ),
        )

        if constraints_enabled:
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**Max allowable depth**")
                max_depth_preset = st.selectbox(
                    "Preset",
                    options=[
                        "30 m  (shallow ice)",
                        "50 m  (moderate ice)",
                        "100 m (thick ice)",
                        "200 m (very thick ice)",
                        "500 m (polar-scale)",
                        "1000 m (unlimited practical)",
                        "Custom",
                    ],
                    index=1,          # default = 50 m
                    key="max_depth_preset",
                    help="Pick a preset or choose Custom to enter any value.",
                )

                if max_depth_preset == "Custom":
                    max_depth = st.number_input(
                        "Custom max depth (m)",
                        min_value=1.0, max_value=5000.0,
                        value=200.0, step=10.0,
                        key="max_depth_custom",
                    )
                else:
                    # Parse the number from the preset string
                    max_depth = float(max_depth_preset.split(" m")[0])
                    st.caption(f"Max depth = **{max_depth:.0f} m**")

            with col_right:
                constraint_action = st.radio(
                    "When violated",
                    options=["Warn only", "Auto-correct"],
                    index=0,
                    key="constraint_action",
                    horizontal=True,
                    help=(
                        "• **Warn only** — show warning, edit stands\n"
                        "• **Auto-correct** — fix bedrock after each edit"
                    ),
                )
                min_depth = st.number_input(
                    "Min depth (m)",
                    min_value=0.0, max_value=100.0,
                    value=1.0, step=0.5,
                    key="min_depth_input",
                    help="Ice thickness below this triggers a warning or correction.",
                )
        else:
            constraint_action = "Warn only"
            max_depth = None
            min_depth = None

        coupled = "Coupled" in edit_mode
        edit_surface = ("Surface" in edit_mode) or ("Both" in edit_mode) or coupled
        edit_bedrock = ("Bedrock" in edit_mode) or ("Both" in edit_mode) or coupled

        def _enforce_constraints():
            """Apply depth constraints — only when enabled AND auto-correct."""
            if not constraints_enabled:
                return
            if constraint_action != "Auto-correct":
                return

            may_touch_bedrock = coupled or (edit_surface and edit_bedrock)
            if may_touch_bedrock:
                if max_depth is not None:
                    st.session_state['bedrock_df_mod'] = enforce_max_depth(
                        st.session_state['surface_df_mod'],
                        st.session_state['bedrock_df_mod'],
                        max_depth,
                    )
                if min_depth is not None:
                    st.session_state['bedrock_df_mod'] = enforce_min_depth(
                        st.session_state['surface_df_mod'],
                        st.session_state['bedrock_df_mod'],
                        min_depth,
                    )
                st.session_state['bedrock_df_mod'] = enforce_bedrock_below_surface(
                    st.session_state['surface_df_mod'],
                    st.session_state['bedrock_df_mod'],
                    margin=0.1,
                )
            elif edit_surface and not edit_bedrock:
                st.session_state['bedrock_df_mod'] = enforce_bedrock_below_surface(
                    st.session_state['surface_df_mod'],
                    st.session_state['bedrock_df_mod'],
                    margin=0.1,
                )

        def _show_depth_status():
            """Show current depth statistics — always informational,
            never blocking."""
            _surf = st.session_state.get('surface_df_mod', surface_df)
            _bed = st.session_state.get('bedrock_df_mod', bedrock_df)

            surf_a, bed_a = regrid_to_common_x(_surf, _bed)
            depth = surf_a['Z'].to_numpy() - bed_a['Z'].to_numpy()
            n_pts = len(depth)
            d_max = float(depth.max())
            d_min = float(depth.min())
            d_mean = float(depth.mean())
            n_inverted = int((depth < 0).sum())

            # ── Free mode: just show a compact info badge ──
            if not constraints_enabled:
                if n_inverted > 0:
                    st.error(
                        f"❌ **{n_inverted}** inverted point(s) "
                        f"(bedrock above surface)  ·  "
                        f"depth range: {d_min:.1f} – {d_max:.1f} m  ·  "
                        f"mean: {d_mean:.1f} m"
                    )
                else:
                    st.info(
                        f"📐 Depth: **{d_min:.1f}** – **{d_max:.1f}** m  "
                        f"(mean {d_mean:.1f} m, {n_pts} pts)  ·  "
                        f"Constraints **off** — edit freely"
                    )
                return

            # ── Constrained mode ──
            n_exceed_max = int((depth > max_depth).sum()) if max_depth is not None else 0
            n_below_min = int((depth < min_depth).sum()) if min_depth is not None else 0

            all_ok = (n_inverted == 0 and n_exceed_max == 0 and n_below_min == 0)

            if all_ok:
                st.success(
                    f"✅ All {n_pts} points within constraints  ·  "
                    f"depth: {d_min:.1f} – {d_max:.1f} m  ·  "
                    f"mean: {d_mean:.1f} m"
                )
            else:
                if n_inverted > 0:
                    st.error(
                        f"❌ **{n_inverted}** inverted point(s) "
                        f"(bedrock above surface)"
                    )
                if n_exceed_max > 0:
                    st.warning(
                        f"⚠️ **{n_exceed_max}** point(s) exceed max depth "
                        f"of {max_depth:.0f} m  "
                        f"(actual max = **{d_max:.1f} m**)"
                    )
                if n_below_min > 0:
                    st.warning(
                        f"⚠️ **{n_below_min}** point(s) below min depth "
                        f"of {min_depth:.1f} m  "
                        f"(actual min = **{d_min:.2f} m**)"
                    )
                if constraint_action == "Warn only":
                    st.caption(
                        "💡 Switch to **Auto-correct** or press "
                        "**🔨 Force-Apply** to fix."
                    )

        def _force_apply():
            """One-shot enforcement regardless of current mode."""
            if max_depth is not None:
                st.session_state['bedrock_df_mod'] = enforce_max_depth(
                    st.session_state['surface_df_mod'],
                    st.session_state['bedrock_df_mod'],
                    max_depth,
                )
            if min_depth is not None:
                st.session_state['bedrock_df_mod'] = enforce_min_depth(
                    st.session_state['surface_df_mod'],
                    st.session_state['bedrock_df_mod'],
                    min_depth,
                )
            st.session_state['bedrock_df_mod'] = enforce_bedrock_below_surface(
                st.session_state['surface_df_mod'],
                st.session_state['bedrock_df_mod'],
                margin=0.1,
            )

        # ---- Editing sub-tabs ---------------------------------------
        sub_slider, sub_table, sub_bulk, sub_custom, sub_resample = st.tabs([
            "① Interactive Slider",
            "② Table Editor",
            "③ Bulk Operations",
            "④ Custom Function",
            "⑤ Resample Grid",
        ])

        # ── Depth status bar (always visible) ──
        _show_depth_status()

        # ── Force-Apply button (visible when constraints are on) ──
        if constraints_enabled:
            fa_col1, fa_col2 = st.columns([1, 5])
            with fa_col1:
                if st.button(
                    "🔨 Force-Apply",
                    key="force_apply_btn",
                    type="secondary",
                ):
                    _force_apply()
                    st.rerun()
            with fa_col2:
                st.caption(
                    "One-shot: enforce max-depth, min-depth, and "
                    "bedrock-below-surface right now."
                )

        # ════════════════════════════════════════════════════════════
        # ① Interactive Slider — immediate visual feedback per point
        # ════════════════════════════════════════════════════════════
        with sub_slider:
            st.markdown("#### Interactive Point-by-Point Adjustment")
            st.caption(
                "Select a point index, then drag the Z slider. The profile "
                "plot below updates **instantly** on every slider move — "
                "no button click needed."
            )

            col_sl1, col_sl2 = st.columns(2)

            # ── Surface slider ──
            if edit_surface:
                with col_sl1:
                    st.markdown("##### Surface")
                    surf_mod = st.session_state['surface_df_mod']
                    n_surf = len(surf_mod)

                    surf_pt = st.slider(
                        "Point index",
                        0, max(0, n_surf - 1), 0, 1,
                        key="surf_pt_idx",
                    )

                    if n_surf > 0 and 0 <= surf_pt < n_surf:
                        cur_x_s = float(surf_mod.iloc[surf_pt]['X'])
                        cur_z_s = float(surf_mod.iloc[surf_pt]['Z'])
                        st.write(
                            f"Point **{surf_pt}** / {n_surf - 1}  \n"
                            f"X = `{cur_x_s:.2f}` m  \n"
                            f"Z = `{cur_z_s:.2f}` m"
                        )

                        slider_key_s = f"surf_z_slider_{surf_pt}"
                        new_z_s = st.slider(
                            "Adjust Z (m)",
                            min_value=float(cur_z_s - 500.0),
                            max_value=float(cur_z_s + 500.0),
                            value=float(cur_z_s),
                            step=0.1,
                            format="%.2f",
                            key=slider_key_s,
                        )

                        if abs(new_z_s - cur_z_s) > 0.05:
                            df = st.session_state['surface_df_mod'].copy()
                            df.loc[surf_pt, 'Z'] = new_z_s
                            df = df.sort_values('X').reset_index(drop=True)
                            st.session_state['surface_df_mod'] = df
                            _enforce_constraints()
                            st.rerun()

                        delta_s = new_z_s - cur_z_s
                        if abs(delta_s) > 0.05:
                            st.info(f"ΔZ = {delta_s:+.2f} m (applied)")
                        else:
                            st.caption("No change yet — drag the slider to adjust.")
                    else:
                        st.warning("Surface profile is empty.")

            # ── Bedrock slider ──
            if edit_bedrock:
                with col_sl2:
                    st.markdown("##### Bedrock")
                    bed_mod = st.session_state['bedrock_df_mod']
                    n_bed = len(bed_mod)

                    bed_pt = st.slider(
                        "Point index",
                        0, max(0, n_bed - 1), 0, 1,
                        key="bed_pt_idx",
                    )

                    if n_bed > 0 and 0 <= bed_pt < n_bed:
                        cur_x_b = float(bed_mod.iloc[bed_pt]['X'])
                        cur_z_b = float(bed_mod.iloc[bed_pt]['Z'])
                        st.write(
                            f"Point **{bed_pt}** / {n_bed - 1}  \n"
                            f"X = `{cur_x_b:.2f}` m  \n"
                            f"Z = `{cur_z_b:.2f}` m"
                        )

                        slider_key_b = f"bed_z_slider_{bed_pt}"
                        new_z_b = st.slider(
                            "Adjust Z (m)",
                            min_value=float(cur_z_b - 500.0),
                            max_value=float(cur_z_b + 500.0),
                            value=float(cur_z_b),
                            step=0.1,
                            format="%.2f",
                            key=slider_key_b,
                        )

                        if abs(new_z_b - cur_z_b) > 0.05:
                            df = st.session_state['bedrock_df_mod'].copy()
                            df.loc[bed_pt, 'Z'] = new_z_b
                            df = df.sort_values('X').reset_index(drop=True)
                            st.session_state['bedrock_df_mod'] = df
                            _enforce_constraints()
                            st.rerun()

                        delta_b = new_z_b - cur_z_b
                        if abs(delta_b) > 0.05:
                            st.info(f"ΔZ = {delta_b:+.2f} m (applied)")
                        else:
                            st.caption("No change yet — drag the slider to adjust.")
                    else:
                        st.warning("Bedrock profile is empty.")

            if not edit_surface and not edit_bedrock:
                st.warning("Select an edit mode above to enable editing.")

        # ════════════════════════════════════════════════════════════
        # ② Table Editor — auto-commit via on_change callback
        # ════════════════════════════════════════════════════════════
        with sub_table:
            st.markdown("#### Table Editor (auto-commit)")
            st.caption(
                "Edit Z values directly. Changes are committed **automatically** "
                "when you finish editing a cell (click away or press Enter). "
                "You can also add or delete rows. No 'Apply' button needed."
            )

            def _on_surface_table_change():
                """Auto-commit surface table edits to session_state."""
                edited = st.session_state.get('surface_point_editor')
                if edited is not None and isinstance(edited, pd.DataFrame):
                    edited = (
                        edited.dropna()
                        .sort_values('X')
                        .drop_duplicates('X')
                        .reset_index(drop=True)
                    )
                    if len(edited) > 0:
                        st.session_state['surface_df_mod'] = edited
                        _enforce_constraints()
                        if coupled and 'bedrock_point_editor' in st.session_state:
                            del st.session_state['bedrock_point_editor']

            def _on_bedrock_table_change():
                """Auto-commit bedrock table edits to session_state."""
                edited = st.session_state.get('bedrock_point_editor')
                if edited is not None and isinstance(edited, pd.DataFrame):
                    edited = (
                        edited.dropna()
                        .sort_values('X')
                        .drop_duplicates('X')
                        .reset_index(drop=True)
                    )
                    if len(edited) > 0:
                        st.session_state['bedrock_df_mod'] = edited
                        _enforce_constraints()
                        if (coupled or (edit_surface and edit_bedrock
                                        and constraints_enabled
                                        and constraint_action == "Auto-correct")):
                            if 'bedrock_point_editor' in st.session_state:
                                del st.session_state['bedrock_point_editor']

            col_te1, col_te2 = st.columns(2)

            with col_te1:
                if edit_surface:
                    st.markdown(
                        f"**Surface** "
                        f"({len(st.session_state['surface_df_mod'])} pts)"
                    )
                    st.data_editor(
                        st.session_state['surface_df_mod'].copy(),
                        num_rows="dynamic",
                        use_container_width=True,
                        column_config={
                            "X": st.column_config.NumberColumn(
                                "X (m)", help="Distance along flowline",
                                step=1.0, format="%.2f"),
                            "Z": st.column_config.NumberColumn(
                                "Z (m)", help="Elevation",
                                step=0.1, format="%.2f"),
                        },
                        key="surface_point_editor",
                        on_change=_on_surface_table_change,
                    )

            with col_te2:
                if edit_bedrock:
                    st.markdown(
                        f"**Bedrock** "
                        f"({len(st.session_state['bedrock_df_mod'])} pts)"
                    )
                    st.data_editor(
                        st.session_state['bedrock_df_mod'].copy(),
                        num_rows="dynamic",
                        use_container_width=True,
                        column_config={
                            "X": st.column_config.NumberColumn(
                                "X (m)", help="Distance along flowline",
                                step=1.0, format="%.2f"),
                            "Z": st.column_config.NumberColumn(
                                "Z (m)", help="Elevation",
                                step=0.1, format="%.2f"),
                        },
                        key="bedrock_point_editor",
                        on_change=_on_bedrock_table_change,
                    )

        # ════════════════════════════════════════════════════════════
        # ③ Bulk Operations
        # ════════════════════════════════════════════════════════════
        with sub_bulk:
            st.markdown("#### Bulk Transformations")
            st.caption(
                "Apply transformations to the selected profile(s). "
                "Constraints are applied automatically only in "
                "**Auto-correct** mode."
            )

            if edit_surface and not edit_bedrock:
                bulk_target = "Surface"
            elif edit_bedrock and not edit_surface:
                bulk_target = "Bedrock"
            elif edit_surface and edit_bedrock:
                bulk_target = st.radio(
                    "Target Profile",
                    ["Surface", "Bedrock", "Both"],
                    horizontal=True,
                    key="bulk_target_radio",
                )
            else:
                bulk_target = "Both"

            def _apply_to_target(func, *args):
                if bulk_target in ("Surface", "Both"):
                    st.session_state['surface_df_mod'] = func(
                        st.session_state['surface_df_mod'], *args)
                if bulk_target in ("Bedrock", "Both"):
                    st.session_state['bedrock_df_mod'] = func(
                        st.session_state['bedrock_df_mod'], *args)
                for k in ('surface_point_editor', 'bedrock_point_editor'):
                    if k in st.session_state:
                        del st.session_state[k]
                _enforce_constraints()
                st.rerun()

            col_b1, col_b2 = st.columns(2)

            with col_b1:
                st.markdown("##### Shift (uniform offset)")
                shift_val = st.number_input(
                    "ΔZ (m)", -500.0, 500.0, 0.0, 1.0, key="shift_val")
                if st.button("Apply Shift", key="btn_shift"):
                    _apply_to_target(apply_profile_shift, shift_val)

                st.markdown("##### Scale (stretch / compress)")
                scale_val = st.number_input(
                    "Scale factor", 0.1, 10.0, 1.0, 0.01, key="scale_val")
                scale_anchor = st.selectbox(
                    "Anchor", ["mean", "start", "absolute"], key="scale_anchor")
                if st.button("Apply Scale", key="btn_scale"):
                    _apply_to_target(apply_profile_scale, scale_val, scale_anchor)

                st.markdown("##### Tilt (linear gradient)")
                tilt_val = st.number_input(
                    "Slope (m/km)", -200.0, 200.0, 0.0, 1.0, key="tilt_val")
                if st.button("Apply Tilt", key="btn_tilt"):
                    _apply_to_target(apply_profile_tilt, tilt_val)

            with col_b2:
                st.markdown("##### Smooth (moving average)")
                smooth_w = st.slider(
                    "Window size", 3, 51, 5, 2, key="smooth_w")
                if st.button("Apply Smoothing", key="btn_smooth"):
                    _apply_to_target(apply_profile_smooth, smooth_w)

                st.markdown("##### Gaussian Bump / Trough")
                bump_x = st.number_input(
                    "Center X (m)", float(X_MESH_MIN), float(X_MESH_MAX),
                    float((X_MESH_MIN + X_MESH_MAX) / 2), 10.0, key="bump_x")
                bump_w = st.number_input(
                    "Width σ (m)", 10.0, 2000.0, 200.0, 10.0, key="bump_w")
                bump_a = st.number_input(
                    "Amplitude (m, negative = trough)",
                    -300.0, 300.0, 20.0, 5.0, key="bump_a")
                if st.button("Apply Bump", key="btn_bump"):
                    _apply_to_target(apply_profile_bump, bump_x, bump_w, bump_a)

                st.markdown("##### Step (icefall / riegel)")
                step_x = st.number_input(
                    "Step X (m)", float(X_MESH_MIN), float(X_MESH_MAX),
                    float(X_MESH_MIN + 0.5 * (X_MESH_MAX - X_MESH_MIN)),
                    10.0, key="step_x")
                step_h = st.number_input(
                    "Step height (m)", -300.0, 300.0, 30.0, 5.0, key="step_h")
                if st.button("Apply Step", key="btn_step"):
                    _apply_to_target(apply_profile_step, step_x, step_h)

        # ════════════════════════════════════════════════════════════
        # ④ Custom Function
        # ════════════════════════════════════════════════════════════
        with sub_custom:
            st.markdown("#### Custom Z = f(X, Z_orig)")
            st.caption(
                "Enter a Python expression. Available variables:\n"
                "- `x` — numpy array of X distances (m)\n"
                "- `z_orig` — numpy array of original Z values (m)\n"
                "- `np` — numpy module\n\n"
                "Examples:\n"
                "- `z_orig + 20 * np.sin(2 * np.pi * x / 2000)` — sinusoidal undulation\n"
                "- `z_orig * 0.8` — flatten by 20 %\n"
                "- `np.maximum(z_orig, 3000)` — clip below 3000 m\n"
                "- `z_orig - 0.005 * (x - x[0])**2` — parabolic deepening"
            )
            custom_expr = st.text_input(
                "Expression", value="z_orig",
                key="custom_expr",
                help="Must return an array of the same length as x."
            )
            if st.button("Apply Custom Function", key="btn_custom"):
                try:
                    if edit_surface:
                        st.session_state['surface_df_mod'] = apply_custom_function(
                            st.session_state['surface_df_mod'], custom_expr)
                    if edit_bedrock:
                        st.session_state['bedrock_df_mod'] = apply_custom_function(
                            st.session_state['bedrock_df_mod'], custom_expr)
                    _enforce_constraints()
                    for k in ('surface_point_editor', 'bedrock_point_editor'):
                        if k in st.session_state:
                            del st.session_state[k]
                    st.success("Custom function applied successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Expression error: {e}")

        # ════════════════════════════════════════════════════════════
        # ⑤ Resample
        # ════════════════════════════════════════════════════════════
        with sub_resample:
            st.markdown("#### Resample to Uniform X Grid")
            st.caption(
                "Linearly interpolate both profiles onto a regular grid. "
                "Useful before exporting to FEA solvers that expect "
                "uniform spacing."
            )
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                dx_new = st.number_input(
                    "Grid spacing ΔX (m)", 1.0, 500.0, 50.0, 5.0,
                    key="resample_dx")
                x_lo = st.number_input(
                    "X min (m)", value=float(surface_df['X'].min()),
                    key="resample_xmin")
                x_hi = st.number_input(
                    "X max (m)", value=float(surface_df['X'].max()),
                    key="resample_xmax")
            with col_r2:
                st.write("")
                st.write("")
                if st.button("Resample", key="btn_resample"):
                    st.session_state['surface_df_mod'] = resample_profile(
                        st.session_state['surface_df_mod'], dx_new, x_lo, x_hi)
                    st.session_state['bedrock_df_mod'] = resample_profile(
                        st.session_state['bedrock_df_mod'], dx_new, x_lo, x_hi)
                    _enforce_constraints()
                    for k in ('surface_point_editor', 'bedrock_point_editor'):
                        if k in st.session_state:
                            del st.session_state[k]
                    st.success(f"Resampled to {dx_new:.1f} m spacing.")
                    st.rerun()

        # ---- Reset button -------------------------------------------
        col_rst1, col_rst2 = st.columns([4, 1])
        with col_rst2:
            if st.button("🔄 Reset to Original", key="btn_reset_profiles"):
                st.session_state['surface_df_mod'] = surface_df.copy()
                st.session_state['bedrock_df_mod'] = bedrock_df.copy()
                for k in list(st.session_state.keys()):
                    if k.startswith('surf_z_slider_') or k.startswith('bed_z_slider_'):
                        del st.session_state[k]
                for k in ('surface_point_editor', 'bedrock_point_editor',
                          'surf_pt_idx', 'bed_pt_idx'):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        # ════════════════════════════════════════════════════════════
        # LIVE Visualization: Original vs Modified
        # ════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📈 Original vs Modified Profiles")

        if edit_mode == "Surface only (fixed bedrock)":
            st.info("🔒 **Bedrock is FROZEN** — only the surface is editable. "
                    "The bedrock trace will not change.")
        elif edit_mode == "Bedrock only (fixed surface)":
            st.info("🔒 **Surface is FROZEN** — only the bedrock is editable. "
                    "The surface trace will not change.")
        elif coupled:
            if constraints_enabled:
                st.warning("🔗 **Coupled mode** — editing one profile will "
                           "auto-adjust the other to maintain depth constraints "
                           f"(max={max_depth} m, min={min_depth} m).")
            else:
                st.info("🔗 **Coupled mode** — depth constraints are **off**, "
                        "so both profiles are effectively independent. "
                        "Toggle 'Enable depth constraints' above to enforce "
                        "auto-adjustment.")
        else:
            st.caption("Both profiles are independently editable.")

        surf_mod = st.session_state['surface_df_mod']
        bed_mod = st.session_state['bedrock_df_mod']

        fig_cmp = go.Figure()

        fig_cmp.add_trace(go.Scatter(
            x=surface_df['X'], y=surface_df['Z'],
            name='Original Surface', mode='lines',
            line=dict(color='royalblue', width=2, dash='dot'),
            opacity=0.4,
        ))
        fig_cmp.add_trace(go.Scatter(
            x=surf_mod['X'], y=surf_mod['Z'],
            name='Modified Surface', mode='lines+markers',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6),
            hovertemplate='X: %{x:.1f}<br>Z: %{y:.1f}<extra>Surf</extra>',
        ))
        fig_cmp.add_trace(go.Scatter(
            x=bedrock_df['X'], y=bedrock_df['Z'],
            name='Original Bedrock', mode='lines',
            line=dict(color='saddlebrown', width=2, dash='dot'),
            opacity=0.4,
        ))
        fig_cmp.add_trace(go.Scatter(
            x=bed_mod['X'], y=bed_mod['Z'],
            name='Modified Bedrock', mode='lines+markers',
            line=dict(color='saddlebrown', width=3, dash='dash'),
            marker=dict(size=6),
            hovertemplate='X: %{x:.1f}<br>Z: %{y:.1f}<extra>Bed</extra>',
        ))

        # Shaded ice body — use the common (bedrock) X grid so surf and bed
        # are perfectly aligned and the fill is a clean closed polygon.
        surf_aligned, bed_aligned = regrid_to_common_x(surf_mod, bed_mod)
        depth_arr = surf_aligned['Z'].to_numpy() - bed_aligned['Z'].to_numpy()

        fig_cmp.add_trace(go.Scatter(
            x=np.concatenate([surf_aligned['X'], surf_aligned['X'][::-1]]),
            y=np.concatenate([surf_aligned['Z'], bed_aligned['Z'][::-1]]),
            fill='toself', fillcolor='rgba(135,206,235,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Ice body', showlegend=False, hoverinfo='skip',
        ))

        # Highlight the currently-selected slider point.
        if edit_surface and 'surf_pt_idx' in st.session_state:
            sp = st.session_state['surf_pt_idx']
            if sp < len(surf_mod):
                fig_cmp.add_trace(go.Scatter(
                    x=[surf_mod.iloc[sp]['X']],
                    y=[surf_mod.iloc[sp]['Z']],
                    mode='markers',
                    marker=dict(size=14, color='red', symbol='star',
                                line=dict(width=2, color='darkred')),
                    name='Selected Surface Pt',
                    showlegend=True,
                ))
        if edit_bedrock and 'bed_pt_idx' in st.session_state:
            bp = st.session_state['bed_pt_idx']
            if bp < len(bed_mod):
                fig_cmp.add_trace(go.Scatter(
                    x=[bed_mod.iloc[bp]['X']],
                    y=[bed_mod.iloc[bp]['Z']],
                    mode='markers',
                    marker=dict(size=14, color='orange', symbol='star',
                                line=dict(width=2, color='darkorange')),
                    name='Selected Bedrock Pt',
                    showlegend=True,
                ))

        fig_cmp.update_layout(
            title='Profile Comparison (dashed = original, solid = modified)',
            xaxis_title='Distance X (m)',
            yaxis_title='Elevation Z (m)',
            height=520,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            plot_bgcolor='white',
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ---- Depth profile ------------------------------------------
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=surf_aligned['X'], y=depth_arr,
            name='Ice Depth', mode='lines',
            line=dict(color='teal', width=3),
            fill='tozeroy', fillcolor='rgba(0,128,128,0.15)',
        ))
        if constraints_enabled and max_depth is not None:
            fig_dep.add_hline(y=max_depth, line=dict(color='red', dash='dash'),
                              annotation_text=f"Max = {max_depth} m",
                              annotation_position="top left")
        if constraints_enabled and min_depth is not None:
            fig_dep.add_hline(y=min_depth, line=dict(color='orange', dash='dash'),
                              annotation_text=f"Min = {min_depth} m",
                              annotation_position="bottom left")
        fig_dep.update_layout(
            title='Ice Depth (Surface − Bedrock)',
            xaxis_title='Distance X (m)', yaxis_title='Depth (m)',
            height=320, plot_bgcolor='white',
        )
        st.plotly_chart(fig_dep, use_container_width=True)

        # ============================================================
        # Validation
        # ============================================================
        st.markdown("---")
        st.subheader("✅ Validation Report")

        flags = validate_profiles(surf_mod, bed_mod,
                                  max_depth=max_depth,
                                  min_depth=min_depth)

        v1, v2, v3, v4 = st.columns(4)
        with v1: st.metric("Max Depth", f"{flags['depth_max']:.1f} m")
        with v2: st.metric("Min Depth", f"{flags['depth_min']:.1f} m")
        with v3: st.metric("Mean Depth", f"{flags['depth_mean']:.1f} m")
        with v4: st.metric("Bedrock > Surface", f"{flags['n_above_surface']} pts")

        if flags['n_above_surface'] > 0:
            st.error(
                f"🚨 {flags['n_above_surface']} point(s) have bedrock above "
                "surface — this will create inverted elements! "
                "Enable depth constraints with **Auto-correct**, or press "
                "'🔨 Force-Apply' above."
            )
        elif flags['n_exceeds_max'] > 0:
            if edit_mode == "Surface only (fixed bedrock)":
                st.warning(
                    f"⚠️ {flags['n_exceeds_max']} point(s) exceed the max depth "
                    f"of {max_depth} m. Since bedrock is **frozen**, you must "
                    "either lower the surface or switch to **Coupled** mode to "
                    "auto-adjust the bedrock. In **Warn only** mode this is "
                    "permitted."
                )
            else:
                st.warning(
                    f"⚠️ {flags['n_exceeds_max']} point(s) exceed the max depth "
                    f"of {max_depth} m. Switch to **Auto-correct** or press "
                    "'🔨 Force-Apply' to fix."
                )
        elif flags['n_below_min'] > 0:
            st.warning(
                f"⚠️ {flags['n_below_min']} point(s) are below the min depth "
                f"of {min_depth} m."
            )
        else:
            if constraints_enabled:
                st.success(
                    f"✅ All {flags['n_points']} points satisfy all constraints. "
                    f"Depth range: {flags['depth_min']:.1f}–{flags['depth_max']:.1f} m."
                )
            else:
                st.info(
                    f"📐 Depth constraints are **off**. "
                    f"Depth range: {flags['depth_min']:.1f}–{flags['depth_max']:.1f} m, "
                    f"mean {flags['depth_mean']:.1f} m — free-modification mode."
                )

        # ============================================================
        # Export Modified Profiles
        # ============================================================
        st.markdown("---")
        st.subheader("💾 Export Modified Profiles")

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)

        with col_e1:
            st.download_button(
                "⬇️ Modified Surface (.dat)",
                profile_to_dat_string(surf_mod),
                file_name=f"modified_{os.path.basename(surface_file)}",
                mime="text/plain",
                key="dl_mod_surf_dat",
            )
        with col_e2:
            st.download_button(
                "⬇️ Modified Bedrock (.dat)",
                profile_to_dat_string(bed_mod),
                file_name=f"modified_{os.path.basename(bedrock_file)}",
                mime="text/plain",
                key="dl_mod_bed_dat",
            )
        with col_e3:
            combined_csv = pd.DataFrame({
                'X': surf_aligned['X'],
                'Surface_Z': surf_aligned['Z'],
                'Bedrock_Z': bed_aligned['Z'],
                'Depth': depth_arr,
            }).to_csv(index=False, float_format='%.6f')
            st.download_button(
                "⬇️ Combined (CSV)",
                combined_csv,
                file_name="modified_profiles.csv",
                mime="text/csv",
                key="dl_mod_csv",
            )
        with col_e4:
            depth_csv = pd.DataFrame({
                'X': surf_aligned['X'],
                'Depth': depth_arr,
            }).to_csv(index=False, float_format='%.6f')
            st.download_button(
                "⬇️ Depth Profile (CSV)",
                depth_csv,
                file_name="ice_depth_profile.csv",
                mime="text/csv",
                key="dl_depth_csv",
            )

        # ---- Toggle: feed modified profiles into FEA ----------------
        st.markdown("---")
        use_mod = st.checkbox(
            "🔀 Use modified profiles for FEA mesh deformation",
            value=st.session_state.get('use_modified_profiles', True),
            key="use_modified_profiles_cb",
            help=(
                "When checked, the 'Deformed 3D Mesh' tab and the "
                "mesh.nodes export will use the modified profiles instead "
                "of the original .dat files."
            ),
        )
        st.session_state['use_modified_profiles'] = use_mod

        if use_mod:
            st.info(
                "👉 Modified profiles are **active**. Switch to the "
                "'Deformed 3D Mesh' tab to see the effect."
            )

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
                st.plotly_chart(fig, use_container_width=True)
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
                st.plotly_chart(fig2d, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render 2D figure: {e}")

    # ================================================================
    # Tab 1b: Modified Profiles — dedicated customizable figure
    # ================================================================
    with tab_mod:
        st.subheader("🔄 Modified Profiles")

        has_mod = (
            'surface_df_mod' in st.session_state
            and 'bedrock_df_mod' in st.session_state
        )

        if not has_mod:
            st.info(
                "No modified profiles yet.  Go to the "
                "**✏️ Profile Editor** tab to modify the surface or bedrock, "
                "then return here."
            )
        else:
            surf_mod = st.session_state['surface_df_mod']
            bed_mod = st.session_state['bedrock_df_mod']

            # ── Check if profiles actually differ from originals ──
            def _max_abs_diff(a, b):
                """Max |ΔZ| between two profiles (on a's X grid)."""
                a_s = a.sort_values('X')
                b_s = b.sort_values('X')
                x_a = a_s['X'].to_numpy()
                z_a = a_s['Z'].to_numpy()
                z_b = np.interp(x_a, b_s['X'].to_numpy(), b_s['Z'].to_numpy())
                return float(np.max(np.abs(z_a - z_b)))

            diff_surf = _max_abs_diff(surf_mod, surface_df)
            diff_bed = _max_abs_diff(bed_mod, bedrock_df)
            profiles_differ = (diff_surf > 0.01 or diff_bed > 0.01)

            if not profiles_differ:
                st.caption(
                    "ℹ️  Modified profiles are currently identical to the "
                    "originals.  The figure below shows the unmodified state."
                )

            # ─────────────────────────────────────────────────
            #  Customization controls
            # ─────────────────────────────────────────────────
            st.markdown("#### 🎛️ Figure Options")

            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1:
                show_ice_thickness = st.checkbox(
                    "Ice thickness subplot",
                    value=True,
                    key="mod_show_thickness",
                    help="Lower panel: ice thickness = surface − bedrock.",
                )
            with col_c2:
                show_orig_ghost = st.checkbox(
                    "Original as reference",
                    value=True,
                    key="mod_show_orig_ghost",
                    help="Faded dotted lines showing the original profiles.",
                )
            with col_c3:
                show_markers = st.checkbox(
                    "Point markers",
                    value=True,
                    key="mod_show_markers",
                    help="Show markers at each data point.",
                )
            with col_c4:
                show_depth_refs = st.checkbox(
                    "Reference grid lines",
                    value=True,
                    key="mod_show_depth_refs",
                    help="Light dotted lines at regular Z / thickness intervals.",
                )

            # ── Thickness contour interval ──
            if show_ice_thickness and show_depth_refs:
                thickness_contour_m = st.slider(
                    "Thickness contour interval (m)",
                    min_value=5, max_value=100, value=10, step=5,
                    key="mod_thickness_contour",
                    help="Spacing of horizontal reference lines on the thickness panel.",
                )

            # ─────────────────────────────────────────────────
            #  Figure
            # ─────────────────────────────────────────────────
            fig_mod = make_modified_flowline_fig(
                surf_mod, bed_mod, vis_config,
                surface_orig=surface_df if show_orig_ghost else None,
                bedrock_orig=bedrock_df if show_orig_ghost else None,
                show_ice_thickness=show_ice_thickness,
                show_orig_ghost=show_orig_ghost,
                show_markers=show_markers,
                show_depth_refs=show_depth_refs,
            )
            st.plotly_chart(fig_mod, use_container_width=True)

            # ─────────────────────────────────────────────────
            #  Summary statistics
            # ─────────────────────────────────────────────────
            st.markdown("### 📏 Modified Profile Statistics")

            # Ice thickness
            surf_s = surf_mod.sort_values('X').drop_duplicates('X')
            bed_s = bed_mod.sort_values('X').drop_duplicates('X')
            x_common = bed_s['X'].to_numpy()
            z_surf_c = np.interp(
                x_common, surf_s['X'].to_numpy(), surf_s['Z'].to_numpy()
            )
            z_bed_c = bed_s['Z'].to_numpy()
            ice_thick = z_surf_c - z_bed_c

            # Approximate cross-sectional area (trapezoidal rule)
            dx = np.diff(x_common)
            h_avg = (ice_thick[:-1] + ice_thick[1:]) / 2.0
            xsec_area = float(np.sum(h_avg * dx))  # m²

            stat_c1, stat_c2, stat_c3, stat_c4, stat_c5 = st.columns(5)
            with stat_c1:
                st.metric(
                    "Max ice thickness",
                    f"{ice_thick.max():.1f} m",
                    delta=f"{ice_thick.max() - (surface_df['Z'].max() - bedrock_df['Z'].min()):.1f} m vs orig"
                    if profiles_differ else None,
                )
            with stat_c2:
                st.metric("Mean ice thickness", f"{ice_thick.mean():.1f} m")
            with stat_c3:
                st.metric("Min ice thickness", f"{ice_thick.min():.1f} m")
            with stat_c4:
                st.metric("Cross-section area", f"{xsec_area:,.0f} m²")
            with stat_c5:
                st.metric(
                    "Profile length",
                    f"{x_common[-1] - x_common[0]:.0f} m",
                )

            # ── Delta from original ──
            if profiles_differ:
                st.markdown("#### ΔZ from Original")
                delta_c1, delta_c2, delta_c3, delta_c4 = st.columns(4)
                with delta_c1:
                    st.metric("Max ΔZ surface", f"{diff_surf:+.2f} m")
                with delta_c2:
                    mean_dz_s = float(
                        np.mean(
                            surf_mod.sort_values('X')['Z'].to_numpy()
                            - surface_df.sort_values('X')['Z'].to_numpy()[
                                :len(surf_mod)
                            ]
                        )
                    )
                    st.metric("Mean ΔZ surface", f"{mean_dz_s:+.2f} m")
                with delta_c3:
                    st.metric("Max ΔZ bedrock", f"{diff_bed:+.2f} m")
                with delta_c4:
                    mean_dz_b = float(
                        np.mean(
                            bed_mod.sort_values('X')['Z'].to_numpy()
                            - bedrock_df.sort_values('X')['Z'].to_numpy()[
                                :len(bed_mod)
                            ]
                        )
                    )
                    st.metric("Mean ΔZ bedrock", f"{mean_dz_b:+.2f} m")

            # ─────────────────────────────────────────────────
            #  Profile data tables
            # ─────────────────────────────────────────────────
            with st.expander("📋 Profile data tables", expanded=False):
                tab_st, tab_bt, tab_ht = st.tabs(
                    ["Surface", "Bedrock", "Ice Thickness"]
                )
                with tab_st:
                    st.dataframe(
                        surf_mod.sort_values('X').round(4).reset_index(drop=True),
                        use_container_width=True,
                    )
                with tab_bt:
                    st.dataframe(
                        bed_mod.sort_values('X').round(4).reset_index(drop=True),
                        use_container_width=True,
                    )
                with tab_ht:
                    thick_df = pd.DataFrame({
                        'X': x_common,
                        'Ice_Thickness_m': np.round(ice_thick, 4),
                    })
                    st.dataframe(thick_df, use_container_width=True)

            # ─────────────────────────────────────────────────
            #  Validation
            # ─────────────────────────────────────────────────
            validation = validate_profiles(surf_mod, bed_mod)
            if validation['is_valid']:
                st.success(
                    "✅ Modified profiles are physically valid "
                    "(bedrock below surface everywhere)."
                )
            else:
                st.error(
                    f"❌ Validation failed: bedrock pierces surface at "
                    f"**{validation['n_above_surface']}** point(s). "
                    f"Min depth = {validation['depth_min']:.2f} m."
                )

            # ─────────────────────────────────────────────────
            #  Download modified .dat files
            # ─────────────────────────────────────────────────
            st.markdown("### 💾 Download Modified Profiles")
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button(
                    "⬇️ Modified Surface (.dat)",
                    data=profile_to_dat_string(surf_mod),
                    file_name="modified_surface.dat",
                    mime="text/plain",
                    key="dl_mod_surface",
                )
            with dl_c2:
                st.download_button(
                    "⬇️ Modified Bedrock (.dat)",
                    data=profile_to_dat_string(bed_mod),
                    file_name="modified_bedrock.dat",
                    mime="text/plain",
                    key="dl_mod_bedrock",
                )

    # --------------------------------------------------------------
    # Tab 2 — Deformed 3D Mesh
    # --------------------------------------------------------------
    with tab2:
        st.subheader("Deformed 3D Mesh with Y-Variations")

        if st.session_state.get('use_modified_profiles', False):
            surf_deform = st.session_state.get('surface_df_mod', surface_df)
            bed_deform = st.session_state.get('bedrock_df_mod', bedrock_df)
            st.info(
                f"🔀 Using **modified** profiles for deformation  \n"
                f"Surface: {len(surf_deform)} pts · "
                f"Bedrock: {len(bed_deform)} pts"
            )
        else:
            surf_deform = surface_df
            bed_deform = bedrock_df
            st.caption("Using **original** .dat profiles for deformation.")

        with st.spinner("Deforming mesh using parallel processing..."):
            n_jobs = min(os.cpu_count() or 4, 8)
            idx_chunks = np.array_split(np.arange(len(nodes_orig)), n_jobs)
            chunks = [
                nodes_orig.iloc[idx].reset_index(drop=True).copy()
                for idx in idx_chunks
            ]

            results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(deform_chunk)(chunk, bed_deform, surf_deform,
                                      profile_type, params)
                for chunk in chunks
            )
            nodes_deformed = pd.concat(results, ignore_index=True)

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
            st.dataframe(thickness_stats, use_container_width=True, hide_index=True)

        try:
            fig_def = make_3d_mesh_fig(
                nodes_deformed, 'Z_new', vis_config,
                title=f"Deformed 3D Mesh — {profile_type}"
            )
            st.plotly_chart(fig_def, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render deformed 3D figure: {e}")

        st.session_state['nodes_deformed'] = nodes_deformed

    # --------------------------------------------------------------
    # Tab 3 — Net Elevation Angle (Z-X)
    # --------------------------------------------------------------
    if st.session_state.get('use_modified_profiles', False):
        angles_surface = st.session_state.get('surface_df_mod', surface_df)
        angles_bedrock = st.session_state.get('bedrock_df_mod', bedrock_df)
    else:
        angles_surface = surface_df
        angles_bedrock = bedrock_df
    angles_df = compute_elevation_angles(angles_surface, angles_bedrock)
    st.session_state['angles_df'] = angles_df

    with tab3:
        st.subheader("📐 Net Elevation Angle θ = arctan(dZ/dX) along the X-axis")
        st.caption(
            "Numerical derivative computed with second-order central differences "
            "in the interior and one-sided differences at the endpoints (via "
            "`np.gradient`), which handles non-uniform X spacing in the `.dat` files."
        )
        if st.session_state.get('use_modified_profiles', False):
            st.info("🔀 Angles computed on **modified** profiles.")
        else:
            st.caption("Angles computed on **original** profiles.")

        try:
            fig_ang = make_angle_fig(angles_df, vis_config)
            st.plotly_chart(fig_ang, use_container_width=True)
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
            use_container_width=True,
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
            csv_mesh = st.session_state['nodes_deformed'][
                ['ID', 'Flag', 'X', 'Y', 'Z_new']
            ].to_csv(index=False, sep=' ', header=False, float_format='%.6f')
            st.download_button(
                "⬇️ Download Deformed mesh.nodes",
                csv_mesh,
                file_name="mesh.nodes",
                key="download_mesh_tab4",
            )

        if 'angles_df' in st.session_state:
            st.markdown("---")
            st.markdown(
                "**Elevation-angle table** "
                "(also available on the *Elevation Angles* tab)"
            )
            csv_angles_tab4 = st.session_state['angles_df'].to_csv(
                index=False, float_format='%.8f'
            )
            st.download_button(
                "⬇️ Download elevation_angles.csv",
                csv_angles_tab4,
                file_name="elevation_angles.csv",
                mime="text/csv",
                key="download_angles_tab4",
            )

        # ---- Modified profile exports (mirror of Profile Editor tab) ----
        st.markdown("---")
        st.subheader("📄 Modified Profile Exports")

        if ('surface_df_mod' in st.session_state
                and 'bedrock_df_mod' in st.session_state):
            surf_m = st.session_state['surface_df_mod']
            bed_m = st.session_state['bedrock_df_mod']

            col_x1, col_x2, col_x3 = st.columns(3)
            with col_x1:
                st.download_button(
                    "⬇️ Modified Surface (.dat)",
                    profile_to_dat_string(surf_m),
                    file_name=f"modified_{os.path.basename(surface_file)}",
                    mime="text/plain",
                    key="dl_tab4_mod_surf",
                )
            with col_x2:
                st.download_button(
                    "⬇️ Modified Bedrock (.dat)",
                    profile_to_dat_string(bed_m),
                    file_name=f"modified_{os.path.basename(bedrock_file)}",
                    mime="text/plain",
                    key="dl_tab4_mod_bed",
                )
            with col_x3:
                surf_a, bed_a = regrid_to_common_x(surf_m, bed_m)
                depth_a = surf_a['Z'].to_numpy() - bed_a['Z'].to_numpy()
                combined = pd.DataFrame({
                    'X': surf_a['X'],
                    'Surface_Z': surf_a['Z'],
                    'Bedrock_Z': bed_a['Z'],
                    'Depth': depth_a,
                })
                st.download_button(
                    "⬇️ Combined Profiles (CSV)",
                    combined.to_csv(index=False, float_format='%.6f'),
                    file_name="modified_profiles_combined.csv",
                    mime="text/csv",
                    key="dl_tab4_combined",
                )

            st.caption(
                f"Surface: {len(surf_m)} points · "
                f"Bedrock: {len(bed_m)} points · "
                f"Max depth: {depth_a.max():.1f} m · "
                f"Min depth: {depth_a.min():.1f} m"
            )
        else:
            st.info(
                "No modified profiles yet. Visit the '✏️ Profile Editor' tab."
            )
