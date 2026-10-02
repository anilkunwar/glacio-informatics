import streamlit as st
import os
import glob
import zipfile
import gc
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import meshio
import warnings

warnings.filterwarnings("ignore")

MEM_BUDGET_MB = 1024.0

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Elmer Glacier Viewer",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_NAME = "himalayan_glacier3d"
DATA_DIR = os.path.join(SCRIPT_DIR, DEFAULT_DIR_NAME)

COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
             'Blues', 'Reds', 'Greens', 'Jet', 'Rainbow']

# =============================================
# MEMORY UTILITIES
# =============================================
def get_rss_mb():
    """Current resident memory (MB). psutil if installed, else /proc
    (Linux / Streamlit Cloud), else getrusage peak as last resort."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1e3   # kB -> MB
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    except Exception:
        return None

# =============================================
# DATA EXTRACTION & LOADING
# =============================================
@st.cache_data
def ensure_data_dir(directory: str) -> str:
    """Self-extract <directory>.zip on first load if the folder is missing."""
    if os.path.isdir(directory) and os.listdir(directory):
        return directory

    base = os.path.basename(directory)
    zip_paths = [
        os.path.join(SCRIPT_DIR, f"{base}.zip"),
        os.path.join(SCRIPT_DIR, "data", f"{base}.zip"),
    ]
    for zpath in zip_paths:
        if not os.path.isfile(zpath):
            continue
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                tops = {n.split("/")[0] for n in names}
                if tops == {base}:
                    zf.extractall(SCRIPT_DIR)
                else:
                    os.makedirs(directory, exist_ok=True)
                    zf.extractall(directory)
            return directory
        except Exception as e:
            st.error(f"Failed to extract {zpath}: {e}")
    return directory


@st.cache_resource(show_spinner=False)
def load_skeleton(directory: str, prefix: str):
    """Tiny, permanent structure: file list + geometry + field catalogue.
    Reads ONLY the first file. No time-series data is kept in RAM."""
    if not os.path.isdir(directory):
        return None

    pvtu = sorted(set(glob.glob(os.path.join(directory, "**", f"{prefix}*.pvtu"),
                                recursive=True)))
    files = pvtu if pvtu else sorted(set(
        glob.glob(os.path.join(directory, "**", f"{prefix}*.vtu"), recursive=True)))
    if not files:
        return None

    try:
        m0 = meshio.read(files[0])
    except Exception as e:
        st.error(f"Failed to read {os.path.basename(files[0])}: {e}")
        return None

    points = np.asarray(m0.points, dtype=np.float32)
    triangles = None
    for block in m0.cells:
        if block.type == "triangle":
            triangles = np.asarray(block.data, dtype=np.int32)
            break

    field_info = {}
    for key, arr in m0.point_data.items():
        a = np.asarray(arr)
        comps = 1 if a.ndim == 1 else int(np.prod(a.shape[1:]))
        if comps == 1:
            kind = "scalar"
        elif a.ndim == 2 and comps <= 3:
            kind = "vector"
        else:
            kind = "tensor"
        field_info[key] = {"comps": comps, "kind": kind}

    del m0  # free the meshio object (cells, float64 arrays, ...)

    n_pts = len(points)
    total_comps = sum(v["comps"] for v in field_info.values())
    return {
        "files": files,
        "n_timesteps": len(files),
        "points": points,             # read-only, shared — never mutate
        "triangles": triangles,       # read-only, shared
        "field_info": field_info,
        "n_pts": n_pts,
        "total_comps": total_comps,
        "ts_bytes": n_pts * max(total_comps, 1) * 4,
    }


def _read_timestep(path: str, n_pts: int):
    """Read ONE timestep file; return its point-data fields as float32.
    Arrays are shared from the cache — do NOT mutate them."""
    try:
        mesh = meshio.read(path)
    except Exception as e:
        st.warning(f"Failed to read {os.path.basename(path)}: {e}")
        return {}
    if mesh.points.shape[0] != n_pts:
        st.warning(f"{os.path.basename(path)}: {mesh.points.shape[0]} points "
                   f"(expected {n_pts}); skipping.")
        return {}
    out = {}
    for key, arr in mesh.point_data.items():
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)   # tensors -> flattened
        out[key] = a
    return out

# Two LRU policies: normal (≤6 timesteps, 30 min) vs low-memory (≤2, 5 min)
load_ts        = st.cache_resource(max_entries=6, ttl=1800, show_spinner=False)(_read_timestep)
load_ts_lowmem = st.cache_resource(max_entries=2, ttl=300,  show_spinner=False)(_read_timestep)


def decimate_triangles(points, values, triangles, max_tris):
    """Random triangle subsampling with vertex remapping — keeps a VALID mesh
    and shrinks both RAM and the Plotly JSON payload."""
    if len(triangles) <= max_tris:
        return points, values, triangles
    rng = np.random.default_rng(0)                     # stable across reruns
    keep = rng.choice(len(triangles), size=max_tris, replace=False)
    tris = triangles[keep]
    used = np.unique(tris.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], values[used], remap[tris].astype(np.int32)

# =============================================
# STREAMLIT APP
# =============================================
def main():
    st.title("🏔️ Elmer Glacier 3D Diagnostic Viewer")
    st.caption("🧠 Memory-budgeted (< 1 GB): lazy timesteps + LRU cache + mesh decimation")

    # ---------- Sidebar ----------
    st.sidebar.header("⚙️ Configuration")
    prefix = st.sidebar.text_input("File Prefix", value="Stokes_ELA400_3D_diagnostic")

    # ---------- Memory panel ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"🧠 Memory (budget {MEM_BUDGET_MB:.0f} MB)")
    mem_slot  = st.sidebar.empty()   # filled after data load
    prog_slot = st.sidebar.empty()

    low_mem = st.sidebar.checkbox(
        "Low-Memory Mode", value=True,
        help="Keeps ≤2 timesteps in RAM, caps plot size, defers the 2D plot.")

    if st.sidebar.button("🗑️ Free memory now"):
        load_skeleton.clear()
        load_ts.clear()
        load_ts_lowmem.clear()
        st.cache_data.clear()
        gc.collect()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Rendering Controls")
    z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 100.0, 10.0, 1.0,
                               help="Exaggerate Z to see ice thickness.")
    default_pts = 60_000 if low_mem else 150_000
    max_points = st.sidebar.number_input("Max Points (decimation)",
                                         5_000, 500_000, default_pts, 5_000)
    if low_mem and max_points > 80_000:
        max_points = 80_000
        st.sidebar.caption("Capped at 80k in Low-Memory Mode.")

    # ---------- Load skeleton (cheap, permanent) ----------
    data_dir = ensure_data_dir(DATA_DIR)
    with st.spinner("Scanning mesh files..."):
        skel = load_skeleton(data_dir, prefix)

    # Live RAM meter (refreshes on every rerun)
    mem = get_rss_mb()
    if mem is not None:
        mem_slot.metric("Server RAM in use", f"{mem:.0f} / {MEM_BUDGET_MB:.0f} MB")
        prog_slot.progress(min(mem / MEM_BUDGET_MB, 1.0))
        if mem > MEM_BUDGET_MB:
            st.sidebar.error("Over budget — press 'Free memory' and keep "
                             "Low-Memory Mode on.")

    if skel is None:
        st.error(f"No `.vtu`/`.pvtu` files found matching prefix `{prefix}`.")
        with st.expander("🔍 Debug Info", expanded=True):
            st.write(f"Looking in: `{data_dir}`")
            if not os.path.isdir(data_dir):
                st.write("❌ Directory does not exist. Ensure your data is committed "
                         "to GitHub or drop a `.zip` in the repo root.")
        return

    n_ts = skel["n_timesteps"]
    st.success(f"✅ Found {n_ts} timestep(s); mesh has {skel['n_pts']:,} nodes, "
               f"{len(skel['field_info'])} fields.")
    with st.expander("📁 Files & memory plan"):
        st.write([os.path.basename(f) for f in skel["files"]])
        keep = 2 if low_mem else 6
        skel_mb = (skel["points"].nbytes +
                   (skel["triangles"].nbytes if skel["triangles"] is not None else 0)) / 1e6
        st.caption(f"Per-timestep cache ≈ {skel['ts_bytes']/1e6:.1f} MB × ≤{keep} entries "
                   f"(≈ {skel['ts_bytes']*keep/1e6:.0f} MB max) "
                   f"+ skeleton ≈ {skel_mb:.1f} MB.")

    available_fields = list(skel["field_info"].keys())
    if not available_fields:
        st.error("No point-data fields found in the VTU/PVTU file(s).")
        return

    # ---------- Controls ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        default_field = "Velocity" if "Velocity" in available_fields else available_fields[0]
        field = st.selectbox("Select Field", available_fields,
                             index=available_fields.index(default_field))
    with col2:
        if n_ts > 1:
            timestep = st.slider("Timestep", 0, n_ts - 1, 0)
        else:
            timestep = 0
            st.info("Only 1 timestep — slider disabled.")
    with col3:
        colormap = st.selectbox("Colormap", COLORMAPS, index=0)

    # ---------- LAZY: load ONLY the selected timestep ----------
    loader = load_ts_lowmem if low_mem else load_ts
    with st.spinner(f"Reading timestep {timestep + 1}..."):
        ts_fields = loader(skel["files"][timestep], skel["n_pts"])

    if field not in ts_fields:
        st.error(f"Field `{field}` is missing in "
                 f"`{os.path.basename(skel['files'][timestep])}`.")
        return

    # ---------- Geometry (copy ONLY if exaggerating) ----------
    pts = skel["points"]
    if z_exag != 1.0:
        pts = pts.copy()
        pts[:, 2] *= z_exag

    kind = skel["field_info"][field]["kind"]
    raw = ts_fields[field]                       # shared cache array — read-only
    if kind == "scalar":
        values, label = raw, field
    else:
        values, label = np.linalg.norm(raw, axis=1), f"{field} (Magnitude)"

    valid_mask = ~np.isnan(values)
    plot_pts = pts[valid_mask]
    plot_vals = values[valid_mask]
    if len(plot_pts) == 0:
        st.warning("No valid (non-NaN) values to plot for this field/timestep.")
        return

    # ---------- Triangle path: NaN-filter + BUDGETED decimation ----------
    triangles = None
    if skel["triangles"] is not None:
        tris = skel["triangles"]
        tri_keep = valid_mask[tris].all(axis=1)
        tris = tris[tri_keep]
        remap = np.full(len(valid_mask), -1, dtype=np.int64)
        remap[valid_mask] = np.arange(int(valid_mask.sum()))
        triangles = remap[tris].astype(np.int32)

        max_tris = 2 * int(max_points)
        if len(triangles) > max_tris:
            plot_pts, plot_vals, triangles = decimate_triangles(
                plot_pts, plot_vals, triangles, max_tris)
            st.caption(f"Mesh decimated to {len(triangles):,} triangles / "
                       f"{len(plot_pts):,} vertices to fit the memory budget.")
    else:
        # Alphahull path — decimate points
        cap = min(int(max_points), 50_000)      # browser-safety cap
        if len(plot_pts) > cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(plot_pts), cap, replace=False)
            plot_pts, plot_vals = plot_pts[idx], plot_vals[idx]

    cmin, cmax = float(np.min(plot_vals)), float(np.max(plot_vals))
    auto_scale = st.checkbox("Auto Color Scale", value=True)
    if not auto_scale:
        c1, c2 = st.columns(2)
        cmin = c1.number_input("Min Limit", value=cmin, format="%.3e")
        cmax = c2.number_input("Max Limit", value=cmax, format="%.3e")

    # =============================================
    # PLOTLY 3D
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
            hovertemplate=(f'<b>{label}:</b> %{{intensity:.3e}}<br>'
                           'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>')))
    else:
        fig.add_trace(go.Mesh3d(
            x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2],
            alphahull=5,
            intensity=plot_vals, colorscale=colormap, intensitymode='vertex',
            cmin=cmin, cmax=cmax, opacity=0.9,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
            hovertemplate=(f'<b>{label}:</b> %{{intensity:.3e}}<br>'
                           'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>')))

    fig.update_layout(
        height=700, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.6)),
                   xaxis=dict(title="X (m)"), yaxis=dict(title="Y (m)"),
                   zaxis=dict(title="Z (m, exaggerated)")))
    st.plotly_chart(fig, use_container_width=True)
    del fig                       # release arrays + JSON before the next section
    if low_mem:
        gc.collect()

    # =============================================
    # MATPLOTLIB (opt-in — an expander still renders on every rerun)
    # =============================================
    with st.expander("🖼️ 2D Top-Down Projection (Matplotlib)"):
        if st.checkbox("Render 2D projection", value=not low_mem,
                       help="Off by default in Low-Memory Mode."):
            fig2, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(plot_pts[:, 0], plot_pts[:, 1], c=plot_vals,
                            s=2, cmap=colormap.lower())
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title(f"{label} (Top-Down)")
            plt.colorbar(sc, ax=ax, label=label)
            st.pyplot(fig2)
            plt.close(fig2)
            del fig2, ax

    # =============================================
    # STATISTICS (computed on the decimated set — cheap)
    # =============================================
    with st.expander("📊 Field Statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min",  f"{np.min(plot_vals):.3e}")
        c2.metric("Max",  f"{np.max(plot_vals):.3e}")
        c3.metric("Mean", f"{np.mean(plot_vals):.3e}")
        c4.metric("Std Dev", f"{np.std(plot_vals):.3e}")

if __name__ == "__main__":
    main()
