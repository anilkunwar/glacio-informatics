import streamlit as st
import os
import glob
import zipfile
import gc
import sys
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import meshio
import warnings

warnings.filterwarnings("ignore")
import streamlit as st
import os
import zipfile
import tempfile
import glob
import numpy as np
import meshio

st.set_page_config(page_title="VTU to NPZ Converter", page_icon="⚙️", layout="wide")

st.title("⚙️ Elmer VTU/PVTU to NPZ Converter")
st.markdown("""
This tool converts heavy, multi-file Elmer outputs into a single, highly compressed `.npz` file.  
💡 **Pro Tip:** If your total VTU file size is > 150 MB, **run this script locally** on your machine to avoid Streamlit Cloud's 1 GB RAM limit during conversion.
""")

# =============================================
# FILE UPLOAD & EXTRACTION
# =============================================
uploaded_files = st.file_uploader(
    "Upload VTU/PVTU files (or a single .zip containing them)",
    accept_multiple_files=True,
    type=["vtu", "pvtu", "zip"]
)

if not uploaded_files:
    st.info("👈 Please upload your Elmer output files to begin.")
    st.stop()

# Create a temporary directory to work in
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

        # Discover all VTU/PVTU files
        vtu_files = sorted(glob.glob(os.path.join(temp_dir, "**/*.vtu"), recursive=True))
        pvtu_files = sorted(glob.glob(os.path.join(temp_dir, "**/*.pvtu"), recursive=True))
        
        # Prioritize .pvtu if both exist (parallel output)
        files_to_process = pvtu_files if pvtu_files else vtu_files

        if not files_to_process:
            st.error("❌ No .vtu or .pvtu files found in the upload.")
            st.stop()

        st.success(f"✅ Found {len(files_to_process)} file(s) to process.")

    # =============================================
    # PROCESSING (Memory-Efficient Loop)
    # =============================================
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
        
        # Read mesh
        mesh = meshio.read(filepath)
        
        if t == 0:
            points = mesh.points.astype(np.float32)
            n_pts = len(points)
            
            # Extract surface triangles if they exist
            for block in mesh.cells:
                if block.type == "triangle":
                    triangles = block.data.astype(np.int32)
                    break
            
            # Initialize field arrays based on the first file's schema
            for key, arr in mesh.point_data.items():
                if np.issubdtype(arr.dtype, np.number):
                    comps = 1 if arr.ndim == 1 else arr.shape[1]
                    # Shape: (n_timesteps, n_points, comps)
                    shape = (n_timesteps, n_pts, comps)
                    fields_dict[key] = np.full(shape, np.nan, dtype=np.float32)
        
        # Populate fields for this timestep
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

    # =============================================
    # SAVE TO NPZ
    # =============================================
    save_dict = {
        "points": points,
        "triangles": triangles if triangles is not None else np.array([]),
        "n_timesteps": np.array([n_timesteps])
    }
    
    for key, arr in fields_dict.items():
        save_dict[f"field_{key}"] = arr

    output_filename = "glacier_data.npz"
    output_path = os.path.join(temp_dir, output_filename)
    
    np.savez_compressed(output_path, **save_dict)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    st.success(f"🎉 Successfully created `{output_filename}` ({file_size_mb:.2f} MB)")

    # Provide download button
    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Download glacier_data.npz",
            data=f,
            file_name="glacier_data.npz",
            mime="application/octet-stream"
        )

finally:
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
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
    """Current resident memory (MB).

    Priority: psutil (if installed) -> /proc/self/status (Linux / Streamlit
    Cloud) -> resource.getrusage peak as last resort. Correctly handles the
    kB-vs-bytes difference of ru_maxrss between Linux and macOS.
    """
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
        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: kB ; macOS: bytes
        return val / 1e3 if sys.platform != "darwin" else val / 1e6
    except Exception:
        return None

# =============================================
# DATA EXTRACTION
# =============================================
@st.cache_data(max_entries=2, show_spinner=False)
def ensure_data_dir(directory: str) -> str:
    """Self-extract <directory>.zip on first load if the folder is missing
    OR present but empty. Returns the resolved directory path. Pure — no UI."""
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
        except Exception:
            continue
    return directory

# =============================================
# SKELETON LOADER (cached, bounded, PURE)
# =============================================
@st.cache_resource(max_entries=5, show_spinner=False)
def load_skeleton(directory: str, prefix: str):
    """Tiny, permanent structure: file list + geometry + field catalogue.

    Reads ONLY the first file. Returns either a skeleton dict or an error dict
    ``{"__error__": msg}``. Emits no Streamlit UI so it can live safely inside
    cache_resource without cache-poisoning or stale-UI side effects.
    """
    if not os.path.isdir(directory):
        return {"__error__": f"Directory does not exist: {directory}"}

    pvtu = sorted(set(glob.glob(os.path.join(directory, "**", f"{prefix}*.pvtu"),
                                recursive=True)))
    files = pvtu if pvtu else sorted(set(
        glob.glob(os.path.join(directory, "**", f"{prefix}*.vtu"), recursive=True)))
    if not files:
        return {"__error__": f"No .vtu/.pvtu files match prefix '{prefix}' "
                             f"under {directory}"}

    try:
        m0 = meshio.read(files[0])
    except Exception as e:
        return {"__error__": f"Failed to read {os.path.basename(files[0])}: {e}"}

    points = np.asarray(m0.points, dtype=np.float32)
    points.setflags(write=False)   # shared across sessions — read-only

    triangles = None
    for block in m0.cells:
        if block.type == "triangle":
            tri = np.asarray(block.data, dtype=np.int32)
            tri.setflags(write=False)
            triangles = tri
            break

    # Classify only numeric fields; skip strings/objects outright.
    field_info = {}
    for key, arr in m0.point_data.items():
        a = np.asarray(arr)
        if not np.issubdtype(a.dtype, np.number):
            continue                                  # skip string/object fields
        comps = 1 if a.ndim == 1 else int(np.prod(a.shape[1:]))
        field_info[key] = {
            "comps": comps,
            "kind": "scalar" if comps == 1 else ("vector" if comps <= 3 else "tensor"),
        }

    del m0   # free meshio object (cells, float64 arrays, ...)

    n_pts = len(points)
    total_comps = sum(v["comps"] for v in field_info.values())
    return {
        "files": files,
        "n_timesteps": len(files),
        "points": points,
        "triangles": triangles,
        "field_info": field_info,
        "n_pts": n_pts,
        "total_comps": total_comps,
        "ts_bytes": n_pts * max(total_comps, 1) * 4,
    }

# =============================================
# PER-TIMESTEP LOADERS (PURE, TWO LRU POLICIES)
# =============================================
def _read_timestep(path: str, n_pts: int):
    """Pure worker: returns {field: ndarray} or {"__error__": msg}.

    Arrays are float32, shared from the cache — do NOT mutate in the caller.
    Note: scalar fields may arrive as (n,) OR (n, 1); the downstream
    ``compute_field_values`` helper normalizes both.
    """
    try:
        mesh = meshio.read(path)
    except Exception as e:
        return {"__error__": f"Failed to read {os.path.basename(path)}: {e}"}

    if mesh.points.shape[0] != n_pts:
        return {"__error__": (
            f"{os.path.basename(path)}: {mesh.points.shape[0]} points "
            f"(expected {n_pts}); file skipped.")}

    out = {}
    for key, arr in mesh.point_data.items():
        a = np.asarray(arr)
        if not np.issubdtype(a.dtype, np.number):
            continue                                  # skip non-numeric fields
        a = a.astype(np.float32, copy=False)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)             # tensors -> (n, comps)
        out[key] = a
    return out


# Two DISTINCT function identities -> two distinct Streamlit caches.
@st.cache_resource(max_entries=6, ttl=1800, show_spinner=False)
def load_ts_normal(path: str, n_pts: int):
    return _read_timestep(path, n_pts)


@st.cache_resource(max_entries=2, ttl=300, show_spinner=False)
def load_ts_lowmem(path: str, n_pts: int):
    return _read_timestep(path, n_pts)

# =============================================
# FIELD NORMALIZATION (shape-safe)
# =============================================
def compute_field_values(raw, field_name, n_pts):
    """Normalize any point-data array to (values_1D, label, valid_mask).

    Robust against:
      * scalars stored as (n,) OR (n, 1)   <- the source of the IndexError
      * vectors / tensors (n, k)           -> magnitude
      * integer fields (np.isnan -> TypeError)
      * +/-inf (np.isnan misses them)
      * arrays whose length != reference mesh (explicit error, not a crash)
    """
    a = np.asarray(raw)
    if a.ndim == 0:
        raise ValueError(f"Field '{field_name}' is a single constant — cannot plot.")
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"Field '{field_name}' is non-numeric ({a.dtype}) — skipped.")

    a2 = a.reshape(a.shape[0], -1)            # (n, comps): ravels (n,1) AND tensors

    if a2.shape[0] != n_pts:
        raise ValueError(
            f"Field '{field_name}' has {a2.shape[0]:,} values but the reference "
            f"mesh has {n_pts:,} nodes. The timestep files don't all share the "
            f"same mesh (mesh adaptation? a bare mesh file matching the prefix?). "
            f"Press '🗑️ Free memory now' and check the file list."
        )

    if a2.shape[1] == 1:
        values, label = a2[:, 0], field_name
    else:
        values = np.linalg.norm(a2, axis=1)
        label = f"{field_name} (Magnitude, {a2.shape[1]} comps)"

    try:
        valid_mask = np.isfinite(values)          # catches NaN AND +/-inf
    except TypeError:                              # integer dtypes etc.
        valid_mask = np.ones(values.shape, dtype=bool)

    return values, label, valid_mask

# =============================================
# TRIANGLE-AWARE DECIMATION
# =============================================
def decimate_triangles(points, values, triangles, max_tris):
    """Random triangle subsampling with vertex remapping.

    Keeps a VALID mesh and shrinks both RAM and the Plotly JSON payload.
    RNG is seeded so the view is stable across reruns.
    """
    if len(triangles) <= max_tris:
        return points, values, triangles
    rng = np.random.default_rng(0)
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

    # ---------- Sidebar: configuration ----------
    st.sidebar.header("⚙️ Configuration")
    prefix = st.sidebar.text_input("File Prefix", value="Stokes_ELA400_3D_diagnostic")

    # ---------- Sidebar: memory panel ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"🧠 Memory (budget {MEM_BUDGET_MB:.0f} MB)")
    mem_slot  = st.sidebar.empty()
    prog_slot = st.sidebar.empty()

    low_mem = st.sidebar.checkbox(
        "Low-Memory Mode", value=True,
        help="Keeps ≤2 timesteps in RAM, caps plot size, defers the 2D plot.")

    if st.sidebar.button("🗑️ Free memory now"):
        load_skeleton.clear()
        load_ts_normal.clear()
        load_ts_lowmem.clear()
        st.cache_data.clear()
        gc.collect()
        st.rerun()

    # ---------- Sidebar: rendering controls ----------
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Rendering Controls")
    z_exag = st.sidebar.slider("Z Exaggeration", 1.0, 100.0, 10.0, 1.0,
                               help="Exaggerate Z to see ice thickness.")

    upper = 80_000 if low_mem else 500_000
    default_pts = min(60_000 if low_mem else 150_000, upper)
    max_points = st.sidebar.number_input(
        "Max Points (decimation)", 5_000, upper, default_pts, 5_000,
        help="Cap applied via triangle- or point-subsampling. "
             f"Upper bound is {upper:,} in {'Low-Memory' if low_mem else 'Normal'} Mode.")

    # ---------- Load skeleton (cheap, permanent, bounded) ----------
    data_dir = ensure_data_dir(DATA_DIR)
    with st.spinner("Scanning mesh files..."):
        skel = load_skeleton(data_dir, prefix)

    # ---------- Live RAM meter ----------
    mem = get_rss_mb()
    if mem is not None:
        mem_slot.metric("Server RAM in use", f"{mem:.0f} / {MEM_BUDGET_MB:.0f} MB")
        prog_slot.progress(min(mem / MEM_BUDGET_MB, 1.0))
        if mem > MEM_BUDGET_MB:
            st.sidebar.error("Over budget — press 'Free memory' and keep "
                             "Low-Memory Mode on.")

    # ---------- Handle skeleton errors ----------
    if skel is None or "__error__" in skel:
        msg = skel["__error__"] if skel and "__error__" in skel else "Unknown error."
        st.error(msg)
        with st.expander("🔍 Debug Info", expanded=True):
            st.write(f"Looking in: `{data_dir}`")
            if not os.path.isdir(data_dir):
                st.write("❌ Directory does not exist. Ensure your data is committed "
                         "to GitHub or drop a `.zip` in the repo root.")
            else:
                st.write(f"Directory exists. Searching for prefix `{prefix}`.")
        return

    n_ts = skel["n_timesteps"]
    st.success(f"✅ Found {n_ts} timestep(s); mesh has {skel['n_pts']:,} nodes, "
               f"{len(skel['field_info'])} fields.")

    with st.expander("📁 Files & memory plan"):
        st.write([os.path.basename(f) for f in skel["files"]])
        keep = 2 if low_mem else 6
        skel_mb = (skel["points"].nbytes +
                   (skel["triangles"].nbytes if skel["triangles"] is not None else 0)) / 1e6
        st.caption(
            f"Upper bound per timestep ≈ {skel['ts_bytes']/1e6:.1f} MB "
            f"× ≤{keep} cached entries (≈ {skel['ts_bytes']*keep/1e6:.0f} MB max) "
            f"+ skeleton ≈ {skel_mb:.1f} MB. Actual RSS is usually lower "
            f"because most fields are small."
        )

    available_fields = list(skel["field_info"].keys())
    if not available_fields:
        st.error("No point-data fields found in the VTU/PVTU file(s).")
        return

    # ---------- Main controls ----------
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
    loader = load_ts_lowmem if low_mem else load_ts_normal
    with st.spinner(f"Reading timestep {timestep + 1}..."):
        ts_fields = loader(skel["files"][timestep], skel["n_pts"])

    if ts_fields is None or "__error__" in ts_fields:
        st.error(ts_fields["__error__"] if ts_fields else "Timestep load failed.")
        return

    if field not in ts_fields:
        st.error(f"Field `{field}` is missing in "
                 f"`{os.path.basename(skel['files'][timestep])}`.")
        return

    # ---------- Debug expander (shape inspection) ----------
    with st.expander("🐛 Field array shapes (debug)", expanded=False):
        st.write({k: {"shape": tuple(np.asarray(v).shape),
                      "dtype": str(np.asarray(v).dtype)}
                  for k, v in ts_fields.items()})
        st.write({"mesh points": tuple(skel["points"].shape),
                  "n_pts": skel["n_pts"]})

    # ---------- Geometry (copy ONLY if exaggerating) ----------
    pts = skel["points"]                       # read-only shared array
    if z_exag != 1.0:
        pts = pts.copy()
        pts[:, 2] *= z_exag

    # ---------- Normalize the chosen field to a 1-D (values, mask) ----------
    raw = ts_fields[field]
    try:
        values, label, valid_mask = compute_field_values(raw, field, skel["n_pts"])
    except ValueError as e:
        st.error(str(e))
        return

    plot_pts = pts[valid_mask]                 # mask is guaranteed 1-D, length n_pts
    plot_vals = values[valid_mask]
    if len(plot_pts) == 0:
        st.warning("No finite values to plot for this field/timestep.")
        return

    # ---------- Triangle path: NaN-filter + budgeted decimation ----------
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
        # Alphahull path — decimate points. Hard cap for browser safety.
        cap = min(int(max_points), 50_000)
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
    del fig
    if low_mem:
        gc.collect()

    # =============================================
    # MATPLOTLIB (opt-in)
    # =============================================
    with st.expander("🖼️ 2D Top-Down Projection (Matplotlib)"):
        if st.checkbox("Render 2D projection", value=not low_mem,
                       help="Off by default in Low-Memory Mode."):
            fig2, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(plot_pts[:, 0], plot_pts[:, 1], c=plot_vals,
                            s=2, cmap=colormap.lower())
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{label} (Top-Down)")
            plt.colorbar(sc, ax=ax, label=label)
            st.pyplot(fig2)
            plt.close(fig2)
            del fig2, ax, sc

    # =============================================
    # STATISTICS (computed on the decimated set — cheap)
    # =============================================
    with st.expander("📊 Field Statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min",     f"{np.min(plot_vals):.3e}")
        c2.metric("Max",     f"{np.max(plot_vals):.3e}")
        c3.metric("Mean",    f"{np.mean(plot_vals):.3e}")
        c4.metric("Std Dev", f"{np.std(plot_vals):.3e}")

if __name__ == "__main__":
    main()
