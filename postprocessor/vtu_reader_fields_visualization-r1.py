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
