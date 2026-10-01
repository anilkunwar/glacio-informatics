import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Profile Transformer", layout="wide")
st.title("🔄 Glacier Profile Data Transformer")
st.markdown("""
This app converts raw CSV profile data into the `.dat` format required by the Glacier Mesh Deformer app.
""")

# Input area
st.header("1. Input Data")
input_text = st.text_area(
    "Paste your CSV/TXT data here (including headers):", 
    height=200,
    placeholder="0 | x | y\n1 | 376.383 | 5048.309\n2 | 494.464 | 5033.816\n..."
)
uploaded_file = st.file_uploader("Or upload a CSV/TXT file:", type=['csv', 'txt'])

# Transformation options
st.sidebar.header("⚙️ Transformation Options")
scale_to_zero = st.sidebar.checkbox("Shift X-axis to start at 0", value=True)
offset = st.sidebar.number_input("X-Offset (if not scaling to 0):", value=8000, step=100, disabled=scale_to_zero)

# Process the data
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
elif input_text:
    df = pd.read_csv(StringIO(input_text), sep=None, engine='python')

if df is not None:
    # Clean column names (lowercase, strip whitespace)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    st.write("### Original Parsed Data (Preview)")
    st.dataframe(df.head(), use_container_width=True)
    
    # Check for required columns
    if 'x' in df.columns and 'y' in df.columns:
        # Create new transformed dataframe
        trans_df = pd.DataFrame()
        
        x_vals = pd.to_numeric(df['x'], errors='coerce')
        
        if scale_to_zero:
            # X_new = X_max - X (Shifts min X to 0, flips to keep lower X at lower Z)
            trans_df['X'] = x_vals.max() - x_vals
        else:
            # X_new = Offset - X (Original method)
            trans_df['X'] = offset - x_vals
            
        # Rename Y to Z
        trans_df['Z'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Drop any rows that became NaN
        trans_df = trans_df.dropna()
        
        # CRITICAL: Sort by X ascending. 
        trans_df = trans_df.sort_values('X').reset_index(drop=True)
        trans_df = trans_df.drop_duplicates(subset='X').reset_index(drop=True)
        
        st.write("### Transformed Data (Preview)")
        st.dataframe(trans_df.head(), use_container_width=True)
        st.caption("ℹ️ *Note: The table above shows a preview. The download buttons below contain ALL rows.*")
        
        # Plotting Original vs Transformed
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Profile")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', name='Original', line=dict(color='blue')))
            fig1.update_layout(xaxis_title="Original X", yaxis_title="Y (Elevation)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Transformed Profile")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=trans_df['X'], y=trans_df['Z'], mode='lines+markers', name='Transformed', line=dict(color='red')))
            fig2.update_layout(xaxis_title="New X (Starts at 0)", yaxis_title="Z (Elevation)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
        st.header("2. Export Data")
        
        # Dynamically determine the default filename based on the uploaded file
        default_filename = "transformed_profile.dat"
        if uploaded_file is not None:
            # Extract base name without extension (e.g., "bedrock" from "bedrock.csv")
            base_name = uploaded_file.name.rsplit('.', 1)[0]
            default_filename = f"{base_name}.dat"
            
        file_name = st.text_input("Output Filename:", default_filename)
        
        # Convert to .dat format (space-separated, no header, no index, FULL ROWS)
        dat_str = trans_df.to_csv(sep=' ', index=False, header=False, float_format='%.4f')
        
        st.download_button(
            label="⬇️ Download Transformed .dat File",
            data=dat_str,
            file_name=file_name,
            mime='text/plain'
        )
        
        # Allow full rows CSV download as requested
        csv_filename = file_name.rsplit('.', 1)[0] + ".csv"
        csv_str = trans_df.to_csv(index=False, float_format='%.4f')
        
        st.download_button(
            label="⬇️ Download Transformed .csv File (Full Rows)",
            data=csv_str,
            file_name=csv_filename,
            mime='text/csv'
        )
        
        # Show a preview of the raw text format
        with st.expander("Preview raw .dat text format (First 500 chars)"):
            st.code(dat_str[:500] + ("\n..." if len(dat_str) > 500 else ""), language='plaintext')
            
    else:
        st.error("❌ Could not find 'x' and 'y' columns in your data. Please ensure your headers contain 'x' and 'y' (case-insensitive).")
else:
    st.info("👈 Paste your data above or upload a file to begin.")
