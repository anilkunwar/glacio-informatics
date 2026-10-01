import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Profile Transformer", layout="wide")
st.title("🔄 Glacier Profile Data Transformer")
st.markdown("""
This app converts raw CSV profile data into the `.dat` format required by the Glacier Mesh Deformer app.
**Transformations applied:**
1. Extracts `x` and `y` columns (ignoring index columns).
2. Renames `y` to `Z` (elevation).
3. Applies `X_new = 8000 - X` to flip the X axis.
4. Sorts `X` in strictly ascending order (required for `np.interp`).
5. Exports as a space-separated `.dat` file without headers.
""")

# Input area
st.header("1. Input Data")
input_text = st.text_area(
    "Paste your CSV/TXT data here (including headers):", 
    height=200,
    placeholder="0 | x | y\n1 | 376.383 | 5048.309\n2 | 494.464 | 5033.816\n..."
)
uploaded_file = st.file_uploader("Or upload a CSV/TXT file:", type=['csv', 'txt'])

offset = st.number_input("Enter X-Offset value:", value=8000, step=100)

# Process the data
df = None
if uploaded_file is not None:
    # Auto-detect separator (handles commas, pipes, tabs, spaces)
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
elif input_text:
    # Parse pasted text
    df = pd.read_csv(StringIO(input_text), sep=None, engine='python')

if df is not None:
    # Clean column names (lowercase, strip whitespace)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    st.write("### Original Parsed Data")
    st.dataframe(df.head(), use_container_width=True)
    
    # Check for required columns
    if 'x' in df.columns and 'y' in df.columns:
        # Create new transformed dataframe
        trans_df = pd.DataFrame()
        
        # Apply X transformation: X^T = 8000 - X
        trans_df['X'] = offset - pd.to_numeric(df['x'], errors='coerce')
        
        # Rename Y to Z
        trans_df['Z'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Drop any rows that became NaN due to formatting issues
        trans_df = trans_df.dropna()
        
        # CRITICAL: Sort by X ascending. 
        # If X=0 becomes 8000 and X=2500 becomes 5500, np.interp needs strictly increasing X.
        trans_df = trans_df.sort_values('X').reset_index(drop=True)
        
        # Drop duplicate X values just in case
        trans_df = trans_df.drop_duplicates(subset='X').reset_index(drop=True)
        
        st.write("### Transformed Data (Ready for .dat)")
        st.dataframe(trans_df.head(), use_container_width=True)
        
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
            fig2.update_layout(xaxis_title="New X (8000 - X)", yaxis_title="Z (Elevation)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
        # Convert to .dat format (space-separated, no header, no index)
        dat_str = trans_df.to_csv(sep=' ', index=False, header=False, float_format='%.4f')
        
        st.header("2. Export Data")
        file_name = st.text_input("Output Filename:", "surface_transformed.dat")
        
        st.download_button(
            label="⬇️ Download Transformed .dat File",
            data=dat_str,
            file_name=file_name,
            mime='text/plain'
        )
        
        # Show a preview of the raw text format
        with st.expander("Preview raw .dat text format"):
            st.code(dat_str[:500] + ("\n..." if len(dat_str) > 500 else ""), language='plaintext')
            
    else:
        st.error("❌ Could not find 'x' and 'y' columns in your data. Please ensure your headers contain 'x' and 'y' (case-insensitive).")
else:
    st.info("👈 Paste your data above or upload a file to begin.")
