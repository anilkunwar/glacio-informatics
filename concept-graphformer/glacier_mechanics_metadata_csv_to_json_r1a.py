"""
Scopus CSV to JSON Converter - Multi-file, Multi-section
- Read CSV files from local 'metadatabase' folder (if present)
- Also accept user uploads
- All articles get unique IDs and are combined into a single JSON
- Supports deduplication by DOI/Scopus ID
- Custom folder path input via UI with os.path.join path handling
- Debug info for troubleshooting path issues
- FIXED: Proper NaN/Inf sanitization to ensure strict JSON compliance (null instead of NaN)
"""

import streamlit as st
import pandas as pd
import json
import uuid
import re
import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO
from typing import Optional, List, Tuple, Dict, Any


def sanitize_for_json(obj):
    """
    Recursively replace NaN, Inf, -Inf, numpy NA, pandas NA/NaT with None.
    Ensures output is strictly JSON-compliant (no bare NaN literals).
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.generic):
        # Catches numpy scalars: np.nan, np.int64, np.float32, etc.
        if np.issubdtype(type(obj), np.floating) or np.issubdtype(type(obj), np.complexfloating):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        # For integer numpy types, convert to native Python
        return obj.item()
    else:
        # Catch pandas NA, NaT, and any other null-like singletons
        try:
            import pandas as pd
            if pd.isna(obj) and obj is not None and not isinstance(obj, (str, bool, int, float)):
                return None
        except Exception:
            pass
        return obj


def parse_single_csv_section(section_content: str, source_filename: str, section_index: int) -> List[Dict[str, Any]]:
    """
    Parse one CSV section, return list of dicts with unique IDs.

    Args:
        section_content: Raw CSV text for one section
        source_filename: Name of the source file for tracking
        section_index: Index of this section within the file

    Returns:
        List of article dictionaries with metadata fields added
    """
    try:
        # Read CSV with UTF-8 encoding, handle missing values, force string dtype to preserve leading zeros
        df = pd.read_csv(StringIO(section_content), encoding='utf-8', dtype=str, keep_default_na=False)

        # Replace empty/whitespace-only cells AND common missing-value strings with None
        df = df.replace(r'^\s*$', None, regex=True)
        df = df.replace(['NaN', 'nan', 'NAN', 'NA', 'N/A', 'null', 'NULL', 'None', 'none'], None)

        # Final pandas-level sweep: any remaining pandas/numpy NA → None
        df = df.where(df.notna(), None)

        records = df.to_dict(orient='records')

        # CRITICAL: sanitize so no float NaN survives into JSON
        records = [sanitize_for_json(record) for record in records]

        # Add tracking metadata to each record
        for record in records:
            record['unique_id'] = str(uuid.uuid4())
            record['source_file'] = source_filename
            record['section_index'] = section_index
            record['import_timestamp'] = datetime.now().isoformat()

        return records
    except pd.errors.EmptyDataError:
        st.warning(f"⚠️ Empty CSV section {section_index} in {source_filename}")
        return []
    except pd.errors.ParserError as e:
        st.warning(f"⚠️ CSV parse error in section {section_index} of {source_filename}: {e}")
        return []
    except UnicodeDecodeError as e:
        st.warning(f"⚠️ Encoding error in {source_filename}: {e}. Try saving as UTF-8.")
        return []
    except Exception as e:
        st.warning(f"⚠️ Failed to parse section {section_index} in {source_filename}: {type(e).__name__}: {e}")
        return []


def parse_multi_section_csv(file_content: str, source_filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split file by blank lines, parse each chunk as CSV.

    Args:
        file_content: Full text content of the CSV file
        source_filename: Name of the source file

    Returns:
        Tuple of (all_articles list, summary dict with sections_parsed and total_articles)
    """
    # Split by one or more blank lines (handles \n\n, \n \n, etc.)
    sections = re.split(r'\n\s*\n', file_content.strip())
    all_articles: List[Dict[str, Any]] = []
    sections_parsed = 0

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Basic validation: must have comma and newline to be CSV
        if ',' in section and '\n' in section:
            articles = parse_single_csv_section(section, source_filename, i)
            if articles:
                all_articles.extend(articles)
                sections_parsed += 1
            else:
                st.warning(f"⚠️ Section {i+1} in `{source_filename}` parsed but yielded no articles")
        else:
            st.warning(f"⚠️ Skipping section {i+1} in `{source_filename}` – not valid CSV format (missing comma or newline)")

    summary = {
        "sections_parsed": sections_parsed,
        "total_articles": len(all_articles),
        "total_sections_found": len([s for s in sections if s.strip()])
    }
    return all_articles, summary


def parse_single_csv_file(file_content: str, source_filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse a CSV file (may be multi-section or single).

    Args:
        file_content: Full text content of the CSV file
        source_filename: Name of the source file

    Returns:
        Tuple of (articles list, summary dict)
    """
    file_content_stripped = file_content.strip()

    # Detect multi-section by looking for blank line separators
    if re.search(r'\n\s*\n', file_content_stripped):
        st.info(f"📄 `{source_filename}`: multi-section format detected")
        return parse_multi_section_csv(file_content, source_filename)
    else:
        # Single-section CSV
        articles = parse_single_csv_section(file_content_stripped, source_filename, 0)
        summary = {
            "sections_parsed": 1 if articles else 0,
            "total_articles": len(articles),
            "total_sections_found": 1
        }
        return articles, summary


def get_csv_files_from_folder(folder_path: str) -> List[Path]:
    """
    Scan a folder for all .csv files using os.path.join for path construction.

    Args:
        folder_path: Path to folder (relative or absolute)

    Returns:
        List of Path objects for CSV files found
    """
    # Use os.path.join for cross-platform path construction
    normalized_path = os.path.normpath(folder_path)

    if not os.path.exists(normalized_path):
        return []
    if not os.path.isdir(normalized_path):
        st.warning(f"⚠️ Path '{folder_path}' exists but is not a directory")
        return []

    # List all CSV files in the folder
    csv_files = []
    try:
        for filename in os.listdir(normalized_path):
            if filename.lower().endswith('.csv'):
                file_full_path = os.path.join(normalized_path, filename)
                if os.path.isfile(file_full_path):
                    csv_files.append(Path(file_full_path))
    except PermissionError:
        st.warning(f"⚠️ Permission denied reading folder: {normalized_path}")
        return []
    except Exception as e:
        st.warning(f"⚠️ Error scanning folder {normalized_path}: {e}")
        return []

    # Sort for consistent ordering
    csv_files.sort(key=lambda p: p.name)

    if not csv_files:
        st.info(f"📁 Folder '{folder_path}' contains no .csv files")
    else:
        st.success(f"📁 Found {len(csv_files)} CSV file(s) in '{folder_path}'")

    return csv_files


def find_metadatabase_folder() -> str:
    """
    Auto-detect the metadatabase folder by checking common paths.
    Uses os.path.join for all path operations.

    Returns:
        String path to the first valid metadatabase folder found, or default fallback
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define candidate paths using os.path.join
    candidates = [
        os.path.join(script_dir, "metadatabase"),
        os.path.join(script_dir, "database", "metadatabase"),
        os.path.join(script_dir, "data", "metadatabase"),
        os.path.join(script_dir, "metadatabase", "csv"),
        os.path.join(os.getcwd(), "metadatabase"),
        os.path.join(os.getcwd(), "database", "metadatabase"),
        os.path.join(os.getcwd(), "data", "metadatabase"),
        "metadatabase",  # relative to cwd
        os.path.join("database", "metadatabase"),  # relative to cwd
        os.path.join("data", "metadatabase"),  # relative to cwd
    ]

    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.exists(normalized) and os.path.isdir(normalized):
            # Verify it contains CSV files
            has_csv = any(f.lower().endswith('.csv') for f in os.listdir(normalized))
            if has_csv:
                return path

    # Return default fallback
    return os.path.join("database", "metadatabase")


def process_file_from_path(file_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read a CSV file from disk and parse it.

    Args:
        file_path: Path object pointing to the CSV file

    Returns:
        Tuple of (articles list, summary dict)
    """
    filename = file_path.name
    file_str_path = str(file_path)

    try:
        # Try multiple encodings in order of preference
        content = None
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_str_path, 'r', encoding=encoding) as f:
                    content = f.read()
                if encoding != 'utf-8':
                    st.info(f"ℹ️ `{filename}` read with fallback encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError("Could not decode file with any supported encoding")

        articles, summary = parse_single_csv_file(content, filename)
        return articles, summary

    except FileNotFoundError:
        st.error(f"❌ File not found: `{file_str_path}`")
        return [], {"error": f"File not found: {file_str_path}"}
    except PermissionError:
        st.error(f"❌ Permission denied reading: `{file_str_path}`")
        return [], {"error": f"Permission denied: {file_str_path}"}
    except Exception as e:
        st.error(f"❌ Error reading `{file_str_path}`: {type(e).__name__}: {e}")
        return [], {"error": f"{type(e).__name__}: {str(e)}"}


def process_uploaded_files(uploaded_files: List) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process user-uploaded files.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        Tuple of (all_articles list, file_summaries dict)
    """
    all_articles: List[Dict[str, Any]] = []
    file_summaries: Dict[str, Any] = {}
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Read and decode file content
            file_bytes = uploaded_file.read()
            content = None

            # Try UTF-8 first
            try:
                content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Try fallback encodings
                for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        content = file_bytes.decode(encoding)
                        st.info(f"ℹ️ `{uploaded_file.name}` decoded with fallback: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue

            if content is None:
                raise UnicodeDecodeError("utf-8", file_bytes[:100], 0, 100, "Could not decode with any encoding")

            articles, summary = parse_single_csv_file(content, uploaded_file.name)
            all_articles.extend(articles)
            file_summaries[uploaded_file.name] = summary

            if articles:
                st.success(f"✅ Upload `{uploaded_file.name}` → {summary['total_articles']} articles (from {summary['sections_parsed']} sections)")
            else:
                st.warning(f"⚠️ No articles found in upload `{uploaded_file.name}`")

        except UnicodeDecodeError as e:
            st.error(f"❌ Encoding error in `{uploaded_file.name}`: {e}")
            file_summaries[uploaded_file.name] = {"error": f"Encoding: {e}"}
        except Exception as e:
            st.error(f"❌ Error processing upload `{uploaded_file.name}`: {type(e).__name__}: {e}")
            file_summaries[uploaded_file.name] = {"error": f"{type(e).__name__}: {str(e)}"}

    return all_articles, file_summaries


def process_folder_files(folder_csv_paths: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process all CSV files from the local folder.

    Args:
        folder_csv_paths: List of Path objects for CSV files

    Returns:
        Tuple of (all_articles list, file_summaries dict)
    """
    all_articles: List[Dict[str, Any]] = []
    file_summaries: Dict[str, Any] = {}

    for file_path in folder_csv_paths:
        articles, summary = process_file_from_path(file_path)
        all_articles.extend(articles)
        file_summaries[file_path.name] = summary

        if "error" in summary:
            st.error(f"❌ Folder file `{file_path.name}`: {summary['error']}")
        elif articles:
            st.success(f"✅ Folder `{file_path.name}` → {summary['total_articles']} articles (from {summary['sections_parsed']} sections)")
        else:
            st.warning(f"⚠️ No articles found in folder file `{file_path.name}`")

    return all_articles, file_summaries


def deduplicate_articles(articles: List[Dict[str, Any]], key_field: str = 'DOI') -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove duplicate articles based on a key field.

    Args:
        articles: List of article dictionaries
        key_field: Field name to use for deduplication (default: 'DOI')

    Returns:
        Tuple of (deduplicated list, number of duplicates removed)
    """
    if not articles:
        return [], 0

    seen: set = set()
    unique: List[Dict[str, Any]] = []
    duplicates_removed = 0

    for article in articles:
        # Try multiple potential ID fields in order of preference
        key = None
        id_fields = [key_field, 'Scopus ID', 'EID', 'DOI', 'Article ID', 'unique_id']

        for field in id_fields:
            if field in article and article[field] and str(article[field]).strip():
                key = f"{field}:{str(article[field]).strip().lower()}"
                break

        if key is None:
            # Fallback to unique_id if no other identifier found
            key = article.get('unique_id')

        if key and key not in seen:
            seen.add(key)
            unique.append(article)
        else:
            duplicates_removed += 1

    return unique, duplicates_removed


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    # Page configuration
    st.set_page_config(
        page_title="Scopus Multi-CSV to JSON Converter",
        page_icon="📚",
        layout="wide",
        menu_items={
            'Get Help': 'https://github.com/anilkunwar/batteryenergydensity-knowledge',
            'Report a bug': 'https://github.com/anilkunwar/batteryenergydensity-knowledge/issues',
            'About': "### Scopus CSV to JSON Converter\n\nConvert Scopus export CSV files (including multi-section formats) to a unified JSON with unique IDs."
        }
    )

    # Header
    st.title("📚 Scopus CSV to JSON Converter")
    st.markdown(
        "**Convert Scopus export CSV files to unified JSON format**\n\n"
        "**Features:**\n"
        "- ✅ Auto-detect CSV files in `metadatabase` folder (with path auto-detection)\n"
        "- ✅ Upload additional CSV files (multiple allowed)\n"
        "- ✅ Handle multi-section CSV exports (separated by blank lines)\n"
        "- ✅ Assign unique UUID v4 to every article\n"
        "- ✅ Optional deduplication by DOI, Scopus ID, or EID\n"
        "- ✅ Single combined JSON output with full metadata\n"
        "- ✅ Cross-platform path handling with os.path.join\n"
        "- ✅ **FIXED**: Strict JSON compliance — all missing values output as `null` (no bare `NaN`)"
    )

    # Sidebar: Instructions and settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Folder path configuration
        st.subheader("📁 Source Folder")

        # Auto-detect and set default path
        default_folder = find_metadatabase_folder()

        folder_path_input = st.text_input(
            "CSV folder path (relative or absolute)",
            value=default_folder,
            help="Enter path to folder containing Scopus CSV exports. Auto-detected from common locations."
        )

        # Show resolved absolute path
        resolved_path = os.path.normpath(os.path.join(os.getcwd(), folder_path_input)) if not os.path.isabs(folder_path_input) else os.path.normpath(folder_path_input)
        st.caption(f"Resolved: `{resolved_path}`")

        # Deduplication option
        st.subheader("🔁 Deduplication")
        enable_dedup = st.checkbox(
            "Remove duplicate articles",
            value=False,
            help="Check to remove articles with matching DOI, Scopus ID, or EID"
        )
        dedup_field = st.selectbox(
            "Primary key for deduplication",
            options=['DOI', 'Scopus ID', 'EID', 'Title', 'Authors'],
            index=0,
            disabled=not enable_dedup,
            help="Field to use as primary identifier when removing duplicates"
        )

        # Debug info toggle
        st.subheader("🔍 Debug")
        show_debug = st.checkbox("Show working directory info", value=False)

        if show_debug:
            st.markdown("**Path Diagnostics**")
            st.code(f"📍 os.getcwd(): {os.getcwd()}")
            st.code(f"📄 Script dir: {os.path.dirname(os.path.abspath(__file__))}")
            st.code(f"📁 Input path: {folder_path_input}")
            st.code(f"🔗 Resolved: {resolved_path}")
            st.code(f"✅ Exists: {os.path.exists(resolved_path)}")
            st.code(f"📂 Is dir: {os.path.isdir(resolved_path) if os.path.exists(resolved_path) else 'N/A'}")

            if os.path.exists(resolved_path) and os.path.isdir(resolved_path):
                try:
                    csv_files = [f for f in os.listdir(resolved_path) if f.lower().endswith('.csv')]
                    st.code(f"📄 CSV files: {len(csv_files)}")
                    for cf in csv_files[:5]:
                        fp = os.path.join(resolved_path, cf)
                        size = os.path.getsize(fp)
                        st.caption(f"  • {cf} ({format_file_size(size)})")
                except Exception as e:
                    st.code(f"⚠️ Error listing: {e}")

        # Usage instructions
        st.header("📖 How to Use")
        st.markdown(
            "1. **Folder mode**: Place Scopus CSV exports in the folder specified above\n"
            "2. **Upload mode**: Use the file uploader below for additional files\n"
            "3. **Configure**: Enable deduplication if needed\n"
            "4. **Convert**: Click the button to process all sources\n"
            "5. **Download**: Get your combined JSON file\n\n"
            "**Note**: Multi-section CSVs (with blank line separators) are auto-detected."
        )

        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Refresh Folder Scan", use_container_width=True):
            st.rerun()

        if st.button("🧹 Clear Session State", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main content area
    st.divider()

    # Folder scanning
    st.subheader("📁 Folder Source")

    # Validate and scan folder
    folder_csv_paths = []
    folder_valid = False

    if folder_path_input:
        normalized_input = os.path.normpath(folder_path_input)
        if os.path.isabs(normalized_input):
            check_path = normalized_input
        else:
            check_path = os.path.normpath(os.path.join(os.getcwd(), folder_path_input))

        if os.path.exists(check_path) and os.path.isdir(check_path):
            folder_valid = True
            folder_csv_paths = get_csv_files_from_folder(check_path)
        else:
            st.warning(f"⚠️ Folder not found: `{folder_path_input}`\n\nResolved path: `{check_path}`")

    if folder_csv_paths:
        with st.expander(f"View {len(folder_csv_paths)} CSV file(s) found", expanded=False):
            for fp in folder_csv_paths:
                try:
                    size = os.path.getsize(str(fp))
                    st.text(f"• {fp.name} ({format_file_size(size)})")
                except:
                    st.text(f"• {fp.name}")

    # File upload section
    st.subheader("📤 Upload Additional Files")
    uploaded_files = st.file_uploader(
        "Select CSV files to upload (multiple allowed)",
        type=['csv'],
        accept_multiple_files=True,
        help="These files will be merged with folder sources. Each file can contain multiple sections separated by blank lines."
    )

    if uploaded_files:
        with st.expander(f"View {len(uploaded_files)} uploaded file(s)", expanded=False):
            for uf in uploaded_files:
                st.text(f"• {uf.name} ({format_file_size(uf.size)})")

    # Convert button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        convert_btn = st.button("🚀 Convert to JSON", type="primary", use_container_width=True)

    # Processing logic
    if convert_btn:
        all_articles: List[Dict[str, Any]] = []
        file_summaries: Dict[str, Any] = {}
        processing_errors: List[str] = []

        # Progress indicator
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        total_sources = len(folder_csv_paths) + (len(uploaded_files) if uploaded_files else 0)
        if total_sources == 0:
            st.warning("⚠️ No source files found. Add files to the folder or upload CSV files to continue.")
            st.info("💡 Tip: Check the sidebar debug info to verify your folder path is correct.")
            return

        progress_bar = st.progress(0)
        processed_count = 0

        # Process folder files
        if folder_csv_paths and folder_valid:
            status_placeholder.info(f"🔄 Processing {len(folder_csv_paths)} file(s) from folder: `{folder_path_input}`...")
            folder_articles, folder_summaries = process_folder_files(folder_csv_paths)
            all_articles.extend(folder_articles)
            file_summaries.update(folder_summaries)
            processed_count += len(folder_csv_paths)
            progress_bar.progress(processed_count / max(total_sources, 1))

        # Process uploaded files
        if uploaded_files:
            status_placeholder.info(f"🔄 Processing {len(uploaded_files)} uploaded file(s)...")
            upload_articles, upload_summaries = process_uploaded_files(uploaded_files)
            all_articles.extend(upload_articles)
            file_summaries.update(upload_summaries)
            processed_count += len(uploaded_files)
            progress_bar.progress(processed_count / max(total_sources, 1))

        status_placeholder.empty()
        progress_bar.empty()

        # Deduplication step
        if enable_dedup and all_articles:
            original_count = len(all_articles)
            all_articles, removed_count = deduplicate_articles(all_articles, key_field=dedup_field)
            if removed_count > 0:
                st.info(f"🔁 Deduplication: Removed {removed_count} duplicate(s), {len(all_articles)} unique article(s) remaining")
            else:
                st.success(f"🔁 No duplicates found based on '{dedup_field}'")

        # Final results
        if not all_articles:
            st.error("❌ No articles found in any source files. Please check:")
            st.markdown("""
            - ✅ CSV files have proper headers and data rows
            - ✅ Files are UTF-8 encoded (or try latin-1)
            - ✅ Multi-section files use blank lines as separators
            - ✅ Folder path is correct (check sidebar debug info)
            """)
            return

        # Success summary
        total_articles = len(all_articles)
        total_files_with_data = sum(1 for s in file_summaries.values() if s.get("total_articles", 0) > 0)
        total_json_size = len(json.dumps(all_articles, ensure_ascii=False, allow_nan=False))

        st.success(f"✅ **Success!** Processed **{total_articles:,} article(s)** from **{total_files_with_data} file(s)**")
        st.caption(f"Output size: ~{format_file_size(total_json_size)} (uncompressed JSON)")

        # Per-file summary table
        summary_data = []
        for fname, summary in file_summaries.items():
            summary_data.append({
                "File": fname,
                "Sections Parsed": summary.get("sections_parsed", 0),
                "Total Sections": summary.get("total_sections_found", "N/A"),
                "Articles": summary.get("total_articles", 0),
                "Status": "❌ Error" if "error" in summary else "✅ OK"
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.subheader("📊 Processing Summary")

            # Style the dataframe
            def highlight_errors(val):
                if isinstance(val, str) and "Error" in val:
                    return "color: #dc3545; font-weight: 600"
                return ""

            styled_df = summary_df.style.map(highlight_errors, subset=["Status"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Preview section
        with st.expander("🔍 Preview: First 5 Articles (Unique IDs)", expanded=False):
            for i, article in enumerate(all_articles[:5], 1):
                # Try common title field names
                title = (article.get('Title') or article.get('title') or 
                        article.get('Document title') or article.get('Document Title') or 
                        article.get('Article Title') or 'N/A')

                authors = (article.get('Authors') or article.get('authors') or 
                          article.get('Author(s)') or article.get('Author names') or 'N/A')

                year = (article.get('Year') or article.get('publicationYear') or 
                       article.get('Publication Year') or article.get('Year of Publication') or 'N/A')

                source = article.get('source_file', 'unknown')
                uid = article.get('unique_id', 'N/A')

                st.markdown(f"**{i}.** `{uid[:8]}...`")
                st.markdown(f"📝 **{str(title)[:120]}{'...' if len(str(title)) > 120 else ''}**")
                st.caption(f"👥 {authors} | 📅 {year} | 📁 {source}")
                st.divider()

        # JSON output
        st.subheader("💾 Download Combined JSON")

        # Generate JSON with proper formatting — FIXED: allow_nan=False enforces strict JSON
        combined_json = json.dumps(all_articles, indent=2, ensure_ascii=False, allow_nan=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"scopus_combined_{timestamp}.json"

        # Download button with size info
        json_size_kb = len(combined_json.encode('utf-8')) / 1024
        st.download_button(
            label=f"⬇️ Download JSON ({total_articles:,} articles, {json_size_kb:.1f} KB)",
            data=combined_json,
            file_name=combined_filename,
            mime="application/json",
            use_container_width=True
        )

        # Optional: Show full JSON preview (collapsible)
        with st.expander("📄 Full JSON Preview (First Article)", expanded=False):
            if all_articles:
                st.json(all_articles[0], expanded=False)

        # Copy instructions
        st.caption("💡 Tip: For large datasets, use the download button. For small outputs, you can copy directly from the JSON preview above.")

        # Export options
        with st.expander("🔧 Advanced Export Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                # Minified JSON option — FIXED: allow_nan=False
                minified_json = json.dumps(all_articles, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                minified_filename = f"scopus_minified_{timestamp}.json"
                st.download_button(
                    label="⬇️ Download Minified JSON",
                    data=minified_json,
                    file_name=minified_filename,
                    mime="application/json",
                    use_container_width=True
                )
            with col_b:
                # CSV export of unique IDs only
                id_df = pd.DataFrame([{'unique_id': a['unique_id'], 'title': a.get('Title') or a.get('title') or 'N/A', 'source': a.get('source_file')} for a in all_articles])
                id_csv = id_df.to_csv(index=False)
                id_filename = f"scopus_ids_{timestamp}.csv"
                st.download_button(
                    label="⬇️ Download IDs Only (CSV)",
                    data=id_csv,
                    file_name=id_filename,
                    mime="text/csv",
                    use_container_width=True
                )

    # Footer
    st.divider()
    st.caption(
        f"Scopus CSV to JSON Converter • "
        f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
        f"Working dir: {os.getcwd()} • "
        "All articles assigned UUID v4 unique identifiers • Strict JSON compliance enforced"
    )

    # Hidden debug: log key metrics to console for server deployments
    if st.session_state.get('_logged', False) is False:
        print(f"[ScopusConverter] Initialized at {datetime.now().isoformat()}")
        st.session_state['_logged'] = True


if __name__ == "__main__":
    main()
