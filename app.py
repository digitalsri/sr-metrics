import fitz  # PyMuPDF
import re
import os
import io
import time
import json
import pandas as pd
from typing import Dict, List
from PIL import Image
import streamlit as st
import google.generativeai as genai
import requests
import tempfile
from collections import defaultdict

# ----------------- APP CONFIGURATION -----------------

st.set_page_config(page_title="Climate Heroes KPI Extractor", page_icon="🌍", layout="wide")

# --- Model & API Config ---
AVAILABLE_MODELS = {
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (Recommended)",
    "gemini-2.5-flash": "Gemini 2.5 Flash (High Quality)",
}
DEFAULT_MODEL_KEY = "gemini-2.5-flash-lite"

# --- KPI & Ranking Definitions ---
KPI_DEFINITIONS = {
    "Scope 1 Emissions": {"keywords": ["scope 1", "scope-1", "direct ghg"], "default_selected": True},
    "Scope 2 Emissions": {"keywords": ["scope 2", "scope-2", "indirect energy"], "default_selected": True},
    "Scope 3 Emissions": {"keywords": ["scope 3", "scope-3", "value chain"], "default_selected": True},
    "Total Energy Usage": {"keywords": ["total energy consumption", "energy usage"], "default_selected": False},
    "Emissions Intensity": {"keywords": ["emissions intensity", "carbon intensity"], "default_selected": False},
    "Energy Intensity": {"keywords": ["energy intensity"], "default_selected": False}
}
EMISSIONS_KPIS = ["Scope 1 Emissions", "Scope 2 Emissions", "Scope 3 Emissions"]
GHG_UNITS = ["tco2e", "mt co2e", "co2-e", "metric tons of co2"]

# ----------------- API CONFIGURATION -----------------

@st.cache_resource
def configure_api():
    api_key = st.secrets.get("GEMINI_API_KEY_R5Y") or st.secrets.get("GOOGLE_API_KEY_RK") or st.secrets.get("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key); return True
        except Exception: return False
    return False

api_key_configured = configure_api()

# ----------------- CORE LOGIC & RANKING -----------------

@st.cache_data(show_spinner=False)
def find_and_rank_candidate_pages(pdf_content: bytes, kpis_to_search: List[str], target_year: int) -> Dict:
    candidate_pages = defaultdict(lambda: {'score': 0})
    years_to_check = [str(y) for y in range(target_year, target_year - 3, -1)]

    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text").lower()
            if not text.strip(): continue

            score = 0
            if page.find_tables(): score += 25
            if sum(1 for year in years_to_check if year in text) >= 2: score += 20
            if any(unit in text for unit in GHG_UNITS): score += 15
            
            selected_emissions_kpis = [kpi for kpi in EMISSIONS_KPIS if kpi in kpis_to_search]
            emissions_kpis_on_page = {kpi for kpi in selected_emissions_kpis if any(kw in text for kw in KPI_DEFINITIONS[kpi]['keywords'])}
            score += 10 * len(emissions_kpis_on_page)

            if score > 0:
                all_kpis_on_page = emissions_kpis_on_page.union(
                    {kpi for kpi in kpis_to_search if kpi not in EMISSIONS_KPIS and any(kw in text for kw in KPI_DEFINITIONS[kpi]['keywords'])}
                )
                if all_kpis_on_page:
                    candidate_pages[page_num]['score'] = score
                    candidate_pages[page_num]['kpis'] = all_kpis_on_page

    ranked_candidates = {kpi: [] for kpi in kpis_to_search}
    if not candidate_pages: return ranked_candidates
    
    sorted_pages = sorted(candidate_pages.items(), key=lambda item: -item[1]['score'])
    for page_num, data in sorted_pages:
        for kpi in data.get('kpis', []):
            ranked_candidates[kpi].append({"page": page_num, "score": data['score']})
            
    return ranked_candidates

def generate_prompt(kpi: str, target_year: int) -> str:
    kpi_specific_instructions = {
        "Scope 1 Emissions": "Look for a row labeled 'Scope 1' or 'Total Scope 1'. Prioritize the total, pre-calculated value over sub-categories.",
        "Scope 2 Emissions": "Search for 'Scope 2' emissions. You MUST identify if 'market-based' and/or 'location-based' values are provided. Extract each type as a separate object.",
        "Scope 3 Emissions": "Look for a row labeled 'Scope 3' or 'Total Scope 3'. Your goal is to find the single, pre-calculated total for Scope 3. Do NOT sum the individual categories yourself."
    }
    return f"""
    You are an expert AI assistant for extracting sustainability data. Analyze the image to extract the following metric.
    **Target KPI:** {kpi} **Target Year:** {target_year}
    **Instructions:**
    1. Locate `{kpi}` in the image, focusing on tables.
    2. Find the value for the year `{target_year}`. If not available, use the most recent year shown.
    3. Follow these rules: {kpi_specific_instructions.get(kpi, "Extract the total value for the specified KPI.")}
    4. You MUST return a JSON array of objects.
    **Required JSON Format**:
    ```json
    [
      {{"kpi": "{kpi}", "value": <number, or null>, "unit": "<string>", "year": <number>, "variant": "<'market-based' or 'location-based', otherwise null>", "confidence": <float, 0.0-1.0>}}
    ]
    ```
    """

def extract_kpi_with_gemini(image: Image.Image, kpi: str, target_year: int, model_id: str) -> List[Dict]:
    if not api_key_configured: return [{"kpi": kpi, "error": "API Key not configured."}]
    model = genai.GenerativeModel(model_id)
    prompt = generate_prompt(kpi, target_year)
    try:
        response = model.generate_content([prompt, image])
        json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if not json_match: return [{"kpi": kpi, "error": "No valid JSON array found in AI response."}]
        parsed_json = json.loads(json_match.group(0))
        return parsed_json if isinstance(parsed_json, list) else [parsed_json]
    except Exception as e:
        return [{"kpi": kpi, "error": f"API call or parsing failed: {str(e)}"}]

def page_to_image(doc: fitz.Document, page_num: int, dpi: int) -> Image.Image:
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def reset_session_state():
    for key in list(st.session_state.keys()):
        if key not in ['dpi_setting', 'scan_pages']: del st.session_state[key]
    st.session_state.kpi_selections = {kpi: data["default_selected"] for kpi, data in KPI_DEFINITIONS.items()}
    st.session_state.results, st.session_state.pdf_url = [], ""
    st.session_state.diagnostics = {'total_pdf_pages': 0, 'pages_analyzed': 0, 'api_calls': 0, 'kpi_success_rate': "0 / 0"}

# ----------------- STREAMLIT UI -----------------

def main():
    st.title("🌍 Sustainability KPI Extractor")
    st.success("Automate the extraction of key sustainability metrics with AI power. Empowering Our Climate Heroes!")

    with st.sidebar:
        st.header("⚙️ Settings & Info")
        if 'diagnostics' not in st.session_state: reset_session_state()
        model_display_name = st.selectbox("Select AI Model", options=list(AVAILABLE_MODELS.values()), index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY))
        st.session_state.dpi_setting = st.slider("PDF Rendering DPI", 100, 300, 150, 10, help="Higher DPI improves image quality.")
        st.session_state.scan_pages = st.slider("Max pages to scan per KPI", 1, 5, 3, 1, help="More pages increases thoroughness and cost.")
        
        st.markdown("---")
        st.header("📊 Current Run Stats")
        d = st.session_state.diagnostics
        c1, c2 = st.columns(2)
        c1.metric("Total Pages in PDF", d['total_pdf_pages'])
        c1.metric("API Calls Made", d['api_calls'])
        c2.metric("Pages Analyzed by AI", d['pages_analyzed'])
        c2.metric("KPI Success Rate", d['kpi_success_rate'])
        with st.expander("ℹ️ How Confidence Scores Work"):
            st.markdown("- **High (80%+)**: 🟢 Clear data.\n- **Medium (60-79%)**: 🟠 Good.\n- **Low (<60%)**: 🔴 Uncertain.")

    if 'pdf_url' not in st.session_state: st.session_state.pdf_url = ""
    with st.container(border=True):
        st.subheader("📄 1. Provide Report Details")
        c1, c2 = st.columns([3, 1])
        st.session_state.pdf_url = c1.text_input("Public PDF URL", value=st.session_state.pdf_url, placeholder="https://sustainability.atmeta.com/...")
        target_year = c2.number_input("Target Year", 2015, 2030, 2023)

    with st.container(border=True):
        st.subheader("🎯 2. Select KPIs to Target")
        kpi_cols = st.columns(3)
        for i, kpi in enumerate(KPI_DEFINITIONS.keys()):
            kpi_cols[i % 3].checkbox(kpi, value=st.session_state.kpi_selections.get(kpi, True), key=kpi)
    selected_kpis = [kpi for kpi, selected in st.session_state.items() if kpi in KPI_DEFINITIONS and selected]

    # **FIX 2: Inject CSS for the 'Start a New Analysis' button**
    st.markdown("""
        <style>
        /* Targets the 'Start a New Analysis' button (any button that isn't primary) */
        .stButton > button:not([kind="primary"]) {
            background-color: #f0f2f6;      /* A standard Streamlit light grey */
            color: #4f4f4f;                 /* A pleasing dark grey for text and icon */
            border: 1px solid #dcdcdc;      /* A subtle border */
        }
        .stButton > button:not([kind="primary"]):hover {
            background-color: #e6e6e6;
            color: #4f4f4f;
            border-color: #c0c0c0;
        }
        </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    extract_clicked = c1.button("🚀 Extract KPIs", type="primary", use_container_width=True, disabled=not api_key_configured)
    if c2.button("↻ Start a New Analysis", use_container_width=True): reset_session_state(); st.rerun()
    if not api_key_configured: st.warning("🚨 Gemini API key not configured. Please add it to your Streamlit secrets.")

    if extract_clicked:
        if not st.session_state.pdf_url: st.warning("Please provide a PDF URL."); st.stop()
        if not selected_kpis: st.warning("Please select at least one KPI."); st.stop()
        
        start_time = time.time()
        try:
            with st.spinner("Downloading and analyzing report..."):
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(st.session_state.pdf_url, headers=headers, timeout=30)
                response.raise_for_status()
                pdf_content = response.content
                
                candidate_pages = find_and_rank_candidate_pages(pdf_content, selected_kpis, target_year)
            
            with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                st.session_state.diagnostics['total_pdf_pages'] = len(doc)
                model_id = next(k for k, v in AVAILABLE_MODELS.items() if v == model_display_name)
                all_results, processed_pages = [], set()

                with st.status(f"Extracting {len(selected_kpis)} KPIs...", expanded=True) as status:
                    for kpi in selected_kpis:
                        status.write(f"Processing: **{kpi}**")
                        pages_to_scan = candidate_pages.get(kpi, [])
                        if not pages_to_scan:
                            all_results.append({"kpi": kpi, "value": None, "error": "No relevant pages found."}); continue

                        for page_info in pages_to_scan[:st.session_state.scan_pages]:
                            page_num = page_info["page"]
                            status.write(f"- Analyzing page `{page_num}` (Score: {page_info['score']}) for **{kpi}**...")
                            img = page_to_image(doc, page_num, dpi=st.session_state.dpi_setting)
                            st.session_state.diagnostics['api_calls'] = st.session_state.diagnostics.get('api_calls', 0) + 1
                            extraction_results = extract_kpi_with_gemini(img, kpi, target_year, model_id)
                            
                            processed_pages.add(page_num)
                            for res in extraction_results: res["page"] = page_num; all_results.append(res)
                            if any((r.get("confidence", 0) or 0) > 0.85 for r in extraction_results): break 
                
                st.session_state.results = all_results
                st.session_state.diagnostics['pages_analyzed'] = len(processed_pages)

                found_kpis = len({r.get('kpi') for r in all_results if pd.notna(r.get('value'))})
                if selected_kpis: st.session_state.diagnostics['kpi_success_rate'] = f"{found_kpis}/{len(selected_kpis)} ({found_kpis/len(selected_kpis):.0%})"

            total_time = time.time() - start_time
            st.success(f"Extraction complete in {total_time:.2f} seconds! Displaying results...")

        except requests.exceptions.RequestException as e: st.error(f"Failed to download PDF: {e}. Check URL.")
        except Exception as e: st.error(f"An unexpected error occurred: {e}")

    if st.session_state.get('results'):
        st.markdown("---"); st.subheader("📊 Extraction Results")
        df = pd.DataFrame(st.session_state.results).drop_duplicates(subset=['kpi', 'variant', 'value', 'year']).reset_index(drop=True)
        
        for _, row in df.iterrows():
            kpi_name = row.get('kpi')
            if pd.notna(row.get('variant')) and row.get('variant') != 'null': kpi_name = f"{kpi_name} ({row.get('variant').title()})"
            
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                if pd.notna(row.get('value')):
                    c1.metric(label=kpi_name, value=f"{row.get('value')} {row.get('unit', '')}", help=f"Year: {row.get('year')}")
                else:
                    c1.metric(label=kpi_name, value="Not Found", help=str(row.get('error', 'No value extracted.')))
                
                confidence = row.get('confidence', 0) or 0
                color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.6 else "red"
                c2.markdown(f"**Confidence:** <span style='color:{color};'>**{confidence:.0%}**</span>", unsafe_allow_html=True)
                
                if pd.notna(row.get('page')):
                    with st.expander(f"Verify on Page {int(row['page'])}"):
                        try:
                            verify_resp = requests.get(st.session_state.pdf_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
                            verify_resp.raise_for_status()
                            with fitz.open(stream=verify_resp.content, filetype="pdf") as verify_doc:
                                st.image(page_to_image(verify_doc, int(row['page']), dpi=200), use_container_width=True)
                        except Exception as e: st.error(f"Could not load verification image: {e}")
        
        export_df_data = [{'KPI': r.get('kpi'), 'Variant': r.get('variant'), 'Value': r.get('value'), 'Unit': r.get('unit'), 'Year': r.get('year'), 'Confidence': f"{r.get('confidence', 0):.2f}", 'Source Page': r.get('page')} for _, r in df.iterrows() if pd.notna(r.get('value'))]
        
        if export_df_data:
            st.markdown("---"); st.subheader("📥 Export Results")
            output = io.BytesIO()
            pd.DataFrame(export_df_data).to_excel(output, index=False, sheet_name='KPI_Results')
            st.download_button(label="📥 Download as Excel", data=output.getvalue(), file_name=f"KPI_Extraction_{target_year}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

if __name__ == "__main__":
    main()