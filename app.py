import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import io

# --- DESIGN CONFIGURATION ---
PURPLE = '#6B67DA'
DARK_PURPLE = '#38358E'
BLACK_PURPLE = '#211E52'
DARK_GREY = '#4A4A4A'

st.set_page_config(page_title="Indexed Chart Creator", layout="wide")

# Custom CSS for UI Distinction and Header Styling
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: #f0f2f6;
        padding-top: 2rem;
    }}
    .stButton>button {{
        width: 100%;
        border-radius: 5px;
        background-color: {PURPLE};
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem;
    }}
    .stButton>button:hover {{
        background-color: {DARK_PURPLE};
        color: white;
    }}
    /* Title Styling */
    .app-title {{
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1px;
        color: {BLACK_PURPLE};
        margin-bottom: 0px;
        line-height: 1.1;
    }}
    /* Attribution Styling - Smaller font below title */
    .app-attribution {{
        font-size: 24px;
        font-weight: 600;
        color: {BLACK_PURPLE};
        margin-top: 0px;
        margin-bottom: 10px;
    }}
    /* Subtitle updated to Black */
    .app-subtitle {{
        color: #000000;
        font-size: 18px;
        margin-bottom: 5px;
        font-weight: normal;
    }}
    /* Custom Bolder Divider */
    .bold-divider {{
        height: 3px;
        background-color: #e6e9ef;
        border: none;
        margin-top: 10px;
        margin-bottom: 25px;
    }}
    /* Anti-Crop Fix for Logo */
    [data-testid="stImage"] {{
        padding: 5px;
        background-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

# Set global font configuration
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# --- HEADER AREA ---
st.image("https://github.com/justinxtsui/Index-chart-maker/blob/main/Screenshot%202026-02-06%20at%2016.51.25.png?raw=true", width=400) 

st.markdown('<div class="app-title">Dexter ( ◡̀_◡́)ᕤ </div>', unsafe_allow_html=True)
st.markdown('<div class="app-attribution">by JT</div>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Turn fundraising exports into indexed time series charts (For internal use only)</p>', unsafe_allow_html=True)
st.markdown('<hr class="bold-divider">', unsafe_allow_html=True)

# --- SIDEBAR LOGIC FLOW ---
with st.sidebar:
    st.header("Let's go!")
    
    uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.divider()

            st.header("Select data to analysis")

            time_column = st.selectbox("1. Select Time Column", options=data.columns.tolist())
            
            time_period = st.radio("2. Time Period Type", options=['Year', 'Month', 'Quarter'], horizontal=True)

            temp_proc = data.copy()
            if time_period == 'Year':
                temp_proc[time_column] = temp_proc[time_column].astype(str).str.strip()
                temp_proc[time_column] = pd.to_numeric(temp_proc[time_column], errors='coerce')
                if temp_proc[time_column].isna().any() or temp_proc[time_column].max() > 3000:
                    temp_proc[time_column] = pd.to_datetime(data[time_column], errors='coerce').dt.year
            elif time_period == 'Month':
                temp_proc[time_column] = pd.to_datetime(temp_proc[time_column], errors='coerce')
            elif time_period == 'Quarter':
                temp_dt = pd.to_datetime(temp_proc[time_column], errors='coerce')
                temp_proc[time_column] = temp_dt.dt.to_period('Q')
            
            temp_proc = temp_proc.dropna(subset=[time_column])
            all_times = sorted(temp_proc[time_column].unique())

            col_start, col_end = st.columns(2)
            with col_start:
                start_time = st.selectbox("Start (100)", options=all_times, index=0)
            with col_end:
                end_time = st.selectbox("End Period", options=all_times, index=len(all_times)-1)

            st.divider()

            available_cols = [col for col in data.columns if col != time_column]
            value_columns = st.multiselect("3. Values to Plot", options=["Row Count"] + available_cols)

            st.write("4. Categorization")
            use_category = st.checkbox("Split by Category Column")
            selected_categories, category_column, include_overall = [], None, False
            
            if use_category:
                category_column = st.selectbox("Category Column", options=[c for c in data.columns if c != time_column])
                unique_cats = data[category_column].unique().tolist()
                selected_categories = st.multiselect(f"Specific Values", options=unique_cats)
                include_overall = st.checkbox("Include Overall Trend", value=True)

            st.divider()

            st.header("Labels")
            custom_chart_title = st.text_input("Chart Title", value=f"Indexed Trend")
            custom_y_label = st.text_input("Y Axis Label", value="Index Change (%)")
            
            custom_labels = {}
            if value_columns:
                st.write("**Line Key Labels**")
                keys_to_label = []
                if use_category and selected_categories:
                    if include_overall: 
                        for v in value_columns: keys_to_label.append(f"Overall - {v}")
                    for cat in selected_categories: 
                        for v in value_columns: keys_to_label.append(f"{cat} - {v}")
                else:
                    for v in value_columns: keys_to_label.append(v)
                
                for key in keys_to_label:
                    custom_labels[key] = st.text_input(f"Label: {key}", value=key)

            st.divider()

            st.header("Design & Export")
            show_all_labels = st.checkbox("Show value labels on chart", value=True)
            only_final_label = st.checkbox("Only show final year value", value=False)
            export_format = st.selectbox("Format", options=['PNG', 'SVG (Vectorized)'])

        except Exception as e:
            st.sidebar.error(f"Setup Error: {e}")

# --- MAIN CHARTING LOGIC ---
if uploaded_file is not None and value_columns:
    try:
        temp_chart_data = data.copy()
        if time_period == 'Year':
            temp_chart_data[time_column] = pd.to_numeric(temp_chart_data[time_column].astype(str).str.strip(), errors='coerce')
            if temp_chart_data[time_column].isna().any():
                temp_chart_data[time_column] = pd.to_datetime(data[time_column], errors='coerce').dt.year
        elif time_period == 'Month':
            temp_chart_data[time_column] = pd.to_datetime(temp_chart_data[time_column], errors='coerce')
        elif time_period == 'Quarter':
            temp_dt_c = pd.to_datetime(temp_chart_data[time_column], errors='coerce')
            temp_chart_data[time_column] = temp_dt_c.dt.to_period('Q')
        
        temp_chart_data = temp_chart_data.dropna(subset=[time_column])
        if "Row Count" in value_columns: temp_chart_data["Row Count"] = 1

        plot_groups = {}
        if use_category and selected_categories:
            if include_overall:
                ov_agg = temp_chart_data.groupby(time_column)[value_columns].sum().reset_index()
                for v in value_columns: plot_groups[f"Overall - {v}"] = ov_agg[[time_column, v]]
            for cat in selected_categories:
                c_agg = temp_chart_data[temp_chart_data[category_column] == cat].groupby(time_column)[value_columns].sum().reset_index()
                for v in value_columns: plot_groups[f"{cat} - {v}"] = c_agg[[time_column, v]]
        else:
            std_agg = temp_chart_data.groupby(time_column)[value_columns].sum().reset_index()
            for v in value_columns: plot_groups[v] = std_agg[[time_column, v]]

        processed_lines = []
        for orig_label, df in plot_groups.items():
            filtered = df[(df[time_column] >= start_time) & (df[time_column] <= end_time)].sort_values(time_column).reset_index(drop=True)
            if not filtered.empty:
                v_col = filtered.columns[1]
                base_val = filtered[filtered[time_column] == start_time][v_col].values[0]
                filtered['Idx'] = (filtered[v_col] / base_val * 100) if base_val != 0 else 100.0
                processed_lines.append({'label': orig_label, 'data': filtered})

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        colors = [PURPLE, DARK_PURPLE, DARK_GREY, '#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A825', '#2E7D32']
        x_vals_str = [str(t) for t in sorted(all_times) if start_time <= t <= end_time]
        x_pos = np.arange(len(x_vals_str))
        font_size = int(max(7, min(14, 150 / len(x_vals_str))))

        for i, line_obj in enumerate(processed_lines):
            color = colors[i % len(colors)]
            ax.plot(x_pos, line_obj['data']['Idx'], marker='o', label=custom_labels.get(line_obj['label'], line_obj['label']), 
                    color=color, linewidth=2.5, markersize=4)

        if show_all_labels and processed_lines:
            num_points = len(x_pos)
            for idx in range(num_points):
                current_values = []
                for line_obj in processed_lines:
                    if idx < len(line_obj['data']):
                        current_values.append(line_obj['data']['Idx'][idx])
                    else:
                        current_values.append(None)
                valid_vals = [v for v in current_values if v is not None]
                if not valid_vals: continue
                max_at_pos = max(valid_vals)
                min_at_pos = min(valid_vals)

                for i, line_obj in enumerate(processed_lines):
                    if idx >= len(line_obj['data']): continue
                    if only_final_label and idx != len(line_obj['data']) - 1: continue
                    
                    val = line_obj['data']['Idx'][idx]
                    perc = int(round(val - 100))
                    txt = f"{'+' if perc > 0 else ''}{perc}%"
                    color = colors[i % len(colors)]

                    if only_final_label:
                        ax.text(idx + 0.1, val, txt, ha='left', va='center', color=color, fontweight='bold', fontsize=font_size)
                    else:
                        if val == max_at_pos: va, v_off = 'bottom', 3
                        elif val == min_at_pos: va, v_off = 'top', -6
                        else: va, v_off = 'bottom', 3
                        ax.text(idx, val + v_off, txt, ha='center', va=va, color=color, fontweight='bold', fontsize=font_size)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_vals_str, fontsize=font_size)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v-100))}%"))
        ax.set_ylabel(custom_y_label, fontsize=font_size+2, fontweight='bold', color=DARK_GREY)
        ax.tick_params(axis='both', labelsize=font_size, length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), frameon=False, prop={'size': font_size})
        plt.title(custom_chart_title, fontsize=21, fontweight='bold', pad=60, color=BLACK_PURPLE)
        
        if only_final_label: 
            ax.set_xlim(right=len(x_pos)-0.5+0.8)
        else:
            ax.set_xlim(left=-0.5, right=len(x_pos)-0.5)
            
        plt.tight_layout()
        st.pyplot(fig)

        with st.sidebar:
            buf = io.BytesIO()
            fmt = "svg" if "SVG" in export_format else "png"
            fig.savefig(buf, format=fmt, dpi=300, bbox_inches='tight', transparent=True)
            st.download_button(f"📥 Download {export_format}", buf.getvalue(), f"chart.{fmt}", f"image/{fmt}")

    except Exception as e:
        st.error(f"Visualization Error: {e}")
elif uploaded_file is not None:
    st.info("👈 Please select 'Values to Plot' in the sidebar to generate the chart.")
else:
    st.info("👈 Please upload your data file in the sidebar to begin.")
