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

# Custom CSS for polished Sidebar and Buttons
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
    </style>
    """, unsafe_allow_html=True)

# Set global font configuration
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# --- MAIN AREA HEADER ---
st.title('📊 Indexed Chart Creator')

# --- SIDEBAR LOGIC FLOW ---
with st.sidebar:
    st.header("🛠 Configuration")
    uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # 1. Load Data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.divider()

            # 2. Select Time Column
            time_column = st.selectbox("1. Select Time Column", options=data.columns.tolist())
            
            # 3. Select Time Period Type
            time_period = st.radio("2. Time Period Type", options=['Year', 'Month', 'Quarter'], horizontal=True)

            # Pre-process time for range selection
            temp_proc = data.copy()
            if time_period == 'Year':
                temp_proc[time_column] = temp_proc[time_column].astype(str).str.strip()
                temp_proc[time_column] = pd.to_numeric(temp_proc[time_column], errors='coerce')
                if temp_proc[time_column].isna().any() or temp_proc[time_column].max() > 3000:
                    temp_proc[time_column] = pd.to_datetime(data[time_column], errors='coerce').dt.year
            elif time_period == 'Month':
                temp_proc[time_column] = pd.to_datetime(temp_proc[time_column], errors='coerce')
            elif time_period == 'Quarter':
                temp_proc[time_column] = pd.to_datetime(temp_proc[time_column], errors='coerce').dt.to_period('Q')
            
            temp_proc = temp_proc.dropna(subset=[time_column])
            all_times = sorted(temp_proc[time_column].unique())

            # 4. Select Time Range
            col_start, col_end = st.columns(2)
            with col_start:
                start_time = st.selectbox("Start (100)", options=all_times, index=0)
            with col_end:
                end_time = st.selectbox("End Period", options=all_times, index=len(all_times)-1)

            st.divider()

            # 5. Select Values to Plot
            available_cols = [col for col in data.columns if col != time_column]
            value_columns = st.multiselect("3. Values to Plot", options=["Row Count"] + available_cols)

            # 6. Select Category to Split
            st.write("4. Categorization")
            use_category = st.checkbox("Split by Category Column")
            selected_categories = []
            category_column = None
            include_overall = False
            
            if use_category:
                category_column = st.selectbox("Category Column", options=[c for c in data.columns if c != time_column])
                unique_cats = data[category_column].unique().tolist()
                selected_categories = st.multiselect(f"Specific {category_column} Values", options=unique_cats)
                include_overall = st.checkbox("Include Overall Trend", value=True)

            st.divider()

            # 7. Design Elements
            st.write("🎨 Design & Export")
            show_all_labels = st.checkbox("Show value labels on chart", value=True)
            export_format = st.selectbox("Format for Adobe/Web", options=['PNG', 'SVG (Vectorized)'])

        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN CHARTING LOGIC ---
if uploaded_file is not None and value_columns:
    try:
        # Re-apply processing to temp_data for charting
        temp_data = data.copy()
        if time_period == 'Year':
            temp_data[time_column] = temp_data[time_column].astype(str).str.strip()
            temp_data[time_column] = pd.to_numeric(temp_data[time_column], errors='coerce')
            if temp_data[time_column].isna().any() or temp_data[time_column].max() > 3000:
                temp_data[time_column] = pd.to_datetime(data[time_column], errors='coerce').dt.year
        elif time_period == 'Month':
            temp_data[time_column] = pd.to_datetime(temp_data[time_column], errors='coerce')
        elif time_period == 'Quarter':
            temp_data[time_column] = pd.to_datetime(temp_data[time_column], errors='coerce').dt.to_period('Q')
        
        temp_data = temp_data.dropna(subset=[time_column])
        if "Row Count" in value_columns:
            temp_data["Row Count"] = 1

        plot_groups = {}
        if use_category and selected_categories:
            if include_overall:
                overall_agg = temp_data.groupby(time_column)[value_columns].sum().reset_index()
                for v_col in value_columns:
                    plot_groups[f"Overall - {v_col}"] = overall_agg[[time_column, v_col]]
            for cat in selected_categories:
                cat_data = temp_data[temp_data[category_column] == cat]
                cat_agg = cat_data.groupby(time_column)[value_columns].sum().reset_index()
                for v_col in value_columns:
                    plot_groups[f"{cat} - {v_col}"] = cat_agg[[time_column, v_col]]
        else:
            standard_agg = temp_data.groupby(time_column)[value_columns].sum().reset_index()
            for v_col in value_columns:
                plot_groups[v_col] = standard_agg[[time_column, v_col]]

        final_plot_data = {}
        for label, group_df in plot_groups.items():
            filtered = group_df[(group_df[time_column] >= start_time) & (group_df[time_column] <= end_time)].copy()
            filtered = filtered.sort_values(time_column).reset_index(drop=True)
            if not filtered.empty:
                v_col = filtered.columns[1]
                base_row = filtered[filtered[time_column] == start_time]
                if not base_row.empty:
                    base_val = base_row[v_col].values[0]
                    filtered['Index'] = (filtered[v_col] / base_val * 100) if (base_val != 0 and pd.notna(base_val)) else 100.0
                    final_plot_data[label] = filtered

        # Drawing
        fig, ax = plt.subplots(figsize=(16, 8))
        colors = [PURPLE, DARK_PURPLE, DARK_GREY, '#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A825', '#2E7D32']
        
        x_vals_str = [str(t) for t in all_times if start_time <= t <= end_time]
        x_pos = np.arange(len(x_vals_str))
        font_size = int(max(7, min(14, 150 / len(x_vals_str))))

        for i, (label, df) in enumerate(final_plot_data.items()):
            color = colors[i % len(colors)]
            ax.plot(x_pos, df['Index'], marker='o', label=label, color=color, linewidth=2.5, markersize=8)
            
            if show_all_labels:
                indices = df['Index'].tolist()
                for idx, val in enumerate(indices):
                    if pd.isna(val): continue
                    perc = int(round(val - 100))
                    txt = f"{'+' if perc > 0 else ''}{perc}%"
                    if idx == 0: va, v_offset = 'bottom', 3
                    else: va, v_offset = ('bottom', 3) if val >= indices[idx-1] else ('top', -6)
                    ax.text(idx, val + v_offset, txt, ha='center', va=va, color=color, fontweight='bold', fontsize=font_size)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_vals_str, fontsize=font_size)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v-100))}%"))
        ax.tick_params(axis='both', which='major', labelsize=font_size, length=0)
        for label in ax.get_yticklabels(): label.set_fontfamily('sans-serif')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), frameon=False, prop={'family': 'sans-serif', 'size': font_size})
        plt.title(f"Indexed Trend ({start_time} = 100)", fontsize=21, fontweight='bold', pad=60, color=BLACK_PURPLE)
        plt.tight_layout()
        
        st.pyplot(fig)

        # Download in Sidebar for clean UI
        with st.sidebar:
            buf = io.BytesIO()
            fmt = "svg" if "SVG" in export_format else "png"
            fig.savefig(buf, format=fmt, dpi=300, bbox_inches='tight')
            st.download_button(f"📥 Download {export_format}", buf.getvalue(), f"chart.{fmt}", f"image/{fmt}")

        with st.expander("View Filtered Data Table"):
            st.write(final_plot_data)

    except Exception as e:
        st.error(f"Visualization Error: {e}")
elif uploaded_file is not None:
    st.info("Please select at least one value to plot from the sidebar.")
else:
    st.info("Upload a file in the sidebar to begin.")
