import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator
import io

# --- DESIGN CONFIGURATION ---
PURPLE = '#6B67DA'
DARK_PURPLE = '#38358E'
BLACK_PURPLE = '#211E52'
DARK_GREY = '#4A4A4A'

st.set_page_config(page_title="Indexed Chart Creator", layout="wide")

# Custom CSS for UI Enhancement
st.markdown(f"""
    <style>
    .main {{
        background-color: #f8f9fa;
    }}
    .stButton>button {{
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: {PURPLE};
        color: white;
    }}
    .stExpander {{
        background-color: white;
        border-radius: 10px;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 24px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Set font configuration
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# --- APP HEADER ---
with st.container():
    st.title('📊 Indexed Chart Creator')
    st.write('Transform raw data into professional, indexed trend visualizations.')
    st.divider()

# --- SIDEBAR: DATA UPLOAD & CONFIG ---
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        st.success("File Ready")
        st.header("⚙️ Chart Configuration")
        
        # We wrap these to keep the variables available for the main logic
        # 1. Time & Value Configuration
        time_column = st.selectbox("Time Column", options=[]) # Placeholder
        time_period = st.selectbox("Time Type", options=['Year', 'Month', 'Quarter', 'Custom'])
        
        # Placeholder for dynamic options after file load
        st.divider()
        st.header("🎨 Styling")
        show_all_labels = st.checkbox("Show Labels on Points", value=True)
        export_format = st.radio("Export Format", options=['PNG', 'SVG'], horizontal=True)

if uploaded_file is not None:
    try:
        # Load data (Same function logic)
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        # Update sidebar options dynamically
        # (This is the only clean way to map UI to data without changing core functions)
        time_column = st.sidebar.selectbox("Select time column", options=data.columns.tolist(), key="time_col_sidebar")
        available_cols = [col for col in data.columns if col != time_column]
        value_columns = st.sidebar.multiselect("Values to plot", options=["Row Count"] + available_cols)

        # Main Layout Columns
        col_main, col_settings = st.columns([3, 1])

        with col_settings:
            st.subheader("Split Logic")
            use_category = st.checkbox("Enable Categorization")
            
            selected_categories = []
            category_column = None
            include_overall = False

            if use_category:
                category_column = st.selectbox("Category Column", options=[c for c in data.columns if c != time_column])
                include_overall = st.checkbox("Include Overall Data", value=True)
                unique_cats = data[category_column].unique().tolist()
                selected_categories = st.multiselect(f"Values in {category_column}", options=unique_cats)
            
            st.divider()
            st.subheader("Range & Baseline")
            
            # Temporary preprocessing to get unique times for the slider/select
            temp_range_data = data.copy()
            if time_period == 'Year':
                temp_range_data[time_column] = pd.to_numeric(temp_range_data[time_column].astype(str).str.strip(), errors='coerce')
                if temp_range_data[time_column].isna().any():
                    temp_range_data[time_column] = pd.to_datetime(data[time_column], errors='coerce').dt.year
            
            all_times = sorted(temp_range_data[time_column].dropna().unique())
            
            if all_times:
                start_time = st.selectbox("Baseline (100)", options=all_times, index=0)
                end_time = st.selectbox("End Period", options=all_times, index=len(all_times)-1)

        with col_main:
            # Data Preview in an expander to save vertical space
            with st.expander("🔍 View Raw Data Preview"):
                st.dataframe(data.head(10), use_container_width=True)

            if value_columns:
                # --- START OF CORE CHARTING FUNCTION LOGIC (UNTOUCHED) ---
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
                        val_col = filtered.columns[1]
                        base_row = filtered[filtered[time_column] == start_time]
                        if not base_row.empty:
                            base_val = base_row[val_col].values[0]
                            filtered['Index'] = (filtered[val_col] / base_val * 100) if (base_val != 0 and pd.notna(base_val)) else 100.0
                            final_plot_data[label] = filtered

                # --- CHART PREVIEW CARD ---
                st.subheader("📈 Visualization")
                fig, ax = plt.subplots(figsize=(16, 8), facecolor='#f8f9fa')
                ax.set_facecolor('#f8f9fa')
                colors = [PURPLE, DARK_PURPLE, DARK_GREY, '#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A825', '#2E7D32']
                
                x_vals_str = [str(t) for t in sorted(all_times) if start_time <= t <= end_time]
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
                ax.legend(loc='upper right', bbox_to_anchor=(1, 1.1), frameon=False, prop={'family': 'sans-serif', 'size': font_size})
                plt.title(f"Indexed Trend ({start_time} = 100)", fontsize=18, fontweight='bold', pad=40, color=BLACK_PURPLE)
                plt.tight_layout()
                
                st.pyplot(fig)

                # Export Download Button
                buf = io.BytesIO()
                fig.savefig(buf, format=export_format.lower(), dpi=300, bbox_inches='tight')
                st.download_button(
                    label=f"💾 Download Chart as {export_format}",
                    data=buf.getvalue(),
                    file_name=f"indexed_chart.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}"
                )
            else:
                st.info("👈 Please select value columns in the sidebar to generate the chart.")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
else:
    # Empty State
    st.info("Welcome! Please upload a data file in the sidebar to get started.")
    st.image("https://img.icons8.com/clouds/500/000000/data-configuration.png", width=200)
