import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator
import io

# Design configuration
PURPLE = '#6B67DA'
DARK_PURPLE = '#38358E'
BLACK_PURPLE = '#211E52'
DARK_GREY = '#4A4A4A'

# Set font configuration
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

st.title('Indexed Chart Creator')
st.write('Upload your data and create professional indexed charts')

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.success(f"File loaded successfully! {len(data)} rows found.")
        
        with st.expander("Preview Data"):
            st.dataframe(data.head(10))
        
        st.subheader("1. Time & Value Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            time_column = st.selectbox("Select time column", options=data.columns.tolist())
            time_period = st.selectbox("Time period type", options=['Year', 'Month', 'Quarter', 'Custom'])
        
        with col2:
            available_cols = [col for col in data.columns if col != time_column]
            value_columns = st.multiselect("Select value columns to plot", options=["Row Count"] + available_cols)

        # Categorization Feature
        st.subheader("2. Categorization (Split Index)")
        use_category = st.checkbox("Enable Categorization / Split by Column")
        
        selected_categories = []
        category_column = None
        include_overall = False

        if use_category:
            cat_col1, cat_col2 = st.columns(2)
            with cat_col1:
                category_column = st.selectbox("Select category column", options=[c for c in data.columns if c != time_column])
                include_overall = st.checkbox("Include Overall Data (Total trend)", value=True)
            with cat_col2:
                unique_cats = data[category_column].unique().tolist()
                selected_categories = st.multiselect(f"Select values from {category_column}", options=unique_cats)

        if value_columns:
            temp_data = data.copy()
            
            # IMPROVED TIME CONVERSION LOGIC
            if time_period == 'Year':
                # Convert to string, strip whitespace, then extract digits and convert to int
                temp_data[time_column] = temp_data[time_column].astype(str).str.strip()
                # Use pd.to_numeric to get integers, coerce errors to NaN to drop them
                temp_data[time_column] = pd.to_numeric(temp_data[time_column], errors='coerce')
                
                # Check if it looks like it might still be a full date string (length > 4)
                # and only apply dt.year if the column wasn't successfully converted to simple years
                if temp_data[time_column].isna().any() or temp_data[time_column].max() > 3000:
                   temp_data[time_column] = pd.to_datetime(data[time_column], errors='coerce').dt.year
            
            elif time_period == 'Month':
                temp_data[time_column] = pd.to_datetime(temp_data[time_column], errors='coerce')
            elif time_period == 'Quarter':
                temp_data[time_column] = pd.to_datetime(temp_data[time_column], errors='coerce').dt.to_period('Q')
            
            temp_data = temp_data.dropna(subset=[time_column])
            if "Row Count" in value_columns:
                temp_data["Row Count"] = 1

            # Prepare plotting data structure
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

            # Time Range Selection
            all_times = sorted(temp_data[time_column].unique())
            st.subheader("3. Time Range & Indexing")
            tr_col1, tr_col2 = st.columns(2)
            with tr_col1:
                start_time = st.selectbox("Start time (baseline = 100)", options=all_times, index=0)
            with tr_col2:
                end_time = st.selectbox("End time", options=all_times, index=len(all_times)-1)

            show_all_labels = st.checkbox("Show labels on all points", value=True)

            # Process final indexed data
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

            # Chart Drawing
            st.subheader("Chart Preview")
            fig, ax = plt.subplots(figsize=(20, 10))
            colors = [PURPLE, DARK_PURPLE, DARK_GREY, '#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A825', '#2E7D32']
            
            x_vals_str = [str(t) for t in sorted(all_times) if start_time <= t <= end_time]
            x_pos = np.arange(len(x_vals_str))
            font_size = int(max(7, min(21, 150 / len(x_vals_str))))

            for i, (label, df) in enumerate(final_plot_data.items()):
                color = colors[i % len(colors)]
                ax.plot(x_pos, df['Index'], marker='o', label=label, color=color, linewidth=2.5, markersize=8)
                
                if show_all_labels:
                    indices = df['Index'].tolist()
                    for idx, val in enumerate(indices):
                        if pd.isna(val): continue
                        perc = int(round(val - 100))
                        txt = f"{'+' if perc > 0 else ''}{perc}%"
                        
                        if idx == 0:
                            va, v_offset = 'bottom', 3
                        else:
                            if val >= indices[idx-1]:
                                va, v_offset = 'bottom', 3
                            else:
                                va, v_offset = 'top', -6
                        
                        ax.text(idx, val + v_offset, txt, 
                                ha='center', va=va, color=color, fontweight='bold', fontsize=font_size)

            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_vals_str, fontsize=font_size)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v-100))}%"))
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), frameon=False, prop={'size': 12})
            plt.title(f"Indexed Trend ({start_time} = 100)", fontsize=21, fontweight='bold', pad=60, color=BLACK_PURPLE)
            plt.tight_layout()
            
            st.pyplot(fig)

            # Export
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            st.download_button("Download PNG", buf.getvalue(), "indexed_chart.png", "image/png")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a file to begin.")
