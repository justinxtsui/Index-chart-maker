import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from streamlit_sortables import sort_items

# --- THE FIX FOR EDITABLE SVG TEXT ---
plt.rcParams['svg.fonttype'] = 'none' 

# --- CONFIGURATION & PALETTE ---
PURPLE, DARK_PURPLE, LIGHT_PURPLE = '#6B67DA', '#38358E', '#BBBAF6'
WHITE_PURPLE, BLACK_PURPLE, YELLOW = '#EAEAFF', '#211E52', '#FFB914'

CATEGORY_COLORS = [PURPLE, DARK_PURPLE, LIGHT_PURPLE, BLACK_PURPLE]
SPLIT_LINE_PALETTE = [PURPLE, DARK_PURPLE, BLACK_PURPLE, YELLOW]

PREDEFINED_COLORS = {
    'Purple': PURPLE, 'Dark Purple': DARK_PURPLE, 'Light Purple': LIGHT_PURPLE,
    'White Purple': WHITE_PURPLE, 'Black Purple': BLACK_PURPLE, 'Yellow': YELLOW
}

SINGLE_BAR_COLOR, PREDICTION_SHADE_COLOR = '#BBBAF6', WHITE_PURPLE
DEFAULT_LINE_COLOR, TITLE_COLOR, DEFAULT_TITLE = '#000000', '#000000', 'Grant Funding and Deal Count Over Time'

st.set_page_config(page_title="Time Series Chart Generator", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR SIDEBAR & BRANDING ---
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: #f8f9fb;
        border-right: 1px solid #e6e9ef;
    }}
    [data-testid="stSidebar"] h2 {{
        color: {DARK_PURPLE};
        font-size: 1.2rem;
        border-bottom: 2px solid {LIGHT_PURPLE};
        padding-bottom: 5px;
        margin-top: 20px;
    }}
    .app-title {{
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1px;
        color: {BLACK_PURPLE};
        margin-bottom: 0px;
        line-height: 1.1;
    }}
    .app-attribution {{
        font-size: 24px;
        font-weight: 600;
        color: {BLACK_PURPLE};
        margin-top: 0px;
        margin-bottom: 10px;
    }}
    .app-subtitle {{
        color: #000000;
        font-size: 18px;
        margin-bottom: 5px;
        font-weight: normal;
    }}
    .bold-divider {{
        height: 3px;
        background-color: #e6e9ef;
        border: none;
        margin-top: 10px;
        margin-bottom: 25px;
    }}
    </style>
    """, unsafe_allow_html=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Public Sans', 'DejaVu Sans']

# --- HELPER FUNCTIONS ---

def format_currency(value):
    value = float(value)
    if value == 0: return "£0"
    neg, x_abs = value < 0, abs(value)
    if x_abs >= 1e9: unit, divisor = "b", 1e9
    elif x_abs >= 1e6: unit, divisor = "m", 1e6
    elif x_abs >= 1e3: unit, divisor = "k", 1e3
    else: unit, divisor = "", 1.0
    scaled = x_abs / divisor
    s = f"{scaled:.3g}"
    if float(s).is_integer(): s = str(int(float(s)))
    return f"{'-' if neg else ''}£{s}{unit}"

def is_dark_color(hex_color):
    try:
        r, g, b = to_rgb(hex_color)
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5
    except: return False

@st.cache_data
def load_data(uploaded_file, sheet_name=None):
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    data.columns = data.columns.str.strip()
    return data

def apply_filter(df, filter_configs):
    if not filter_configs: return df
    temp_df = df.copy()
    for config in filter_configs:
        col, values, include = config['column'], config['values'], config['include']
        if values: temp_df = temp_df[temp_df[col].isin(values)] if include else temp_df[~temp_df[col].isin(values)]
    return temp_df

def process_data(df, date_col, bar_val_col, line_val_col, year_range, line_cat_col, granularity):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='mixed', errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    
    # Handle Bar Values
    bar_col_to_use = "bar_internal_val"
    if bar_val_col == "Row Count":
        df["bar_internal_val"] = 1
    elif bar_val_col:
        df[bar_val_col] = pd.to_numeric(df[bar_val_col], errors='coerce').fillna(0)
        bar_col_to_use = bar_val_col
    else:
        df["bar_internal_val"] = 0

    # Handle Line Values
    line_col_to_use = "line_internal_val"
    if line_val_col == "Row Count":
        df["line_internal_val"] = 1
    elif line_val_col:
        df[line_val_col] = pd.to_numeric(df[line_val_col], errors='coerce').fillna(0)
        line_col_to_use = line_val_col
    else:
        df["line_internal_val"] = 0
    
    chart_data = df[df[date_col].dt.year.between(year_range[0], year_range[1], inclusive='both')].copy()
    if chart_data.empty: return None, "No data available."
    
    chart_data['time_period'] = chart_data[date_col].dt.to_period('Q').astype(str) if granularity == 'Quarterly' else chart_data[date_col].dt.year
    chart_data = chart_data.sort_values(date_col)
    
    # 1. Process Bars (Simple Sum)
    final_data = chart_data.groupby('time_period').agg({bar_col_to_use: 'sum'}).reset_index()
    final_data.rename(columns={bar_col_to_use: 'bar_total'}, inplace=True)

    # 2. Process Lines (Split or Simple)
    if line_cat_col != 'None' and line_cat_col in chart_data.columns:
        line_grouped = chart_data.groupby(['time_period', line_cat_col])[line_col_to_use].sum().reset_index()
        line_pivot = line_grouped.pivot(index='time_period', columns=line_cat_col, values=line_col_to_use).fillna(0)
        line_pivot.columns = [f"line_split_{c}" for c in line_pivot.columns]
        final_data = final_data.merge(line_pivot, on='time_period', how='left').fillna(0)
    else:
        line_metric = chart_data.groupby('time_period')[line_col_to_use].sum().reset_index(name='line_metric')
        final_data = final_data.merge(line_metric, on='time_period', how='left').fillna(0)
        
    return final_data, None

def generate_chart(final_data, bar_val_col, show_bars, show_line, title, y_axis_title, pred_y, line_cat_col, granularity):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_pos, time_labels = np.arange(len(final_data)), final_data['time_period'].values
    font_size = int(max(8, min(22, 150 / len(final_data))))
    
    # Scale Calculation
    if 'bar_total' in final_data.columns:
        y_max = final_data['bar_total'].max()
    else:
        y_max = 1
    y_max = y_max if y_max > 0 else 1

    # Plot Bars
    if show_bars and 'bar_total' in final_data.columns:
        for i in range(len(final_data)):
            val = final_data['bar_total'].iloc[i]
            # Simple bar plot
            ax1.bar(x_pos[i], val, 0.8, color=SINGLE_BAR_COLOR, edgecolor='none', linewidth=0)
            if val > 0:
                label_text = str(int(val)) if bar_val_col == "Row Count" else format_currency(val)
                ax1.text(x_pos[i], y_max*0.01, label_text, ha='center', va='bottom', fontsize=font_size, fontweight='bold', color='#000000')

    ax1.set_xticks(x_pos); ax1.set_xticklabels(time_labels, fontsize=font_size); ax1.set_ylim(0, y_max * 1.15)
    ax1.set_ylabel(y_axis_title, fontsize=16, fontweight='bold')
    
    ax1.tick_params(left=False, labelleft=False, length=0)
    for s in ax1.spines.values():
        s.set_visible(False)

    # Plot Lines
    if show_line:
        ax2 = ax1.twinx()
        line_cols = [c for c in final_data.columns if str(c).startswith('line_split_')] if line_cat_col != 'None' else ['line_metric']
        valid_line_cols = [c for c in line_cols if c in final_data.columns]
        
        if valid_line_cols:
            l_max = final_data.loc[:, valid_line_cols].values.max()
            l_max = l_max if l_max > 0 else 1
            for idx, l_col in enumerate(valid_line_cols):
                lc = SPLIT_LINE_PALETTE[idx % len(SPLIT_LINE_PALETTE)] if line_cat_col != 'None' else DEFAULT_LINE_COLOR
                ax2.plot(x_pos, final_data[l_col].values, color=lc, marker='o', linestyle='-', linewidth=2.5, markersize=8)
                for i, y in enumerate(final_data[l_col].values):
                    label_text = str(int(y)) 
                    ax2.text(x_pos[i], y + l_max*0.05, label_text, ha='center', va='bottom', fontsize=font_size, color=lc, fontweight='bold')
            ax2.axis('off'); ax2.set_ylim(0, l_max * 1.6)

    # Legends
    handles = []
    if show_bars:
        handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=SINGLE_BAR_COLOR, markersize=12, label='Value'))
    if show_line and 'valid_line_cols' in locals():
        if line_cat_col != 'None': 
            handles += [Line2D([0], [0], color=SPLIT_LINE_PALETTE[i % len(SPLIT_LINE_PALETTE)], marker='o', label=f"{str(c).replace('line_split_', '')}") for i, c in enumerate(valid_line_cols)]
        else: 
            handles.append(Line2D([0], [0], color=DEFAULT_LINE_COLOR, marker='o', label='Line Metric'))
    
    if handles: 
        ax1.legend(handles=handles, loc='upper left', frameon=False, prop={'size': 14}, ncol=2)
    
    plt.title(title, fontsize=22, fontweight='bold', pad=30); return fig

# --- APP HEADER AREA ---
st.image("https://github.com/justinxtsui/Index-chart-maker/blob/main/Screenshot%202026-02-06%20at%2016.51.25.png?raw=true", width=250) 

st.markdown('<div class="app-title">Dexter ( ◡̀_◡́)ᕤ </div>', unsafe_allow_html=True)
st.markdown('<div class="app-attribution">by JT</div>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Turn fundraising exports into indexed time series charts (For internal use only)</p>', unsafe_allow_html=True)
st.markdown('<hr class="bold-divider">', unsafe_allow_html=True)

# --- SIDEBAR LOGIC FLOW ---
with st.sidebar:
    # Use placeholder to show warnings at the very top
    warning_placeholder = st.empty()

    st.header("Let's go!")
    
    uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # 1. Select Data
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
            
            # Use top placeholder for error message
            if start_time > end_time:
                warning_placeholder.error("Start Period cannot be after End Period!")
                st.stop()

            # 2. Values
            st.header("Select Values")
            available_cols = [col for col in data.columns if col != time_column]
            value_columns = st.multiselect("3. Values to Plot", options=["Row Count"] + available_cols)

            # 3. Categorization
            st.header("Categorization")
            use_category = st.checkbox("Split by Category Column")
            selected_categories, category_column, include_overall = [], None, False
            
            if use_category:
                category_column = st.selectbox("Category Column", options=[c for c in data.columns if c != time_column])
                unique_cats = data[category_column].unique().tolist()
                selected_categories = st.multiselect(f"Specific Values", options=unique_cats)
                include_overall = st.checkbox("Include Overall Trend", value=True)

            # 4. Labels
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

            # 5. Design & Export
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
            st.download_button(f"Download {export_format}", buf.getvalue(), f"chart.{fmt}", f"image/{fmt}")

    except Exception as e:
        st.error(f"Visualization Error: {e}")
elif uploaded_file is not None:
    st.info("👈 Please select 'Values to Plot' in the sidebar to generate the chart.")
else:
    st.info("👈 Please upload your data file in the sidebar to begin.")
