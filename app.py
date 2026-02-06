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
        
        # Show data preview
        with st.expander("Preview Data"):
            st.dataframe(data.head(10))
        
        # Column selection
        st.subheader("Chart Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time column selection
            time_column = st.selectbox(
                "Select time column",
                options=data.columns.tolist(),
                help="Choose the column that contains time data (year, month, quarter, etc.)"
            )
            
            # Time period type
            time_period = st.selectbox(
                "Time period type",
                options=['Year', 'Month', 'Quarter', 'Custom'],
                help="Select the type of time period"
            )
        
        with col2:
            # Value columns selection (multiple)
            value_columns = st.multiselect(
                "Select value columns to plot",
                options=[col for col in data.columns if col != time_column],
                help="Choose one or more columns to display as lines"
            )
        
        if value_columns:
            # Prepare data
            chart_data = data[[time_column] + value_columns].copy()
            
            # Convert time column to appropriate format
            if time_period == 'Year':
                chart_data[time_column] = pd.to_datetime(chart_data[time_column], errors='coerce').dt.year
            elif time_period == 'Month':
                chart_data[time_column] = pd.to_datetime(chart_data[time_column], errors='coerce')
            elif time_period == 'Quarter':
                chart_data[time_column] = pd.to_datetime(chart_data[time_column], errors='coerce').dt.to_period('Q')
            
            # Remove rows with NaN in time column
            chart_data = chart_data.dropna(subset=[time_column])
            
            # Get unique time values
            unique_times = sorted(chart_data[time_column].unique())
            
            # Year range selection
            st.subheader("Time Range Selection")
            col3, col4 = st.columns(2)
            
            with col3:
                start_time = st.selectbox(
                    "Start time (baseline = 0%)",
                    options=unique_times,
                    index=0
                )
            
            with col4:
                end_time = st.selectbox(
                    "End time",
                    options=unique_times,
                    index=len(unique_times)-1
                )
            
            # Filter data by time range
            if time_period == 'Quarter':
                chart_data_filtered = chart_data[
                    (chart_data[time_column] >= start_time) & 
                    (chart_data[time_column] <= end_time)
                ].copy()
            else:
                chart_data_filtered = chart_data[
                    (chart_data[time_column] >= start_time) & 
                    (chart_data[time_column] <= end_time)
                ].copy()
            
            # Calculate indexed values
            for col in value_columns:
                base_value = chart_data_filtered[chart_data_filtered[time_column] == start_time][col].values[0]
                chart_data_filtered[f'{col}_Index'] = (chart_data_filtered[col] / base_value) * 100
            
            # Chart options
            st.subheader("Chart Options")
            col5, col6 = st.columns(2)
            
            with col5:
                show_labels = st.checkbox("Show percentage labels on lines", value=True)
                chart_title = st.text_input(
                    "Chart title",
                    value=f"Indexed Chart ({start_time} = 100)"
                )
            
            with col6:
                export_format = st.selectbox(
                    "Export format",
                    options=['PNG', 'SVG'],
                    help="Choose the format for exporting the chart"
                )
            
            # Create the chart
            st.subheader("Chart Preview")
            
            fig, ax = plt.subplots(figsize=(20, 10))
            
            x_values = chart_data_filtered[time_column].values
            x_pos = np.arange(len(x_values))
            
            # Calculate font size
            font_size = int(max(7, min(21, 150 / len(chart_data_filtered))))
            
            # Find max and min values
            index_cols = [f'{col}_Index' for col in value_columns]
            max_val = max([chart_data_filtered[col].max() for col in index_cols])
            min_val = min([chart_data_filtered[col].min() for col in index_cols])
            
            # Set y-axis range to show -20% to 20% (symmetric)
            y_min = 80  # -20%
            y_max = 120  # +20%
            
            # Color palette
            colors = [PURPLE, DARK_PURPLE, DARK_GREY, '#FF6B6B', '#4ECDC4', '#45B7D1']
            
            # Plot lines
            for idx, (col, color) in enumerate(zip(value_columns, colors[:len(value_columns)])):
                index_col = f'{col}_Index'
                ax.plot(x_pos, chart_data_filtered[index_col].values, 
                       color=color, marker='o', linestyle='-', 
                       linewidth=2.5, markersize=8, label=col)
            
            # Add value labels if enabled
            if show_labels:
                for idx, (col, color) in enumerate(zip(value_columns, colors[:len(value_columns)])):
                    index_col = f'{col}_Index'
                    y_values = chart_data_filtered[index_col].values
                    
                    for i, y in enumerate(y_values):
                        # Get all values at this x position
                        all_vals = [chart_data_filtered[f'{c}_Index'].values[i] for c in value_columns]
                        
                        # Determine positioning
                        if y == max(all_vals):
                            v_offset = max_val * 0.03
                            va = 'bottom'
                        elif y == min(all_vals):
                            v_offset = -max_val * 0.025
                            va = 'top'
                        else:
                            if abs(y - max(all_vals)) < abs(y - min(all_vals)):
                                v_offset = max_val * 0.03
                                va = 'bottom'
                            else:
                                v_offset = -max_val * 0.025
                                va = 'top'
                        
                        # Add text with percentage format
                        percentage_change = int(y - 100)
                        if percentage_change > 0:
                            label_text = f"+{percentage_change}%"
                        else:
                            label_text = f"{percentage_change}%"
                        
                        ax.text(x_pos[i], y + v_offset, label_text, 
                               ha='center', va=va, fontsize=font_size, 
                               color=color, fontweight='bold')
            
            # Configure x-axis
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_values, fontsize=font_size)
            ax.set_xlim(-0.5, len(x_values) - 0.5)
            
            # Configure y-axis
            ax.set_ylim(y_min - 5, y_max)
            
            def format_percentage(value, pos):
                return f"{int(value - 100)}%"
            
            ax.yaxis.set_major_formatter(FuncFormatter(format_percentage))
            ax.yaxis.set_major_locator(FixedLocator([80, 90, 100, 110, 120]))
            ax.tick_params(left=False, labelleft=True, length=0, labelsize=font_size)
            ax.tick_params(bottom=False, labelbottom=True, length=0, labelsize=font_size)
            
            # Remove all spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), frameon=False, prop={'size': 13}, ncol=1)
            
            # Add title
            plt.title(chart_title, fontsize=21, fontweight='bold', pad=100, color=BLACK_PURPLE)
            
            plt.tight_layout()
            
            # Display chart
            st.pyplot(fig)
            
            # Export functionality
            st.subheader("Export Chart")
            
            # Create download button
            buf = io.BytesIO()
            if export_format == 'PNG':
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                file_extension = 'png'
                mime_type = 'image/png'
            else:
                fig.savefig(buf, format='svg', bbox_inches='tight')
                file_extension = 'svg'
                mime_type = 'image/svg+xml'
            
            buf.seek(0)
            
            st.download_button(
                label=f"Download as {export_format}",
                data=buf,
                file_name=f"indexed_chart.{file_extension}",
                mime=mime_type
            )
            
            plt.close(fig)
        
        else:
            st.info("Please select at least one value column to plot.")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("Please upload a CSV or Excel file to get started.")