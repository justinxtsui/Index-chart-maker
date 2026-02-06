# --- SIDEBAR LOGIC FLOW ---
with st.sidebar:
    st.header("🛠 Configuration")
    
    # Placeholder to maintain logic flow before file is uploaded
    uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # 1. Load Data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.divider()

            # NEW PLACEMENT: Moved "Select data to analysis" here
            st.header("Select data to analysis")

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
                temp_dt = pd.to_datetime(temp_proc[time_column], errors='coerce')
                temp_proc[time_column] = temp_dt.dt.to_period('Q')
            
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
            selected_categories, category_column, include_overall = [], None, False
            
            if use_category:
                category_column = st.selectbox("Category Column", options=[c for c in data.columns if c != time_column])
                unique_cats = data[category_column].unique().tolist()
                selected_categories = st.multiselect(f"Specific Values", options=unique_cats)
                include_overall = st.checkbox("Include Overall Trend", value=True)

            st.divider()

            # LABELS SECTION
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

            # 7. Design Elements
            st.header("🎨 Design & Export")
            show_all_labels = st.checkbox("Show value labels on chart", value=True)
            only_final_label = st.checkbox("Only show final year value", value=False)
            export_format = st.selectbox("Format", options=['PNG', 'SVG (Vectorized)'])

        except Exception as e:
            st.sidebar.error(f"Setup Error: {e}")
