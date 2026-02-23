import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
from sqlalchemy import create_engine

# Function to clean data for training (enhanced to impute if needed)
def clean_training_data(X, y, impute_value=0):
    df_temp = X.copy()
    df_temp['target'] = y
    # Replace inf with a finite value
    df_temp = df_temp.replace([np.inf, -np.inf], impute_value)
    # Drop remaining NaNs
    df_clean = df_temp.dropna()
    X_clean = df_clean.drop(columns=['target'])
    y_clean = df_clean['target']
    return X_clean, y_clean

# App Title with Professional Styling
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Analytics Dashboard")
st.markdown("A professional tool for analyzing and predicting student results. Navigate using the sidebar.")

# Sidebar for Navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Section", ["Login", "Data Configuration", "Data Cleaning", "Visual Analytics", "Prediction Engine", "Export Reports"])

# Global variables using session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'subjects' not in st.session_state:
    st.session_state.subjects = []
if 'student_id_column' not in st.session_state:
    st.session_state.student_id_column = None
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'graph_type' not in st.session_state:
    st.session_state.graph_type = None
if 'users' not in st.session_state:
    st.session_state.users = {'admin': 'password'}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []

# Login Section
if page == "Login":
    st.header("User Login")
    st.markdown("Secure access to the dashboard. Admin can manage users.")
    if not st.session_state.logged_in:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.logged_in = True
                    st.success("Login successful! Navigate to other sections.")
                else:
                    st.error("Invalid credentials. Try again.")
    else:
        st.success(f"Welcome, {username}! You are logged in.")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        if username == 'admin':
            st.subheader("Admin: Add New User")
            with st.form("add_user_form"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                add_submitted = st.form_submit_button("Add User")
                if add_submitted:
                    st.session_state.users[new_user] = new_pass
                    st.success(f"User {new_user} added successfully.")

# Require login for other pages
if not st.session_state.logged_in and page != "Login":
    st.warning("Please log in first from the Login section.")
else:
    # Data Configuration
    if page == "Data Configuration":
        st.header("Data Configuration")
        st.markdown("Upload or connect to your student data source. The system will auto-detect subjects.")

        col1, col2 = st.columns(2)
        with col1:
            data_source = st.selectbox("Data Source", ["CSV/Excel Upload", "SQL Database"])
        with col2:
            if data_source == "CSV/Excel Upload":
                uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])
                if uploaded_file:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    else:
                        st.session_state.df = pd.read_excel(uploaded_file)
                    st.success("Data loaded!")
            elif data_source == "SQL Database":
                db_url = st.text_input("Database URL (e.g., sqlite:///mydb.db)")
                table_name = st.text_input("Table Name")
                if st.button("Load Data"):
                    engine = create_engine(db_url)
                    st.session_state.df = pd.read_sql_table(table_name, engine)
                    st.success("Data loaded!")

        if st.session_state.df is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head(5))

            # Refined subject detection: Exclude more non-subject keywords
            all_columns = st.session_state.df.columns.tolist()
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            excluded_keywords = ['id', 'total', 'count', 'pass', 'fail', 'date', 'time', 'cgpa', 'gpa', 'average', 'sum', 'percent', 'grade']
            likely_subjects = [col for col in numeric_cols if not any(kw in col.lower() for kw in excluded_keywords)]
            st.session_state.numeric_columns = likely_subjects

            st.subheader("Column Selection")
            col3, col4 = st.columns(2)
            with col3:
                id_suggestions = [col for col in all_columns if 'id' in col.lower() or st.session_state.df[col].dtype == 'object']
                st.session_state.student_id_column = st.selectbox("Student ID Column", id_suggestions + all_columns, index=0 if id_suggestions else 0)
            with col4:
                st.session_state.subjects = st.multiselect("Select Subjects (Auto-filtered)", likely_subjects, default=likely_subjects)
                st.info("Only potential subject columns are shown. Edit if needed.")

            st.subheader("Analysis Setup")
            col5, col6 = st.columns(2)
            with col5:
                st.session_state.analysis_type = st.selectbox("Analysis Type", [
                    "Individual Student Analysis", "Class-wise Analysis by Subject", "Class-wise Analysis for All Subjects",
                    "Top Performers Analysis", "Low Performers Analysis", "Correlation Analysis", "Overall Class Statistics",
                    "Grade Trends Over Time", "Comparative Analysis", "At-Risk Students"
                ])
            with col6:
                st.session_state.graph_type = st.selectbox("Graph Type", ["Bar Plot", "Box Plot", "Histogram", "Line Plot", "Pie Chart", "Scatter Plot", "Heatmap"])

    # Data Cleaning
    elif page == "Data Cleaning":
        st.header("Data Cleaning")
        st.markdown("Prepare your data by handling missing values and outliers.")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df.head(5))

            col1, col2 = st.columns(2)
            with col1:
                fill_method = st.selectbox("Missing Values", ["Drop Rows", "Fill Mean", "Fill Median"])
                if st.button("Apply"):
                    if fill_method == "Drop Rows":
                        st.session_state.df = st.session_state.df.dropna(subset=st.session_state.subjects)
                    elif fill_method == "Fill Mean":
                        st.session_state.df[st.session_state.subjects] = st.session_state.df[st.session_state.subjects].fillna(st.session_state.df[st.session_state.subjects].mean())
                    elif fill_method == "Fill Median":
                        st.session_state.df[st.session_state.subjects] = st.session_state.df[st.session_state.subjects].fillna(st.session_state.df[st.session_state.subjects].median())
                    st.success("Applied!")
            with col2:
                if st.button("Remove Outliers (Z-score > 3)"):
                    from scipy import stats
                    numeric_df = st.session_state.df[st.session_state.subjects]
                    st.session_state.df = st.session_state.df[(np.abs(stats.zscore(numeric_df)) < 3).all(axis=1)]
                    st.success("Outliers removed!")
        else:
            st.warning("Load data first.")

    # Visual Analytics (Made inputs optional where possible)
    elif page == "Visual Analytics":
        st.header("Visual Analytics")
        st.markdown("Explore insights for individual students, class-wide analyses, and more with interactive charts.")
        if st.session_state.df is not None and st.session_state.subjects:
            df = st.session_state.df.copy()
            if st.session_state.student_id_column in df.columns:
                df = df.set_index(st.session_state.student_id_column)

            def generate_plot(data, title, graph_type, x=None, y=None):
                x_val = x if x is not None else (data.index if hasattr(data, 'index') and not data.index.empty else None)
                y_val = y if y is not None else (data.columns.tolist() if hasattr(data, 'columns') and not data.columns.empty else None)
                if x_val is None or y_val is None:
                    st.warning("Insufficient data for plot.")
                    return

                # Fix for pie chart, histogram, scatter: ensure y is string if single value
                if graph_type in ["Pie Chart", "Histogram"] and isinstance(y_val, list) and len(y_val) == 1:
                    y_val = y_val[0]

                fig = None
                if graph_type == "Bar Plot":
                    fig = px.bar(data, x=x_val, y=y_val, title=title)
                elif graph_type == "Box Plot":
                    fig = px.box(data, y=y_val, title=title)
                elif graph_type == "Histogram":
                    fig = px.histogram(data, x=y_val, title=title)
                elif graph_type == "Line Plot":
                    fig = px.line(data, x=x_val, y=y_val, title=title)
                elif graph_type == "Pie Chart":
                    fig = px.pie(data, names=x_val, values=y_val, title=title)
                elif graph_type == "Scatter Plot":
                    if isinstance(y_val, list) and len(y_val) >= 2:
                        fig = px.scatter(data, x=y_val[0], y=y_val[1], title=title)
                    else:
                        st.warning("Scatter plot requires at least two columns for x and y.")
                        return
                elif graph_type == "Heatmap":
                    fig = go.Figure(data=go.Heatmap(z=data.values, x=data.columns, y=data.index, colorscale='Viridis'))
                    fig.update_layout(title=title)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            analysis = st.session_state.analysis_type
            with st.expander(f"Analysis: {analysis}", expanded=True):
                if analysis == "Individual Student Analysis":
                    student_id = st.text_input("Enter Student ID (Optional - Leave blank for random example)")
                    if not student_id and len(df.index) > 0:
                        student_id = df.index[0]  # Default to first student if blank
                        st.info(f"No ID provided. Showing example for {student_id}.")
                    if student_id:
                        # Convert to string for matching
                        student_id_str = str(student_id)
                        if student_id_str in df.index.astype(str):
                            student_data = df.loc[df.index.astype(str) == student_id_str, st.session_state.subjects].iloc[0]
                            st.subheader("Individual Student Scores")
                            st.dataframe(student_data)
                            avg = student_data.mean()
                            st.metric("Average Score", f"{avg:.2f}")
                            plot_data = pd.DataFrame({'Subjects': st.session_state.subjects, 'Scores': student_data}).set_index('Subjects')
                            generate_plot(plot_data, f"Scores for {student_id}", st.session_state.graph_type, y='Scores')
                        else:
                            st.warning("Student ID not found. Ensure the ID matches the format in your data (e.g., check for leading zeros or strings).")

                elif analysis == "Class-wise Analysis by Subject":
                    subject = st.selectbox("Select Subject (Optional - Defaults to first)", st.session_state.subjects, index=0)
                    data = df[[subject]]
                    st.subheader("Class-wide Summary for Subject")
                    st.dataframe(data.describe())
                    generate_plot(data, f"Class Performance in {subject}", st.session_state.graph_type, y=subject)

                elif analysis == "Class-wise Analysis for All Subjects":
                    averages = df[st.session_state.subjects].mean()
                    data = pd.DataFrame({'Subjects': st.session_state.subjects, 'Average': averages}).set_index('Subjects')
                    st.subheader("Class Averages Across All Subjects")
                    st.dataframe(data)
                    generate_plot(data, "Class-wide Averages", st.session_state.graph_type, y='Average')

                elif analysis == "Top Performers Analysis":
                    df['Average'] = df[st.session_state.subjects].mean(axis=1)
                    top = df.sort_values('Average', ascending=False).head(10)
                    plot_data = pd.DataFrame({'Student': top.index, 'Average': top['Average']}).set_index('Student')
                    st.subheader("Top 10 Class Performers")
                    st.dataframe(plot_data)
                    generate_plot(plot_data, "Top Performers (Class-wide)", st.session_state.graph_type, y='Average')

                elif analysis == "Low Performers Analysis":
                    df['Average'] = df[st.session_state.subjects].mean(axis=1)
                    low = df.sort_values('Average', ascending=True).head(10)
                    plot_data = pd.DataFrame({'Student': low.index, 'Average': low['Average']}).set_index('Student')
                    st.subheader("Low 10 Class Performers")
                    st.dataframe(plot_data)
                    generate_plot(plot_data, "Low Performers (Class-wide)", st.session_state.graph_type, y='Average')

                elif analysis == "Correlation Analysis":
                    corr = df[st.session_state.subjects].corr()
                    st.subheader("Class-wide Correlation Matrix")
                    st.dataframe(corr)
                    generate_plot(corr, "Subject Correlations (All Students)", "Heatmap")

                elif analysis == "Overall Class Statistics":
                    stats = df[st.session_state.subjects].describe()
                    st.subheader("Overall Class Statistics")
                    st.dataframe(stats)
                    stats_data = stats.loc[['mean', 'std']].T.reset_index().rename(columns={'index': 'Subject'})
                    generate_plot(stats_data, "Class Mean & Std Dev", st.session_state.graph_type, x='Subject', y=['mean', 'std'])

                elif analysis == "Grade Trends Over Time":
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    if date_cols:
                        date_col = st.selectbox("Date Column (Optional - Defaults to first)", date_cols, index=0)
                        df[date_col] = pd.to_datetime(df[date_col])
                        df['Average'] = df[st.session_state.subjects].mean(axis=1)
                        trends = df.groupby(date_col)['Average'].mean().reset_index()
                        st.subheader("Class-wide Grade Trends")
                        st.dataframe(trends.head())
                        generate_plot(trends, "Grade Trends (All Students)", st.session_state.graph_type, x=date_col, y='Average')
                    else:
                        st.warning("No date column detected.")

                elif analysis == "Comparative Analysis":
                    selected_subs = st.multiselect("Subjects to Compare (Optional - Defaults to all)", st.session_state.subjects, default=st.session_state.subjects)
                    if len(selected_subs) >= 2:
                        data = df[selected_subs]
                        st.subheader("Class-wide Comparison")
                        st.dataframe(data.describe())
                        generate_plot(data, "Subject Comparisons (All Students)", st.session_state.graph_type, y=selected_subs)
                    else:
                        st.warning("Select at least 2 subjects.")

                elif analysis == "At-Risk Students":
                    threshold = st.number_input("Failure Threshold (Default: 50)", value=50.0)
                    df['Average'] = df[st.session_state.subjects].mean(axis=1)
                    at_risk = df[df['Average'] < threshold]
                    plot_data = pd.DataFrame({'Student': at_risk.index, 'Average': at_risk['Average']}).set_index('Student')
                    st.subheader("At-Risk Students (Class-wide)")
                    st.dataframe(at_risk)
                    generate_plot(plot_data, "At-Risk Analysis", st.session_state.graph_type, y='Average')

        else:
            st.info("Configure data in the Data Configuration section.")

    # Prediction Engine (Enhanced cleaning and user input for prediction)
    elif page == "Prediction Engine":
        st.header("Prediction Engine")
        st.markdown("Train a model on historical data, then predict marks for a selected subject based on manually entered marks for other subjects.")
        if st.session_state.df is not None and st.session_state.subjects:
            df = st.session_state.df
            target = st.selectbox("Select Subject to Predict", st.session_state.subjects)
            features = [s for s in st.session_state.subjects if s != target]

            if features and st.button("Train Model on Historical Data"):
                X = df[features]
                y = df[target]
                # Clean data before splitting (impute inf with 0)
                X, y = clean_training_data(X, y, impute_value=0)
                if len(X) == 0:
                    st.error("No valid data after cleaning. Please use Data Cleaning section to handle NaNs/infs.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    st.metric("Model Accuracy (MSE on Test Data)", f"{mse:.2f} (Lower is better)")

                    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    st.subheader("Feature Importance (How Other Subjects Influence Prediction)")
                    st.dataframe(importance)
                    fig = px.bar(importance, x='Feature', y='Importance', title="Feature Importance")
                    st.plotly_chart(fig)

                    st.subheader("Actual vs Predicted on Test Data")
                    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
                    fig = px.scatter(pred_df, x='Actual', y='Predicted')
                    st.plotly_chart(fig)

                    # Store model in session state for prediction
                    st.session_state.model = model
                    st.session_state.features = features
                    st.session_state.target = target
                    st.success("Model trained! Use the section below to make predictions.")

            # Prediction section (independent of training, but requires trained model)
            if 'model' in st.session_state:
                st.subheader(f"Predict {st.session_state.target} Based on Other Subjects")
                st.markdown("Enter marks for the following subjects, and the model will predict the score for the selected subject.")
                inputs = {}
                col1, col2 = st.columns(2)
                for i, f in enumerate(st.session_state.features):
                    with (col1 if i % 2 == 0 else col2):
                        inputs[f] = st.number_input(f"Enter mark for {f}", min_value=0.0, max_value=100.0, value=50.0)
                if st.button("Predict Score"):
                    input_df = pd.DataFrame([inputs])
                    # Clean input for prediction
                    input_df = input_df.replace([np.inf, -np.inf], 0).fillna(0)
                    pred = st.session_state.model.predict(input_df)[0]
                    st.metric(f"Predicted {st.session_state.target} Score", f"{pred:.2f}")
        else:
            st.info("Load and configure data first.")

    # Export Reports
    elif page == "Export Reports":
        st.header("Export Reports")
        st.markdown("Generate and download customized reports.")
        if st.session_state.df is not None:
            # Customized PDF: Improved formatting for readability with tabular structure and equal padding
            if st.button("Generate Summary PDF"):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter
                margin = 72  # 1 inch margin on all sides
                table_start_x = margin
                table_start_y = height - margin - 50
                cell_padding = 10
                row_height = 20
                col_width = (width - 2 * margin) / (len(st.session_state.subjects) + 1)  # Equal columns

                # Title
                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(width / 2, height - margin, "Student Performance Summary Report")

                # Key Statistics Table
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, table_start_y, "Key Statistics")

                stats = st.session_state.df[st.session_state.subjects].describe()
                selected_stats = stats.loc[['mean', 'std', 'min', 'max']]
                table_rows = ['Statistic'] + list(selected_stats.columns)
                table_data = [['mean', 'std', 'min', 'max']] + [[f"{selected_stats.loc[stat, col]:.2f}" for col in selected_stats.columns] for stat in selected_stats.index]

                # Draw table grid and text
                table_start_y -= row_height
                for i in range(len(table_data) + 1):  # Horizontal lines
                    c.line(table_start_x, table_start_y - i * row_height, table_start_x + (len(table_rows)) * col_width, table_start_y - i * row_height)
                for j in range(len(table_rows) + 1):  # Vertical lines
                    c.line(table_start_x + j * col_width, table_start_y, table_start_x + j * col_width, table_start_y - len(table_data) * row_height)

                # Fill table headers
                c.setFont("Helvetica-Bold", 10)
                for j, header in enumerate(table_rows):
                    c.drawString(table_start_x + j * col_width + cell_padding, table_start_y - row_height + cell_padding / 2 - 5, header)

                # Fill table data
                c.setFont("Helvetica", 10)
                for i, row in enumerate(table_data):
                    for j, cell in enumerate(row):
                        c.drawString(table_start_x + (j + 1) * col_width + cell_padding, table_start_y - (i + 1) * row_height + cell_padding / 2 - 5, str(cell))

                # Top 5 Subjects Section
                table_start_y -= (len(table_data) + 1) * row_height + 40
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, table_start_y, "Top 5 Subjects by Average")
                table_start_y -= 20
                averages = st.session_state.df[st.session_state.subjects].mean().sort_values(ascending=False).head(5)
                for idx, (sub, avg) in enumerate(averages.items()):
                    c.setFont("Helvetica", 10)
                    c.drawString(margin + cell_padding, table_start_y - idx * row_height, f"{sub}: {avg:.2f}")

                c.save()
                buffer.seek(0)
                b64 = base64.b64encode(buffer.getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="summary_report.pdf">Download Summary PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.download_button("Download Data as CSV", st.session_state.df.to_csv(index=False), "data_export.csv")
        else:
            st.info("No data available for export.")

# Note: Run with `streamlit run app.py`
