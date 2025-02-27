import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from ydata_profiling import ProfileReport
import google.generativeai as genai

# Configure Google Gemini AI API
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# Function to process uploaded file
@st.cache_data
def process_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            return None
    except Exception as e:
        return None

st.title("📊 Data Sweeper - AI-Powered Data Cleaning & Analysis")

upload_files = st.file_uploader("Upload CSV or Excel file(s)", accept_multiple_files=True, type=["csv", "xlsx"])

if upload_files:
    for idx, file in enumerate(upload_files):
        df = process_file(file)
        if df is None:
            st.error(f"Error processing {file.name}. Please upload a valid file.")
            continue
        
        st.write(f"### File: {file.name} ({file.size / 1024:.2f} KB)")
        st.dataframe(df.head())
        
        # Convert column index to a list
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Data Cleaning
        st.subheader("🧹 Data Cleaning")
        
        if st.button(f"Clean {file.name}", key=f"clean_{idx}"):
            df.drop_duplicates(inplace=True)
            
            if numeric_cols:  # Ensure numeric_cols is not empty
                imputer = SimpleImputer(strategy="mean")
                imputed_data = imputer.fit_transform(df[numeric_cols])  # Transform separately
                df[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)  # Assign correctly
                
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols])
                df[numeric_cols] = pd.DataFrame(scaled_data, columns=numeric_cols, index=df.index)  # Assign correctly
            
            st.success("✅ Data cleaned successfully!")

        # Data Visualization
        st.subheader("📊 Data Visualization")
        if numeric_cols:
            chart_type = st.selectbox("Choose Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Box Plot"], key=f"chart_type_{idx}")
            x_col = st.selectbox("Select X-axis Column", numeric_cols, key=f"x_col_{idx}")
            y_col = st.selectbox("Select Y-axis Column", numeric_cols, key=f"y_col_{idx}")
            
            if chart_type in ["Bar", "Line", "Scatter"]:
                fig = getattr(px, chart_type.lower())(df, x=x_col, y=y_col)
            elif chart_type == "Box Plot":
                fig = px.box(df, x=x_col, y=y_col)
            else:
                fig = px.histogram(df, x=x_col)
            
            st.plotly_chart(fig)

        # AI Report
        st.subheader("📑 AI-Powered Report")
        if st.button(f"Generate Report {file.name}", key=f"report_{idx}"):
            profile = ProfileReport(df, explorative=True)
            report_path = f"{file.name}_report.html"
            profile.to_file(report_path)
            with open(report_path, "rb") as f:
                st.download_button("Download Report", f, file_name=report_path, mime="text/html")
            st.success("✅ Report Generated!")

        # Data Export
        st.subheader("🔄 Convert & Download")
        conversion_type = st.radio(f"Convert {file.name} to:", ["CSV", "Excel"], key=f"convert_{idx}")
        if st.button(f"Download {file.name} as {conversion_type}", key=f"download_{idx}"):
            buffer = BytesIO()
            if conversion_type == "CSV":
                df.to_csv(buffer, index=False)
                mime_type = "text/csv"
            else:
                df.to_excel(buffer, index=False)
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            buffer.seek(0)
            st.download_button("Download File", buffer, file_name=f"{file.name}.{conversion_type.lower()}", mime=mime_type)

    # AI Chatbot (Only appears after file upload)
    st.subheader("🤖 Ask AI About Your Data")
    user_query = st.chat_input("Ask a question about your data...")
    if user_query:
        with st.spinner("Thinking... 🤖"):
            try:
                prompt = f"Dataset Overview:\nColumns: {df.columns.tolist()}\nData Types: {df.dtypes.to_string()}\nSample Data:\n{df.head().to_string()}\n\nQuestion: {user_query}"
                response = model.generate_content([prompt])  # Ensure input is a list
                st.write("**AI Response:**", response.text)
            except Exception as e:
                st.error(f"❌ Error generating AI response: {e}")
