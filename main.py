import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from transformers import pipeline
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Load or initialize data
DATA_FILE = 'data.csv'
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=['date', 'study_hours', 'stress_level', 'sleep_hours', 'exercise_min'])
    df.to_csv(DATA_FILE, index=False)

# Simple generative AI using Hugging Face (local, no API key needed)
generator = pipeline('text-generation', model='gpt2')  # Lightweight GPT-like model

# Homepage/Dashboard
st.title("🧠 AI-Powered Study & Wellness Tracker for Olympiad Students")
st.subheader("Aligned with SDG 3: Good Health & Well-Being | SDG 4: Quality Education")
col1, col2 = st.columns(2)
with col1:
    st.image("sdg3.png", width=100) if 'sdg3.png' in os.listdir() else st.write("SDG 3 Icon")
with col2:
    st.image("sdg4.png", width=100) if 'sdg4.png' in os.listdir() else st.write("SDG 4 Icon")

st.write("Welcome! Track your habits, get AI insights, and personalized Olympiad plans. Reduce stress, boost performance.")

# Motivational quote
quotes = ["'The only way to do great work is to love what you do.' - Steve Jobs", "'Success is not final, failure is not fatal: It is the courage to continue that counts.' - Winston Churchill"]
st.info(random.choice(quotes))

# Tabbed interface for neatness
tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Input & Tracker", "🔍 Analysis & Insights", "🗓️ Study Pathway Generator", "📄 Export Report"])

with tab1:
    st.header("Daily Input")
    with st.form("daily_form"):
        study_hours = st.number_input("Study Hours Today", min_value=0.0, max_value=24.0, step=0.5)
        stress_level = st.slider("Stress Level (1-10)", 1, 10)
        sleep_hours = st.number_input("Sleep Hours Last Night", min_value=0.0, max_value=24.0, step=0.5)
        exercise_min = st.number_input("Exercise Minutes Today", min_value=0, step=5)
        submitted = st.form_submit_button("Submit")
        if submitted:
            new_data = pd.DataFrame({
                'date': [datetime.date.today().strftime('%Y-%m-%d')],
                'study_hours': [study_hours],
                'stress_level': [stress_level],
                'sleep_hours': [sleep_hours],
                'exercise_min': [exercise_min]
            })
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success("Data saved! Check Analysis tab for insights.")

with tab2:
    st.header("Analysis & Insights")
    if len(df) < 2:
        st.warning("Enter at least 2 days of data for analysis.")
    else:
        # Linear Regression: Predict stress from study hours
        X = df[['study_hours']]
        y = df['stress_level']
        reg = LinearRegression().fit(X, y)
        pred_stress = reg.predict([[10]])[0]  # Example prediction
        st.write(f"Predicted stress if studying 10 hours: {pred_stress:.1f}/10")
        
        # Decision Tree: Burnout risk (high if sleep <6 and stress >7)
        df['burnout_risk'] = np.where((df['sleep_hours'] < 6) & (df['stress_level'] > 7), 1, 0)
        X_tree = df[['sleep_hours', 'stress_level', 'study_hours']]
        y_tree = df['burnout_risk']
        if len(set(y_tree)) > 1:  # Need variety in data
            tree = DecisionTreeClassifier().fit(X_tree, y_tree)
            risk = tree.predict([[sleep_hours, stress_level, study_hours]])[0]
            st.write("Burnout Risk Today: " + ("High! Rest up." if risk else "Low. Keep going!"))
        
        # Graphs
        fig1 = px.line(df, x='date', y='stress_level', title='Stress Over Time')
        st.plotly_chart(fig1)
        
        fig2 = px.scatter(df, x='study_hours', y='stress_level', trendline='ols', title='Stress vs Study Hours')
        st.plotly_chart(fig2)

with tab3:
    st.header("Study Pathway Generator")
    exam = st.selectbox("Exam", ["IOQM", "Other Olympiad"])
    days_left = st.number_input("Days Left to Exam", min_value=1, step=1)
    strong_subjects = st.text_input("Strong Subjects (comma-separated, e.g., Algebra,Geometry)")
    weak_subjects = st.text_input("Weak Subjects (comma-separated, e.g., Number Theory)")
    
    if st.button("Generate Plan"):
        # Rule-based structure with generative AI for tips
        phases = days_left // 4
        plan = f"Personalized {days_left}-Day Plan for {exam}:\n"
        plan += f"Days 1-{phases}: Focus on strong subjects ({strong_subjects}) + Basics & PYQs\n"
        plan += f"Days {phases+1}-{2*phases}: Tackle weak subjects ({weak_subjects}) + Drills\n"
        plan += f"Days {2*phases+1}-{3*phases}: Mixed mocks + Review\n"
        plan += f"Days {3*phases+1}-{days_left}: Full revision + Rest\n"
        
        # Generative AI for health/motivation tips
        prompt = f"Generate 3 short wellness tips for a student studying {days_left} days for Olympiad, focusing on stress and health."
        tips = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        plan += "\nHealth Reminders:\n" + tips
        
        st.text_area("Your Plan", plan, height=300)
        
        # Progress tracker
        if len(df) > 0:
            progress = (datetime.date.today() - pd.to_datetime(df['date']).min()).days / days_left * 100
            st.progress(min(progress / 100, 1.0))
            st.write(f"Progress: {progress:.1f}%")
        else:
            st.write("No data available for progress tracking.")

with tab4:
    st.header("Export Report as PDF")
    if st.button("Generate PDF"):
        pdf_file = "report.pdf"
        c = canvas.Canvas(pdf_file, pagesize=letter)
        c.drawString(100, 750, "AI Study & Wellness Report")
        c.drawString(100, 730, f"Data Entries: {len(df)}")
        
        # Add graph (save fig and draw)
        plt.figure()
        df['date'] = pd.to_datetime(df['date'])  # Ensure date is in datetime format
        df.plot(x='date', y='stress_level')
        plt.savefig("stress.png")
        plt.close()  # Close the plot to avoid display
        c.drawImage("stress.png", 100, 500, width=400, height=200)
        c.save()
        st.download_button("Download PDF", data=open(pdf_file, 'rb'), file_name=pdf_file)
        st.success("PDF ready!")
