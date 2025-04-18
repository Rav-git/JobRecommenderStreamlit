import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

import pandas as pd
import PyPDF2
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from ftfy import fix_text
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.matcher import Matcher
import os
import csv
from io import BytesIO
import numpy as np

# Custom CSS with proper escaping and formatting
st.markdown("""
    <style>
    /* Global Styles */
    div[data-testid="stToolbar"] {
        display: none;
    }
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stApp {
        background: linear-gradient(135deg, #f0f4fd 0%, #e6eefa 100%);
        background-attachment: fixed;
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Glassmorphism Effects */
    .glass-effect {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Upload Area Styling */
    .upload-area {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 2px dashed rgba(79, 70, 229, 0.3);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 3rem;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(79, 70, 229, 0.1);
        animation: fadeIn 0.8s ease-out forwards;
    }
    .upload-area:hover {
        transform: translateY(-5px);
        border-color: #4f46e5;
        box-shadow: 0 12px 40px rgba(79, 70, 229, 0.15);
    }
    .upload-area::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.4),
            transparent
        );
        transition: 0.6s;
    }
    .upload-area:hover::before {
        left: 100%;
    }
    
    /* Job Card Styling */
    .job-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 32px;
        margin: 28px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(79, 70, 229, 0.1);
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.6s ease-out forwards;
        animation-delay: calc(var(--animation-order) * 0.1s);
    }
    .job-card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 15px 45px rgba(79, 70, 229, 0.15);
        border-color: rgba(79, 70, 229, 0.2);
    }
    .job-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(135deg, #4f46e5, #818cf8);
        border-radius: 0 24px 24px 0;
        opacity: 0;
        transition: 0.4s;
    }
    .job-card:hover::after {
        opacity: 1;
    }
    
    /* Job Header */
    .job-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }
    .job-title {
        color: #1e293b !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        line-height: 1.4 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Match Score */
    .match-score {
        background: #ffffff !important;
        color: #1e293b !important;
        padding: 10px 20px !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        font-size: 15px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2) !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(79, 70, 229, 0.2) !important;
    }
    .match-score:hover {
        background: linear-gradient(135deg, #4f46e5, #818cf8) !important;
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3);
    }
    .match-score span {
        color: #4f46e5 !important;
    }
    .match-score:hover span {
        color: #ffffff !important;
    }
    
    /* Location */
    .job-location {
        color: #475569 !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        margin: 16px 0 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    .job-location span {
        color: #475569 !important;
    }
    .location-icon {
        color: #4f46e5 !important;
    }
    
    /* Apply Button */
    .apply-button {
        background: #ffffff !important;
        color: #4f46e5 !important;
        padding: 14px 32px !important;
        border-radius: 30px !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 15px !important;
        border: 1px solid rgba(79, 70, 229, 0.2) !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.1) !important;
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    .apply-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 0%;
        height: 100%;
        background: linear-gradient(45deg, #4f46e5, #818cf8);
        transition: all 0.4s ease;
        z-index: -1;
        border-radius: 30px;
    }
    .apply-button:hover {
        color: #ffffff !important;
        border-color: transparent !important;
    }
    .apply-button:hover::before {
        width: 100%;
    }
    .apply-button span {
        color: #4f46e5 !important;
        transition: all 0.3s ease;
    }
    .apply-button:hover span {
        color: #ffffff !important;
    }
    
    /* Messages */
    .success-message {
        color: #059669;
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: 500;
        border: 1px solid rgba(5, 150, 105, 0.2);
        box-shadow: 0 8px 32px rgba(5, 150, 105, 0.1);
        animation: fadeIn 0.6s ease-out forwards;
    }
    .error-message {
        color: #dc2626;
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: 500;
        border: 1px solid rgba(220, 38, 38, 0.2);
        box-shadow: 0 8px 32px rgba(220, 38, 38, 0.1);
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    /* Skills Display */
    .skills-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 32px;
        margin: 28px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(79, 70, 229, 0.1);
        animation: fadeIn 0.6s ease-out forwards;
    }
    .skills-container h3 {
        color: #1e293b;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .skills-container h3 span {
        color: #4f46e5;
    }
    .skills-wrapper {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        padding: 8px 0;
    }
    .skill-tag {
        display: inline-flex;
        align-items: center;
        background: #ffffff;
        color: #4f46e5;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid rgba(79, 70, 229, 0.2);
    }
    .skill-tag:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #4f46e5, #818cf8);
        color: #ffffff;
        border-color: transparent;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    
    /* Header and Text Styles */
    h1.gradient-text {
        color: #4f46e5;
        font-size: 4.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        line-height: 1.2;
        letter-spacing: -1px;
        text-shadow: 0 2px 8px rgba(79, 70, 229, 0.15);
        animation: fadeIn 0.8s ease-out forwards;
    }
    .subtitle {
        font-size: 1.4rem;
        color: #475569;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
        animation: fadeIn 0.8s ease-out forwards;
        animation-delay: 0.2s;
    }
    
    /* Footer */
    .footer {
        color: #475569;
        text-align: center;
        padding: 4rem 0;
        margin-top: 4rem;
        font-size: 1.1rem;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-top: 1px solid rgba(79, 70, 229, 0.1);
    }
    .footer span {
        color: #ef4444;
        font-size: 1.2rem;
    }
    
    /* Upload Area */
    .upload-area h3 {
        color: #4f46e5;
        margin-bottom: 1.2rem;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .upload-area p {
        color: #475569;
        font-size: 1.1rem;
    }
    
    /* Loading Animation */
    .loading-animation {
        width: 100%;
        height: 4px;
        background: linear-gradient(to right, #4f46e5, #818cf8, #4f46e5);
        background-size: 200% 100%;
        animation: shimmer 2s infinite linear;
        border-radius: 2px;
        margin: 1rem 0;
    }
    
    /* Streamlit Component Overrides */
    div[data-testid="stFileUploader"] {
        width: 100%;
    }
    div[data-testid="stFileUploader"] > div {
        width: 100%;
    }
    div[data-testid="stFileUploader"] > div > button {
        background-color: #4f46e5 !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFileUploader"] > div > button:hover {
        background-color: #4338ca !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize matcher
matcher = Matcher(nlp.vocab)

# Read skills from CSV file and create patterns
def initialize_skills_matcher():
    try:
        with open('skills.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            skills = [row for row in csv_reader]
            
        # Create pattern dictionaries from skills
        # Split skills by comma and create patterns for each individual skill
        all_skills = []
        for skill_group in skills[0]:
            individual_skills = [s.strip().lower() for s in skill_group.split(',')]
            all_skills.extend(individual_skills)
            
        # Remove duplicates and empty strings
        all_skills = list(set([s for s in all_skills if s]))
        
        # Create patterns for each skill
        skill_patterns = []
        for skill in all_skills:
            # Handle multi-word skills
            if ' ' in skill:
                words = skill.split()
                pattern = [{'LOWER': word.lower()} for word in words]
            else:
                pattern = [{'LOWER': skill.lower()}]
            skill_patterns.append(pattern)
        
        # Add patterns to matcher
        matcher.add('Skills', skill_patterns)
        
    except FileNotFoundError:
        st.error("Skills database not found. Please ensure skills.csv is present.")
        return

def handle_file_upload(uploaded_file):
    if uploaded_file is None:
        return None, "No file uploaded"
    
    try:
        # Check file type
        if not uploaded_file.name.lower().endswith('.pdf'):
            return None, "Please upload a PDF file"
        
        # Read file content
        file_content = uploaded_file.read()
        
        # Create BytesIO object for PyPDF2
        pdf_file = BytesIO(file_content)
        
        # Extract text using PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        return text, None
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def extract_skills(text):
    doc = nlp(text.lower())
    matches = matcher(doc)
    skills = set()
    
    # Extract matched skills and normalize them
    for match_id, start, end in matches:
        skill = doc[start:end].text.lower()
        # Normalize skill name (remove extra spaces, etc.)
        skill = ' '.join(skill.split())
        skills.add(skill)
    
    return sorted(list(skills))

def ngrams(string, n=3):
    string = fix_text(string)
    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()
    string = re.sub(' +', ' ', string).strip()
    string = ' ' + string + ' '
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def create_job_card(row, matching_skills, index):
    """Helper function to create a job card with proper HTML structure"""
    # Format match score and escape special characters
    match_score = max(min(row['match'] * 100, 100), 0)  # Ensure score is between 0-100
    job_title = row['j_tittle'].replace('<', '&lt;').replace('>', '&gt;').strip()
    location = row['company_locations'].replace('<', '&lt;').replace('>', '&gt;').strip()
    apply_link = row['Apply_link'].replace('"', '&quot;').strip()
    
    # Create simplified job card with modern design and proper HTML structure
    card = f"""
    <div class="job-card" style="--animation-order: {index};">
        <div class="job-header">
            <h3 class="job-title" style="color: #1e293b !important;">{job_title}</h3>
            <div class="match-score" style="color: #1e293b !important; background: #ffffff !important;">
                <span style="color: #4f46e5 !important;">🎯</span> {match_score:.1f}%
            </div>
        </div>
        
        <div class="job-location" style="color: #475569 !important;">
            <span class="location-icon" style="color: #4f46e5 !important;">📍</span>
            <span style="color: #475569 !important;">{location}</span>
        </div>
        
        <div style="display: flex; justify-content: flex-end; margin-top: 24px;">
            <a href="{apply_link}" target="_blank" rel="noopener noreferrer" class="apply-button" style="color: #4f46e5 !important; background: #ffffff !important;">
                Apply Now <span style="color: #4f46e5 !important;">🚀</span>
            </a>
        </div>
    </div>
    """
    
    # Clean up whitespace while preserving HTML structure
    card = re.sub(r'\s+', ' ', card.strip())
    card = re.sub(r'>\s+<', '><', card)
    
    return card

def display_skills(skills):
    """Helper function to display skills with proper HTML structure"""
    if not skills:
        return ""
        
    skills_html = """
    <div class="skills-container">
        <h3>
            <span>📋</span>
            Your Professional Skills
        </h3>
        <div class="skills-wrapper">
    """
    
    for skill in skills:
        # Escape any HTML special characters in the skill name
        skill_escaped = skill.replace('<', '&lt;').replace('>', '&gt;').strip()
        skills_html += f'<span class="skill-tag">{skill_escaped}</span>'
    
    skills_html += """
        </div>
    </div>
    """
    return skills_html

def process_resume_and_recommend(uploaded_file, job_data):
    # Extract text from PDF using the new handler
    resume_text, error = handle_file_upload(uploaded_file)
    if error:
        raise Exception(error)
    
    # Extract skills with improved matching
    skills = extract_skills(resume_text)
    if not skills:
        raise Exception("No skills were found in the resume. Please ensure your resume lists your skills clearly.")
    
    # Prepare skills for matching
    skills_text = ' '.join(skills)
    
    # Prepare job descriptions
    job_descriptions = job_data['job_description'].values.astype('U')
    
    try:
        # Calculate skill overlap scores
        job_data['skill_overlap'] = job_data['job_description'].apply(
            lambda x: len(set(extract_skills(x)).intersection(set(skills))) / max(len(skills), 1)
        )
        
        # Calculate text similarity scores
        vectorizer = TfidfVectorizer(
            analyzer='word',
            min_df=1,
            max_df=0.95,
            stop_words='english',
            token_pattern=r'(?u)\b\w\w+\b'
        )
        
        # Create document corpus and calculate similarities
        all_docs = [resume_text] + list(job_descriptions)
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        resume_vector = tfidf_matrix[0]
        job_vectors = tfidf_matrix[1:]
        
        # Calculate cosine similarities
        similarities = []
        for job_vector in job_vectors:
            similarity = resume_vector.dot(job_vector.T).toarray()[0][0]
            similarities.append(similarity)
        
        # Normalize similarities
        similarities = np.array(similarities)
        if similarities.max() != 0:
            similarities = similarities / similarities.max()
        
        # Add similarity scores
        job_data['text_similarity'] = similarities
        
        # Calculate final match score with adjusted weights
        job_data['match'] = (
            job_data['skill_overlap'] * 0.7 +      # Higher weight for direct skill matches
            job_data['text_similarity'] * 0.3      # Lower weight for text similarity
        )
        
        # Sort and return top matches
        recommendations = job_data.sort_values('match', ascending=False).head(10)
        
        return recommendations, skills
        
    except Exception as e:
        raise Exception(f"Error during job matching: {str(e)}")

def main():
    # Initialize skills matcher at startup
    initialize_skills_matcher()
    
    # Hide default Streamlit elements
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    # Header with modern design
    st.markdown("""
        <div style="text-align: center; padding: 6rem 0 4rem;">
            <h1 style="color: #4f46e5; font-size: 4.5rem; font-weight: 800; margin: 0 auto 2rem; line-height: 1.2; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                🎯 Smart Job Recommendation System
            </h1>
            <p style="text-align: center; color: #475569; font-size: 1.4rem; max-width: 800px; margin: 1.5rem auto; line-height: 1.6; font-weight: 400;">
                Upload your resume and let our AI match you with the perfect job opportunities
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="upload-area">
                <h3 style="color: #4f46e5; margin-bottom: 1.2rem; font-size: 1.8rem; font-weight: 700;">
                    📤 Upload Your Resume
                </h3>
                <p style="color: #475569; font-size: 1.1rem;">
                    Drag and drop your PDF resume here to find your perfect match
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Resume", type=['pdf'], key="resume_uploader", label_visibility="collapsed")
        
        if uploaded_file:
            # Show loading animation
            st.markdown('<div class="loading-animation"></div>', unsafe_allow_html=True)
            
            with st.spinner("🔍 Analyzing your resume..."):
                try:
                    # Load and process data
                    job_data = pd.read_csv('linkdindata.csv')
                    recommendations, skills = process_resume_and_recommend(uploaded_file, job_data)
                    
                    # Display skills
                    st.markdown(display_skills(skills), unsafe_allow_html=True)
                    
                    # Display success message
                    st.markdown("""
                        <div class="success-message">
                            ✨ Analysis complete! Here are your personalized job matches
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display job recommendations
                    st.markdown("""
                        <h2 style="color: #1e293b; margin: 3rem 0 2rem; font-size: 2.2rem; font-weight: 700; text-align: center;">
                            🎯 Top Job Recommendations
                        </h2>
                    """, unsafe_allow_html=True)
                    
                    # Display each job card with animation delay
                    for idx, row in recommendations.iterrows():
                        job_skills = extract_skills(row['job_description'])
                        matching_skills = set(skills).intersection(set(job_skills))
                        st.markdown(create_job_card(row, matching_skills, idx), unsafe_allow_html=True)
                        
                except Exception as e:
                    st.markdown(f"""
                        <div class="error-message">
                            ❌ An error occurred: {str(e)}
                        </div>
                    """, unsafe_allow_html=True)
    
    # Footer with modern design
    st.markdown("""
        <div class="footer">
            <p>Built with <span>❤️</span> using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 