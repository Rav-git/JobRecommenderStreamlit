# Smart Job Recommendation System 🎯

A modern job recommendation system built with Streamlit that matches your resume with relevant job opportunities using AI-powered skill extraction and intelligent matching algorithms.

## Features ✨

- PDF Resume Upload & Analysis
- Automatic Skill Extraction
- Smart Job Matching Algorithm
- Modern UI with Glassmorphism Effects
- Real-time Job Recommendations
- Direct Application Links

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/job-recommendation-system.git
cd job-recommendation-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage 💡

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`
3. Upload your resume (PDF format)
4. Get instant job recommendations based on your skills!

## Data Files 📁

- `skills.csv`: Database of professional skills for matching
- `linkdindata.csv`: Job listings database

## Technologies Used 🛠️

- Python 3.10+
- Streamlit
- spaCy
- scikit-learn
- NLTK
- PyPDF2
- pandas
- NumPy

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 