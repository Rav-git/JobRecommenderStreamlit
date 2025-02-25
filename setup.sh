#!/bin/bash

# Make script executable
chmod +x setup.sh

# Download NLTK data
python -m nltk.downloader stopwords

# Download spaCy model
python -m spacy download en_core_web_sm

# Verify spaCy model installation
python -c "import spacy; print(f'spaCy version: {spacy.__version__}'); print(f'Model path: {spacy.util.get_data_path()}'); print('Available models:'); print([p.name for p in spacy.util.get_data_path().glob('*')])" 