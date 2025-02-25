#!/bin/bash

# Download NLTK data
python -m nltk.downloader stopwords

# Download spaCy model
python -m spacy download en_core_web_sm 