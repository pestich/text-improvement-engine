from getsuggestion import GetSuggestion
import argparse
import pandas as pd

# Define constants for configuration.
MAX_NGRAM_SIZE = 4
SPACY_THRESHOLD = 0.5
SBERT_THRESHOLD = 0.9

# Create an ArgumentParser to handle command-line arguments.
parser = argparse.ArgumentParser(description='Text Improvement Engine')
parser.add_argument('--source', type=str, help='Input dir for source df')
parser.add_argument('--target', type=str, help='Output dir for result df', default='target.csv')
args = parser.parse_args()

# Create an instance of the 'getSuggestion' class with specified configuration.
suggestion = GetSuggestion(MAX_NGRAM_SIZE, SPACY_THRESHOLD, SBERT_THRESHOLD)

# Load a CSV file containing standardized phrases for analysis.
standardised_phrases = pd.read_csv('./Standardised terms.csv')

# Preprocess the standardized phrases using the 'getSuggestion' class.
standardised_phrases = suggestion.preprocessing_phrases(standardised_phrases)

# Open and read the source text file specified in the command-line arguments.
with open(args.source, 'r') as f:
    text = f.read()

# Preprocess the source text using the 'getSuggestion' class.
text = suggestion.preprocessing_text(text)

# Generate recommendations for text improvement based on the source text and standardized phrases.
result = suggestion.get_phases_to_replace(text, standardised_phrases)

# Save the result (suggestions) to a CSV file specified in the command-line arguments.
result.to_csv(args.target)


