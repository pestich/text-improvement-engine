# Text Improvement Engine

The Text Improvement Engine is a command-line tool designed for analyzing and enhancing text content based on a list of standardized phrases. This tool takes input in the form of a TXT file and provides output in CSV format. Its primary purpose is to recommend improvements to align the input text with predefined standards.

## Usage

To run the Text Improvement Engine, follow these steps:

1. Make sure you have Python installed on your system.

2. Install the required libraries, `spacy` and `sentence-transformers`, if not already installed. You can install them using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Clone or download the Text Improvement Engine repository.

4. Navigate to the project directory in your terminal.

5. Execute the following command:

   ```bash
   python text_suggestions.py --source <SAMPLE.TXT_PATH> --target <TARGET_PATH>
   ```

   Replace `<SAMPLE.TXT_PATH>` with the path to your input text file and `<TARGET_PATH>` with the desired path for the output CSV file.

## Project Description

The Text Improvement Engine utilizes natural language processing techniques and machine learning to analyze and enhance text. Here's how it works:

1. **Input Text Analysis**: The program begins by analyzing the input text provided in a TXT file.

2. **N-gram Generation**: The complexity of this task lies in determining the appropriate N value for N-gram generation. The engine generates N-grams ranging from 1 to 4 words for the input text.

3. **Standardized Phrase Comparison**: For each N-gram generated, the engine compares the cosine similarity of the N-gram with a list of standardized phrases.

4. **Recommendation Generation**: Based on the cosine similarity scores, the engine recommends improvements to replace phrases in the input text to align it more closely with the standardized phrases.

5. **Output CSV**: The recommended improvements are saved in a CSV file, where each row contains the original text, the suggested improvement, and other relevant information.

## Performance Optimization

The complexity of this task lies in the fact that we do not know in advance what the value of N should be for constructing n-grams. Sometimes examples from "standardized" phrases can be used to replace phrases ranging from 1 to 4 words. Therefore, the algorithm needs to generate n-grams for values of N from 1 to 4 words and then determine the cosine similarity for each phrase in their list. Due to the potentially large number of n-grams generated and compared against the standardized phrases, text analysis is divided into two stages:

Such an approach may require a significant amount of processing time. Therefore, text analysis is divided into two stages:

1. The project uses the pre-trained spaCy model "en_core_web_lg." This model already has vectors for 514k words (300 dimensions). This allows for quickly determining the semantic similarity between two phrases. However, due to the low dimensionality of the vectors, the accuracy of assessing actual semantic similarity is relatively low, but the processing speed for phrases is high. By default, the similarity threshold is set to 0.5, which helps filter out obvious phrases with low cosine similarity. Thus, in the first stage, using the spaCy model, we obtain a list of candidate phrases for replacement.

2. In the second stage, for each phrase from the candidate pairs, we obtain embeddings using the pre-trained "multilingual-e5-base" model from sentence-transformers. The embedding size is 768, allowing for a more precise determination of the semantic similarity between two phrases. If the cosine similarity value is above 0.9, the phrases are added to the resulting list, and for each phrase in the sentence, a pair of candidates with the highest cosine similarity value is selected.

Certainly, such an approach doesn't change the initial time complexity of the algorithm but significantly reduces the actual processing time.