import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
from tqdm import tqdm
from typing import List, Union


class GetSuggestion:
    def __init__(self, max_ngram_size, spacy_threshold, sbert_threshold):
        # Load the spaCy language model for English.
        self.nlp = spacy.load('en_core_web_lg')
        # Load a Sentence Transformer model for embeddings.
        self.model = SentenceTransformer('intfloat/multilingual-e5-base')
        # Set the maximum N-gram size for analysis.
        self.max_ngram_size = max_ngram_size
        # Set the spaCy similarity threshold.
        self.spacy_threshold = spacy_threshold
        # Set the Sentence Transformers similarity threshold.
        self.sbert_threshold = sbert_threshold

    def preprocessing_text(self, text: str) -> spacy.tokens.doc.Doc:
        """
        Input: text (raw text)
        Output: Processed spaCy Doc object

        This method preprocesses input text using spaCy and returns a processed Doc object for further analysis.
        """
        doc = self.nlp(text)
        return doc

    def preprocessing_phrases(self, data: pd.DataFrame) -> List[spacy.tokens.doc.Doc]:
        """
        Input: data (a DataFrame containing phrases)
        Output: List of processed spaCy Doc objects

        This method preprocesses a list of phrases contained in a DataFrame by converting them to lowercase
        and processing them using the spaCy pipeline.
        It returns a list of processed Doc objects.
        """
        data = data.iloc[:, 0].str.lower().tolist()
        data_doc = list(self.nlp.pipe(data))
        return data_doc

    def get_candidates(self, sent: spacy.tokens.doc.Doc, phrase: spacy.tokens.doc.Doc) -> List:
        """
        Input: sent (a sentence as spaCy Doc object), phrase (a standardized phrase as spaCy Doc object)
        Output: List of candidate replacements

        This method generates candidate replacements for a phrase within a sentence based on spaCy similarity.
        It returns a list of candidate replacements.
        """
        cand_list = []
        for n_gram in range(1, self.max_ngram_size):
            for idx in range(len(sent) - n_gram):
                span = sent[idx: idx + n_gram]
                # Check if the similarity between the phrase and N-gram span exceeds the spaCy threshold.
                if phrase.similarity(span) > self.spacy_threshold:
                    cand_list.append([phrase.text, span.text])
        return cand_list

    def check_candidates(self, cand_list: List[str]) -> List[Union[float, str]]:
        """
        Input: List of candidate replacements
        Output: Best matching replacement

        This method checks candidate replacements for similarity using Sentence Transformers.
        It returns the best matching replacement based on a similarity threshold.
        """
        result = []
        for i_pair in cand_list:
            embeddings1 = self.model.encode(i_pair[0])
            embeddings2 = self.model.encode(i_pair[1])
            cosine_score = util.cos_sim(embeddings1, embeddings2)[0][0]
            # If the cosine similarity score exceeds the Sentence Transformers threshold, add it to the result.
            if cosine_score > self.sbert_threshold:
                result.append([float(cosine_score), i_pair[0], i_pair[1]])
        if len(result) > 0:
            # Sort the results by similarity score in descending order.
            result.sort(reverse=True)
            return result[0]

    def get_phases_to_replace(self, doc: spacy.tokens.doc.Doc, phrases: List[spacy.tokens.doc.Doc]) -> pd.DataFrame:
        """
        Input: doc (processed text), phrases (list of standardized phrases)
        Output: DataFrame of recommendations

        This method analyzes the input document and standardized phrases, generates recommendations for replacements,
        and returns the results in a Pandas DataFrame.
        """
        result_list = list()
        for sent in doc.sents:
            for i_phrase in tqdm(phrases):
                temp = dict()
                candidate_list = self.get_candidates(sent, i_phrase)
                result = self.check_candidates(candidate_list)
                if result:
                    temp['sentence'] = sent.text
                    temp['phrase_to_replace'] = result[2]
                    temp['suggestion'] = result[1]
                    temp['cosine_score'] = result[0]
                    result_list.append(temp)
        # Convert the results to a Pandas DataFrame.
        result = pd.DataFrame.from_records(result_list)
        return result
