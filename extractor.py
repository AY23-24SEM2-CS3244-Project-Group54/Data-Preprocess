import fitz  # PyMuPDF
from tqdm import tqdm
import csv
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
import nltk.corpus
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')



def extract_key_info(pdf_path):
    doc = fitz.open(pdf_path)
    first_page_text = doc[0].get_text("text")
    last_page_text = doc[-1].get_text("text")
    full_text = "".join([page.get_text() for page in doc])

    print(first_page_text)
    # Keywords and corresponding outcomes
    keywords_to_outcomes = {
        "dismissed": "Appeal dismissed.",
        "allowed": "Appeal allowed.",
        "accordingly": "Order accordingly.",
        "granted": "Declaration granted."
    }

    # Regex patterns
    decision_date_pattern = re.compile(r"(Decision Date)\s*:\s*(.+)")
    coram_pattern = re.compile(r"(Coram)\s*:\s*(.+)")
    facts_section_pattern = re.compile(r"The facts\s*(.*?)(?=(Version No|\Z))", re.DOTALL)
    capitalized_words_pattern = re.compile(r"([A-Z][\w\s]+)(?=Act \()")
    unique_sequences = set()

    # Extract basic info
    key_info = {
        "File Name": os.path.basename(pdf_path),
        "Decision Date": "",
        "Coram": "",
        "Tribunal/Court": "",
        "Area of Law": "",
        "Outcome": "Outcome not explicitly mentioned",
        "The Facts": "Facts section not found",
        "Unigram Vector": [],
        "word2vec": []
    }

    # Extract Tribunal/Court
    tribunal_court = pdf_path.split('_')[2]
    if tribunal_court:
        if 'SGHC':
            key_info["Tribunal/Court"] = 'High Court'
        elif 'SGCA':
            key_info["Tribunal/Court"] = 'Court of Appeal'

    # Extract Coram
    for match in coram_pattern.finditer(first_page_text):
        key_info[match.group(1)] = match.group(2)
            
    # Match case information
    for match in decision_date_pattern.finditer(first_page_text):
        key_info[match.group(1)] = match.group(2)

    # Extract facts section
    facts_section = facts_section_pattern.search(full_text)
    if facts_section:
        key_info["The Facts"] = facts_section.group(1).strip()

    # Unigram Vector
    key_info["Unigram Vector"] = get_unigram_vector(key_info["The Facts"])

    # Word2Vec
    key_info["word2vec"] = word2vec_converter(key_info["The Facts"])

    # Determine the outcome
    last_page_lines = last_page_text.split('\n')
    for line in reversed(last_page_lines):
        for keyword, outcome in keywords_to_outcomes.items():
            if keyword in line.lower():
                key_info["Outcome"] = outcome
                break
        if key_info["Outcome"] != "Outcome not explicitly mentioned":
            break

    # Extract capitalized words before "Act ("
    for page in doc:
        text = page.get_text()
        matches = re.finditer(capitalized_words_pattern, text)
        for match in matches:
            phrase = match.group(1)
            capitalized_words_sequence = ' '.join(word for word in phrase.split() if word[0].isupper())
            unique_sequences.add(capitalized_words_sequence)

    key_info["Capitalized Words Before 'Act ('"] = '; '.join(unique_sequences)

    doc.close()
    return key_info

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def data_preprocess(text):
    """
    Helper function for data preprocessing
    1. Convert text to lower case
    2. Tokenize text
    3. Removal of stop words
    4. Lemmatize words
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # replace punctuation with none

    tokenized_text = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokenized_text = [word for word in tokenized_text if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text]

def get_vect_avg(vect):
    vect = np.array(vect)
    result = []
    for i in range(vect.shape[1]):
        sum_of_first_elements = vect[:, i].sum()

        # Calculate the number of arrays
        num_arrays = vect.shape[0]

        # Calculate the average
        result.append(sum_of_first_elements / num_arrays)
    return result


def get_unigram_vector(text):
    # Initialize CountVectorizer to convert text into unigram frequency vector
    vectorizer = CountVectorizer(stop_words='english', lowercase=True)

    # Check if text is not empty and not composed solely of stop words
    if text.strip() == '' or all(word in vectorizer.get_stop_words() for word in text.split()):
        # Return an empty list or some placeholder to indicate no valid words were found
        return []
    else:
        try:
            unigram_vector = vectorizer.fit_transform([text]).toarray()[0]
            return unigram_vector.tolist()  # Convert numpy array to list for compatibility
        except ValueError:
            # Handle the case where no valid words are found after preprocessing
            return []

def word2vec_converter(text):
    cleaned_text = data_preprocess(text)
    model = Word2Vec(sentences=[cleaned_text], vector_size=100, window=1, min_count=1, workers=4)

    model.train([cleaned_text],
                total_examples=model.corpus_count,
                epochs=10)  # Number of iterations (epochs) over the corpus
    word_vectors = model.wv

    result = []
    # Get the word vector for a specific word
    for word in cleaned_text:
        word_vector = word_vectors[word].tolist()
        result.append(word_vector)
    
    final_result = get_vect_avg(result)
    return final_result

def save_to_csv(all_info, csv_file_path):
    headers = ["File Name", "Case Number", "Decision Date", "Coram", "Tribunal/Court", "Area of Law","Outcome", "The Facts", "Unigram Vector",
               "word2vec", "Capitalized Words Before 'Act ('"]
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for info in all_info:
            writer.writerow(info)


def batch_process_pdf_folder(folder_path, csv_file_path):
    all_info = []
    pdf_files = [file for file in os.listdir(folder_path)[0:100:5] if file.lower().endswith(".pdf")]
    # Use tqdm to create a progress bar for the loop
    for file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(folder_path, file)
        key_info =  extract_key_info(pdf_path)
        all_info.append(key_info)

    save_to_csv(all_info, csv_file_path)
    print(
        f"All PDF files in {folder_path} have been processed. Key information is saved to {csv_file_path}. Total files processed: {len(all_info)}.")


# Adjust these paths as per your requirements
folder_path = 'data/raw/'
csv_file_path = 'final_data.csv'

batch_process_pdf_folder(folder_path, csv_file_path)
