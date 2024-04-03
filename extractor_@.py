import fitz
import csv
import re
import os
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from unidecode import unidecode
from tqdm import tqdm

# Download NLTK resources if not already downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def extract_key_info(pdf_path, total_vocabulary):
    file_name = os.path.basename(pdf_path)
    facts_text = "Facts section not found"
    doc = fitz.open(pdf_path)

    # Concatenate text content of all pages into a single string
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # Convert text to unigram frequency vector
    unigram_vector = get_unigram_vector(facts_text, total_vocabulary)

    # Calculate TF-IDF scores
    total_vocabulary_tf_idf = set(total_vocabulary.keys())
    tfidf_vectorizer = TfidfVectorizer(vocabulary=total_vocabulary_tf_idf)
    tfidf_vector = tfidf_vectorizer.fit_transform([full_text])
    tfidf_scores = tfidf_vector.toarray()[0]

    key_info = {
        "File Name": file_name,
        "The Facts": facts_text,
        "Unigram Vector": unigram_vector,
        "TF-IDF Scores": tfidf_scores,  # Add TF-IDF scores to the key info dictionary
        "Case Number": re.search(r"Case Number\s*:\s*(.+)", full_text).group(1) if re.search(r"Case Number\s*:\s*(.+)", full_text) else "",
        "Decision Date": re.search(r"Decision Date\s*:\s*(.+)", full_text).group(1) if re.search(r"Decision Date\s*:\s*(.+)", full_text) else "",
        "Tribunal/Court": re.search(r"Tribunal/Court\s*:\s*(.+)", full_text).group(1) if re.search(r"Tribunal/Court\s*:\s*(.+)", full_text) else "",
        "Outcome": "NA"
    }

    # Keywords Lib
    keywords_to_outcomes = {
        "dismissed": "Appeal dismissed.",
        "allowed": "Appeal allowed.",
        "accordingly": "Order accordingly.",
        "granted": "Declaration granted."
    }
    
    # Scanner
    last_page_text = doc[-1].get_text()
    last_page_lines = last_page_text.split('\n')
    for line in reversed(last_page_lines):
        for keyword, outcome in keywords_to_outcomes.items():
            if keyword in line.lower():
                key_info["Outcome"] = outcome
                break
        if key_info["Outcome"] != "Outcome not explicitly mentioned":
            break
    
    doc.close()  # Close the document after extracting all necessary information

    return key_info

def get_unigram_vector(text, total_vocab):
    
    words = preprocess_text(text)

    # Initialize unigram vector with zeros
    unigram_vector = [0] * len(total_vocab)

    # Increment count for each word in the text
    for word in words:
        if word in total_vocab:
            index = total_vocab[word]
            unigram_vector[index] += 1

    return unigram_vector

# This section of code does three things
#   1. Saves legal document content to csv file
#   2. Saves legal document unigram frequency vector to txt file and its pointer to csc file

# def save_to_csv(info, csv_file_path):
#     headers = ["File Name", "Case Number", "Decision Date", "Tribunal/Court", "The Facts", "Unigram Vector", "TF-IDF Scores", "Outcome"]
#     with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
#         writer = csv.DictWriter(file, fieldnames=headers)
#         if file.tell() == 0:
#             writer.writeheader()
#         unigram_vector = info["Unigram Vector"]
#         tfidf_scores = info["TF-IDF Scores"]  # Add TF-IDF scores to the key info dictionary
#         with open("unigram_file.txt", 'a', encoding='utf-8') as unigram_file:
#             ptr = unigram_file.tell()
#             for unigram_freq in unigram_vector:
#                 unigram_file.write(str(unigram_freq))
#             unigram_file.write('\n')
#         info_with_str_vector = info.copy()
#         info_with_str_vector["Unigram Vector"] = str(ptr) # a ptr to the txt file containing the unigram frequency vector for each pdf file     
#         info_with_str_vector["TF-IDF Scores"] = ' '.join(map(str, tfidf_scores))  # Convert TF-IDF scores to a string for CSV
#         writer.writerow(info_with_str_vector)

def save_to_csv(info, csv_file_path):
    headers = ["File Name", "Case Number", "Decision Date", "Tribunal/Court", "The Facts", "Unigram Vector", "TF-IDF Scores", "Outcome"]
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if file.tell() == 0:
            writer.writeheader()
        unigram_vector = ' '.join(map(str, info["Unigram Vector"]))  # Convert unigram vector to a string for CSV
        tfidf_scores = ' '.join(map(str, info["TF-IDF Scores"]))  # Convert TF-IDF scores to a string for CSV
        writer.writerow({
            "File Name": info["File Name"],
            "Case Number": info["Case Number"],
            "Decision Date": info["Decision Date"],
            "Tribunal/Court": info["Tribunal/Court"],
            "The Facts": info["The Facts"],
            "Unigram Vector": unigram_vector,  # Store unigram vector as string in CSV
            "TF-IDF Scores": tfidf_scores,
            "Outcome": info["Outcome"]
        })

def batch_process_pdf_folder(folder_path, csv_file_path, total_vocabulary):
    pdf_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".pdf")]
    
    # Create tqdm instance with total number of PDF files
    with tqdm(total=len(pdf_files), desc="Processing PDF files") as pbar:
        for file in pdf_files:
            pdf_path = os.path.join(folder_path, file)
            curr_info = extract_key_info(pdf_path, total_vocabulary)
            save_to_csv(curr_info, csv_file_path)
            pbar.update(1)  # Update progress bar

    print(f"All PDF files in {folder_path} have been processed. Key information is saved to {csv_file_path}.")

def create_and_save_total_vocab(folder_path):
    total_vocab = {}
    pdf_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".pdf")]

    # Create tqdm instance with total number of PDF files
    with tqdm(total=len(pdf_files), desc="Processing PDF files") as pbar:
        for file in pdf_files:
            pdf_path = os.path.join(folder_path, file)
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()

            words = preprocess_text(full_text)
            for word in words:
                total_vocab[word] = total_vocab.get(word, 0) + 1
            doc.close()

            pbar.update(1)  # Update progress bar

    sorted_vocab = dict(sorted(total_vocab.items()))

    with open("vocabulary.txt", 'w', encoding='utf-8') as file:
        for word, freq in sorted_vocab.items():
            file.write(f"{word} {freq}\n")

    return total_vocab

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenization
    words = word_tokenize(text.lower())
    
    # Part-of-Speech tagging
    tagged_words = pos_tag(words)
    
    # Lemmatization and remove stop words
    words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) if get_wordnet_pos(tag) else word
             for word, tag in tagged_words
             if word not in stop_words and word.isalnum()]

    return words

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return None

if __name__ == "__main__":
    # Hardcoded folder path and CSV file path
    folder_path = "data/test"  # Adjust this path as per your folder structure
    csv_file_path = "extracted_info.csv"  # Adjust this path as per your requirements
    
    # Create total vocabulary
    total_vocabulary = create_and_save_total_vocab(folder_path)
    batch_process_pdf_folder(folder_path, csv_file_path, total_vocabulary)
