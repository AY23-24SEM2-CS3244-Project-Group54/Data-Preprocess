import fitz  # PyMuPDF
from tqdm import tqdm 
import csv
import re
import os
from sklearn.feature_extraction.text import CountVectorizer

def extract_key_info(pdf_path):
    doc = fitz.open(pdf_path)
    first_page_text = doc[0].get_text("text")
    last_page_text = doc[-1].get_text("text")
    full_text = "".join([page.get_text() for page in doc])
    
    # Keywords and corresponding outcomes
    keywords_to_outcomes = {
        "dismissed": "Appeal dismissed.",
        "allowed": "Appeal allowed.",
        "accordingly": "Order accordingly.",
        "granted": "Declaration granted."
    }

    # Regex patterns
    case_info_pattern = re.compile(r"(Case Number|Decision Date|Tribunal/Court)\s*:\s*(.+)")
    facts_section_pattern = re.compile(r"The facts\s*(.*?)(?=(Version No|\Z))", re.DOTALL)
    capitalized_words_pattern = re.compile(r"([A-Z][\w\s]+)(?=Act \()")
    unique_sequences = set()

    # Extract basic info
    key_info = {
        "File Name": os.path.basename(pdf_path),
        "Case Number": "",
        "Decision Date": "",
        "Tribunal/Court": "",
        "Outcome": "Outcome not explicitly mentioned",
        "The Facts": "Facts section not found",
        "Unigram Vector": []
    }

    # Match case information
    for match in case_info_pattern.finditer(first_page_text):
        key_info[match.group(1)] = match.group(2)
    
    # Extract facts section
    facts_section = facts_section_pattern.search(full_text)
    if facts_section:
        key_info["The Facts"] = facts_section.group(1).strip()

    # Unigram Vector
    key_info["Unigram Vector"] = get_unigram_vector(key_info["The Facts"])

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


def save_to_csv(all_info, csv_file_path):
    headers = ["File Name", "Case Number", "Decision Date", "Tribunal/Court", "Outcome", "The Facts", "Unigram Vector", "Capitalized Words Before 'Act ('"]
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for info in all_info:
            writer.writerow(info)

def batch_process_pdf_folder(folder_path, csv_file_path):
    all_info = []
    pdf_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".pdf")]
    # Use tqdm to create a progress bar for the loop
    for file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(folder_path, file)
        key_info = extract_key_info(pdf_path)
        all_info.append(key_info)

    save_to_csv(all_info, csv_file_path)
    print(f"All PDF files in {folder_path} have been processed. Key information is saved to {csv_file_path}. Total files processed: {len(all_info)}.")

# Adjust these paths as per your requirements
folder_path = 'i:/CS3244/test_data/'
csv_file_path = 'i:/CS3244/final_data.csv'

batch_process_pdf_folder(folder_path, csv_file_path)
