import fitz
import csv
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize


def extract_key_info(pdf_path, total_vocabulary):
    file_name = os.path.basename(pdf_path)
    facts_text = "Facts section not found"
    doc = fitz.open(pdf_path)

    # Concatenate text content of all pages into a single string
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # Search for "The facts" section in the concatenated text
    facts_section = re.search(r"The facts\s*(.*?)(?=(Version No|\Z))", full_text, re.DOTALL)
    if facts_section:
        facts_text = facts_section.group(1).strip()
        # Check for bold text in the facts section
        bold_text_found = False
        for line in facts_text.split('\n'):
            if "<b>" in line or "<B>" in line:  # Check for bold HTML tags
                bold_text_found = True
                break
        if bold_text_found:
            facts_text = re.sub(r"(<b>|<B>).*?(<\/b>|<\/B>)", "", facts_text)  # Remove bold HTML tags

    # Convert text to unigram frequency vector
    unigram_vector = get_unigram_vector(facts_text, total_vocabulary)
    # print("Debugging the output unigram array\n")
    # print(unigram_vector)
    # print("Debugging the output unigram array element by element\n")
    # for num in unigram_vector:
    #     print(num)
    # print('\n')
    # print("This is one unigram array")
    
    key_info = {
        "File Name": file_name,
        "The Facts": facts_text,
        "Unigram Vector": unigram_vector,
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
    # print(key_info)
    return key_info

def get_unigram_vector(text, total_vocab):
    # Initialize CountVectorizer with total_vocab as vocabulary
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, vocabulary=total_vocab)
    unigram_vector = vectorizer.fit_transform([text]).toarray()[0]
    # print(text)
    # print("Debugging the output unigram array\n")
    # for num in unigram_vector:
    #     print(num)
    # print('\n')
    # print("This is one unigram array")
    # print(unigram_vector)
    return unigram_vector

# def save_to_csv(info, csv_file_path):
#     headers = ["File Name", "Case Number", "Decision Date", "Tribunal/Court", "The Facts", "Unigram Vector", "Outcome"]
    
#     with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
#         writer = csv.DictWriter(file, fieldnames=headers)
#         print("Debugging the output unigram array\n")
#         for num in info["Unigram Vector"]:
#             print(num)
#         print('\n')
#         print("This is one unigram array")
#         writer.writerow(info)

# This section of code does three things
#   1. Saves legal document content to csv file
#   2. Saves legal document unigram frequency vector to txt file and its pointer to csc file
#   3. Saves 
def save_to_csv(info, csv_file_path):
    headers = ["File Name", "Case Number", "Decision Date", "Tribunal/Court", "The Facts", "Unigram Vector", "Outcome"]
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if file.tell() == 0:
            writer.writeheader()
        unigram_vector = info["Unigram Vector"]
        with open("unigram_file.txt", 'a', encoding='utf-8') as unigram_file:
            ptr = unigram_file.tell()
            for unigram_freq in unigram_vector:
                unigram_file.write(str(unigram_freq))
            unigram_file.write('\n')
        # print("Debugging the output unigram array element by element\n")
        # for num in unigram_vector_str:
        #     print(num)
        # print('\n')
        # print("This is one unigram array")
        info_with_str_vector = info.copy()
        info_with_str_vector["Unigram Vector"] = str(ptr) # a ptr to the txt file containing the unigram frequency vector for each pdf file     
        writer.writerow(info_with_str_vector)

def batch_process_pdf_folder(folder_path, csv_file_path, total_vocabulary):
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            curr_info = extract_key_info(pdf_path, total_vocabulary)
            # print("Debugging the output unigram array\n")
            # for num in curr_info["Unigram Vector"]:
            #     print(num)
            # print('\n')
            # print("This is one unigram array")
            save_to_csv(curr_info, csv_file_path)

    print(f"All PDF files in {folder_path} have been processed. Key information is saved to {csv_file_path}.")

# Create a dictionary of vocabulary and its corresponding frequency in across all legal documents
# def create_and_save_total_vocab(folder_path):
#     total_vocab = {}
#     for file in os.listdir(folder_path):
#         if file.lower().endswith(".pdf"):
#             pdf_path = os.path.join(folder_path, file)
#             doc = fitz.open(pdf_path)
#             full_text = ""
#             for page in doc:
#                 full_text += page.get_text()
#             words = word_tokenize(full_text.lower())
#             for word in words:
#                 total_vocab[word] = total_vocab.get(word, 0) + 1
#             doc.close()
        
#         with open("vocabulary.txt", 'a', encoding='utf-8') as file:
#             for word in total_vocab:
#                 file.write(str(word))
#                 file.write(' ')
#                 file.write(str(total_vocab[word]))
#                 file.write('\n')
#     return total_vocab
    
# def create_and_save_total_vocab(folder_path):
#     total_vocab = {}
#     for file in os.listdir(folder_path):
#         if file.lower().endswith(".pdf"):
#             pdf_path = os.path.join(folder_path, file)
#             doc = fitz.open(pdf_path)
#             full_text = ""
#             for page in doc:
#                 full_text += page.get_text()
#             words = word_tokenize(full_text.lower())
#             for word in words:
#                 total_vocab[word] = total_vocab.get(word, 0) + 1
#             doc.close()
        
#     with open("vocabulary.txt", 'w', encoding='utf-8') as file:
#         for word, freq in total_vocab.items():
#             file.write(f"{word} {freq}\n")

#     return total_vocab

def create_and_save_total_vocab(folder_path):
    total_vocab = set()
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            words = word_tokenize(full_text.lower())
            total_vocab.update(words)
            doc.close()
    return total_vocab

if __name__ == "__main__":
    # Hardcoded folder path and CSV file path
    folder_path = "data/test"  # Adjust this path as per your folder structure
    csv_file_path = "extracted_info.csv"  # Adjust this path as per your requirements

    # Create total vocabulary
    total_vocabulary = create_and_save_total_vocab(folder_path)
    # print(total_vocabulary)
    # Batch process here
    batch_process_pdf_folder(folder_path, csv_file_path, total_vocabulary)
