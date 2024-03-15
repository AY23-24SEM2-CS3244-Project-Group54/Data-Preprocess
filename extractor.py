import fitz  # PyMuPDF
import csv
import re
import os

def extract_key_info(pdf_path):
    key_info = {
        "File Name": os.path.basename(pdf_path),
        "Case Number": "",
        "Decision Date": "",
        "Tribunal/Court": "",
        "Outcome": "Outcome not explicitly mentioned",
        "Capitalized Words Before 'Act ('": ""
    }
    
    # Keywords and corresponding outcomes
    keywords_to_outcomes = {
        "dismissed": "Appeal dismissed.",
        "allowed": "Appeal allowed.",
        "accordingly": "Order accordingly.",
        "granted": "Declaration granted."
    }
    
    # This regex pattern captures the entire phrase leading up to "Act ("
    pattern = re.compile(r"([A-Z][\w\s]+)(?=Act \()")
    unique_sequences = set()

    with fitz.open(pdf_path) as doc:
        first_page_text = doc[0].get_text("text")
        last_page_text = doc[-1].get_text("text")

        # Extract basic info
        case_number_match = re.search(r"Case Number\s*:\s*(.+)", first_page_text)
        decision_date_match = re.search(r"Decision Date\s*:\s*(.+)", first_page_text)
        tribunal_court_match = re.search(r"Tribunal/Court\s*:\s*(.+)", first_page_text)

        key_info["Case Number"] = case_number_match.group(1) if case_number_match else ""
        key_info["Decision Date"] = decision_date_match.group(1) if decision_date_match else ""
        key_info["Tribunal/Court"] = tribunal_court_match.group(1) if tribunal_court_match else ""

        # Scan last few lines for keywords to determine the outcome
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
            matches = re.finditer(pattern, text)
            for match in matches:
                phrase = match.group(1)
                capitalized_words_sequence = ' '.join(word for word in phrase.split() if word[0].isupper())
                unique_sequences.add(capitalized_words_sequence)

    key_info["Capitalized Words Before 'Act ('"] = '; '.join(unique_sequences)
    return key_info


def save_to_csv(all_info, csv_file_path):
    headers = ["File Name", "Case Number", "Decision Date", "Tribunal/Court", "Outcome", "Capitalized Words Before 'Act ('"]
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for info in all_info:
            writer.writerow(info)

def batch_process_pdf_folder(folder_path, csv_file_path):
    all_info = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            key_info = extract_key_info(pdf_path)
            all_info.append(key_info)
    
    save_to_csv(all_info, csv_file_path)
    print(f"All PDF files in {folder_path} have been processed. Key information is saved to {csv_file_path}.")

# PATH
folder_path = 'i:/CS3244/test_data/'  # Adjust to your folder path
csv_file_path = 'i:/CS3244/all.csv'  # Adjust to your CSV file path

# Batch process
batch_process_pdf_folder(folder_path, csv_file_path)
