import fitz  # PyMuPDF
import csv
import re
import os

def extract_key_info(pdf_path):
    file_name = os.path.basename(pdf_path)
    with fitz.open(pdf_path) as doc:
        first_page_text = doc[0].get_text("text")
        last_page_text = doc[-1].get_text("text")
    
    key_info = {
        "File Name": file_name,
        "Case Number": re.search(r"Case Number\s*:\s*(.+)", first_page_text).group(1) if re.search(r"Case Number\s*:\s*(.+)", first_page_text) else "",
        "Decision Date": re.search(r"Decision Date\s*:\s*(.+)", first_page_text).group(1) if re.search(r"Decision Date\s*:\s*(.+)", first_page_text) else "",
        "Tribunal/Court": re.search(r"Tribunal/Court\s*:\s*(.+)", first_page_text).group(1) if re.search(r"Tribunal/Court\s*:\s*(.+)", first_page_text) else "",
        "Outcome": "Outcome not explicitly mentioned"
    }
    
    # Keywords and corresponding outcomes
    keywords_to_outcomes = {
        "dismissed": "Appeal dismissed.",
        "allowed": "Appeal allowed.",
        "accordingly": "Order accordingly.",
        "granted": "Declaration granted."
    }
    
    # Scan last few lines for keywords to determine the outcome
    last_page_lines = last_page_text.split('\n')
    for line in reversed(last_page_lines):
        for keyword, outcome in keywords_to_outcomes.items():
            if keyword in line.lower():
                key_info["Outcome"] = outcome
                break
        if key_info["Outcome"] != "Outcome not explicitly mentioned":
            break

    return key_info

def save_to_csv(all_info, csv_file_path):
    headers = ["File Name", "Case Number", "Decision Date", "Tribunal/Court", "Outcome"]
    
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
folder_path = 'i:/CS3244/test_data/'  # INPUT: Folder PATH
csv_file_path = 'i:/CS3244/batch_extracted_info.csv'  # OUTPUT: CSV file PATH

# Batch process
batch_process_pdf_folder(folder_path, csv_file_path)
