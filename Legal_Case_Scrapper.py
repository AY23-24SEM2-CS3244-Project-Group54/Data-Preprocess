import requests
import os

def download_files():
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'data', r'raw')
    os.makedirs(final_directory, exist_ok=True)

    base_url = 'https://www.elitigation.sg/gd/gd/{year}_{court}_{case_id}/pdf'
    
    years = range(2000, 2016)
    courts = ['SGHC', 'SGCA']
    case_ids = range(1, 430)

    # Iterate over all combinations of years, courts, and case_ids
    for year in years:
        for court in courts:
            for case_id in case_ids:
                pdf_url = base_url.format(year=year, court=court, case_id=case_id)
                response = requests.get(pdf_url)
                #print(response)

                # Check if the request was successful and the content is a PDF
                if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                    fname = os.path.join(final_directory, fr'Case_{year}_{court}_{case_id}.pdf')
                    file = open(fname, "wb")
                    file.write(response.content)
                    file.close()        
                else:
                    # Skip non-existent or non-PDF files
                    continue

download_files()
