import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
import time
import json
import os

TARGET_LAWS = {
    "Technology & Privacy": [
        "https://www.legislation.gov.uk/ukpga/2018/12/data.xml",  
        "https://www.legislation.gov.uk/ukpga/2023/50/data.xml",  
        "https://www.legislation.gov.uk/ukpga/1990/18/data.xml"   
    ],
    "Civil Rights & Information": [
        "https://www.legislation.gov.uk/ukpga/2010/15/data.xml",  
        "https://www.legislation.gov.uk/ukpga/1998/42/data.xml",  
        "https://www.legislation.gov.uk/ukpga/2000/36/data.xml"   
    ],
    "Business & Consumer Law": [
        "https://www.legislation.gov.uk/ukpga/2006/46/data.xml",  
        "https://www.legislation.gov.uk/ukpga/2015/15/data.xml",  
        "https://www.legislation.gov.uk/ukpga/1996/18/data.xml"   
    ],
    "Criminal Law": [
        "https://www.legislation.gov.uk/ukpga/2006/35/data.xml"   
    ],
    "Emerging Tech & State Powers": [
        "https://www.legislation.gov.uk/ukpga/2016/25/data.xml",
        "https://www.legislation.gov.uk/ukpga/2018/18/data.xml"
    ],
    "Environment & Climate": [
        "https://www.legislation.gov.uk/ukpga/2021/30/data.xml",
        "https://www.legislation.gov.uk/ukpga/2008/27/data.xml"
    ],
    "Housing & Tenancy": [
        "https://www.legislation.gov.uk/ukpga/2004/34/data.xml",
        "https://www.legislation.gov.uk/ukpga/1985/70/data.xml"
    ],
    "Health & Safety": [
        "https://www.legislation.gov.uk/ukpga/1974/37/data.xml",
        "https://www.legislation.gov.uk/ukpga/2021/3/data.xml"
    ]
}


def normalize_whitespace(text: str) -> str:
    """Compresses messy whitespace into predictable RAG separators."""
    if not text:
        return ""
    # Compress 3 or more newlines into exactly 2 (\n\n)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Compress multiple spaces into a single space
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def enrich_xml_with_context(body_soup) -> str:
    """Walks the XML tree, stamping Section context onto every paragraph."""
    if not body_soup:
        return ""
        
    enriched_lines = []
    
    # Find every legal Section (wrapped in <P1group> in UK law)
    for section in body_soup.find_all('P1group'):
        
        # Extract the Section Title and Number
        title_tag = section.find('Title')
        num_tag = section.find('Pnumber')
        
        title = title_tag.text.strip() if title_tag else "Untitled"
        sec_num = num_tag.text.strip() if num_tag else "Unknown"
        
        # Create Context Stamp
        context_stamp = f"[Section {sec_num}: {title}]"
        
        # Find all text containers inside this Section
        for para in section.find_all(['P1para', 'P2para', 'P3para', 'BlockText']):
            
            # Flatten internal line breaks within the paragraph
            item_text = para.get_text(separator=' ', strip=True)
            
            if item_text:
                enriched_lines.append(f"{context_stamp} {item_text}")
                
    # Join the enriched paragraphs with \n\n separator
    return "\n\n".join(enriched_lines)


def process_law(url: str, category: str) -> dict:
    print(f"Fetching data from: {url}")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch {url}. Status Code: {response.status_code}")
        return None
        
    # Extract the XML tree
    soup = BeautifulSoup(response.content, "xml") 
    
    # Extract Metadata 
    title_tag = soup.find('dc:title')
    title = title_tag.text if title_tag else "Unknown Title"
    
    parts = url.split('/')
    act_id = f"{parts[-4]}/{parts[-3]}/{parts[-2]}" # e.g., ukpga/2018/12
    year = int(parts[-3]) if parts[-3].isdigit() else 0
    
    # Extract the Body
    body_tag = soup.find('Body')
    if not body_tag:
        print("No <Body> tag found. Skipping.")
        return None
        
    raw_xml_content = str(body_tag)
    
    # Enrich the raw XML with section stamps
    enriched_text = enrich_xml_with_context(body_tag)
    
    # Normalize the whitespace of the enriched text
    final_clean_text = normalize_whitespace(enriched_text)
    
    # Package data
    document = {
        "act_id": act_id,
        "title": title,
        "year": year,
        "category": category,
        "url_source": url,
        "raw_xml_snippet": raw_xml_content, 
        "cleaned_text": final_clean_text 
    }
    
    return document


def populate_english_law():
    os.makedirs("seed_data", exist_ok=True)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    output_filepath = BASE_DIR / Path("seed_data/english_laws.json")
    
    all_processed_laws = []
    total_processed = 0

    print("Starting Offline Scraping for Seed Data...\n")
    
    for category, urls in TARGET_LAWS.items():
        print(f"Processing Category: {category}")
        
        for url in urls:
            doc = process_law(url, category)
            
            if doc:
                all_processed_laws.append(doc)
                print(f"Saved '{doc['title']}' to dataset.")
                total_processed += 1
            
            # sleep to respect the server
            time.sleep(1.5) 
            
        print("-" * 40)
        
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(all_processed_laws, f, ensure_ascii=False, indent=4)
        
    print(f"\nIngestion Complete! Successfully saved {total_processed} laws to {output_filepath}.")


if __name__ == "__main__":
    populate_english_law()