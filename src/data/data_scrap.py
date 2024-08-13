import json
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
import math

def find_total_documents(soup):
    title = soup.title.string.strip()
    print("Title:", title)

    # Extract the number of documents found
    results_intro = soup.find('p', class_='results__results-intro')
    documents_found = results_intro.get_text(strip=True)
    print("Documents found:", documents_found)
    return documents_found
    

def scrape_judgment_xml(
    url: str, order: str = "updated", page: int = 5, search_query: Optional[str] = None
) -> Optional[BeautifulSoup]:
    """Scrapes XML data from the given URL and returns a BeautifulSoup object."""
    full_url=url
    full_url += f"/search?query={search_query}"
    full_url+=f"&page={page}"
    results=[]
    print(full_url)
    #full_url='https://caselaw.nationalarchives.gov.uk/judgments/search?query="planning court"&page=2'
    response = requests.get(full_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        #show_results(soup) 
        total_docs=int(find_total_documents(soup).split()[0])
        number_pages=math.ceil(total_docs/10)
        print(number_pages)
        for i in range(1,number_pages+1):
            full_url=url
            full_url += f"/search?query={search_query}"
            full_url+=f"&page={i}"
            # print(full_url)
            response = requests.get(full_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                results.append(get_results_to_json(soup,i))
    return results

def show_results(soup):
    # Extract the title
    title = soup.title.string.strip()
    print("Title:", title)

    # Extract the number of documents found
    results_intro = soup.find('p', class_='results__results-intro')
    documents_found = results_intro.get_text(strip=True)
    print("Documents found:", documents_found)

    # Extract the list of search results
    result_list = soup.find('ul', class_='judgment-listing__list')
    if result_list:
        search_results = result_list.find_all('li')
        for result in search_results:
            judgment_title = result.find('span', class_='judgment-listing__title').a.get_text(strip=True)
            court_info = result.find('span', class_='judgment-listing__court').get_text(strip=True)
            neutral_citation = result.find('span', class_='judgment-listing__neutralcitation').get_text(strip=True)
            date = result.find('time', class_='judgment-listing__date')['datetime']
            print("Judgment Title:", judgment_title)
            print("Court Information:", court_info)
            print("Neutral Citation:", neutral_citation)
            print("Date:", date)
            print()
    else:
        print("No search results found.")


def get_results_to_json(soup,page):
    results_data = []

    # Extract the list of search results
    result_list = soup.find('ul', class_='judgment-listing__list')
    if result_list:
        search_results = result_list.find_all('li')
        for result in search_results:
            judgment_title = result.find('span', class_='judgment-listing__title').a.get_text(strip=True)
            court_info = result.find('span', class_='judgment-listing__court').get_text(strip=True)
            neutral_citation = result.find('span', class_='judgment-listing__neutralcitation').get_text(strip=True)
            date = result.find('time', class_='judgment-listing__date')['datetime']
            
            # Collect data into a dictionary
            result_data = {
                "Judgment Title": judgment_title,
                "Court Information": court_info,
                "Neutral Citation": neutral_citation,
                "Date": date,
                "page":page
            }
            results_data.append(result_data)
    return results_data

def extract_titles_and_ids(soup: BeautifulSoup) -> Dict[str, str]:
    """Extracts titles and IDs from a BeautifulSoup object representing XML data."""
    results = {}
    items = soup.find_all("entry")
    for item in items:
        title = item.title.get_text(strip=True)
        entry_id = item.id.get_text(strip=True)
        results[title] = entry_id
    return results

def save_results_to_json(data,output_file):
     with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    root_feed_url = "https://caselaw.nationalarchives.gov.uk/judgments"
    page_pagination = 1
    search_query = '"planning+court"'
    xml_data = scrape_judgment_xml(
        url=root_feed_url, page=page_pagination, search_query=search_query
    )
    save_results_to_json(xml_data, "../data/search_results.json")
