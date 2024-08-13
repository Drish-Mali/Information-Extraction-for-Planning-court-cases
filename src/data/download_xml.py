import json
import os
import requests
import re

def generate_url_from_string(input_string):
    # Split the input string by spaces
    input_parts = input_string.split()

    # Check if the input string has 4 parts
    if len(input_parts) == 4:
        # Check if the last part ends with ')'
        if input_parts[-1].endswith(')'):
            date = input_parts[0][1:-1]  # Remove '[' and ']'
            court_type, case_number, court_sub_type = input_parts[1], input_parts[2], input_parts[3][1:-1]  # Remove '(' and ')'
            return f"{court_type.lower()}/{court_sub_type.lower()}/{date}/{case_number}"
        # Check if the last part is a number
        elif input_parts[-1].isdigit():
            date = input_parts[0][1:-1]  # Remove '[' and ']'
            court_type, court_sub_type,case_number = input_parts[1], input_parts[2],input_parts[3]
            return f"{court_type.lower()}/{court_sub_type.lower()}/{date}/{case_number}"

    # Check if the input string has 3 parts
    elif len(input_parts) == 3:
        date = input_parts[0][1:-1]  # Remove '[' and ']'
        court_type, case_number = input_parts[1], input_parts[2]
        return f"{court_type.lower()}/{date}/{case_number}"

    # If none of the conditions match, return an error message
    return "Invalid input format"

def create_folders(max_value, directory):
    for i in range(1, max_value + 1):
        folder_name = f"xml_page_{i}"
        folder_path = os.path.join(directory, folder_name)
        # Create the directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)


def download_judgment(uri, filename):
    base_url = "https://caselaw.nationalarchives.gov.uk"
    endpoint = f"{base_url}/{uri}/data.xml"
    
    response = requests.get(endpoint)
    try:
        
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            print("Failed to download the Judgement")
            print(filename)
            print(uri)    
    except:
        print("error")
        print(filename)
        print(uri)  


def generate_data(json_path):
    judgment_titles = []
    neutral_citations = []
    pages=[]
    urls=[]
    # Read JSON data from file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Loop through each array in the JSON data
    for array in data:
        # Loop through each object in the array
        for obj in array:
            # Extract the title and neutral citation and add them to respective lists
            judgment_titles.append(obj["Judgment Title"])
            neutral_citations.append(obj["Neutral Citation"])
            urls.append(generate_url_from_string(str(obj['Neutral Citation'])))
            pages.append(obj["page"])

    max_value = max(pages)  # Replace with your desired maximum value
    directory = "./data/xml_data"  # Replace with your desired directory path
    create_folders(max_value, directory)
    # Print the extracted titles and neutral citations
    for title, citation ,page,url in zip(judgment_titles, neutral_citations,pages,urls):
        #print("Title:", title)
        title = title.replace('/', '_')
        file_name=title+".xml"
        location="./data/xml_data/xml_page_"+str(page)+"/"
        location=location+file_name
        download_judgment(url,location)
        


if __name__ == "__main__":
    generate_data("./data/search_results.json")