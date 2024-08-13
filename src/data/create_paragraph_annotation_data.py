import os
import pandas as pd

def save_paragraphs_to_folders(paragraph_df, folder_path='.\data\paragraph_annotation'):
    """
    Save paragraphs to separate CSV files in folders based on the file name.

    Args:
    - paragraph_df (DataFrame): DataFrame containing 'file_name' and 'paragraph' columns.
    - output_folder (str): Path to the folder where the CSV files will be saved. Default is 'paragraph_annotation'.
    """
    paragraph_df['introduction']=0
    paragraph_df['fact']=0
    paragraph_df['citation']=0
    paragraph_df['judgment']=0
    #paragraph_df['none']=0
    # Define the regex pattern
    citation_regex = r"\b(?:d)?\s?(?:\w+\s+)*\[\d{4}\]\s\w+\s\d+\s?\(?\w+?\)?"
    fact_regex = r"\b(?:Act|s(?:ection)?|Section)\s*(?:\.)?\s*(\d+)(?:\(\d+\))?\b"

    # Apply a lambda function with vectorized string matching
    paragraph_df["citation"] = paragraph_df["paragraph"].str.contains(citation_regex, regex=True).astype(int)
    paragraph_df["fact"] = paragraph_df["paragraph"].str.contains(fact_regex, regex=True).astype(int)
    #paragraph_df["none"] = ~(paragraph_df["citation"] | paragraph_df["fact"])

    # Create output folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Iterate over unique file names
    for file_name in paragraph_df['file_name'].unique():
        # Get rows for current file name
        filtered_df = paragraph_df[paragraph_df['file_name'] == file_name]
        #print(file_name)
        # Extract page number from file path
        page_number = file_name.split('\\')[-2].split('_')[-1]
        
        # Create folder if it doesn't exist
        page_folder_path = f'{folder_path}\\page_{page_number}'
        #print(page_folder_path)
        os.makedirs(page_folder_path, exist_ok=True)
        
        # Save DataFrame to CSV in the folder
        file_path = f'{page_folder_path}\\{os.path.basename(file_name)[:-4]}.csv'
        # print(file_path)
        filtered_df.to_csv(file_path, index=False)

if __name__=="__main__":
    paragraph_df = pd.read_csv("./data/main_text.csv")

    # Call the function to save paragraphs to folders
    save_paragraphs_to_folders(paragraph_df)
