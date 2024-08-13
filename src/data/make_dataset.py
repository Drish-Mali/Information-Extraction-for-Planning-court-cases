import os
import re
import xmltodict
import xml.etree.ElementTree as ET
import pandas as pd
def get_text(element):
  """
  This function recursively traverses the element tree and gathers text content.
  """
  text = ""
  if element.text:
    text += element.text
  for child in element:
    text += get_text(child)
  if element.tail:
    text += element.tail
  return text

def extract_paragraph_texts(parent_folder):
    error_count = 0
    paragraph_df = pd.DataFrame(columns=['paragraph', 'file_name'])
    error_list = []

    # Iterate through each subfolder (xml_page_1, xml_page_2, etc.) in the parent folder
    for subfolder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        
        # Check if the item in the parent folder is a directory
        if os.path.isdir(subfolder_path):
            # Iterate through each XML file in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".xml"):
                    file_path = os.path.join(subfolder_path, filename)
                    
                    try:
                        root = ET.parse(file_path)
                        # Find the judgmentBody tag
                        judgment_body = root.find('.//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}judgmentBody')

                        # Collect text using the get_text function and store in a list
                        paragraphs = []
                        if judgment_body is not None:
                            for p_tag in judgment_body.iterfind('.//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}p'):
                                paragraphs.append(get_text(p_tag))
                            paragraphs = [s for s in paragraphs if s is not None]
                            
                            paragraphs = [s for s in paragraphs if s.count('-') <= 2]
                            paragraphs = [item for item in paragraphs if not item.isspace() and not all(char in ['\n', '\t'] for char in item)]
                            paragraphs = list(filter(lambda x: any(char.isalnum() for char in x), paragraphs))

                            # Create a DataFrame with a column named "Paragraph"
                            df = pd.DataFrame(paragraphs, columns=['paragraph'])
                            df['file_name'] = file_path
                            paragraph_df = pd.concat([paragraph_df, df], axis=0, ignore_index=True)
                    except Exception as e:
                        print("Error:", e)
                        error_count += 1
                        print(file_path)
                        error_list.append(file_path)
    
    print("Error count:", error_count)
    paragraph_df
    return paragraph_df, error_list

def extract_cover_texts(parent_folder):
    error_count = 0
    file_names = []
    error_list = []
    cover_texts = []

    # Iterate through each subfolder in the parent folder
    for subfolder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        
        # Check if the item in the parent folder is a directory
        if os.path.isdir(subfolder_path):
            # Iterate through each XML file in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".xml"):
                    file_path = os.path.join(subfolder_path, filename)
                    
                    try:
                        root = ET.parse(file_path)
                        # Find the header tag
                        header = root.find('.//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}header')

                        # Collect text from p tags under the header
                        paragraphs = []
                        if header is not None:
                            for p_tag in header.iterfind('.//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}p'):
                                paragraphs.append(get_text(p_tag))

                            # Filter out unwanted text
                            filtered_list = [s for s in paragraphs if s is not None]
                            filtered_list = [s for s in filtered_list if s.count('-') <= 2]
                            filtered_list = [item for item in filtered_list if not item.isspace() and not all(char in ['\n', '\t'] for char in item)]
                            concatenated_string = ''.join(filtered_list)
                            
                            file_names.append(file_path)
                            cover_texts.append(concatenated_string)
                        
                    except Exception as e:
                        print("Error:", e)
                        error_count += 1
                        print(file_path)
                        error_list.append(file_path)

    # Create DataFrame with file names and cover texts
    cover_df = pd.DataFrame({'file_name': file_names, 'cover_text': cover_texts})
    
    print("Error count:", error_count)
    return cover_df, error_list

def get_text_before_introduction(df):
  """
  Gets a string containing all text paragraphs before the row with "Introduction".

  Args:
      df (pandas.DataFrame): The input DataFrame.

  Returns:
      str: The combined string of text paragraphs before the introduction.
  """

  # Find the first row index containing "Introduction" (or None if not found)
  intro_index = df[df['paragraph'].str.contains('Hearing date', case=False)].index.to_list()
  
  if intro_index:
   
    # Get all text paragraphs before the introduction
    text_before_intro = '\n'.join(df['paragraph'].iloc[:intro_index[0]+1])
  elif df[df['paragraph'].str.contains('JUDGMENT', case=False)].index.to_list():
    # If "JUDGMENT" is not found, return an empty string
    intro_index = df[df['paragraph'].str.contains('JUDGMENT', case=False)].index.to_list()
    text_before_intro = '\n'.join(df['paragraph'].iloc[:intro_index[0]+1])
  else:
    text_before_intro = '\n'.join(df['paragraph'].iloc[:])
  return text_before_intro


def process_remaining_files(no_header_list):
    remaining_paragraph = []
    file_path_names = []
    error_list = []
    error_count = 0
    for file_path in no_header_list:
        root = ET.parse(file_path)
        try:
            # Find the judgmentBody tag
            judgment_body = root.find('.//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}judgmentBody')

            # Collect text using the get_text function and store in a list
            paragraphs = []
            if judgment_body is not None:
                for p_tag in judgment_body.iterfind('.//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}p'):
                    paragraphs.append(get_text(p_tag))
                paragraphs = [s for s in paragraphs if s is not None]

                # Create a DataFrame with a column named "Paragraph"
                df = pd.DataFrame(paragraphs, columns=['paragraph'])
                text_before_intro = get_text_before_introduction(df.copy())
                remaining_paragraph.append(text_before_intro)
                file_path_names.append(file_path)
            reamaining_df = pd.DataFrame({"file_name": file_path_names, "cover_text": remaining_paragraph})

        except Exception as e:
            print("error:", e)
            error_count += 1
            print(file_path)
            error_list.append(file_path)

    return reamaining_df

if __name__ == "__main__":
    parent_folder = ".\\data\\xml_data"
    cover_df, error_list = extract_cover_texts(parent_folder)
    paragraph_df, error_list_paragraph= extract_paragraph_texts(parent_folder)
    no_header_list=list(cover_df[cover_df['cover_text']=='']['file_name'])
    reamaining_df=process_remaining_files(no_header_list)
    cover_df['cover_text'] = cover_df.apply(
    lambda x: x['cover_text'] if x['cover_text'].strip() != '' else reamaining_df.loc[reamaining_df['file_name'] == x['file_name'], 'cover_text'].iloc[0],
    axis=1)
    paragraph_df.to_csv("./data/main_text.csv",index=False)
    cover_df.to_csv("./data/cover_text.csv",index=False)

