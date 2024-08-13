import xml.etree.ElementTree as ET
import pandas as pd
import re
import os

def starts_with_number(text):
    return bool(re.match(r'^\d+\.', text))

def extract_paragraph(parent_folder):
    namespace = {
        'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
        'uk': 'https://caselaw.nationalarchives.gov.uk/akn'
    }

    paragraph_df = pd.DataFrame(columns=['paragraph', 'file_name'])
    missing_files = []

    # Iterate through each subfolder in the parent folder
    for subfolder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        
        # Check if the item in the parent folder is a directory
        if os.path.isdir(subfolder_path):
            # Iterate through each XML file in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".xml"):
                    file_path = os.path.join(subfolder_path, filename)
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    segments = []
                    current_segment = []
                    collecting = False

                    # Iterate through all elements in the XML
                    for elem in root.iter():
                        # Check if the element is a <num> tag with a style attribute
                        if elem.tag == f"{{{namespace['akn']}}}num" and 'style' in elem.attrib:
                            if collecting:
                                segments.append(' '.join(current_segment).strip())
                                current_segment = []
                            collecting = True  # Start collecting after finding the target <num> tag
                            if elem.text:
                                current_segment.append(elem.text.strip())  # Append the number text

                        if collecting:
                            if elem.tag != f"{{{namespace['akn']}}}num":
                                if elem.text:
                                    current_segment.append(elem.text.strip())
                                if elem.tail:
                                    current_segment.append(elem.tail.strip())

                    # Add the last segment if any
                    if current_segment:
                        segments.append(' '.join(current_segment).strip())

                    # Initialize a list to store combined segments
                    combined_segments = []
                    current_paragraph = ""

                    # Iterate through the segments and combine them
                    for segment in segments:
                        if starts_with_number(segment):
                            if current_paragraph:
                                combined_segments.append(current_paragraph.strip())
                            current_paragraph = segment
                        else:
                            current_paragraph += " " + segment

                    # Append the last paragraph if any
                    if current_paragraph:
                        combined_segments.append(current_paragraph.strip())

                    # Add combined segments to DataFrame
                    for paragraph in combined_segments:
                        new_row = pd.DataFrame({'paragraph': [paragraph], 'file_name': [file_path]})
                        paragraph_df = pd.concat([paragraph_df, new_row], ignore_index=True)

                    if not combined_segments:
                        missing_files.append(file_path)

    print("Missing Files:")
    for file_path in missing_files:
        print(file_path)

    return paragraph_df,missing_files


if __name__=='__main__':
    parent_folder = '.\\data\\xml_data'
    paragraph_df,missing_files = extract_paragraph(parent_folder)

    print(len(list(paragraph_df['file_name'].unique())))
    print(len(missing_files))

    paragraph_counts = paragraph_df['file_name'].value_counts()

    # Filter out the files that have only one paragraph
    files_with_one_paragraph = paragraph_counts[paragraph_counts ==1].index

    # Display the filenames
    files_with_one_paragraph_list = files_with_one_paragraph.tolist()
    print(files_with_one_paragraph_list)
    print(len(files_with_one_paragraph_list))


    # Filter out rows with file names that appear only once
    filtered_df = paragraph_df[~paragraph_df['file_name'].isin(files_with_one_paragraph_list)].reset_index(drop=True)
    print(filtered_df.shape)
    annotated_df = pd.read_csv(".\\data\\main_text.csv")
    print(len(filtered_df['file_name'].unique()))
    print(len(files_with_one_paragraph_list))
    # Filter rows based on the file names in filtered_files
    new_list=files_with_one_paragraph_list+missing_files
    new_filtered_df = annotated_df[annotated_df['file_name'].isin(new_list)][['paragraph','file_name']]

    print(new_filtered_df.shape)
    filtered_df = pd.concat([filtered_df, new_filtered_df], ignore_index=True)
    filtered_df.to_csv(".\\data\\new_main_text.csv",index=False)
    print(filtered_df.shape)
    print(len(list(filtered_df['file_name'].unique())))