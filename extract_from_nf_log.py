import pandas as pd
import re

def extractProcessDefinitions(file_path):
    """
    Extracts process definitions from a Nextflow log file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the Nextflow log file.

    Returns:
    - pd.DataFrame: A DataFrame containing the process names and their associated paths.
    """
    # Define the regex pattern to match the relevant line
    pattern = r"Workflow process definitions \[dsl.\]:\s*(.*)"

    # Initialize an empty list to store the extracted data
    data = []

    # Open the file and search for the relevant line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the content after "[dsl.]:"
                content = match.group(1)

                # Split the content into path and process name pairs
                entries = re.findall(r"([^,]+)\s*\[([^\]]+)\]", content)
                for path, processes in entries:
                    # Split the process names and add them to the data list
                    process_list = [process.strip() for process in processes.split(",")]
                    for process in process_list:
                        data.append({"path": path.strip(), "process_name": process})

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df