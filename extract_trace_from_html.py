import pandas as pd
import json
import re

def extract_trace_data(file_path):
    """
    Extracts trace data from an HTML file and converts it into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the HTML file containing the trace data.

    Returns:
    - pd.DataFrame: A DataFrame containing the parsed and converted trace data.
    """

    # Load the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Use regular expression to find the trace data
    # The pattern searches for the 'trace' array within the JavaScript object
    match = re.search(r'window\.data\s*=\s*\{.*?"trace":(\[.*?\])\s*,\s*"summary"', html_content, re.DOTALL)

    if match:
        # Extract the JSON content from the match
        trace_data_json = match.group(1).strip()

        # Replace escaped characters in the "script" key values
        # This ensures that any escaped slashes or single quotes are correctly formatted
        trace_data_json = trace_data_json.replace('\\/', '/').replace("\\'", "'")

        # Attempt to parse the entire JSON array at once
        try:
            # Load the JSON data using json.loads
            trace_data_parsed = json.loads(trace_data_json)

            # Convert the parsed JSON data to a DataFrame
            df = pd.DataFrame(trace_data_parsed)

            # Exclude non-"COMPLETED" process execution.
            nb_entries = df.shape[0]
            df.drop(df[df['status']!='COMPLETED'].index, inplace=True)
            if (nb_entries - df.shape[0]) > 0:
                print(f'**WARNING**: {nb_entries - df.shape[0]} process executions were ignored because their status was not COMPLETED.')

            # Define column conversion rules
            # This dictionary maps column types to their respective columns
            conversion_rules = {
                'datetime': ['submit', 'start', 'complete'],
                'timedelta': ['time', 'duration', 'realtime'],
                'integer': ['exit', 'task_id', 'native_id', 'attempt', 'cpus', 'vol_ctxt', 'inv_ctxt'],
                'longlong': ['peak_rss', 'peak_vmem', 'rchar', 'wchar', 'vmem', 'rss', 'syscr', 'syscw', 'read_bytes', 'write_bytes'],
                'float': ['%cpu', '%mem'],
                'numeric_with_nan': ['memory']
            }

            # Apply conversions based on the defined rules
            for col_type, columns in conversion_rules.items():
                for column in columns:
                    if column in df.columns:
                        if col_type == 'datetime':
                            df[column] = pd.to_datetime(df[column].astype('longlong'), unit='ms')
                        elif col_type == 'timedelta':
                            df[column] = pd.to_timedelta(df[column].astype('longlong'), unit='ms')
                        elif col_type == 'integer':
                            df[column] = df[column].astype('int')
                        elif col_type == 'longlong':
                            df[column] = df[column].astype('longlong')
                        elif col_type == 'float':
                            df[column] = df[column].astype('float')
                        elif col_type == 'numeric_with_nan':
                            df[column] = pd.to_numeric(df[column].replace('-', None), errors='coerce')

            return df
        except json.JSONDecodeError as e:
            # Print the error and a portion of the JSON string for inspection if parsing fails
            print(f"Error parsing JSON: {e}")
            print(f"Context around the error:\n{trace_data_json[:1000]}")
            return None
    else:
        print("Trace data not found.")
        return None
