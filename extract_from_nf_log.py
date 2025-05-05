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


def extractResolvedProcessNames(file_path):
    """
    Extracts resolved process names from a Nextflow log file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the Nextflow log file.

    Returns:
    - pd.DataFrame: A DataFrame containing the resolved names, process names, and their associated paths.
    """
    # Define the regex pattern to match the relevant line
    pattern = r"Resolved process names:\s*(.*)"

    # Initialize an empty list to store the extracted data
    data = []

    # Open the file and search for the relevant line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the content after "Resolved process names:"
                content = match.group(1)

                # Split the content into path, process name, and resolved names
                entries = re.findall(r"([^,]+)=([^=]+)=\[([^\]]+)\]", content)
                for path, process_name, resolved_names in entries:
                    # Split the resolved names and add them to the data list
                    resolved_list = [resolved.strip() for resolved in resolved_names.split(",")]
                    for resolved in resolved_list:
                        data.append({
                            "path": path.strip(),
                            "process_name": process_name.strip(),
                            "resolved_name": resolved
                        })

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df


def extractProcessInputs(file_path):
    """
    Extracts process inputs and outputs from a Nextflow log file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the Nextflow log file.

    Returns:
    - pd.DataFrame: A DataFrame containing the resolved process name, input arguments, and output arguments.
    """
    # Define the regex pattern to match the relevant line
    pattern = r"Starting process > ([^\s]+) \((.*?)\) -> \((.*?)\)"

    # Initialize an empty list to store the extracted data
    data = []

    def parse_arguments(arg_string):
        """
        Recursively parse arguments, handling nested tuples.

        :param arg_string: The string containing arguments.
        :return: A list of parsed arguments.
        """
        arguments = []
        for arg in re.findall(r"(\w+):(\[.*?\]|[^\s,]+)", arg_string):
            arg_type, arg_value = arg
            if arg_value.startswith("[") and arg_value.endswith("]"):
                # Parse nested arguments for tuples
                nested_args = parse_arguments(arg_value[1:-1])  # Remove square brackets and parse recursively
                arguments.append({"type": arg_type, "value": nested_args})
            else:
                arguments.append({"type": arg_type, "value": arg_value})
        return arguments

    # Open the file and search for the relevant lines
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the resolved process name, input arguments, and output arguments
                resolved_process_name = match.group(1)
                input_args = match.group(2)
                output_args = match.group(3)

                # Parse the input arguments
                inputs = parse_arguments(input_args)

                # Parse the output arguments
                outputs = []
                for arg in re.findall(r"(\w+):([^\s,]+)", output_args):
                    outputs.append({"type": arg[0], "value": arg[1]})

                # Add the extracted data to the list
                data.append({
                    "resolved_process_name": resolved_process_name,
                    "inputs": inputs,
                    "outputs": outputs
                })

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df


def extractExecutionParameters(file_path):
    """
    Extracts execution parameters from a Nextflow log file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the Nextflow log file.

    Returns:
    - pd.DataFrame: A DataFrame containing the resolved process name and its associated input values.
    """
    # Define the regex pattern to match the relevant line
    pattern = r"TRACE nextflow\.processor\.TaskProcessor - Invoking task > ([^\s]+) with params=TaskStartParams\[id=[0-9]*; index=([0-9]*)\]; values=\[(.*)"

    # Initialize an empty list to store the extracted data
    data = []

    def parse_values(value_string):
        """
        Parse the values, handling tuples as lists.

        :param value_string: The string containing the values.
        :return: A list of parsed values.
        """
        values = []
        for value in re.findall(r"(\[.*?\]|[^\s,]+)", value_string):
            if value.startswith("[") and value.endswith("]"):
                # Parse nested tuples as lists
                nested_values = re.findall(r"[^\s,\[\]]+", value[1:-1])
                values.append(nested_values)
            else:
                values.append(value)
        return values

    # Open the file and search for the relevant lines
    with open(file_path, 'r', encoding='utf-8') as file:
        multiline_buffer = None
        resolved_process_name = None
        instance_number = None

        for line in file:
            if multiline_buffer is not None:
                # Append to the buffer until we find the closing bracket
                multiline_buffer += line.strip()
                if multiline_buffer.endswith("]"):
                    # Parse the buffered input values
                    parsed_values = parse_values(multiline_buffer)
                    data.append({
                        "instance_number": instance_number,
                        "resolved_process_name": resolved_process_name,
                        "input_values": parsed_values
                    })
                    multiline_buffer = None  # Reset the buffer
                continue

            match = re.search(pattern, line)
            if match:
                # Extract the resolved process name and start buffering input values
                resolved_process_name = match.group(1)
                instance_number = match.group(2)
                multiline_buffer = match.group(3).strip()

                # If the line already ends with "]", parse it immediately
                if multiline_buffer.endswith("]"):
                    parsed_values = parse_values(multiline_buffer)
                    data.append({
                        "instance_number": instance_number,
                        "resolved_process_name": resolved_process_name,
                        "input_values": parsed_values
                    })
                    multiline_buffer = None  # Reset the buffer

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df
