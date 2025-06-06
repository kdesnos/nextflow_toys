import pandas as pd
import re
from datetime import datetime, timedelta


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


def extractRunName(file_path):
    """
    Extracts the run name from a Nextflow log file.

    Parameters:
    - file_path (str): The path to the Nextflow log file.

    Returns:
    - str: The extracted run name.
    """
    # Define the regex pattern to match the relevant line
    pattern = r"DEBUG nextflow\.Session - Run name: (.+)"

    # Open the file and search for the relevant line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Return the extracted run name
                return match.group(1)

    # Return None if no run name is found
    return None


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
                    # Remove the trailing ']' before parsing
                    multiline_buffer = multiline_buffer[:-1]
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
                    # Remove the trailing ']' before parsing
                    multiline_buffer = multiline_buffer[:-1]
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


def extractPipelineParameters(file_path):
    """
    Extracts pipeline parameters from a Nextflow log file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the Nextflow log file.

    Returns:
    - pd.DataFrame: A DataFrame containing parameter names, their corresponding values, types, and reformatted values (if applicable).
    """
    parameters = []
    inside_params_section = False

    def parse_value(value):
        """
        Parses a value, converting it to the appropriate type (number, string, or list).

        :param value: The string representation of the value.
        :return: The parsed value, its type, and a reformatted value if applicable.
        """
        value = value.strip()
        reformatted_value = None

        # Handle lists in exotic formats
        if value.startswith("'") and value.endswith("'") and " " in value:
            # Space-separated values within single quotes
            items = value.strip("'").split()
            if all(item.isdigit() for item in items):
                reformatted_value = [int(item) for item in items]
                return value, "List[Integer]", reformatted_value
            else:
                reformatted_value = [item for item in items]
                return value, "List[String]", reformatted_value
        elif value.startswith("[") and value.endswith("]"):
            # Standard list format
            items = value[1:-1].split(",")
            parsed_items = []
            for item in items:
                item = item.strip().strip("'\"")  # Remove quotes around each item
                if item.isdigit():
                    parsed_items.append(int(item))
                else:
                    parsed_items.append(item)
            if all(isinstance(x, int) for x in parsed_items):
                return value, "List[Integer]", parsed_items
            return value, "List[String]", parsed_items
        elif value.isdigit():
            return value, "Integer", int(value)
        elif value.strip('\'"').startswith("/"):
            # Recognize paths starting with "/"
            return value.strip('\'"'), "Path", value.strip('\'"')
        try:
            return value, "Real", float(value)
        except ValueError:
            if value.lower() in ["true", "false"]:
                return value, "Boolean", value.lower() == "true"
            else:
                # Return as string (remove quotes if present)
                return value.strip("\"'"), "String", value.strip("\"'")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()

            if stripped_line == "params {":
                inside_params_section = True
                continue

            if inside_params_section:
                if stripped_line == "}":
                    inside_params_section = False
                    break

                # Parse parameter line
                if "=" in stripped_line:
                    param_name, param_value = map(str.strip, stripped_line.split("=", 1))
                    parsed_value, value_type, reformatted_value = parse_value(param_value)
                    parameters.append({
                        "param_name": param_name,
                        "original_value": parsed_value,
                        "type": value_type,
                        "reformatted_value": reformatted_value
                    })

    # Only return the results if the param section was properly closed.
    if not inside_params_section and len(parameters) > 0:
        # Convert the parameters into a pandas DataFrame
        df = pd.DataFrame(parameters)
        return df
    else:
        return None


def extract_task_info_from_lines(lines):
    """
    Extracts process names from a set of log lines.

    :param lines: List of log lines to process.
    :type lines: list of str
    :return: A list of process names found in the log lines.
    :rtype: list of str
    """
    process_names = []
    # Define the regex pattern to match the log line
    pattern = re.compile(
        r"DEBUG n\.processor\.TaskPollingMonitor - Task completed > TaskHandler\[(jobId: \d+; )?id: \d+; name: (?P<process_name>[^;]+?)( \(\d*\))?; status: COMPLETED; exit: 0;"
    )

    for line in lines:
        match = pattern.search(line)
        if match:
            process_name = match.group("process_name")
            process_names.append(process_name)

    return process_names

def extract_trace_file_path_from_lines(lines):
    """
    Extracts the path to the trace file from a set of log lines.

    :param lines: List of log lines to process.
    :type lines: list of str
    :return: The path to the trace file if found, otherwise None.
    :rtype: str or None
    """
    # Define the regex pattern to match the trace file path
    pattern = re.compile(
        r"DEBUG nextflow\.trace\.TraceFileObserver - Workflow started -- trace file: (?P<trace_file_path>.+)"
    )

    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group("trace_file_path")

    return None

def contains_execution_complete_message(lines):
    """
    Checks if any line in the parsed log lines contains the message:
    "DEBUG nextflow.script.ScriptRunner - > Execution complete -- Goodbye".

    :param lines: List of log lines to process.
    :type lines: list of str
    :return: True if the message is found, otherwise False.
    :rtype: bool
    """
    # Define the regex pattern to match the specific log message
    pattern = re.compile(
        r"DEBUG nextflow\.script\.ScriptRunner - > Execution complete -- Goodbye"
    )

    # Check each line for the pattern
    for line in lines:
        if pattern.search(line):
            return True

    return False