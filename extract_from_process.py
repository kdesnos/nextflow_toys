import re

def extract_process_parameters_hints(file_path, process_name):
    """
    Looks for a line containing '@TimePredictionParams:' followed by parameter names.

    :param file_path: Path to the Nextflow file.
    :param process_name: Name of the process to search for.
    :return: A list of parameter names (strings) found in the '@TimePredictionParams:' line.
    """
    hints = []
    inside_process = False
    brace_count = 0

    # Regex to match "params.PARAM_NAME" where PARAM_NAME contains alphanumeric characters and underscores
    param_regex = re.compile(r"params\.([a-zA-Z0-9_]+)")
    hints_regex = re.compile(r"@TimePredictionParams:\s*(.*)")

    try:
        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                # Check if the line starts the process definition
                if not inside_process and stripped_line.startswith(f"process {process_name} {{"):
                    inside_process = True
                    brace_count += 1
                    continue

                # If inside the process, track braces and look for parameters
                if inside_process:
                    # Count opening and closing braces
                    brace_count += stripped_line.count("{")
                    brace_count -= stripped_line.count("}")

                    # If the closing brace of the process is reached, exit
                    if brace_count == 0:
                        inside_process = False
                        break

                    # Look for the '@TimePredictionParams:' line
                    hints_match = hints_regex.search(stripped_line)
                    if hints_match:
                        hints_line = hints_match.group(1)
                        hints.extend(param_regex.findall(hints_line))

    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while parsing the file: {e}")

    return list(set(hints))  # Return unique hints

def extract_process_parameters(file_path, process_name):
    """
    Extracts the list of parameters used within a specific process in a Nextflow file.

    :param file_path: Path to the Nextflow file.
    :param process_name: Name of the process to search for.
    :return: A list of parameter names (strings) used within the process.
    """
    parameters = []
    inside_process = False
    brace_count = 0

    # Regex to match "params.PARAM_NAME" where PARAM_NAME contains alphanumeric characters and underscores
    param_regex = re.compile(r"params\.([a-zA-Z0-9_]+)")

    try:
        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                # Check if the line starts the process definition
                if not inside_process and stripped_line.startswith(f"process {process_name} {{"):
                    inside_process = True
                    brace_count += 1
                    continue

                # If inside the process, track braces and look for parameters
                if inside_process:
                    # Count opening and closing braces
                    brace_count += stripped_line.count("{")
                    brace_count -= stripped_line.count("}")

                    # If the closing brace of the process is reached, exit
                    if brace_count == 0:
                        inside_process = False
                        break

                    # Look for parameters using the regex
                    matches = param_regex.findall(stripped_line)
                    parameters.extend(matches)

    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while parsing the file: {e}")

    return list(set(parameters))  # Return unique parameters


if __name__ == "__main__":
    # Example usage
    file_path = "./dat/calibration.nf"  # Replace with the path to your Nextflow file
    process_name = "determine_flux_cal_solns"  # Replace with the process name you want to extract parameters from

    try:
        parameters = extract_process_parameters(file_path, process_name)
        print(f"Parameters used in process '{process_name}': {parameters}")
    except Exception as e:
        print(e)

    file_path =  './dat/correlate.nf'
    process_name = "do_correlation"
    try:
        hints = extract_process_parameters_hints(file_path, process_name)
        print(f"Time prediction parameters hints in process '{process_name}': {hints}")
    except Exception as e:
        print(e)