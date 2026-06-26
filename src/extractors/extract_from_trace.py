import pandas as pd
import re

def parse_duration(duration_str):
    """
    Parses a duration string with mixed units (e.g., '22ms', '5m 2.5s', '2h 3.25m 22s')
    and converts it into a pandas Timedelta object.

    :param duration_str: The duration string to parse.
    :type duration_str: str
    :return: A pandas Timedelta object representing the duration.
    :rtype: pd.Timedelta
    """
    if not isinstance(duration_str, str):
        return pd.NaT

    # Define regex to extract hours, minutes, seconds, and milliseconds (supporting floating-point values)
    pattern = re.compile(
        r"((?P<hours>\d+(\.\d+)?)h)?\s*((?P<minutes>\d+(\.\d+)?)m)?\s*((?P<seconds>\d+(\.\d+)?)s)?\s*((?P<milliseconds>\d+(\.\d+)?)ms)?"
    )
    match = pattern.fullmatch(duration_str.strip())
    if not match:
        return pd.NaT

    # Extract matched groups and convert to floats (default to 0 if not present)
    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    milliseconds = float(match.group("milliseconds") or 0)

    # Convert to pandas Timedelta
    return pd.Timedelta(
        hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
    )


def extract_trace_data_from_lines(lines):
    """
    Extracts trace data from a list of lines and converts it into a pandas DataFrame.

    :param lines: List of strings representing lines from a file.
    :type lines: list of str
    :return: A pandas DataFrame containing the parsed and typed trace data.
    :rtype: pd.DataFrame
    """
    # Ensure there are lines to process
    if not lines or len(lines) < 1:
        return None

    # Define the expected header columns
    expected_columns = [
        "task_id", "hash", "native_id", "name", "status", "exit", "submit",
        "duration", "realtime", "%cpu", "peak_rss", "peak_vmem", "rchar", "wchar"
    ]

    # Check if the first line matches the expected header
    first_line = lines[0].strip().split("\t")
    if all(col in expected_columns for col in first_line):
        header = first_line
        data_lines = [line.strip().split("\t") for line in lines[1:] if line.strip()]
    else:
        header = expected_columns
        data_lines = [line.strip().split("\t") for line in lines if line.strip()]

    # Create a DataFrame from the data
    df = pd.DataFrame(data_lines, columns=header)

    # Clean up the "name" column to remove optional numbers in parentheses
    if "name" in df.columns:
        df["name"] = df["name"].str.replace(r"\s*\(\d+\)$", "", regex=True)

    # Define column conversion rules
    conversion_rules = {
        'datetime': ['submit'],
        'timedelta': ['duration', 'realtime'],
        'integer': ['task_id', 'native_id', 'exit'],
        'float': ['%cpu'],
        'memory': ['peak_rss', 'peak_vmem', 'rchar', 'wchar']
    }

    # Apply conversions based on the defined rules
    for col_type, columns in conversion_rules.items():
        for column in columns:
            if column in df.columns:
                if col_type == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif col_type == 'timedelta':
                    df[column] = df[column].apply(parse_duration)
                elif col_type == 'integer':
                    df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                elif col_type == 'float':
                    df[column] = pd.to_numeric(df[column].str.replace('%', ''), errors='coerce') / 100.0
                elif col_type == 'memory':
                    df[column] = pd.to_numeric(df[column].str.replace('[a-zA-Z]', '', regex=True), errors='coerce')

    return df