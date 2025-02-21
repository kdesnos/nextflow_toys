import pandas as pd

def generate_nextflow_config(trace_df, output_config_file):
    """
    Generates a Nextflow configuration file based on the maximum realtime value for each process.

    Parameters:
    - trace_df (pd.DataFrame): DataFrame containing 'process' and 'realtime' columns.
    - output_config_file (str): Path to the output configuration file.

    Returns:
    - None: Writes the configuration to the specified file.
    """

    # Calculate the maximum realtime for each unique process
    max_realtime_df = trace_df.groupby('process')['realtime'].max().reset_index()

    # Open the output file for writing
    with open(output_config_file, 'w') as file:
        # Write the process block header
        file.write("process {\n")

        # Write the time configuration for each process
        for index, row in max_realtime_df.iterrows():
            process = row['process']
            max_time_seconds = row['realtime'].total_seconds()
            file.write(f"    withName: '{process}' {{\n")
            file.write(f"        time = {{ ({max_time_seconds + 60}) * Math.pow(1.2, task.attempt)) * 1.s }}\n")
            file.write("    }\n\n")

        # Write the process block footer
        file.write("}\n")

# Example usage
# generate_nextflow_config(trace_df, 'path/to/output_config_file.config')
