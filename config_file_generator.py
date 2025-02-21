import pandas as pd
import datetime

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
        file.write(f"// Config file generated on the {datetime.datetime.now()}\n\n")
        file.write("process {\n")
        file.write("    // Default error handling strategy for all process errors.\n")
        file.write("    errorStrategy = 'retry'\n")
        file.write("    maxRetries = 3\n\n")

        # Write the time configuration for each process
        file.write("    // Per-process time limit corresponding to worst case execution time from traces.\n")
        file.write("    // 60 extra seconds are added to account for the early termination of jobs by Slurm,\n")
        file.write("    // as requested by Nextflow job with the B:USR2 signal.\n")
        for index, row in max_realtime_df.iterrows():
            process = row['process']
            max_time_seconds = row['realtime'].total_seconds()
            file.write(f"    withName: '{process}' {{\n")
            file.write(f"        time = {{ ({max_time_seconds} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}\n")
            file.write("    }\n\n")

        # Write the process block footer
        file.write("}\n")

# Example usage
# generate_nextflow_config(trace_df, 'path/to/output_config_file.config')
