from math import ceil
import pandas as pd
import datetime

def generate_nextflow_config(trace_df, output_config_file):
    """
    Generates a Nextflow configuration file based on the maximum realtime and peak_rss values for each process.

    Parameters:
    - trace_df (pd.DataFrame): DataFrame containing 'process', 'realtime', and 'peak_rss' columns.
    - output_config_file (str): Path to the output configuration file.

    Returns:
    - None: Writes the configuration to the specified file.
    """

    # Calculate the maximum realtime and peak_rss for each unique process
    max_realtime_df = trace_df.groupby('process')['realtime'].max().reset_index()
    max_rss_df = trace_df.groupby('process')['peak_rss'].max().reset_index()

    # Merge the two DataFrames on the 'process' column
    merged_df = pd.merge(max_realtime_df, max_rss_df, on='process')

    # Identify processes with at least one None value in the memory column
    processes_with_none_memory = trace_df[trace_df['memory'].isnull()]['process'].unique()

    # Add a column to indicate whether to print the memory configuration
    merged_df['printMemory'] = ~merged_df['process'].isin(processes_with_none_memory)

    # Open the output file for writing
    with open(output_config_file, 'w') as file:
        # Write the process block header
        file.write(f"// Config file generated on the {datetime.datetime.now()}\n\n")
        file.write("process {\n")
        file.write("    // Default error handling strategy for all process errors.\n")
        file.write("    errorStrategy = 'retry'\n")
        file.write("    maxRetries = 3\n\n")

        # Write the time and memory configuration for each process
        file.write("    // Per-process time and memory limits corresponding to worst case execution from traces.\n")
        file.write("    // 60 extra seconds are added to account for the early termination of jobs by Slurm,\n")
        file.write("    // as requested by Nextflow job with the B:USR2 signal.\n")
        for index, row in merged_df.iterrows():
            process = row['process']
            max_time_seconds = ceil(row['realtime'].total_seconds())
            max_rss_mb = ceil(row['peak_rss'] / (1024 * 1024)) if row['peak_rss'] > 0 else 1  # Convert peak_rss to MB
            file.write(f"    withName: '{process}' {{\n")
            file.write(f"        time = {{ ({max_time_seconds} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}\n")
            if row['printMemory']:
                file.write(f"        memory = {max_rss_mb}.MB \n")
            else:
                file.write(f"        // Memory config not printed because no memory config was previously give for this process.\n")
                file.write(f"        // This is most likely the result of an alternative memory config given with clusterOptions.\n")
                file.write(f"        // memory = {max_rss_mb}.MB \n")
            file.write("    }\n\n")

        # Write the process block footer
        file.write("}\n")

# Example usage
# generate_nextflow_config(trace_df, 'path/to/output_config_file.config')
