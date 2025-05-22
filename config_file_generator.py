from math import ceil
from pathlib import Path
import pandas as pd
import datetime

from nf_trace_db_manager import NextflowTraceDBManager
from nf_trace_db_analyzer import analyze_process_execution_correlation, anova_on_process_execution_times, extract_execution_time_linear_reg, extract_execution_time_quantile_reg, get_execution_times_distribution_charasteristics


def generate_nextflow_config_from_trace(trace_df, output_config_file):
    """
    Generates a Nextflow configuration file based on the maximum realtime and peak_rss values for each process.

    Parameters:
    - trace_df (pd.DataFrame): DataFrame containing 'process', 'realtime', and 'peak_rss' columns.
    - output_config_file (str): Path to the output configuration file.

    Returns:
    - None: Writes the configuration to the specified file.
    """
    max_nb_retries = 3
    memory_margin = 1.10

    # Calculate the maximum realtime and peak_rss for each unique process
    max_realtime_df = trace_df.groupby('process')['realtime'].max().reset_index()
    max_rss_df = trace_df.groupby('process')['peak_rss'].max().reset_index()
    pre_alloc_memory_df = trace_df.groupby('process')['memory'].max().reset_index()

    # Merge the two DataFrames on the 'process' column
    merged_df = pd.merge(max_realtime_df, max_rss_df, on='process')
    merged_df = pd.merge(merged_df, pre_alloc_memory_df, on='process')

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
        file.write(f"    maxRetries = {max_nb_retries}\n\n")

        # Write the time and memory configuration for each process
        file.write("    // Per-process time and memory limits corresponding to worst case execution from traces.\n")
        file.write("    // 60 extra seconds are added to account for the early termination of jobs by Slurm,\n")
        file.write("    // as requested by Nextflow job with the B:USR2 signal.\n")
        file.write("    // A safety margin of 25% is added to the original memory constraint at first try.\n")
        file.write("    // Memory constraint is relaxed at each retry until it reach the one successfuly used in traces.\n")
        for index, row in merged_df.iterrows():
            process = row['process']
            max_time_seconds = ceil(row['realtime'].total_seconds())
            max_rss_mb = ceil(row['peak_rss'] / (1024 * 1024)) if row['peak_rss'] > 0 else 1  # Convert peak_rss to MB
            successful_mem_mb = ceil(row['memory'] / (1024 * 1024)) if row['memory'] > 0 else 1  # Convert memory to MB
            file.write(f"    withName: '{process}' {{\n")
            file.write(f"        time = {{ ({max_time_seconds} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}\n")
            # Check if observed mem footprint is greater than the max_rss_mb. In which case, just increase at each retry.
            if successful_mem_mb > max_rss_mb * memory_margin:
                memory_config = f"memory = {{ ({max_rss_mb}*{memory_margin} + ({successful_mem_mb} - {max_rss_mb}*{memory_margin}) * (task.attempt - 1) / (process.maxRetries - 1)) * 1.MB }}"
            else:
                memory_config = f"memory = {{ ({max_rss_mb} * Math.pow({memory_margin}, task.attempt))  * 1.MB }}"

            if row['printMemory']:
                file.write(f"        {memory_config}\n")
            else:
                file.write(f"        // Memory config not printed because no memory config was previously give for this process.\n")
                file.write(f"        // This is most likely the result of an alternative memory config given with clusterOptions.\n")
                file.write(f"        // {memory_config}\n")
            file.write("    }\n\n")

        # Write the process block footer
        file.write("}\n")

# Example usage
# generate_nextflow_config(trace_df, 'path/to/output_config_file.config')


def generate_nextflow_config_from_db(db_manager: NextflowTraceDBManager, output_config_file: Path):
    """
    Generate a Nextflow configuration file based on linear regression models and statistical analysis.
    """

    max_nb_retries = 3

    # Step 1: Perform ANOVA analysis
    anova_results = anova_on_process_execution_times(db_manager)

    print("\n## ANOVA results:")
    print(anova_results)

    stats_based_config = []
    model_based_config = []

    # Step 2.1: For processes and resolved processes with non-impactful traces, extract statistical information
    for _, row in anova_results.iterrows():
        if not row['trace_significant'] and not row['resolved_significant']:

            # Per process stats
            process_stats = get_execution_times_distribution_charasteristics(db_manager, row['process_name'])
            stats_based_config.append({
                'process_name': row['process_name'],
                'mean_time': process_stats['mean_time'].iloc[0],
                'std_dev_time': process_stats['std_dev_time'].iloc[0],
                'min_time': process_stats['min_time'].iloc[0],
                'max_time': process_stats['max_time'].iloc[0]
            })
        elif not row['trace_significant'] and row['resolved_significant']:
            # Per resolved process stats
            # Find process names
            rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(row['process_name'])
            for rp in rp_list:
                process_stats = get_execution_times_distribution_charasteristics(db_manager, rp.name, is_resolved_name=True)
                stats_based_config.append({
                    'process_name': rp.name,
                    'mean_time': process_stats['mean_time'].iloc[0],
                    'std_dev_time': process_stats['std_dev_time'].iloc[0],
                    'min_time': process_stats['min_time'].iloc[0],
                    'max_time': process_stats['max_time'].iloc[0]
                })

        # Step 2.2: For processes and resolved processes with impactful traces, parameter-dependent prediction
        elif row['trace_significant'] and not row['resolved_significant']:
            # Per process prediction
            model = extract_execution_time_linear_reg(db_manager, process_name=row['process_name'], is_resolved_name=False)
            model_based_config.append({
                'process_name': row['process_name'],
                'model': model
            })
        else:  # row['trace_significant'] and row['resolved_significant']:
            # Per resolved process prediction
            rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(row['process_name'])
            for rp in rp_list:
                # Find process names
                model = extract_execution_time_linear_reg(db_manager, process_name=rp.name, is_resolved_name=True)
                model_based_config.append({
                    'process_name': rp.name,
                    'model': model
                })

    # Convert the collected stats into a pandas DataFrame
    stats_based_config = pd.DataFrame(stats_based_config)
    model_based_config = pd.DataFrame(model_based_config)

    # Step 3: Generate the Nextflow configuration file
    with open(output_config_file, 'w') as file:
        # Write the process block header
        file.write(f"// Config file generated from a traces database on the {datetime.datetime.now()}\n\n")
        file.write("process {\n")
        file.write("    // Default error handling strategy for all process errors.\n")
        file.write("    errorStrategy = 'retry'\n")
        file.write(f"    maxRetries = {max_nb_retries}\n\n")

        # Print content for stats-based configuration
        file.write("    // Per-process time limits corresponding observed mean execution time plus 2 times the std dev.\n")
        file.write("    // 60 extra seconds are added to account for the early termination of jobs by Slurm,\n")
        file.write("    // as requested by Nextflow job with the B:USR2 signal.\n")

        for index, row in stats_based_config.iterrows():
            process = row['process_name']
            mean_time = row['mean_time'] / 1000.0  # Convert to seconds
            std_dev_time = row['std_dev_time'] / 1000.0  # Convert to seconds

            file.write(f"    withName: '{process}' {{\n")
            file.write(f"        time = {{ ({mean_time:.2f} + 2.0 * {std_dev_time:.2f} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}\n")
            file.write("    }\n\n")

        # Print content for parameter-dependent configuration
        file.write("    // Per-process time limits corresponding to parameter-dependent execution time.\n")
        file.write("    // 2 times the RMSE of the linear regression is added to the expression to encompass 95% of values.\n")
        file.write("    // 60 extra seconds are added to account for the early termination of jobs by Slurm,\n")
        file.write("    // as requested by Nextflow job with the B:USR2 signal.\n")

        for index, row in model_based_config.iterrows():
            process = row['process_name']
            model = row['model']['model']
            params = row['model']['selected_parameters']
            rmse = row['model']['rmse']

            # Generate the expression for the linear regression model
            # Divide by 1000.0 to convert to seconds
            expression = f"{(model.intercept_):.0f} + " + " + ".join(
                [f"({coef:.0f} * {f'params.{p}' if not params[p]['type'] == 'Boolean' else f'(params.{p} ? 1.0 : 0.0)'} )" for coef,
                    p in zip(model.coef_, params.keys())]  # converted in seconds
            )

            # Add RMSE to the expression
            expression += f" + 2.0 * {rmse:.0f}"

            # Divide by 1000.0 to convert to seconds
            expression = f"({expression}) / 1000.0"

            file.write(f"    withName: '{process}' {{\n")
            file.write(f"        time = {{ ({expression} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}\n")
            file.write("    }\n\n")

        # Footer for the process block
        file.write("}\n")


if __name__ == "__main__":
    # Example usage
    db_manager = NextflowTraceDBManager("./dat/nf_trace_db.sqlite")
    db_manager.connect()
    output_config_file = Path("./dat/celebi_from_db.config")

    generate_nextflow_config_from_db(db_manager, output_config_file)
