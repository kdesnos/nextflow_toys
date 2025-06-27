from math import ceil
from pathlib import Path
import pandas as pd
import datetime

from nf_trace_db_manager import NextflowTraceDBManager
from nf_trace_db_analyzer import build_execution_metric_predictors


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


def generate_nextflow_config_from_db(db_manager: NextflowTraceDBManager, output_config_file: Path, models=None):
    """
    Generate a Nextflow configuration file based on linear regression models and statistical analysis.

    This function builds execution predictors using statistical analysis and linear regression models.
    It then generates a Nextflow configuration file with time and memory constraints for each process.

    :param db_manager: An instance of NextflowTraceDBManager to interact with the database.
    :param output_config_file: A Path object specifying the output file for the generated configuration.
    :param models: Optional list of dataframes [stats_based_time, model_based_time, no_time_model, stats_based_memory, model_based_memory, no_memory_model].
    :return: None. Writes the configuration to the specified file.
    """
    max_nb_retries = 5
    memory_margin = 1.10  # Same margin as in generate_nextflow_config_from_trace

    # Use pre-built models if provided, else build them
    if models is not None:
        stats_based_time, model_based_time, no_time_model, stats_based_memory, model_based_memory, no_memory_model = models
    else:
        stats_based_time, model_based_time, no_time_model = build_execution_metric_predictors(db_manager, metric="time")
        stats_based_memory, model_based_memory, no_memory_model = build_execution_metric_predictors(db_manager, metric="memory")

    # Create a consolidated dictionary of process configs
    process_configs = {}

    # Add processes from stats-based time models
    for _, row in stats_based_time.iterrows():
        process_name = row['process_name']
        if process_name not in process_configs:
            process_configs[process_name] = {
                'process_name': process_name,
                'time_model_type': None,
                'time_model': None,
                'memory_model_type': None,
                'memory_model': None,
                'allocated_memory': None  # Initialize allocated memory
            }
        process_configs[process_name]['time_model_type'] = 'stats'
        process_configs[process_name]['time_model'] = {
            'mean_time': row['mean_time'],
            'std_dev_time': row['std_dev_time']
        }

    # Add processes from model-based time models
    for _, row in model_based_time.iterrows():
        process_name = row['process_name']
        if process_name not in process_configs:
            process_configs[process_name] = {
                'process_name': process_name,
                'time_model_type': None,
                'time_model': None,
                'memory_model_type': None,
                'memory_model': None,
                'allocated_memory': None  # Initialize allocated memory
            }
        # Use class name to differentiate model types
        model_class = row['model']['model'].__class__.__name__
        if model_class == 'AmdahlLinearRegressor':
            process_configs[process_name]['time_model_type'] = 'amdhal_reg'
        else:
            process_configs[process_name]['time_model_type'] = 'linear_reg'
        process_configs[process_name]['time_model'] = row['model']

    # Add processes from stats-based memory models
    for _, row in stats_based_memory.iterrows():
        process_name = row['process_name']
        if process_name not in process_configs:
            process_configs[process_name] = {
                'process_name': process_name,
                'time_model_type': None,
                'time_model': None,
                'memory_model_type': None,
                'memory_model': None,
                'allocated_memory': None  # Initialize allocated memory
            }
        process_configs[process_name]['memory_model_type'] = 'stats'
        process_configs[process_name]['memory_model'] = {
            'mean_memory': row['mean_memory'],
            'std_dev_memory': row['std_dev_memory'],
            'max_memory': row['max_memory']
        }
        
        # Get allocated memory data for this process
        execution_data = db_manager.process_executions_manager.getExecutionMetricsForProcessAndTraces(
            process_name, is_resolved_name=row['resolved_significant'])
        
        if not execution_data.empty and 'allocated_mem' in execution_data.columns:
            # Filter out zero or null allocated memory values and get the minimum
            non_zero_allocated = execution_data[execution_data['allocated_mem'] > 0]['allocated_mem']
            if not non_zero_allocated.empty:
                min_allocated_mem = non_zero_allocated.min()
                process_configs[process_name]['allocated_memory'] = min_allocated_mem

    # Add processes from model-based memory models
    for _, row in model_based_memory.iterrows():
        process_name = row['process_name']
        if process_name not in process_configs:
            process_configs[process_name] = {
                'process_name': process_name,
                'time_model_type': None,
                'time_model': None,
                'memory_model_type': None,
                'memory_model': None,
                'allocated_memory': None  # Initialize allocated memory
            }
        process_configs[process_name]['memory_model_type'] = 'model'
        process_configs[process_name]['memory_model'] = row['model']
        
        # Get allocated memory data for this process if not already retrieved
        if process_configs[process_name]['allocated_memory'] is None:
            execution_data = db_manager.process_executions_manager.getExecutionMetricsForProcessAndTraces(
                process_name, is_resolved_name=row['resolved_significant'])
            
            if not execution_data.empty and 'allocated_mem' in execution_data.columns:
                # Filter out zero or null allocated memory values and get the minimum
                non_zero_allocated = execution_data[execution_data['allocated_mem'] > 0]['allocated_mem']
                if not non_zero_allocated.empty:
                    min_allocated_mem = non_zero_allocated.min()
                    process_configs[process_name]['allocated_memory'] = min_allocated_mem

    # Generate the Nextflow configuration file
    with open(output_config_file, 'w') as file:
        # Write the process block header
        file.write(f"// Config file generated from a traces database on the {datetime.datetime.now()}\n\n")

        # TIME CONFIGURATION TYPES
        file.write("// TIME CONFIGURATION TYPES:\n")
        file.write("// - stats-based: mean execution time plus 2.6 times the standard deviation\n")
        file.write("// - model-based: parameter-dependent linear regression model plus 2.6 times the RMSE\n\n")

        # Include all original comment blocks for stats-based configuration
        file.write("// Per-process time limits corresponding observed mean execution time plus 2.6 times the std dev.\n")
        file.write("// 60 extra seconds are added to account for the early termination of jobs by Slurm,\n")
        file.write("// as requested by Nextflow job with the B:USR2 signal.\n\n")

        # Include all original comment blocks for parameter-dependent configuration
        file.write("// Per-process time limits corresponding to parameter-dependent execution time.\n")
        file.write("// 2.6 times the RMSE of the linear regression is added to the expression to encompass 99% of values.\n")
        file.write("// 60 extra seconds are added to account for the early termination of jobs by Slurm,\n")
        file.write("// as requested by Nextflow job with the B:USR2 signal.\n\n")

        # MEMORY CONFIGURATION TYPES
        file.write("// MEMORY CONFIGURATION TYPES:\n")
        file.write("// - stats-based: memory starts at observed max with safety margin and increases toward smallest successful allocation\n")
        file.write("// - model-based: parameter-dependent linear regression model plus safety margin\n\n")

        file.write("process {\n")
        file.write("    // Default error handling strategy for all process errors.\n")
        file.write("    errorStrategy = 'retry'\n")
        file.write(f"    maxRetries = {max_nb_retries}\n\n")

        # Process each configuration entry
        for process_name, config in process_configs.items():
            file.write(f"    withName: '{process_name}' {{\n")

            # Time configuration
            if config['time_model_type'] == 'stats':
                mean_time = config['time_model']['mean_time'] / 1000.0  # Convert to seconds
                std_dev_time = config['time_model']['std_dev_time'] / 1000.0  # Convert to seconds
                file.write(
                    f"        time = {{ ({mean_time:.2f} + 2.6 * {std_dev_time:.2f} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}  // stats-based\n")

            elif config['time_model_type'] == 'amdhal_reg':
                model = config['time_model']['model']
                params = config['time_model']['selected_parameters']
                rmse = config['time_model']['rmse']
                param_names = [get_param_expr_str(p, params[p]['type']) for p in params.keys()]
                expression = model.get_expression(param_names)
                expression += f" + 2.6 * {rmse:.0f}"
                expression = f"({expression}) / 1000.0"
                file.write(f"        time = {{ ({expression} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}  // model-based\n")

            elif config['time_model_type'] == 'linear_reg':
                model = config['time_model']['model']
                params = config['time_model']['selected_parameters']
                rmse = config['time_model']['rmse']
                expression = f"{(model.intercept_):.0f} + " + " + ".join(
                    [f"({coef:.0f} * {get_param_expr_str(p, params[p]['type'])} )" for coef, p in zip(model.coef_, params.keys())]
                )
                expression += f" + 2.6 * {rmse:.0f}"
                expression = f"({expression}) / 1000.0"
                file.write(f"        time = {{ ({expression} + 60.0) * Math.pow(1.25, (task.attempt - 1)) * 1.s }}  // model-based\n")

            # Memory configuration
            if config['memory_model_type'] == 'stats':
                max_memory_mb = ceil(config['memory_model']['max_memory'] / (1024 * 1024)) if config['memory_model']['max_memory'] > 0 else 1
                allocated_memory = config['allocated_memory']
                

                allocated_memory_mb = ceil(allocated_memory / (1024 * 1024))
                if allocated_memory_mb > max_memory_mb * memory_margin:
                    memory_config = f"memory = {{ ({max_memory_mb}*{memory_margin} + ({allocated_memory_mb} - {max_memory_mb}*{memory_margin}) * (task.attempt - 1) / (process.maxRetries - 1)) * 1.MB }}  // stats-based with allocated memory"
                else:
                    # Observed memory memory is somehow more than the allocated successful max memory.
                    memory_config = f"memory = {{ ({max_memory_mb} * Math.pow({memory_margin}, task.attempt))  * 1.MB }}  // stats-based"
               
                
                file.write(f"        {memory_config}\n")

            elif config['memory_model_type'] == 'model':
                model = config['memory_model']['model']
                params = config['memory_model']['selected_parameters']
                rmse = config['memory_model']['rmse']
                allocated_memory = config['allocated_memory']

                # Explicitly differentiate between AmdahlLinearRegressor and standard LinearRegression
                if model.__class__.__name__ == 'AmdahlLinearRegressor':
                    param_names = [get_param_expr_str(p, params[p]['type']) for p in params.keys()]
                    expression = model.get_expression(param_names)
                else:
                    expression = f"{(model.intercept_):.0f} + " + " + ".join(
                        [f"({coef:.0f} * {get_param_expr_str(p, params[p]['type'])} )" for coef, p in zip(model.coef_, params.keys())]
                    )

                # Add RMSE, convert to MB, and apply Math.ceil() in the base expression
                base_expression = f"Math.ceil(({expression} + 2.6 * {rmse:.0f}) / (1024 * 1024))"
                allocated_memory_mb = ceil(allocated_memory / (1024 * 1024))
                
                # Create a model-based expression using ternary operator to handle different cases
                memory_config = (
                    f"memory = {{ "
                    f"({allocated_memory_mb} < {base_expression}*{memory_margin}) ? "
                    f"{base_expression} * Math.pow({memory_margin}, task.attempt) : "
                    f"{base_expression}*{memory_margin} + ({allocated_memory_mb} - {base_expression}*{memory_margin}) * "
                    f"(task.attempt - 1) / (process.maxRetries - 1)"
                    f" }} * 1.MB  // model-based with allocated memory"
                )

    
                file.write(f"        {memory_config}\n")

            file.write("    }\n\n")

        # Footer for the process block
        file.write("}\n")


def generate_markdown_summary(db_manager, output_markdown_file, metric="time", models=None):
    """
    Generate a markdown file summarizing the predictors obtained from build_execution_predictors.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param output_markdown_file: Path to the output markdown file.
    :param metric: The metric to analyze ("time" or "memory").
    :param models: Optional list of dataframes [stats_based_config, model_based_config, no_model], as returned by build_execution_metric_predictors.
    :return: None. Writes the summary to the specified markdown file.
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Build execution predictors for the specified metric
    if models is None:
        stats_based_config, model_based_config, no_model = build_execution_metric_predictors(db_manager, metric=metric)
    else:
        stats_based_config, model_based_config, no_model = models

    # Set up metric-specific formatting
    metric_unit = "seconds" if metric == "time" else "MB"
    metric_divisor = 1000.0 if metric == "time" else (1024 * 1024)  # Convert ms to s or bytes to MB

    # Prepare the markdown content
    markdown_lines = [
        f"# {metric.capitalize()} Predictors Summary",
        "",
        f"| Process Name | Predictor Type | Per Trace | Per Resolved | Parameters | RMSE/Std Dev ({metric_unit}) | Prediction Expression | Hinted Params |",
        "|--------------|----------------|-----------|--------------|------------|-------------:|------------------------|----------------|"
    ]

    # Add rows for stats-based predictors
    for _, row in stats_based_config.iterrows():
        process_name = row["process_name"]
        predictor_type = "std dev"
        per_trace = row["trace_significant"]
        per_resolved = row["resolved_significant"]
        parameters = "N/A"

        # Format values based on metric
        if metric == "time":
            std_dev = f"{row[f'std_dev_{metric}'] / metric_divisor:.2f}"
            expression = f"{row[f'mean_{metric}'] / metric_divisor:.2f}"
        else:  # memory
            std_dev = f"{row[f'std_dev_{metric}'] / metric_divisor:.2f}"
            mean_memory = row[f'mean_{metric}'] / metric_divisor
            max_memory = row[f'max_{metric}'] / metric_divisor
            expression = f"mean: {mean_memory:.2f}, max: {max_memory:.2f}"

        # Get hinted parameters for the process
        hinted_params = db_manager.process_params_hints_manager.getHintedParamNamesByProcessName(
            process_name, is_resolved_name=row["resolved_significant"])
        hinted_params_str = ", ".join(hinted_params) if hinted_params else "None"

        markdown_lines.append(
            f"| {process_name} | {predictor_type} | {per_trace} | {per_resolved} | {parameters} | {std_dev} | `{expression}` | {hinted_params_str} |"
        )

    # Add rows for model-based predictors
    for _, row in model_based_config.iterrows():
        process_name = row["process_name"]
        model = row["model"]
        params = model["selected_parameters"]
        per_trace = row["trace_significant"]
        per_resolved = row["resolved_significant"]
        rmse = f"{model['rmse'] / metric_divisor:.2f}"
        model_class = model["model"].__class__.__name__
        if model_class == 'AmdahlLinearRegressor':
            predictor_type = "amdhal_reg"
            param_names = [get_param_expr_str(p, params[p]['type']) for p in params.keys()]
            expression = model["model"].get_expression(param_names)
        else:
            predictor_type = "linear_reg"
            expression = model["expression"]
        
        # Replace 'inverted_nbCores' with '(1 / nbCores)' in the expression for markdown clarity
        expression = expression.replace("inverted_nbCores", "(1 / nbCores)")
        parameters = ", ".join([
            "(1 / nbCores)" if p == "inverted_nbCores" else p for p in params.keys()
        ])
        # Get hinted parameters for the process
        hinted_params = db_manager.process_params_hints_manager.getHintedParamNamesByProcessName(
            process_name, is_resolved_name=row["resolved_significant"])
        hinted_params_str = ", ".join(hinted_params) if hinted_params else "None"
        markdown_lines.append(
            f"| {process_name} | {predictor_type} | {per_trace} | {per_resolved} | {parameters} | {rmse} | `{expression}` | {hinted_params_str} |"
        )

    # Add a note for processes without models
    if not no_model.empty:
        markdown_lines.append("\n## Processes Without Models")
        markdown_lines.append("\n| Process Name | Reason | Per Trace | Per Resolved |")
        markdown_lines.append("|-------------|--------|-----------|--------------|")

        for _, row in no_model.iterrows():
            process_name = row["process_name"]
            markdown_lines.append(
                f"| {process_name} | No model | {row['trace_significant']} | {row['resolved_significant']} |"
            )

    # Write the markdown content to the file
    with open(output_markdown_file, "w") as file:
        file.write("\n".join(markdown_lines))

    print(f"{metric.capitalize()} markdown summary written to {output_markdown_file}")


def get_param_expr_str(param_name, param_type=None):
    """
    Returns the appropriate string for a parameter in a regression expression for Nextflow config.
    Handles special cases for Boolean, nbCores, and inverted_nbCores.
    """
    if param_type == "Boolean":
        return f"(params.{param_name} ? 1.0 : 0.0)"
    if param_name == "nbCores":
        return "task.cpus"
    if param_name == "inverted_nbCores":
        return "1 / task.cpus"
    return f"params.{param_name}"


if __name__ == "__main__":
    # Example usage
    db_manager = NextflowTraceDBManager("./dat/nf_trace_db.sqlite")
    db_manager.connect()
    output_config_file = Path("./dat/celebi_from_db.config")

    # Build models once for reuse
    print("Building execution time predictor...")
    time_models = build_execution_metric_predictors(db_manager, metric="time")
    print("Building execution memory predictor...")
    memory_models = build_execution_metric_predictors(db_manager, metric="memory")

    print("Generating Markdown summaries...")
    generate_markdown_summary(db_manager, "./dat/celebi_time_predictors_summary.md", metric="time", models=time_models)
    generate_markdown_summary(db_manager, "./dat/celebi_memory_predictors_summary.md", metric="memory", models=memory_models)
    print("Generating Nextflow configuration from database...")
    generate_nextflow_config_from_db(db_manager, output_config_file, models=time_models + memory_models)
