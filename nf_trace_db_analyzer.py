import pandas as pd
from nf_trace_db_manager import NextflowTraceDBManager


def analyze_process_execution_time_consistency(
    db_manager,
    tolerance=0.1,
    std_dev_threshold=15000,
    process_names=None,
    resolved_process_names=None,
    trace_names=None,
    group_by_resolved_name=False,
    group_by_trace_name=False,
    quantile=0.80  # Default quantile value
):
    """
    Analyze the consistency of execution times for processes in the Processes table, ignoring outliers.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param tolerance: The relative tolerance (default: 0.1, i.e., 10%) for the coefficient of variation.
    :param std_dev_threshold: The absolute threshold for the standard deviation (default: 15000 milliseconds or 15 seconds).
    :param process_names: Optional list of process names to filter the results. If None, all processes are included.
    :param resolved_process_names: Optional list of resolved process names to filter the results. If None, all resolved processes are included.
    :param trace_names: Optional list of trace names to filter the results. If None, all traces are included.
    :param group_by_resolved_name: If True, include resolved process names in the grouping.
    :param group_by_trace_name: If True, include trace names in the grouping.
    :param quantile: The quantile range for outlier removal (default: 0.80, meaning 0.1 and 0.9 are used for IQR).
    :return: A Pandas DataFrame with grouped results, computed statistics, and constancy criteria.
    """
    # SQL query to retrieve execution times and grouping columns
    query = f"""
        SELECT t.name AS trace_name,
            p.name AS process_name,
            rpn.name AS resolved_name,
            pe.time,
            pe.cpu
        FROM Processes p
        LEFT JOIN (
            ResolvedProcessNames rpn
            JOIN ProcessExecutions pe ON rpn.rId = pe.rId
        ) ON rpn.pId = p.pId
        LEFT JOIN Traces t ON pe.tId = t.tId
    """

    # Add WHERE clauses for filtering
    filters = []
    params = []

    if process_names is not None:
        placeholders = ",".join("?" for _ in process_names)
        filters.append(f"p.name IN ({placeholders})")
        params.extend(process_names)

    if resolved_process_names is not None:
        placeholders = ",".join("?" for _ in resolved_process_names)
        filters.append(f"rpn.name IN ({placeholders})")
        params.extend(resolved_process_names)

    if trace_names is not None:
        placeholders = ",".join("?" for _ in trace_names)
        filters.append(f"t.name IN ({placeholders})")
        params.extend(trace_names)

    if filters:
        query += " WHERE " + " AND ".join(filters)

    # Execute the query and fetch the results
    cursor = db_manager.connection.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()

    # Create a Pandas DataFrame from the query results
    column_names = ["trace_name", "process_name", "resolved_name", "time", "cpu"]
    df = pd.DataFrame(results, columns=column_names)

    # Remove outliers using the IQR method for each group
    # with the quantile range determined by the `quantile` parameter
    def remove_outliers(group):
        if (quantile == 1.00):
            # Skip outlier removal if quantile is 1.00
            return group
        lower_quantile = (1 - quantile) / 2
        upper_quantile = 1 - lower_quantile
        d1 = group.quantile(lower_quantile)
        d9 = group.quantile(upper_quantile)
        iqr = d9 - d1
        lower_bound = d1 - 1.5 * iqr
        upper_bound = d9 + 1.5 * iqr
        return group[(group >= lower_bound) & (group <= upper_bound)]

    # Determine the grouping columns
    group_cols = []
    if group_by_trace_name:
        group_cols.append("trace_name")
    if group_by_resolved_name:
        group_cols.append("resolved_name")
        group_cols.append("process_name")  # Include process_name to retain it when grouping by resolved_name
    else:
        group_cols.append("process_name")  # Default to grouping by process_name if resolved_name is not activated
    group_cols.append("cpu")  # Always group by cpu

    # Apply outlier removal to the 'time' column only, then merge back with group columns
    df["time"] = df.groupby(group_cols)["time"].transform(remove_outliers)

    # Group by the selected columns and compute statistics
    # Use dropna=False to keep Process (or ResolvedProcess) names with NaN execution times
    grouped = df.groupby(group_cols, dropna=False)["time"]
    stats = grouped.agg(
        mean_time="mean",
        std_dev_time="std",
        execution_count="count"
    ).reset_index()

    # Calculate the coefficient of variation
    stats["coefficient_of_variation"] = stats["std_dev_time"] / stats["mean_time"]

    # Add a column for the constancy criteria
    stats["is_constant"] = (
        (stats["coefficient_of_variation"] <= tolerance) | (stats["std_dev_time"] <= std_dev_threshold)
    )

    # Ensure all required columns are present in the DataFrame
    if "trace_name" not in stats:
        stats["trace_name"] = "*"
    if "resolved_name" not in stats:
        stats["resolved_name"] = "*"

    # Reorder the columns to match the specified order
    stats = stats[
        ["trace_name", "process_name", "resolved_name", "is_constant", "execution_count",
         "mean_time", "std_dev_time", "coefficient_of_variation", "cpu"]
    ]

    return stats


def identify_process_execution_time_consistency(
    db_manager,
    tolerance=0.1,
    std_dev_threshold=15000,
    quantile=0.80
):
    """
    Identify processes with consistent execution times based on the specified criteria.
    This function does not print anything and returns the analysis tables.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param tolerance: The relative tolerance for the coefficient of variation.
    :param std_dev_threshold: The absolute threshold for the standard deviation.
    :param quantile: The quantile range for outlier removal (default: 0.80, meaning 0.1 and 0.9 are used for IQR).
    :return: A tuple of DataFrames: process_consistency_analysis, per_trace_analysis, per_resolved_analysis, per_resolved_per_trace_analysis.
    """
    # Identify processes with consistent execution times
    process_consistency_analysis = analyze_process_execution_time_consistency(
        db_manager, tolerance, std_dev_threshold, quantile=quantile
    )
    executed_process_analysis = process_consistency_analysis[~process_consistency_analysis["mean_time"].isna()]

    # For processes with inconsistent execution times, check consistency per trace
    inconsistent_processes = executed_process_analysis[executed_process_analysis["is_constant"] == False][
        "process_name"
    ].unique()

    # Perform per-trace analysis only for inconsistent processes
    per_trace_analysis = analyze_process_execution_time_consistency(
        db_manager, tolerance, std_dev_threshold, process_names=inconsistent_processes, group_by_trace_name=True, quantile=quantile
    )

    # Identify processes that are consistent across all traces
    consistent_processes_per_trace = (
        per_trace_analysis.groupby("process_name")["is_constant"]
        .all()
        .reset_index()
        .rename(columns={"is_constant": "is_consistent_per_trace"})
    )

    # Filter only processes that are consistent per trace
    consistent_processes_per_trace = consistent_processes_per_trace[
        consistent_processes_per_trace["is_consistent_per_trace"] == True
    ]

    # Filter the per_trace_analysis table to include only rows with process_name in consistent_processes_per_trace
    per_trace_analysis["is_constant_per_trace"] = per_trace_analysis["process_name"].isin(consistent_processes_per_trace["process_name"])

    # Remove processes that are consistent across all traces from the inconsistent_processes list
    inconsistent_processes = [process for process in inconsistent_processes if process not in consistent_processes_per_trace["process_name"].values]

    # Identify resolved processes that are consistent across all traces
    per_resolved_analysis = analyze_process_execution_time_consistency(
        db_manager, tolerance, std_dev_threshold, process_names=inconsistent_processes, group_by_resolved_name=True, quantile=quantile
    )

    consistent_per_resolved_analysis = per_resolved_analysis[per_resolved_analysis["is_constant"] == True]

    # Identify resolved processes that are consistent within each trace
    inconsistent_resolved_processes = [resolved_process for resolved_process in per_resolved_analysis["resolved_name"]
                                       if resolved_process not in consistent_per_resolved_analysis["resolved_name"].values]

    per_resolved_per_trace_analysis = analyze_process_execution_time_consistency(
        db_manager, tolerance, std_dev_threshold, resolved_process_names=inconsistent_resolved_processes, group_by_resolved_name=True, group_by_trace_name=True, quantile=quantile
    )

    return process_consistency_analysis, per_trace_analysis, per_resolved_analysis, per_resolved_per_trace_analysis


def print_process_execution_time_consistency(
    process_consistency_analysis,
    per_trace_analysis,
    per_resolved_analysis,
    per_resolved_per_trace_analysis
):
    """
    Reproduce the same prints as the original identify_process_execution_time_consistency function.

    :param process_consistency_analysis: DataFrame with process-level consistency analysis.
    :param per_trace_analysis: DataFrame with per-trace consistency analysis.
    :param per_resolved_analysis: DataFrame with resolved process-level consistency analysis.
    :param per_resolved_per_trace_analysis: DataFrame with resolved process per-trace consistency analysis.
    """
    # Process-level consistency analysis
    executed_process_analysis = process_consistency_analysis[~process_consistency_analysis["mean_time"].isna()]
    consistent_processes = executed_process_analysis[executed_process_analysis["is_constant"] == True]
    print(f"\n## Consistent processes: {len(consistent_processes)}")
    print(executed_process_analysis.sort_values(by=["is_constant", "process_name"], ascending=[False, True]))

    # Per-trace consistency analysis
    consistent_per_trace = per_trace_analysis[per_trace_analysis["is_constant_per_trace"] == True]
    consistent_processes_count = consistent_per_trace.groupby("process_name").ngroups  # Group by process_name to avoid double-counting
    print(f"\n## Processes consistent within each individual traces: {consistent_processes_count}")
    print(per_trace_analysis.sort_values(by=["is_constant_per_trace", "process_name"], ascending=[False, True]))

    # Resolved process-level consistency analysis
    consistent_per_resolved_analysis = per_resolved_analysis[per_resolved_analysis["is_constant"] == True]
    print(f"\n## Resolved processes consistent across all traces: {len(consistent_per_resolved_analysis)}")
    print(per_resolved_analysis.sort_values(by=["is_constant", "process_name", "resolved_name"], ascending=[False, True, True]))

    # Resolved process per-trace consistency analysis
    consistent_per_resolved_per_trace = per_resolved_per_trace_analysis[per_resolved_per_trace_analysis["is_constant"] == True]
    print(f"\n## Resolved processes consistent within each individual traces: {len(consistent_per_resolved_per_trace)}")
    print(consistent_per_resolved_per_trace.sort_values(by=["is_constant", "process_name", "resolved_name"], ascending=[False, True, True]))

    inconsistent_per_resolved_per_trace = per_resolved_per_trace_analysis[per_resolved_per_trace_analysis["is_constant"] == False]
    print(f"\n## Inconsistent processes: {len(inconsistent_per_resolved_per_trace)}")
    print(inconsistent_per_resolved_per_trace.sort_values(by=["is_constant", "process_name", "resolved_name"], ascending=[False, True, True]))

    # Unexecuted processes
    unexecuted_processes = process_consistency_analysis[process_consistency_analysis["mean_time"].isna()]
    print(f"\n## Processes that are never executed: {len(unexecuted_processes)}")
    print(unexecuted_processes.sort_values(by=["process_name"], ascending=[True]))


def summarize_consistency_analysis(
    process_consistency_analysis,
    per_trace_analysis,
    per_resolved_analysis,
    per_resolved_per_trace_analysis
):
    """
    Summarize the consistency analysis into a single DataFrame.

    :param process_consistency_analysis: DataFrame with process-level consistency analysis.
    :param per_trace_analysis: DataFrame with per-trace consistency analysis.
    :param per_resolved_analysis: DataFrame with resolved process-level consistency analysis.
    :param per_resolved_per_trace_analysis: DataFrame with resolved process per-trace consistency analysis.
    :return: A single DataFrame summarizing the consistency analysis.
    """
    # Keep only non-executed processes
    not_executed = process_consistency_analysis[process_consistency_analysis["mean_time"].isna()].copy()
    not_executed["consistency_level"] = "Not Executed"

    # Keep only processes that are constant
    constant = process_consistency_analysis[process_consistency_analysis["is_constant"] == True].copy()
    constant["consistency_level"] = "Constant"

    # Keep only processes that are consistent per trace
    per_trace = per_trace_analysis[per_trace_analysis["is_constant_per_trace"] == True].copy()
    per_trace["consistency_level"] = "Per trace"

    # Keep only resolved processes that are consistent across all traces
    per_resolved = per_resolved_analysis[per_resolved_analysis["is_constant"] == True].copy()
    per_resolved["consistency_level"] = "Per resolved"

    # Keep only resolved processes that are consistent within each trace
    per_resolved_and_trace = per_resolved_per_trace_analysis[per_resolved_per_trace_analysis["is_constant"] == True].copy()
    per_resolved_and_trace["consistency_level"] = "Per resolved and trace"

    # Keep only inconsistent processes
    inconsistent_per_resolved_per_trace = per_resolved_per_trace_analysis[per_resolved_per_trace_analysis["is_constant"] == False].copy()
    inconsistent_per_resolved_per_trace["consistency_level"] = "Inconstant"

    # Combine all the DataFrames
    summary = pd.concat([not_executed, constant, per_trace, per_resolved, per_resolved_and_trace,
                        inconsistent_per_resolved_per_trace], ignore_index=True)

    # Drop unnecessary columns to avoid duplication
    summary = summary[[
        "trace_name", "process_name", "resolved_name", "consistency_level",
        "execution_count", "mean_time", "std_dev_time", "coefficient_of_variation", "cpu"
    ]]

    # Fill missing values for columns that may not exist in all tables
    summary["trace_name"] = summary["trace_name"].fillna("*")
    summary["resolved_name"] = summary["resolved_name"].fillna("*")

    return summary


def identify_variable_pipeline_numerical_parameters(db_manager, trace_names):
    """
    Finds the parameters in the PipelineParams table that are numerical and have
    different values across the specified traces.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param trace_names: List of trace names to check for variable parameters.
    :return: A list of names of parameters that are numerical and have different values across the specified traces.
    """
    # SQL query to get the list of pipeline parameters used for the specified traces
    trace_filter = ", ".join(f'"{name}"' for name in trace_names)
    query = """
        SELECT pp.name, t.name, ppv.value, pp.type
        FROM PipelineParams pp
        JOIN (
        	SELECT ppv.paramId FROM PipelineParamValues ppv 
        	JOIN Traces t ON ppv.tId = t.tId 
        	JOIN PipelineParams pp ON pp.paramId = ppv.paramId
        	WHERE pp.type IN ('Integer', 'Real', 'Boolean', 'List[Real]', 'List[Integer]', 'List[Boolean]')
        	AND t.name in ({})
        	GROUP BY ppv.paramId
        	HAVING COUNT(DISTINCT ppv.value) > 1   
        ) AS J ON pp.paramId = J.paramId
        JOIN PipelineParamValues ppv ON J.paramId = ppv.paramId 
        JOIN Traces t ON t.tId = ppv.tId
        WHERE t.name in ({});
        """.format(trace_filter, trace_filter)
    
    # Execute the query and fetch the results
    cursor = db_manager.connection.cursor() 
    cursor.execute(query)
    results = cursor.fetchall()

    #Create a DataFrame from the query results
    column_names = ["param_name", "trace_name", "value", "type"]   
    df = pd.DataFrame(results, columns=column_names)
    df = df.sort_values(by=["param_name", "trace_name", "value"], ascending=[True, True, True])

    print(df)   


if __name__ == "__main__":
    skip_load_files = True
    # List of HTML and log file paths to load into the database
    files_to_load = [
        {
            "html_file": "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_report.html",
            "log_file": "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_log.log"
        },
        {
            "html_file": "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_report.html",
            "log_file": "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_log.log"
        },
        {
            "html_file": "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_report.html",
            "log_file": "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_log.log"
        }
        # Add more file pairs as needed
    ]

    # Initialize the database manager with the path to the SQLite database
    db_manager = NextflowTraceDBManager("nf_trace_db.sqlite")

    # Establish a connection to the database
    db_manager.connect()
    print("Connected to the database.")

    if not skip_load_files:
        # Check if the database is empty and create tables if necessary
        if db_manager.isDatabaseEmpty():
            db_manager.createTables()
        else:
            db_manager.createTables(force=True)
        print("Tables created successfully.")

        # Ignore ParamInputs and ExecParams for now to speed up tests
        db_manager.process_inputs_manager = None
        db_manager.process_exec_params_manager = None

        # Iterate over the list of files and load them into the database
        for file_pair in files_to_load:
            html_file_path = file_pair["html_file"]
            log_file_path = file_pair["log_file"]
            db_manager.addAllFromFiles(html_file_path, log_file_path)

    # Print database information
    pd.set_option('display.max_rows', 500)
    db_manager.printDBInfo()
    results_python = analyze_process_execution_time_consistency(db_manager)
    print(results_python)

    results_python = analyze_process_execution_time_consistency(db_manager, process_names=["do_correlation"], group_by_resolved_name=True)
    print(results_python)

    results_python = analyze_process_execution_time_consistency(
        db_manager,
        process_names=["do_correlation"],
        group_by_resolved_name=False,
        group_by_trace_name=True)
    print(results_python)

    results_python = analyze_process_execution_time_consistency(
        db_manager,
        process_names=["do_correlation"],
        group_by_resolved_name=True,
        group_by_trace_name=True)
    print(results_python)

    process_consistency_analysis, per_trace_analysis, per_resolved_analysis, per_resolved_per_trace_analysis = identify_process_execution_time_consistency(
        db_manager,
        tolerance=0.1,
        std_dev_threshold=15000,
        quantile=0.90
    )

    print_process_execution_time_consistency(
        process_consistency_analysis,
        per_trace_analysis,
        per_resolved_analysis,
        per_resolved_per_trace_analysis
    )

    summary = summarize_consistency_analysis(
        process_consistency_analysis,
        per_trace_analysis,
        per_resolved_analysis,
        per_resolved_per_trace_analysis
    )

    # Define the custom order for the consistency_level column
    consistency_order = [
        "Constant", "Per trace", "Per resolved", "Per resolved and trace", "Inconstant", "Not Executed"
    ]
    summary["consistency_level"] = pd.Categorical(
        summary["consistency_level"], categories=consistency_order, ordered=True
    )

    # Sort the summary by the custom order of consistency_level, then by process_name and resolved_name
    sorted_summary = summary.sort_values(
        by=["consistency_level", "process_name", "resolved_name"],
        ascending=[True, True, True]
    )

    print("\n## Summary of consistency analysis:")
    print(sorted_summary)

    db_manager.close()
    print("Connection closed.")
