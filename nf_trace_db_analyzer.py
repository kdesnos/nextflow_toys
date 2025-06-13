import pandas as pd
import numpy as np
from scipy.stats import pearsonr, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning, ConvergenceWarning

from nf_trace_db_manager import NextflowTraceDBManager


def get_metric_default_threshold(metric):
    """Returns default threshold values for different metrics"""
    if metric == "time":
        return 15000  # 15 seconds in ms
    elif metric == "memory":
        return 100 * 1024 * 1024  # 100 MB in bytes
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_metric_tolerance(metric):
    """Returns default tolerance values for different metrics"""
    if metric == "time":
        return 0.1  # 10% variation
    elif metric == "memory":
        return 0.2  # 20% variation for memory
    else:
        raise ValueError(f"Unknown metric: {metric}")


def format_metric_value(value, metric):
    """Format metric value to human-readable format

    :param value: The metric value to format.
    :param metric: The type of metric ("time" or "memory").
    :return: A string with the formatted metric value, or None if the input value is NaN.
    :raises ValueError: If the metric is unknown.
    """
    if pd.isna(value):
        return None

    if metric == "time":
        # Convert milliseconds to readable format (HH:MM:SS)
        seconds = value / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    elif metric == "memory":
        # Convert bytes to appropriate unit
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(value)
        unit_index = 0

        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1

        return f"{size:.2f} {units[unit_index]}"

    else:
        raise ValueError(f"Unknown metric: {metric}")


def analyze_process_execution_metric_consistency(
    db_manager,
    metric="time",
    tolerance=None,
    threshold=None,
    process_names=None,
    resolved_process_names=None,
    trace_names=None,
    group_by_resolved_name=False,
    group_by_trace_name=False,
    quantile=0.80
):
    """
    Analyze the consistency of execution metrics for processes in the Processes table, ignoring outliers.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param metric: The metric to analyze ("time" or "memory").
    :param tolerance: The relative tolerance for the coefficient of variation. If None, uses metric default.
    :param threshold: The absolute threshold for the standard deviation. If None, uses metric default.
    :param process_names: Optional list of process names to filter the results.
    :param resolved_process_names: Optional list of resolved process names to filter the results.
    :param trace_names: Optional list of trace names to filter the results.
    :param group_by_resolved_name: If True, include resolved process names in the grouping.
    :param group_by_trace_name: If True, include trace names in the grouping.
    :param quantile: The quantile range for outlier removal (e.g., 0.80 for 10th to 90th percentile).
    :return: A Pandas DataFrame with grouped results, computed statistics, and constancy criteria.
    :raises ValueError: If the metric is not "time" or "memory".
    """
    # Validation for metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Use default values if not specified
    if tolerance is None:
        tolerance = get_metric_tolerance(metric)
    if threshold is None:
        threshold = get_metric_default_threshold(metric)

    # SQL query to retrieve execution metrics and grouping columns
    query = f"""
        SELECT t.name AS trace_name,
            p.name AS process_name,
            rpn.name AS resolved_name,
            pe.{metric},
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
    column_names = ["trace_name", "process_name", "resolved_name", metric, "cpu"]
    df = pd.DataFrame(results, columns=column_names)

    # Remove outliers using the IQR method for each group
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

    # Apply outlier removal to the metric column only, then merge back with group columns
    df[metric] = df.groupby(group_cols)[metric].transform(remove_outliers)

    # Group by the selected columns and compute statistics
    # Use dropna=False to keep Process (or ResolvedProcess) names with NaN metrics
    grouped = df.groupby(group_cols, dropna=False)[metric]

    mean_col = f"mean_{metric}"
    std_dev_col = f"std_dev_{metric}"

    stats = grouped.agg(**{
        mean_col: "mean",
        std_dev_col: "std",
        "execution_count": "count"
    }).reset_index()

    # Calculate the coefficient of variation
    stats["coefficient_of_variation"] = stats[std_dev_col] / stats[mean_col]

    # Add a column for the constancy criteria
    stats["is_constant"] = (
        (stats["coefficient_of_variation"] <= tolerance) | (stats[std_dev_col] <= threshold)
    )

    # Ensure all required columns are present in the DataFrame
    if "trace_name" not in stats:
        stats["trace_name"] = "*"
    if "resolved_name" not in stats:
        stats["resolved_name"] = "*"

    # Reorder the columns for consistent output
    stats = stats[
        ["trace_name", "process_name", "resolved_name", "is_constant", "execution_count",
         mean_col, std_dev_col, "coefficient_of_variation", "cpu"]
    ]

    return stats


def identify_process_execution_metric_consistency(
    db_manager,
    metric="time",
    tolerance=None,
    threshold=None,
    quantile=0.80
):
    """
    Identify processes with consistent execution metrics based on the specified criteria.
    This function does not print anything and returns the analysis tables.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param metric: The metric to analyze ("time" or "memory").
    :param tolerance: The relative tolerance for the coefficient of variation. If None, uses metric default.
    :param threshold: The absolute threshold for the standard deviation. If None, uses metric default.
    :param quantile: The quantile range for outlier removal.
    :return: A tuple of DataFrames:
             (process_consistency_analysis, per_trace_analysis,
              per_resolved_analysis, per_resolved_per_trace_analysis).
    """
    # Use default values if not specified
    if tolerance is None:
        tolerance = get_metric_tolerance(metric)
    if threshold is None:
        threshold = get_metric_default_threshold(metric)

    # Identify processes with consistent execution metrics
    process_consistency_analysis = analyze_process_execution_metric_consistency(
        db_manager, metric, tolerance, threshold, quantile=quantile
    )

    mean_col = f"mean_{metric}"
    executed_process_analysis = process_consistency_analysis[~process_consistency_analysis[mean_col].isna()]

    # For processes with inconsistent execution metrics, check consistency per trace
    inconsistent_processes = executed_process_analysis[executed_process_analysis["is_constant"] == False][
        "process_name"
    ].unique()

    # Perform per-trace analysis only for inconsistent processes
    per_trace_analysis = analyze_process_execution_metric_consistency(
        db_manager, metric, tolerance, threshold,
        process_names=inconsistent_processes,
        group_by_trace_name=True,
        quantile=quantile
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
    per_resolved_analysis = analyze_process_execution_metric_consistency(
        db_manager, metric, tolerance, threshold,
        process_names=inconsistent_processes,
        group_by_resolved_name=True,
        quantile=quantile
    )

    consistent_per_resolved_analysis = per_resolved_analysis[per_resolved_analysis["is_constant"] == True]

    # Identify resolved processes that are consistent within each trace
    inconsistent_resolved_processes = [resolved_process for resolved_process in per_resolved_analysis["resolved_name"]
                                       if resolved_process not in consistent_per_resolved_analysis["resolved_name"].values]

    per_resolved_per_trace_analysis = analyze_process_execution_metric_consistency(
        db_manager, metric, tolerance, threshold,
        resolved_process_names=inconsistent_resolved_processes,
        group_by_resolved_name=True,
        group_by_trace_name=True,
        quantile=quantile
    )

    return process_consistency_analysis, per_trace_analysis, per_resolved_analysis, per_resolved_per_trace_analysis


def print_process_execution_metric_consistency(
    process_consistency_analysis,
    per_trace_analysis,
    per_resolved_analysis,
    per_resolved_per_trace_analysis,
    metric="time"
):
    """
    Print a summary of the process execution metric consistency analysis.

    :param process_consistency_analysis: DataFrame with process-level consistency analysis.
    :param per_trace_analysis: DataFrame with per-trace consistency analysis.
    :param per_resolved_analysis: DataFrame with resolved process-level consistency analysis.
    :param per_resolved_per_trace_analysis: DataFrame with resolved process per-trace consistency analysis.
    :param metric: The metric that was analyzed ("time" or "memory").
    """
    mean_col = f"mean_{metric}"

    # Process-level consistency analysis
    executed_process_analysis = process_consistency_analysis[~process_consistency_analysis[mean_col].isna()]
    consistent_processes = executed_process_analysis[executed_process_analysis["is_constant"] == True]
    print(f"\n## Consistent processes for {metric}: {len(consistent_processes)}")
    print(executed_process_analysis.sort_values(by=["is_constant", "process_name"], ascending=[False, True]))

    # Per-trace consistency analysis
    consistent_per_trace = per_trace_analysis[per_trace_analysis["is_constant_per_trace"] == True]
    consistent_processes_count = consistent_per_trace.groupby("process_name").ngroups
    print(f"\n## Processes with consistent {metric} within each individual trace: {consistent_processes_count}")
    print(per_trace_analysis.sort_values(by=["is_constant_per_trace", "process_name"], ascending=[False, True]))

    # Resolved process-level consistency analysis
    consistent_per_resolved_analysis = per_resolved_analysis[per_resolved_analysis["is_constant"] == True]
    print(f"\n## Resolved processes with consistent {metric} across all traces: {len(consistent_per_resolved_analysis)}")
    print(per_resolved_analysis.sort_values(by=["is_constant", "process_name", "resolved_name"], ascending=[False, True, True]))

    # Resolved process per-trace consistency analysis
    consistent_per_resolved_per_trace = per_resolved_per_trace_analysis[per_resolved_per_trace_analysis["is_constant"] == True]
    print(f"\n## Resolved processes with consistent {metric} within each individual trace: {len(consistent_per_resolved_per_trace)}")
    print(consistent_per_resolved_per_trace.sort_values(by=["is_constant", "process_name", "resolved_name"], ascending=[False, True, True]))

    inconsistent_per_resolved_per_trace = per_resolved_per_trace_analysis[per_resolved_per_trace_analysis["is_constant"] == False]
    print(f"\n## Processes with inconsistent {metric}: {len(inconsistent_per_resolved_per_trace)}")
    print(inconsistent_per_resolved_per_trace.sort_values(by=["is_constant", "process_name", "resolved_name"], ascending=[False, True, True]))

    # Unexecuted processes
    unexecuted_processes = process_consistency_analysis[process_consistency_analysis[mean_col].isna()]
    print(f"\n## Processes that are never executed: {len(unexecuted_processes)}")
    print(unexecuted_processes.sort_values(by=["process_name"], ascending=[True]))


def summarize_metric_consistency_analysis(
    process_consistency_analysis,
    per_trace_analysis,
    per_resolved_analysis,
    per_resolved_per_trace_analysis,
    metric="time"
):
    """
    Summarize the consistency analysis into a single DataFrame.

    :param process_consistency_analysis: DataFrame with process-level consistency analysis.
    :param per_trace_analysis: DataFrame with per-trace consistency analysis.
    :param per_resolved_analysis: DataFrame with resolved process-level consistency analysis.
    :param per_resolved_per_trace_analysis: DataFrame with resolved process per-trace consistency analysis.
    :param metric: The metric that was analyzed ("time" or "memory").
    :return: A single Pandas DataFrame summarizing the consistency analysis.
    """
    mean_col = f"mean_{metric}"
    std_dev_col = f"std_dev_{metric}"

    # Keep only non-executed processes
    not_executed = process_consistency_analysis[process_consistency_analysis[mean_col].isna()].copy()
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
        "execution_count", mean_col, std_dev_col, "coefficient_of_variation", "cpu"
    ]]

    # Fill missing values for columns that may not exist in all tables
    summary["trace_name"] = summary["trace_name"].fillna("*")
    summary["resolved_name"] = summary["resolved_name"].fillna("*")

    return summary


def get_metric_distribution_characteristics(db_manager, process_name, metric="time", is_resolved_name=False, trace_names=None):
    """
    Get the distribution characteristics of execution metrics for a given process.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param process_name: The name of the process or resolved process to analyze.
    :param metric: The metric to analyze ("time" or "memory").
    :param is_resolved_name: If True, treat process_name as a resolved process name.
    :param trace_names: A list of trace names to filter the analysis. If None, all traces are included.
    :return: A Pandas DataFrame with distribution characteristics (mean, std_dev, min, max).
    :raises ValueError: If the metric is not "time" or "memory".
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # SQL query to get the execution metrics for the specified process
    query = f"""
        SELECT pe.{metric}, t.name AS trace_name
        FROM ProcessExecutions pe
        JOIN ResolvedProcessNames rpn ON pe.rId = rpn.rId
        JOIN Processes p ON rpn.pId = p.pId
        JOIN Traces t ON pe.tId = t.tId
        WHERE {"p" if not is_resolved_name else "rpn"}.name = ?
    """
    params = [process_name]

    # Add filtering for trace names if provided
    if trace_names:
        placeholders = ",".join("?" for _ in trace_names)
        query += f" AND t.name IN ({placeholders})"
        params.extend(trace_names)

    cursor = db_manager.connection.cursor()
    cursor.execute(query, params)
    execution_data = cursor.fetchall()

    # Convert execution data to a DataFrame
    execution_col = f"execution_{metric}"
    execution_df = pd.DataFrame(execution_data, columns=[execution_col, "trace_name"])

    # Calculate distribution characteristics
    mean_val = execution_df[execution_col].mean()
    std_dev_val = execution_df[execution_col].std(ddof=0)
    min_val = execution_df[execution_col].min()
    max_val = execution_df[execution_col].max()

    # Create column names based on metric
    mean_col = f"mean_{metric}"
    std_dev_col = f"std_dev_{metric}"
    min_col = f"min_{metric}"
    max_col = f"max_{metric}"

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        "process_name": [process_name],
        mean_col: [mean_val],
        std_dev_col: [std_dev_val],
        min_col: [min_val],
        max_col: [max_val]
    })

    return summary_df


def analyze_process_execution_metric_correlation(db_manager, process_name, metric="time", skip_warnings=False):
    """
    Analyze the correlation between the execution metrics of a given process and varying pipeline parameters.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param process_name: The name of the process to analyze.
    :param metric: The metric to analyze ("time" or "memory").
    :param skip_warnings: If True, suppress warnings about low data points or unique parameter values.
    :return: A Pandas DataFrame with parameter correlations, sorted by absolute correlation.
    :raises ValueError: If the metric is not "time" or "memory".
    :raises Exception: If no varying pipeline parameters or no metric data for the process is found.
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Step 1: Retrieve varying pipeline parameters and their values across traces
    varying_params = identify_variable_pipeline_numerical_parameters(db_manager)
    if varying_params.empty:
        raise Exception("No varying pipeline parameters found.")

    # Step 2: Retrieve all execution metrics for the given process
    query = f"""
        SELECT pe.{metric}, t.name AS trace_name
        FROM ProcessExecutions pe
        JOIN ResolvedProcessNames rpn ON pe.rId = rpn.rId
        JOIN Processes p ON rpn.pId = p.pId
        JOIN Traces t ON pe.tId = t.tId
        WHERE p.name = ?;
    """
    cursor = db_manager.connection.cursor()
    cursor.execute(query, (process_name,))
    execution_data = cursor.fetchall()

    if not execution_data:
        raise Exception(f"No {metric} data found for process '{process_name}'.")

    # Convert execution data to a DataFrame
    execution_col = f"execution_{metric}"
    execution_df = pd.DataFrame(execution_data, columns=[execution_col, "trace_name"])

    # Step 3: Pivot the varying_params DataFrame to have one row per trace and columns for each parameter
    param_df = varying_params.pivot(index="trace_name", columns="param_name", values="value").reset_index()

    # Step 4: Merge execution metrics with varying parameter values
    merged_df = execution_df.merge(param_df, on="trace_name", how="inner")

    # Step 5: Convert parameter values to numerical types based on the "type" column
    for param in param_df.columns:
        if param == "trace_name":
            continue

        # Check the type of the parameter and convert accordingly
        param_type = varying_params[varying_params["param_name"] == param]["type"].iloc[0]
        if param_type == "Boolean":
            # Convert Boolean values to 0 (False) and 1 (True)
            merged_df[param] = merged_df[param].map({"True": 1, "False": 0})
        elif param_type == "Integer":
            # Convert to integers
            merged_df[param] = merged_df[param].astype(int)
        elif param_type == "Real":
            # Convert to floats
            merged_df[param] = merged_df[param].astype(float)
        else:
            raise Exception(f"Unsupported parameter type '{param_type}' for parameter '{param}'.")

    # Step 6: Analyze correlation between execution metrics and each parameter
    correlations = []
    num_data_points = len(merged_df)

    # Warning if the number of data points is too small
    if not skip_warnings and num_data_points < 30:
        print(f"Warning: Low number of data points ({num_data_points}). Correlation results may not be reliable.")

    for param in param_df.columns:
        if param == "trace_name":
            continue

        # Check the number of unique values for the parameter
        num_unique_values = merged_df[param].nunique()
        if not skip_warnings and num_unique_values < 5:
            print(f"Warning: Parameter '{param}' has only {num_unique_values} unique values. Correlation may not be reliable.")

        # Calculate Pearson correlation coefficient and p-value
        correlation, p_value = pearsonr(merged_df[execution_col], merged_df[param])
        correlations.append({
            "process_name": process_name,
            "parameter": param,
            "correlation": correlation,
            "abs_correlation": abs(correlation),
            "p_value": p_value,
            "r_squared": correlation ** 2,  # Coefficient of determination
            "type": varying_params[varying_params["param_name"] == param]["type"].iloc[0]
        })

    # Convert the correlation results to a Pandas DataFrame
    correlation_df = pd.DataFrame(correlations)

    # Sort the DataFrame by absolute correlation in descending order
    correlation_df = correlation_df.sort_values(by="abs_correlation", ascending=False)

    return correlation_df


def extract_metric_linear_reg(db_manager, process_name, metric="time", top_n=3, rmse_threshold=None,
                              is_resolved_name=False, print_info=False, trace_names=None):
    """
    Extract a regression model for predicting execution metrics as a function of the best parameters.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param process_name: The name of the process to analyze.
    :param metric: The metric to analyze ("time" or "memory").
    :param top_n: Maximum number of top parameters to use for the regression model.
    :param rmse_threshold: RMSE threshold for stopping parameter addition. If None, uses metric default.
    :param is_resolved_name: If True, treat process_name as a resolved process name.
    :param print_info: If True, print the current expression and RMSE during the process.
    :param trace_names: Optional list of trace names to filter the data.
    :return: A dictionary containing the regression model, expression, RMSE, selected parameters, and metric.
             Example: {'model': LinearRegression_object, 'expression': 'str_formula', 'rmse': float_value,
                       'selected_parameters': {'param_name': {'type': 'param_type'}}, 'metric': 'metric_name'}
    :raises ValueError: If the metric is not "time" or "memory".
    :raises Exception: If no varying pipeline parameters or no metric data for the process is found.
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Use default threshold if not specified
    if rmse_threshold is None:
        rmse_threshold = get_metric_default_threshold(metric)

    # Step 1: Retrieve varying pipeline parameters
    varying_params = identify_variable_pipeline_numerical_parameters(db_manager, trace_names=trace_names)
    if varying_params.empty:
        raise Exception("No varying pipeline parameters found.")

    # Step 2: Retrieve execution metrics for the given process
    execution_metrics = db_manager.process_executions_manager.getExecutionMetricsForProcessAndTraces(
        process_name, trace_names=trace_names, is_resolved_name=is_resolved_name
    )

    if execution_metrics.empty:
        raise Exception(f"No {metric} data found for process '{process_name}'.")

    # Convert execution metrics to a DataFrame
    execution_col = f"execution_{metric}"
    # Filter the execution metrics to keep only the relevant columns
    if metric == "memory":
        execution_df = execution_metrics[["memory", "trace_name"]]
        execution_df = execution_df.rename(columns={"memory": execution_col})
    else:
        execution_df = execution_metrics[["execution_time", "trace_name"]]
        execution_df = execution_df.rename(columns={"execution_time": execution_col})

    # Step 3: Pivot the varying_params DataFrame to have one row per trace and columns for each parameter
    param_df = varying_params.pivot(index="trace_name", columns="param_name", values="value").reset_index()

    # Step 4: Merge execution metrics with varying parameter values
    merged_df = execution_df.merge(param_df, on="trace_name", how="inner")

    # Step 5: Filter parameters based on hints from ProcessParamsHintsTableManager
    hinted_params = db_manager.process_params_hints_manager.getHintedParamNamesByProcessName(
        process_name, is_resolved_name=is_resolved_name
    )
    if hinted_params:
        available_params = [param for param in param_df.columns if param in hinted_params]
    else:
        available_params = list(param_df.columns)
        available_params.remove("trace_name")

    # Step 6: Prepare data for regression
    y = merged_df[execution_col].values
    selected_params = []
    selected_params_info = {}  # Dictionary to store parameter info (name and type)
    rmse = float(2**31 - 1)  # Initialize with a large value
    expression = ""
    model = None

    # Step 7: Iteratively add parameters to the model
    for _ in range(top_n):
        best_param = None
        best_rmse = float(2**31 - 1)
        best_model = None
        best_expression = ""
        best_param_type = None

        # If no parameters are left to test, break the loop
        if not available_params:
            if print_info:
                print("No more parameters available to test. Stopping.")
            break

        # Test each available parameter
        for param in available_params:
            current_params = selected_params + [param]
            X = merged_df[current_params].values

            # Fit a linear regression model
            temp_model = LinearRegression()
            temp_model.fit(X, y)

            # Evaluate the model
            y_pred = temp_model.predict(X)
            temp_rmse = np.sqrt(mean_squared_error(y, y_pred))

            # Round RMSE values for comparison
            rounded_temp_rmse = round(temp_rmse)
            rounded_best_rmse = round(best_rmse)

            # Get the parameter type
            param_type = varying_params[varying_params["param_name"] == param]["type"].iloc[0]

            # Keep track of the best parameter, prioritizing Boolean > Integer > Real
            if (
                rounded_temp_rmse < rounded_best_rmse
                or (rounded_temp_rmse == rounded_best_rmse and best_param_type == "Real" and param_type in ["Boolean", "Integer"])
                or (rounded_temp_rmse == rounded_best_rmse and best_param_type == "Integer" and param_type == "Boolean")
            ):
                best_rmse = temp_rmse
                best_param = param
                best_model = temp_model
                best_param_type = param_type
                coefficients = best_model.coef_
                intercept = best_model.intercept_
                best_expression = f"{intercept:.2f} + " + " + ".join(
                    [f"({coef:.2f} * {p})" for coef, p in zip(coefficients, current_params)]
                )

        # Stop if no improvement in RMSE
        if best_param is None or round(best_rmse) >= round(rmse):
            if print_info:
                print(f"Best RMSE: {round(best_rmse):.0f} is not better than current RMSE: {round(rmse):.0f}. Stopping.")
            break

        # Update the selected parameters and model if the best parameter improves the RMSE
        selected_params.append(best_param)
        # Store parameter info (name and type)
        selected_params_info[best_param] = {
            "type": varying_params[varying_params["param_name"] == param]["type"].iloc[0]
        }

        available_params.remove(best_param)
        rmse = best_rmse
        model = best_model
        expression = best_expression

        # Print the current expression and RMSE
        if print_info:
            print(f"Current expression: {expression}")
            print(f"Current RMSE: {round(rmse):.0f}")
            print(f"Metric units: {metric}")

        # Stop if RMSE is below the threshold
        if rmse <= rmse_threshold:
            if print_info:
                print(f"RMSE: {round(rmse):.0f} is below the threshold of {rmse_threshold}. Stopping.")
            break

    # Return model info
    model_info = {
        "model": model,
        "expression": expression,
        "rmse": rmse,
        "selected_parameters": selected_params_info,
        "metric": metric
    }

    return model_info


def anova_on_process_execution_metrics(db_manager, metric="time", effect_threshold=None, tolerance=None, trace_names=None):
    """
    For each process, performs ANOVA on execution metrics with factors: trace_name and resolved_name.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param metric: The metric to analyze ("time" or "memory").
    :param effect_threshold: The threshold for considering an effect significant. If None, uses metric default.
    :param tolerance: The threshold for coefficient of variation. If None, uses metric default.
    :param trace_names: A list of trace names to filter the analysis.
    :return: A Pandas DataFrame with ANOVA results for each process.
             Columns include: process_name, test (type of ANOVA), F-statistics, p-values or CV,
             effect sizes, significance flags, and number of observations (n).
    :raises ValueError: If the metric is not "time" or "memory".
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Use default values if not specified
    if effect_threshold is None:
        effect_threshold = get_metric_default_threshold(metric)
    if tolerance is None:
        tolerance = get_metric_tolerance(metric)

    processes = db_manager.process_manager.getAllProcesses()
    results = []

    for process in processes:
        query = f"""
            SELECT pe.{metric}, t.name as trace_name, rpn.name as resolved_name
            FROM ProcessExecutions pe
            JOIN ResolvedProcessNames rpn ON pe.rId = rpn.rId
            JOIN Traces t ON pe.tId = t.tId
            WHERE rpn.pId = ?
        """
        params = [process.pId]

        # Add filtering for trace names if provided
        if trace_names:
            placeholders = ",".join("?" for _ in trace_names)
            query += f" AND t.name IN ({placeholders})"
            params.extend(trace_names)

        cursor = db_manager.connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=[metric, "trace_name", "resolved_name"])
        execs_per_trace = df.groupby("trace_name")[metric].count()
        execs_per_resolved = df.groupby("resolved_name")[metric].count()
        enough_per_trace = (execs_per_trace > 1).any()
        enough_per_resolved = (execs_per_resolved > 1).any()
        n_traces = df["trace_name"].nunique()
        n_resolved = df["resolved_name"].nunique()

        # Compute effect sizes
        trace_effect = (df.groupby("trace_name")[metric].mean().max()
                        - df.groupby("trace_name")[metric].mean().min()) if n_traces > 1 else None
        resolved_effect = (df.groupby("resolved_name")[metric].mean().max()
                           - df.groupby("resolved_name")[metric].mean().min()) if n_resolved > 1 else None

        effect_col = f"{metric}_effect"

        if enough_per_trace and enough_per_resolved and n_traces > 1 and n_resolved > 1:
            # Two-way ANOVA
            model = smf.ols(f'{metric} ~ C(trace_name) + C(resolved_name)', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            results.append({
                "process_name": process.name,
                "test": "anova2w",
                "trace_F": anova_table.loc["C(trace_name)", "F"],
                "trace_p_or_cv": anova_table.loc["C(trace_name)", "PR(>F)"],
                effect_col: trace_effect,
                "trace_significant": (
                    anova_table.loc["C(trace_name)", "PR(>F)"] < 0.05
                    and trace_effect is not None
                    and trace_effect >= effect_threshold
                ),
                "resolved_F": anova_table.loc["C(resolved_name)", "F"],
                "resolved_p": anova_table.loc["C(resolved_name)", "PR(>F)"],
                f"resolved_{effect_col}": resolved_effect,
                "resolved_significant": (
                    anova_table.loc["C(resolved_name)", "PR(>F)"] < 0.05
                    and resolved_effect is not None
                    and resolved_effect >= effect_threshold
                ),
                "n": len(df)
            })
        elif enough_per_trace and n_traces > 1:
            # One-way ANOVA on trace_name
            groups = [g[metric].values for _, g in df.groupby("trace_name") if len(g) > 1]
            if len(groups) > 1:
                F, p = f_oneway(*groups)
            else:
                F, p = None, None
            results.append({
                "process_name": process.name,
                "test": "anova1w",
                "trace_F": F,
                "trace_p_or_cv": p,
                effect_col: trace_effect,
                "trace_significant": (
                    p is not None and p < 0.05
                    and trace_effect is not None
                    and trace_effect >= effect_threshold
                ),
                "resolved_F": None,
                "resolved_p": None,
                f"resolved_{effect_col}": None,
                "resolved_significant": None,
                "n": len(df)
            })
        elif enough_per_resolved and n_resolved > 1:
            # One-way ANOVA on resolved_name
            groups = [g[metric].values for _, g in df.groupby("resolved_name") if len(g) > 1]
            if len(groups) > 1:
                F, p = f_oneway(*groups)
            else:
                F, p = None, None
            results.append({
                "process_name": process.name,
                "test": "anova1w",
                "trace_F": None,
                "trace_p_or_cv": None,
                effect_col: None,
                "trace_significant": None,
                "resolved_F": F,
                "resolved_p": p,
                f"resolved_{effect_col}": resolved_effect,
                "resolved_significant": (
                    p is not None and p < 0.05
                    and resolved_effect is not None
                    and resolved_effect >= effect_threshold
                ),
                "n": len(df)
            })
        else:
            # Not enough data for ANOVA, use standard deviation
            std_dev = df[metric].std()
            coeff_of_variation = std_dev / df[metric].mean() if df[metric].mean() != 0 else float('inf')
            results.append({
                "process_name": process.name,
                "test": "CV",
                "trace_F": None,
                "trace_p_or_cv": coeff_of_variation,
                effect_col: std_dev,
                "trace_significant": coeff_of_variation > tolerance and std_dev > effect_threshold,
                "resolved_F": None,
                "resolved_p": None,
                f"resolved_{effect_col}": None,
                "resolved_significant": None,
                "n": len(df)
            })

    return pd.DataFrame(results)


def build_execution_metric_predictors(db_manager, metric="time", trace_names=None):
    """
    Build execution metric predictors based on ANOVA analysis and linear regression models.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param metric: The metric to predict ("time" or "memory").
    :param trace_names: A list of trace names to filter the analysis.
    :return: A tuple of two DataFrames: (stats_based_config, model_based_config).
             stats_based_config: DataFrame with statistical summaries for processes/resolved processes
                                 where trace effects are not significant.
             model_based_config: DataFrame with linear regression model information for processes/resolved
                                 processes where trace effects are significant.
    :raises ValueError: If the metric is not "time" or "memory".
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Step 1: Perform ANOVA analysis
    anova_results = anova_on_process_execution_metrics(db_manager, metric=metric, trace_names=trace_names)

    stats_based_config = []
    model_based_config = []

    mean_col = f"mean_{metric}"
    std_dev_col = f"std_dev_{metric}"
    min_col = f"min_{metric}"
    max_col = f"max_{metric}"
    effect_col = f"{metric}_effect"

    # Step 2.1: For processes and resolved processes with non-impactful traces, extract statistical information
    for _, row in anova_results.iterrows():
        if not row['trace_significant'] and not row['resolved_significant']:
            # Per process stats
            process_stats = get_metric_distribution_characteristics(
                db_manager, row['process_name'], metric=metric, trace_names=trace_names
            )
            stats_based_config.append({
                'process_name': row['process_name'],
                mean_col: process_stats[mean_col].iloc[0],
                std_dev_col: process_stats[std_dev_col].iloc[0],
                min_col: process_stats[min_col].iloc[0],
                max_col: process_stats[max_col].iloc[0],
                'trace_significant': row['trace_significant'],
                'resolved_significant': row['resolved_significant'],
                'metric': metric
            })
        elif not row['trace_significant'] and row['resolved_significant']:
            # Per resolved process stats
            rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(row['process_name'])
            for rp in rp_list:
                process_stats = get_metric_distribution_characteristics(
                    db_manager, rp.name, metric=metric, is_resolved_name=True, trace_names=trace_names
                )
                stats_based_config.append({
                    'process_name': rp.name,
                    mean_col: process_stats[mean_col].iloc[0],
                    std_dev_col: process_stats[std_dev_col].iloc[0],
                    min_col: process_stats[min_col].iloc[0],
                    max_col: process_stats[max_col].iloc[0],
                    'trace_significant': row['trace_significant'],
                    'resolved_significant': row['resolved_significant'],
                    'metric': metric
                })

        # Step 2.2: For processes and resolved processes with impactful traces, parameter-dependent prediction
        elif row['trace_significant'] and not row['resolved_significant']:
            # Per process prediction
            model = extract_metric_linear_reg(
                db_manager,
                process_name=row['process_name'],
                metric=metric,
                is_resolved_name=False,
                trace_names=trace_names
            )
            model_based_config.append({
                'process_name': row['process_name'],
                'model': model,
                'trace_significant': row['trace_significant'],
                'resolved_significant': row['resolved_significant'],
                'metric': metric
            })
        else:  # row['trace_significant'] and row['resolved_significant']:
            # Per resolved process prediction
            rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(row['process_name'])
            for rp in rp_list:
                model = extract_metric_linear_reg(
                    db_manager,
                    process_name=rp.name,
                    metric=metric,
                    is_resolved_name=True,
                    trace_names=trace_names
                )
                model_based_config.append({
                    'process_name': rp.name,
                    'model': model,
                    'trace_significant': row['trace_significant'],
                    'resolved_significant': row['resolved_significant'],
                    'metric': metric
                })

    # Convert the collected stats into DataFrames
    stats_based_config = pd.DataFrame(stats_based_config)
    model_based_config = pd.DataFrame(model_based_config)

    return stats_based_config, model_based_config


def predict_metric_from_param_values(db_manager, stats_based_config, model_based_config, pipeline_params_df, metric="time"):
    """
    Predicts execution metrics for processes based on pipeline parameters.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param stats_based_config: DataFrame from build_execution_metric_predictors with statistics.
    :param model_based_config: DataFrame from build_execution_metric_predictors with regression models.
    :param pipeline_params_df: DataFrame with parameter values (columns: 'param_name', 'reformatted_value', 'type').
    :param metric: The metric to predict ("time" or "memory").
    :return: DataFrame with process names and their predicted metrics, RMSE, confidence interval, and formatted value.
    :raises ValueError: If the metric is not "time" or "memory", or if config DataFrames are for a different metric.
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Verify that configs are for the requested metric
    if stats_based_config is not None and not stats_based_config.empty:
        if 'metric' in stats_based_config.columns and stats_based_config['metric'].iloc[0] != metric:
            raise ValueError(f"stats_based_config is for {stats_based_config['metric'].iloc[0]}, not {metric}")

    if model_based_config is not None and not model_based_config.empty:
        if 'metric' in model_based_config.columns and model_based_config['metric'].iloc[0] != metric:
            raise ValueError(f"model_based_config is for {model_based_config['metric'].iloc[0]}, not {metric}")

    # Column names for the specific metric
    mean_col = f"mean_{metric}"
    std_dev_col = f"std_dev_{metric}"
    predicted_col = f"predicted_{metric}"
    formatted_col = f"formatted_{metric}"

    # Convert pipeline parameters to a dictionary for easier lookup
    param_values = {}
    for _, row in pipeline_params_df.iterrows():
        param_name = row['param_name']
        # Always use reformatted value (assuming it's always available)
        value = row['reformatted_value']

        # Only convert Boolean values to 0/1 format for model compatibility
        if row['type'] == 'Boolean' and isinstance(value, bool):
            value = 1 if value else 0

        param_values[param_name] = value

    predictions = []
    processed_stats = []  # Track processes that already have statistics-based predictions

    # Process statistics-based predictions (constant metrics)
    if stats_based_config is not None and not stats_based_config.empty:
        for _, row in stats_based_config.iterrows():
            process_name = row['process_name']
            mean_value = row[mean_col]
            std_dev = row[std_dev_col]
            trace_significant = row.get('trace_significant', False)
            resolved_significant = row.get('resolved_significant', False)

            processed_stats.append(process_name)

            prediction = {
                'process_name': process_name,
                predicted_col: mean_value,
                'rmse': std_dev,  # Use std_dev as the uncertainty measure
                'confidence_interval': [
                    max(0, mean_value - 2 * std_dev),  # Lower bound (prevent negative values)
                    mean_value + 2 * std_dev           # Upper bound
                ]
            }

            # Add formatted representation
            prediction[formatted_col] = format_metric_value(mean_value, metric)
            predictions.append(prediction)

            # Add predictions for resolved processes if this is a process-level statistic
            if db_manager is not None and not trace_significant and not resolved_significant:
                # Get all resolved process names for this process
                rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(process_name)

                for rp in rp_list:
                    # Skip if this resolved process already has statistics
                    if rp.name in processed_stats:
                        continue

                    # Add a prediction for this resolved process using the same statistics
                    prediction = {
                        'process_name': rp.name,
                        predicted_col: mean_value,
                        'rmse': std_dev,
                        'confidence_interval': [
                            max(0, mean_value - 2 * std_dev),
                            mean_value + 2 * std_dev
                        ]
                    }

                    # Add formatted representation
                    prediction[formatted_col] = format_metric_value(mean_value, metric)
                    predictions.append(prediction)
                    processed_stats.append(rp.name)

    # Process model-based predictions (variable metrics)
    if model_based_config is not None and not model_based_config.empty:
        processed_models = []

        for _, row in model_based_config.iterrows():
            process_name = row['process_name']
            model_info = row['model']
            trace_significant = row['trace_significant']
            resolved_significant = row['resolved_significant']
            processed_models.append(process_name)

            # Add prediction for this process
            add_model_metric_prediction(predictions, process_name, model_info, param_values, metric)

            # Add predictions for resolved processes if this is a process-level model
            if db_manager is not None and trace_significant and not resolved_significant:
                # Get all resolved process names for this process
                rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(process_name)

                for rp in rp_list:
                    # Skip if this resolved process already has a model
                    if rp.name in processed_models:
                        continue

                    # Add a prediction for this resolved process using the same model
                    add_model_metric_prediction(predictions, rp.name, model_info, param_values, metric)

    # Convert to DataFrame and return
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


def add_model_metric_prediction(predictions, process_name, model_info, param_values, metric="time"):
    """
    Helper function to add a model-based prediction to the predictions list.
    Modifies the 'predictions' list in-place.

    :param predictions: List to append the prediction to.
    :param process_name: The name of the process.
    :param model_info: Dictionary containing model information (model, rmse, selected_parameters, metric).
    :param param_values: Dictionary of parameter values for prediction.
    :param metric: The metric to predict ("time" or "memory").
    :raises ValueError: If the metric is not "time" or "memory", or if model_info's metric mismatches.
    """
    # Validate metric parameter
    if metric not in ["time", "memory"]:
        raise ValueError("Metric must be either 'time' or 'memory'")

    # Check if model_info's metric matches requested metric
    if 'metric' in model_info and model_info['metric'] != metric:
        raise ValueError(f"Model is for {model_info['metric']}, not {metric}")

    model = model_info['model']
    rmse = model_info['rmse']
    selected_params = list(model_info['selected_parameters'].keys())

    predicted_col = f"predicted_{metric}"
    formatted_col = f"formatted_{metric}"

    # Prepare input array for prediction
    X_pred = np.array([[param_values.get(param, 0) for param in selected_params]])

    # Make prediction
    predicted_value = model.predict(X_pred)[0]

    prediction = {
        'process_name': process_name,
        predicted_col: predicted_value,
        'rmse': rmse,
        'confidence_interval': [
            max(0, predicted_value - 2 * rmse),  # Lower bound (prevent negative values)
            predicted_value + 2 * rmse           # Upper bound
        ]
    }

    # Add formatted representation
    prediction[formatted_col] = format_metric_value(predicted_value, metric)
    predictions.append(prediction)


def identify_variable_pipeline_numerical_parameters(db_manager, trace_names=None):
    """
    Finds the parameters in the PipelineParams table that are numerical and have
    different values across all traces. Converts parameter values to their appropriate
    numerical types based on their type.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param trace_names: Optional list of trace names to filter the results.
    :return: A Pandas DataFrame with the parameters that are variable across the specified traces.
             Columns: "param_name", "trace_name", "value" (converted to numerical type), "type".
    :raises Exception: If an unsupported parameter type is encountered.
    """
    # SQL query to get the list of pipeline parameters used for the specified traces
    query = """
        SELECT pp.name AS param_name, t.name AS trace_name, ppv.value, pp.type
        FROM PipelineParams pp
        JOIN (
            SELECT ppv.paramId FROM PipelineParamValues ppv
            JOIN Traces t ON ppv.tId = t.tId
            JOIN PipelineParams pp ON pp.paramId = ppv.paramId
            WHERE pp.type IN ('Integer', 'Real', 'Boolean')
            GROUP BY ppv.paramId
            HAVING COUNT(DISTINCT ppv.value) > 1
        ) AS J ON pp.paramId = J.paramId
        JOIN PipelineParamValues ppv ON J.paramId = ppv.paramId
        JOIN Traces t ON t.tId = ppv.tId
    """

    # Add filtering for trace names if provided
    params = []
    if trace_names:
        placeholders = ",".join("?" for _ in trace_names)
        query += f" WHERE t.name IN ({placeholders})"
        params.extend(trace_names)

    # Execute the query and fetch the results
    cursor = db_manager.connection.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()

    # Create a DataFrame from the query results
    column_names = ["param_name", "trace_name", "value", "type"]
    df = pd.DataFrame(results, columns=column_names)

    # Convert parameter values to their appropriate numerical types
    def convert_value(row):
        if row["type"] == "Boolean":
            return 1 if row["value"] == "True" else 0
        elif row["type"] == "Integer":
            return int(row["value"])
        elif row["type"] == "Real":
            return float(row["value"])
        else:
            raise Exception(f"Unsupported parameter type '{row['type']}' for parameter '{row['param_name']}'.")

    df["value"] = df.apply(convert_value, axis=1)

    return df


def predict_execution_metrics(db_manager, pipeline_params_df, metrics=None):
    """
    Predict execution metrics (time and/or memory) for processes based on pipeline parameters.

    :param db_manager: NextflowTraceDBManager instance.
    :param pipeline_params_df: DataFrame with pipeline parameters (columns: 'param_name', 'reformatted_value', 'type').
    :param metrics: List of metrics to predict (e.g., ["time", "memory"]). Defaults to ["time"].
    :return: Dictionary of DataFrames with predictions for each metric.
             Keys are metric names (e.g., "time", "memory"), values are DataFrames
             containing predictions for that metric.
    :raises ValueError: If an unsupported metric is requested.
    """
    if metrics is None:
        metrics = ["time"]

    results = {}

    for metric in metrics:
        if metric not in ["time", "memory"]:
            raise ValueError(f"Unsupported metric: {metric}. Must be 'time' or 'memory'.")

        # Build predictors for this metric
        stats_config, model_config = build_execution_metric_predictors(db_manager, metric=metric)

        # Generate predictions
        predictions = predict_metric_from_param_values(
            db_manager,
            stats_config,
            model_config,
            pipeline_params_df,
            metric=metric
        )

        results[metric] = predictions

    return results


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
        },
        {
            "html_file": "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_report.html",
            "log_file": "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_log.log"
        },
        {
            "html_file": "./dat/250515_241226_CELEBI/karol_241226_ult_2025-05-15_13_41_42_report.html",
            "log_file": "./dat/250515_241226_CELEBI/karol_241226_ult_2025-05-15_13_41_42_log.log"
        }
        # Add more file pairs as needed
    ]

    # Initialize the database manager with the path to the SQLite database
    db_manager = NextflowTraceDBManager("./dat/nf_trace_db.sqlite")

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

    # Setup pandas display params
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_rows", None)

    # Print database information
    db_manager.printDBInfo()

    # Run the same analyses as in the original file but using the multi-metric functions
    print("\n## Running analyses with time metric:")

    # ANOVA on process execution times
    anova_results = anova_on_process_execution_metrics(db_manager, metric="time")
    print("\n## ANOVA results on process execution times:")
    anova_results = anova_results.sort_values(by=["process_name"], ascending=[True])
    print(anova_results)

    # Correlation analysis for "generate_binconfig"
    df = analyze_process_execution_metric_correlation(db_manager, "generate_binconfig", metric="time")
    print("\n## Correlation results:")
    print(df)

    # Linear regression for "generate_binconfig"
    extract_metric_linear_reg(
        db_manager,
        process_name="generate_binconfig",
        metric="time",
        top_n=3,
        rmse_threshold=10000,
        print_info=True
    )

    # Linear regression for specific resolved processes
    extract_metric_linear_reg(
        db_manager,
        process_name="fcal1:corr_fcal:do_correlation",
        metric="time",
        top_n=9,
        rmse_threshold=10000,
        is_resolved_name=True,
        print_info=True
    )

    extract_metric_linear_reg(
        db_manager,
        process_name="frb:corr_field:do_correlation",
        metric="time",
        top_n=9,
        rmse_threshold=10000,
        is_resolved_name=True,
        print_info=True
    )

    # Example of how to use the predict_execution_metrics function
    print("\n## Example of predicting execution metrics:")
    # This would require pipeline_params_df to be defined
    # predictions = predict_execution_metrics(db_manager, pipeline_params_df, metrics=["time"])
    # print(predictions["time"])

    db_manager.close()
    print("Connection closed.")
