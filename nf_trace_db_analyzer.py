import pandas as pd
from nf_trace_db_manager import NextflowTraceDBManager
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning, ConvergenceWarning


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


def identify_variable_pipeline_numerical_parameters(db_manager, trace_names=None):
    """
    Finds the parameters in the PipelineParams table that are numerical and have
    different values across all traces. Converts parameter values to their appropriate
    numerical types based on their type.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param trace_names: Optional list of trace names to filter the results. If None, all traces are included.
    :return: A DataFrame with the parameters that are variable across the specified traces,
             with values converted to their appropriate numerical types.
    """
    # SQL query to get the list of pipeline parameters used for the specified traces
    query = """
        SELECT pp.name AS param_name, t.name AS trace_name, ppv.value, pp.type
        FROM PipelineParams pp
        JOIN (
            SELECT ppv.paramId FROM PipelineParamValues ppv
            JOIN Traces t ON ppv.tId = t.tId
            JOIN PipelineParams pp ON pp.paramId = ppv.paramId
            WHERE pp.type IN ('Integer', 'Real', 'Boolean') -- For now, ignore 'List[Real]', 'List[Integer]', 'List[Boolean]'
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

    # Return the DataFrame with variable parameters and converted values
    return df


def get_execution_times_distribution_charasteristics(db_manager, process_name, is_resolved_name=False, trace_names=None):
    """
    Get the distribution characteristics of execution times for a given process.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param process_name: The name of the process or resolved process to analyze.
    :param is_resolved_name: If True, treat process_name as a resolved process name.
    :param trace_names: A list of trace names to filter the analysis. If None, all traces are included.
    :return: A Pandas DataFrame with distribution characteristics.
    """
    # SQL query to get the execution times for the specified process
    query = f"""
        SELECT pe.time, t.name AS trace_name
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
    execution_df = pd.DataFrame(execution_data, columns=["execution_time", "trace_name"])

    # Calculate distribution characteristics
    mean_time = execution_df["execution_time"].mean()
    std_dev_time = execution_df["execution_time"].std(ddof=0)
    min_time = execution_df["execution_time"].min()
    max_time = execution_df["execution_time"].max()

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        "process_name": [process_name],
        "mean_time": [mean_time],
        "std_dev_time": [std_dev_time],
        "min_time": [min_time],
        "max_time": [max_time]
    })

    return summary_df


def analyze_process_execution_correlation(db_manager, process_name, skip_warnings=False):
    """
    Analyze the correlation between the execution times of a given process and varying pipeline parameters.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param process_name: The name of the process to analyze.
    :param skip_warnings: If True, suppress warnings about low data points or unique parameter values.
    :return: A Pandas DataFrame with parameter names, correlation coefficients, p-values, and R² values, sorted by correlation.
    """
    # Step 1: Retrieve varying pipeline parameters and their values across traces
    varying_params = identify_variable_pipeline_numerical_parameters(db_manager)
    if varying_params.empty:
        raise Exception("No varying pipeline parameters found.")

    # Step 2: Retrieve all execution times for the given process
    query = """
        SELECT pe.time, t.name AS trace_name
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
        raise Exception(f"No execution times found for process '{process_name}'.")

    # Convert execution data to a DataFrame
    execution_df = pd.DataFrame(execution_data, columns=["execution_time", "trace_name"])

    # Step 3: Pivot the varying_params DataFrame to have one row per trace and columns for each parameter
    param_df = varying_params.pivot(index="trace_name", columns="param_name", values="value").reset_index()

    # Step 4: Merge execution times with varying parameter values
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

    # Step 6: Analyze correlation between execution times and each parameter
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
        correlation, p_value = pearsonr(merged_df["execution_time"], merged_df[param])
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


def extract_execution_time_linear_reg(db_manager, process_name, top_n=3, rmse_threshold=15000,
                                      is_resolved_name=False, print_info=False, trace_names=None):
    """
    Extract an expression for predicting execution time as a function of the best parameters.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param process_name: The name of the process to analyze.
    :param top_n: Maximum number of top parameters to use for the regression model.
    :param rmse_threshold: RMSE threshold for stopping the iterative parameter addition.
    :param is_resolved_name: If True, treat process_name as a resolved process name.
    :param print_info: If True, print the current expression and RMSE during the process.
    :param trace_names: Optional list of trace names to filter the data. If None, all traces are included.
    :return: A dictionary containing the regression model, the expression, and evaluation metrics.
    """
    # Step 1: Retrieve varying pipeline parameters
    varying_params = identify_variable_pipeline_numerical_parameters(db_manager, trace_names=trace_names)
    if varying_params.empty:
        raise Exception("No varying pipeline parameters found.")

    # Step 2: Retrieve execution times for the given process
    execution_times = db_manager.process_executions_manager.getExecutionTimesForProcessAndTraces(
        process_name, trace_names=trace_names, is_resolved_name=is_resolved_name
    )

    if execution_times.empty:
        raise Exception(f"No execution times found for process '{process_name}'.")

    # Convert execution times to a DataFrame
    execution_df = execution_times[["execution_time", "trace_name"]]

    # Step 3: Pivot the varying_params DataFrame to have one row per trace and columns for each parameter
    param_df = varying_params.pivot(index="trace_name", columns="param_name", values="value").reset_index()

    # Step 4: Merge execution times with varying parameter values
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
    y = merged_df["execution_time"].values
    selected_params = []
    selected_params_info = {}  # Dictionary to store parameter info (name and type)
    all_expressions = []  # Store all expressions created during iterations
    all_rmse = []  # Store all RMSE values
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
            "type": varying_params[varying_params["param_name"] == best_param]["type"].iloc[0]
        }

        available_params.remove(best_param)
        rmse = best_rmse
        model = best_model
        expression = best_expression

        # Store this iteration's expression and RMSE
        all_expressions.append(expression)
        all_rmse.append(rmse)

        # Print the current expression and RMSE
        if print_info:
            print(f"Current expression: {expression}")
            print(f"Current RMSE: {round(rmse):.0f}")

        # Stop if RMSE is below the threshold
        if rmse <= rmse_threshold:
            if print_info:
                print(f"RMSE: {round(rmse):.0f} is below the threshold of {rmse_threshold}. Stopping.")
            break

    return {
        "model": model,
        "expression": expression,
        "all_expressions": all_expressions,
        "all_rmse": all_rmse,
        "rmse": rmse,
        "selected_parameters": selected_params_info  # Now contains parameter names and types
    }


def anova_on_process_execution_times(db_manager, effect_threshold_ms=15000, tolerance=0.1, trace_names=None):
    """
    For each process in the Processes table, performs a two-way ANOVA on execution times,
    with factors: run (trace_name) and resolved process name (resolved_name).
    Falls back to one-way ANOVA or standard deviation check if only one factor is possible.
    Returns a DataFrame summarizing the results for each process.
    :param db_manager: An instance of NextflowTraceDBManager.
    :param effect_threshold_ms: The threshold in milliseconds for considering an effect significant.
    :param tolerance: The threshold for coefficient of variation for the fallback test (default: 0.1).
    :param trace_names: A list of trace names to filter the analysis. If None, all traces are included.
    :return: A Pandas DataFrame with results for each process.
    """
    processes = db_manager.process_manager.getAllProcesses()
    results = []

    for process in processes:
        query = """
            SELECT pe.time, t.name as trace_name, rpn.name as resolved_name
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

        df = pd.DataFrame(rows, columns=["time", "trace_name", "resolved_name"])
        execs_per_trace = df.groupby("trace_name")["time"].count()
        execs_per_resolved = df.groupby("resolved_name")["time"].count()
        enough_per_trace = (execs_per_trace > 1).any()
        enough_per_resolved = (execs_per_resolved > 1).any()
        n_traces = df["trace_name"].nunique()
        n_resolved = df["resolved_name"].nunique()

        # Compute effect sizes
        trace_effect = df.groupby("trace_name")["time"].mean().max() - df.groupby("trace_name")["time"].mean().min() if n_traces > 1 else None
        resolved_effect = df.groupby("resolved_name")["time"].mean().max(
        ) - df.groupby("resolved_name")["time"].mean().min() if n_resolved > 1 else None

        if enough_per_trace and enough_per_resolved and n_traces > 1 and n_resolved > 1:
            # Two-way ANOVA
            model = smf.ols('time ~ C(trace_name) + C(resolved_name)', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            results.append({
                "process_name": process.name,
                "test": "anova2w",
                "trace_F": anova_table.loc["C(trace_name)", "F"],
                "trace_p_or_cv": anova_table.loc["C(trace_name)", "PR(>F)"],
                "trace_effect_ms": trace_effect,
                "trace_significant": (
                    anova_table.loc["C(trace_name)", "PR(>F)"] < 0.05 and trace_effect is not None and trace_effect >= effect_threshold_ms
                ),
                "resolved_F": anova_table.loc["C(resolved_name)", "F"],
                "resolved_p": anova_table.loc["C(resolved_name)", "PR(>F)"],
                "resolved_effect_ms": resolved_effect,
                "resolved_significant": (
                    anova_table.loc["C(resolved_name)", "PR(>F)"] < 0.05 and resolved_effect is not None and resolved_effect >= effect_threshold_ms
                ),
                "n": len(df)
            })
        elif enough_per_trace and n_traces > 1:
            # One-way ANOVA on trace_name
            groups = [g["time"].values for _, g in df.groupby("trace_name") if len(g) > 1]
            if len(groups) > 1:
                F, p = f_oneway(*groups)
            else:
                F, p = None, None
            results.append({
                "process_name": process.name,
                "test": "anova1w",
                "trace_F": F,
                "trace_p_or_cv": p,
                "trace_effect_ms": trace_effect,
                "trace_significant": (
                    p is not None and p < 0.05 and trace_effect is not None and trace_effect >= effect_threshold_ms
                ),
                "resolved_F": None,
                "resolved_p": None,
                "resolved_effect_ms": None,
                "resolved_significant": None,
                "n": len(df)
            })
        elif enough_per_resolved and n_resolved > 1:
            # One-way ANOVA on resolved_name
            groups = [g["time"].values for _, g in df.groupby("resolved_name") if len(g) > 1]
            if len(groups) > 1:
                F, p = f_oneway(*groups)
            else:
                F, p = None, None
            results.append({
                "process_name": process.name,
                "test": "anova1w",
                "trace_F": None,
                "trace_p_or_cv": None,
                "trace_effect_ms": None,
                "trace_significant": None,
                "resolved_F": F,
                "resolved_p": p,
                "resolved_effect_ms": resolved_effect,
                "resolved_significant": (
                    p is not None and p < 0.05 and resolved_effect is not None and resolved_effect >= effect_threshold_ms
                ),
                "n": len(df)
            })
        else:
            # Not enough data for ANOVA, use standard deviation
            std_dev = df["time"].std()
            coeff_of_variation = std_dev / df["time"].mean()
            results.append({
                "process_name": process.name,
                "test": "CV",
                "trace_F": None,
                "trace_p_or_cv": coeff_of_variation,
                "trace_effect_ms": std_dev,
                "trace_significant": coeff_of_variation > tolerance and std_dev > effect_threshold_ms,
                "resolved_F": None,
                "resolved_p": None,
                "resolved_effect_ms": None,
                "resolved_significant": None,
                "n": len(df)
            })

    return pd.DataFrame(results)


def build_execution_predictors(db_manager, trace_names=None):
    """
    Build execution predictors based on ANOVA analysis and linear regression models.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param trace_names: A list of trace names to filter the analysis. If None, all traces are included.
    :return: A tuple of two DataFrames: stats_based_config and model_based_config.
    """
    # Step 1: Perform ANOVA analysis
    anova_results = anova_on_process_execution_times(db_manager, trace_names=trace_names)

    stats_based_config = []
    model_based_config = []

    # Step 2.1: For processes and resolved processes with non-impactful traces, extract statistical information
    for _, row in anova_results.iterrows():
        if not row['trace_significant'] and not row['resolved_significant']:
            # Per process stats
            process_stats = get_execution_times_distribution_charasteristics(
                db_manager, row['process_name'], trace_names=trace_names
            )
            stats_based_config.append({
                'process_name': row['process_name'],
                'mean_time': process_stats['mean_time'].iloc[0],
                'std_dev_time': process_stats['std_dev_time'].iloc[0],
                'min_time': process_stats['min_time'].iloc[0],
                'max_time': process_stats['max_time'].iloc[0],
                'trace_significant': row['trace_significant'],
                'resolved_significant': row['resolved_significant']
            })
        elif not row['trace_significant'] and row['resolved_significant']:
            # Per resolved process stats
            rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(row['process_name'])
            for rp in rp_list:
                process_stats = get_execution_times_distribution_charasteristics(
                    db_manager, rp.name, is_resolved_name=True, trace_names=trace_names
                )
                stats_based_config.append({
                    'process_name': rp.name,
                    'mean_time': process_stats['mean_time'].iloc[0],
                    'std_dev_time': process_stats['std_dev_time'].iloc[0],
                    'min_time': process_stats['min_time'].iloc[0],
                    'max_time': process_stats['max_time'].iloc[0],
                    'trace_significant': row['trace_significant'],
                    'resolved_significant': row['resolved_significant']
                })

        # Step 2.2: For processes and resolved processes with impactful traces, parameter-dependent prediction
        elif row['trace_significant'] and not row['resolved_significant']:
            # Per process prediction
            model = extract_execution_time_linear_reg(
                db_manager, process_name=row['process_name'], is_resolved_name=False, trace_names=trace_names
            )
            model_based_config.append({
                'process_name': row['process_name'],
                'model': model,
                'trace_significant': row['trace_significant'],
                'resolved_significant': row['resolved_significant']
            })
        else:  # row['trace_significant'] and row['resolved_significant']:
            # Per resolved process prediction
            rp_list = db_manager.resolved_process_manager.getResolvedProcessesOfProcess(row['process_name'])
            for rp in rp_list:
                model = extract_execution_time_linear_reg(
                    db_manager, process_name=rp.name, is_resolved_name=True, trace_names=trace_names
                )
                model_based_config.append({
                    'process_name': rp.name,
                    'model': model,
                    'trace_significant': row['trace_significant'],
                    'resolved_significant': row['resolved_significant']
                })

    # Convert the collected stats into DataFrames
    stats_based_config = pd.DataFrame(stats_based_config)
    model_based_config = pd.DataFrame(model_based_config)

    return stats_based_config, model_based_config


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

    anova_results = anova_on_process_execution_times(db_manager)
    print("\n## ANOVA results on process execution times:")
    anova_results = anova_results.sort_values(by=["process_name"], ascending=[True])
    print(anova_results)

    df = analyze_process_execution_correlation(db_manager, "generate_binconfig")
    print("\n## Correlation results:")
    print(df)

    extract_execution_time_linear_reg(db_manager, "generate_binconfig", top_n=3, rmse_threshold=10000, print_info=True)

    extract_execution_time_linear_reg(
        db_manager,
        "fcal1:corr_fcal:do_correlation",
        top_n=9,
        rmse_threshold=10000,
        is_resolved_name=True,
        print_info=True)

    extract_execution_time_linear_reg(
        db_manager,
        "frb:corr_field:do_correlation",
        top_n=9,
        rmse_threshold=10000,
        is_resolved_name=True,
        print_info=True)

    db_manager.close()
    print("Connection closed.")
