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
    group_by_trace_name=False
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

    if process_names:
        placeholders = ",".join("?" for _ in process_names)
        filters.append(f"p.name IN ({placeholders})")
        params.extend(process_names)

    if resolved_process_names:
        placeholders = ",".join("?" for _ in resolved_process_names)
        filters.append(f"rpn.name IN ({placeholders})")
        params.extend(resolved_process_names)

    if trace_names:
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
    # with the 1st and 9th deciles as the lower and upper bounds
    def remove_outliers(group):
        d1 = group.quantile(0.10)
        d9 = group.quantile(0.90)
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


if __name__ == "__main__":
    # Initialize the database manager with the path to the SQLite database
    db_manager = NextflowTraceDBManager("nf_trace_db.sqlite")

    # Establish a connection to the database
    db_manager.connect()
    print("Connected to the database.")

    # Check if the database is empty and create tables if necessary
    if db_manager.isDatabaseEmpty():
        db_manager.createTables()
    else:
        db_manager.createTables(force=True)
    print("Tables created successfully.")

    # Ignore ParamInputs and ExecParams for now to speed up tests
    db_manager.process_inputs_manager = None
    db_manager.process_exec_params_manager = None

    # List of HTML and log file paths to load into the database
    files_to_load = [
        {
            "html_file": "c:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_210912_ult_2025-04-22_14_03_39_report.html",
            "log_file": "C:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_210912_ult_2025-04-22_14_03_39_log.log"
        },
        {
            "html_file": "c:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_250313_2025-05-08_10_36_28_report.html",
            "log_file": "c:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_250313_2025-05-08_10_36_28_log.log"
        }
        # Add more file pairs as needed
    ]

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

    db_manager.close()
    print("Connection closed.")
