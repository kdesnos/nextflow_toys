import pandas as pd
from nf_trace_db_manager import NextflowTraceDBManager


def check_all_processes_execution_time(
    db_manager, 
    tolerance=0.1, 
    std_dev_threshold=15000,  # Default threshold updated to 15 seconds (15000 milliseconds)
    process_names=None, 
    resolved_process_names=None, 
    group_by_resolved_name=False
):
    """
    Check if the execution time for every process in the Processes table is constant, ignoring outliers.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param tolerance: The relative tolerance (default: 0.1, i.e., 10%) for the coefficient of variation.
    :param std_dev_threshold: The absolute threshold for the standard deviation (default: 15000 milliseconds or 15 seconds).
    :param process_names: Optional list of process names to filter the results. If None, all processes are included.
    :param resolved_process_names: Optional list of resolved process names to filter the results. If None, all resolved processes are included.
    :param group_by_resolved_name: If True, group results by resolved process names instead of process names.
    :return: A Pandas DataFrame with process names (or resolved names), computed statistics, and constancy criteria.
    """
    # SQL query to retrieve execution times and process names or resolved names
    group_column = "rpn.name AS resolved_name" if group_by_resolved_name else "p.name AS process_name"
    query = f"""
        SELECT {group_column}, pe.time
        FROM Processes p
        LEFT JOIN ResolvedProcessNames rpn ON p.pId = rpn.pId
        LEFT JOIN ProcessExecutions pe ON rpn.rId = pe.rId
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

    if filters:
        query += " WHERE " + " AND ".join(filters)

    # Execute the query and fetch the results
    cursor = db_manager.connection.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()

    # Create a Pandas DataFrame from the query results
    column_name = "resolved_name" if group_by_resolved_name else "process_name"
    df = pd.DataFrame(results, columns=[column_name, "time"])

    # Remove outliers using the IQR method for each group
    def remove_outliers(group):
        q1 = group["time"].quantile(0.25)
        q3 = group["time"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[(group["time"] >= lower_bound) & (group["time"] <= upper_bound)]

    df = df.groupby(column_name, group_keys=False).apply(remove_outliers)

    # Group by the selected column and compute statistics
    grouped = df.groupby(column_name)["time"]
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

    # Reorder the columns
    stats = stats[
        [column_name, "execution_count", "is_constant", "mean_time", "std_dev_time", "coefficient_of_variation"]
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

    # Add metadata from an HTML file to the Traces table
    html_file_path = "c:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_210912_ult_2025-04-22_14_03_39_report.html"
    # Add process definitions
    log_file = "C:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_210912_ult_2025-04-22_14_03_39_nextflow_logs.log"

    # Ignore ParamInputs and ExecParams for now to speed up tests
    db_manager.process_inputs_manager = None
    db_manager.process_exec_params_manager = None

    db_manager.addAllFromFiles(html_file_path, log_file)

    # Print database information
    db_manager.printDBInfo()

    results_python = check_all_processes_execution_time(db_manager)
    print(results_python)

    results_python = check_all_processes_execution_time(db_manager, process_names=["do_correlation"], group_by_resolved_name=True)
    print(results_python)

    db_manager.close()
    print("Connection closed.")
