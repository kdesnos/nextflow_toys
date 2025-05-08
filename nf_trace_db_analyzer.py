import numpy as np
import pandas as pd
import time
from nf_trace_db_manager import NextflowTraceDBManager

def check_all_processes_execution_time(db_manager, tolerance=0.1, std_dev_threshold=30000):
    """
    Check if the execution time for every process in the Processes table is constant.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param tolerance: The relative tolerance (default: 0.1, i.e., 10%) for the coefficient of variation.
    :param std_dev_threshold: The absolute threshold for the standard deviation (default: 30000 milliseconds or 30 seconds).
    :return: A Pandas DataFrame with process names, computed statistics, and constancy criteria.
    """
    # SQL query to retrieve all execution times and process names
    query = """
        SELECT p.name AS process_name, pe.time
        FROM Processes p
        LEFT JOIN ResolvedProcessNames rpn ON p.pId = rpn.pId
        LEFT JOIN ProcessExecutions pe ON rpn.rId = pe.rId;
    """

    # Execute the query and fetch the results
    cursor = db_manager.connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    # Create a Pandas DataFrame from the query results
    df = pd.DataFrame(results, columns=["process_name", "time"])

    # Group by process_name and compute statistics
    grouped = df.groupby("process_name")["time"]
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
        ["process_name", "execution_count", "is_constant", "mean_time", "std_dev_time", "coefficient_of_variation"]
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

    try:
        results_python = check_all_processes_execution_time(db_manager)
        print(results_python)
    except Exception as e:
        print(f"Error: {e}")

    db_manager.close()
    print("Connection closed.")
