import os
import sqlite3
from process_exec_params_table_manager import ProcessExecParamsTableManager
from process_inputs_table_manager import ProcessInputsTableManager
from trace_table_manager import TraceTableManager
from processes_table_manager import ProcessesTableManager
from resolved_process_names_table_manager import ResolvedProcessNamesTableManager
from process_executions_table_manager import ProcessExecutionTableManager


class NextflowTraceDBManager:
    def __init__(self, db_path):
        """
        Initialize the NextflowTraceDBManager with a connection to the specified SQLite database.

        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None
        self.trace_manager = None
        self.process_manager = None
        self.resolved_process_manager = None  # Add ResolvedProcessNamesTableManager attribute
        self.process_executions_manager = None
        self.process_inputs_manager = None
        self.process_exec_params_manager = None

    def connect(self):
        """
        Establish a connection to the SQLite database.
        """
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            # Activate foreign_keys support on all connections as it is not persistent.
            self.connection.execute("PRAGMA foreign_keys = ON;")
            self.trace_manager = TraceTableManager(self.connection)
            self.process_manager = ProcessesTableManager(self.connection)
            self.resolved_process_manager = ResolvedProcessNamesTableManager(self.connection)  # Initialize ResolvedProcessNamesTableManager
            self.process_executions_manager = ProcessExecutionTableManager(self.connection)
            self.process_inputs_manager = ProcessInputsTableManager(self.connection)
            self.process_exec_params_manager = ProcessExecParamsTableManager(self.connection)

    def isConnected(self):
        """
        Check if the database connection is established.

        :return: True if the database is connected, False otherwise.
        """
        return self.connection is not None

    def isDatabaseEmpty(self):
        """
        Check if the connected SQLite database is empty.

        :return: True if the database is empty, False otherwise.
        :raises Exception: If the database is not connected.
        """
        if not self.isConnected():
            raise Exception("Database is not connected.")

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return len(tables) == 0

    def createTables(self, force=False):
        """
        Create tables in the SQLite database using the SQL script file.

        :param force: If True, delete the database file and recreate it.
        :raises Exception: If the database is not connected and force is False.
        """
        if not self.isConnected() and not force:
            raise Exception("Database is not connected.")

        if force:
            # Close the connection if it's open
            if self.connection:
                self.close()

            # Delete the database file
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                print(f"Database file '{self.db_path}' deleted.")

            # Reconnect to recreate the database
            self.connect()

        # Read and execute the SQL script to create tables
        with open(os.path.join(os.path.dirname(__file__), "nextflow_trace_DB_creation_script.sql"), "r") as sql_file:
            sql_script = sql_file.read()
            cursor = self.connection.cursor()
            cursor.executescript(sql_script)

        self.connection.commit()
        print("Tables created successfully.")

    def close(self):
        """
        Close the connection to the SQLite database.
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def getUserVersion(self):
        """
        Retrieve the user_version of the SQLite database.

        :return: The user_version of the database as an integer.
        :raises Exception: If the database is not connected.
        """
        if not self.isConnected():
            raise Exception("Database is not connected.")

        cursor = self.connection.cursor()
        cursor.execute("PRAGMA user_version;")
        version = cursor.fetchone()[0]
        return version

    def addMetadataFromHtml(self, html_file_path):
        """
        Wrapper method to add metadata from an HTML file to the Traces table.

        :param html_file_path: The path to the HTML file.
        :return: The Trace ID (tId) of the added metadata.
        """
        return self.trace_manager.addMetadataToTraceTable(html_file_path)

    def addProcessDefinitionsFromLog(self, log_file_path):
        """
        Wrapper method to add process definitions from a log file to the Processes table.

        :param log_file_path: The path to the log file.
        """
        self.process_manager.addProcessDefinitionsToTable(log_file_path)

    def addResolvedProcessNamesFromLog(self, log_file_path):
        """
        Wrapper method to add resolved process names from a log file to the ResolvedProcessNames table.

        :param log_file_path: The path to the log file.
        """
        self.resolved_process_manager.addResolvedProcessNamesToTable(self, log_file_path)

    def addProcessExecutionsFromHtml(self, html_file_path, trace_id=None):
        """
        Wrapper method to add process executions from an HTML file to the ProcessExecutions table.

        :param html_file_path: The path to the HTML file.
        :param trace_id: Optionnaly, the Trace ID (tId) to associate with the process executions. If not provided, it will be retrieved from the HTML file.
        """
        self.process_executions_manager.addProcessExecutionsFromFile(self, html_file_path, trace_id)

    def addProcessInputsFromLog(self, log_file_path):
        """
        Wrapper method to add process inputs from log file to the ProcessInputs table.

        :param log_file_path: The path to the log file.
        """
        self.process_inputs_manager.addInputsFromLog(self, log_file_path)


# Main prog
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

    # Retrieve and print the user_version of the database
    user_version = db_manager.getUserVersion()
    print(f"Database user_version: {user_version}")

    # Add metadata from an HTML file to the Traces table
    html_file_path = "c:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_210912_ult_2025-02-21_15_23_50_report.html"
    db_manager.addMetadataFromHtml(html_file_path)
    html_file_path = "c:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_210912_ult_2025-04-22_14_03_39_report.html"
    # Not added, as it will automatically be when adding traces.

    # Retrieve all trace entries
    all_traces = db_manager.trace_manager.getAllTraces()
    print("All trace entries:")
    for trace in all_traces:
        print(trace)

    # Add process definitions
    log_file = "C:\\Users\\Karol\\Desktop\\Sandbox\\pipelines\\karol_210912_ult_2025-04-22_14_03_39_nextflow_logs.log"
    db_manager.addProcessDefinitionsFromLog(log_file)

    # Retrieve all process entries
    all_processes = db_manager.process_manager.getAllProcesses()
    print("All process entries:")
    for process in all_processes:
        print(process)

    db_manager.addResolvedProcessNamesFromLog(log_file)

    # Retrieve all resolved process names
    all_process_names = db_manager.resolved_process_manager.getAllResolvedProcessNames()
    print("All resolved process names:")
    for process_name in all_process_names:
        print(process_name)

    db_manager.addProcessExecutionsFromHtml(html_file_path)

    # Retrieve all process executions names
    all_process_executions = db_manager.process_executions_manager.getAllProcessExecutions()
    print("All process executions:")
    for process_execution in all_process_executions:
        print(process_execution)

    db_manager.addProcessInputsFromLog(log_file)
    all_process_inputs = db_manager.process_inputs_manager.getAllProcessInputs()
    print("All process inputs")
    for process_input in all_process_inputs:
        print(process_input)

    db_manager.close()
    print("Connection closed.")
