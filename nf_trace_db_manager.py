import os
import sqlite3
from process_exec_params_table_manager import ProcessExecParamsTableManager
from process_inputs_table_manager import ProcessInputsTableManager
from trace_table_manager import TraceTableManager
from processes_table_manager import ProcessesTableManager
from resolved_process_names_table_manager import ResolvedProcessNamesTableManager
from process_executions_table_manager import ProcessExecutionTableManager
from pipeline_params_table_manager import PipelineParamsTableManager
from pipeline_param_values_table_manager import PipelineParamValuesTableManager
from process_params_hints_table_manager import ProcessParamsHintsTableManager


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
        self.resolved_process_manager = None
        self.process_executions_manager = None
        self.process_inputs_manager = None
        self.process_exec_params_manager = None
        self.pipeline_params_manager = None
        self.pipeline_param_values_manager = None
        self.process_params_hints_manager = None  # Add ProcessParamsHintsTableManager attribute

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
            self.resolved_process_manager = ResolvedProcessNamesTableManager(self.connection)
            self.process_executions_manager = ProcessExecutionTableManager(self.connection)
            self.process_inputs_manager = ProcessInputsTableManager(self.connection)
            self.process_exec_params_manager = ProcessExecParamsTableManager(self.connection)
            self.pipeline_params_manager = PipelineParamsTableManager(self.connection)
            self.pipeline_param_values_manager = PipelineParamValuesTableManager(self.connection)
            self.process_params_hints_manager = ProcessParamsHintsTableManager(self.connection)  # Initialize ProcessParamsHintsTableManager

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

    def addProcessExecParamsFromLog(self, log_file_path):
        """
        Wrapper method to add process execution parameters from log file to the ProcessExecParams table.

        :param log_file_path: The path to the log file.
        """
        self.process_exec_params_manager.addExecutionParamsFromLog(self, log_file_path)

    def addPipelineParamsFromLog(self, log_file_path):
        """
        Wrapper method to add pipeline parameters from log file to the PipelineParams table.

        :param log_file_path: The path to the log file.
        """
        self.pipeline_params_manager.addPipelineParamsFromLog(self, log_file_path)

    def addPipelineParamValuesFromLog(self, log_file_path):
        """
        Wrapper method to add pipeline parameter values from a log file to the PipelineParamValues table.

        :param log_file_path: The path to the log file.
        """
        self.pipeline_param_values_manager.addPipelineParamValuesFromLog(self, log_file_path)

    def addProcessParamHintsFromCode(self, root_folder=None, prefix=None):
        """
        Extract process parameter hints from Nextflow files and store them in the ProcessParamsHints table.

        :param root_folder: (Optional) A path to a root folder where process files are located.
        :param prefix: (Optional) A prefix path to replace with the root folder.
        """
        self.process_params_hints_manager.addProcessParamHintsFromCode(self, root_folder=root_folder, prefix=prefix)

    def addAllFromFiles(self, html_file_path, log_file_path):
        """
        Add all relevant data to the database from the given HTML and log files.

        Adds are skipped if the corresponding TableManager is set to None or if its dependencies are not initialized.

        The ProcessParamsHintsTableManager is not included in this method as it is not directly related to the HTML or log files.

        Dependencies:
        - Metadata addition requires TraceTableManager.
        - Process definitions require ProcessesTableManager.
        - Resolved process names require ResolvedProcessNamesTableManager and ProcessesTableManager.
        - Process executions require ProcessExecutionTableManager, ResolvedProcessNamesTableManager, and TraceTableManager.
        - Process inputs require ProcessInputsTableManager and ResolvedProcessNamesTableManager.
        - Process execution parameters require ProcessExecParamsTableManager, ResolvedProcessNamesTableManager, ProcessExecutionTableManager, and TraceTableManager.
        - Pipeline parameters require PipelineParamsTableManager.
        - Pipeline parameter values require PipelineParamValuesTableManager, PipelineParamsTableManager, and TraceTableManager.

        :param html_file_path: The path to the HTML file.
        :param log_file_path: The path to the log file.
        """
        # Add metadata from the HTML file to the Traces table
        if self.trace_manager is not None:
            tId = self.addMetadataFromHtml(html_file_path)
        else:
            print("Skipping metadata addition: TraceTableManager is not initialized.")
            return

        # Add process definitions from the log file
        if self.process_manager is not None:
            self.addProcessDefinitionsFromLog(log_file_path)
        else:
            print("Skipping process definitions addition: ProcessesTableManager is not initialized.")

        # Add resolved process names from the log file
        if self.resolved_process_manager is not None and self.process_manager is not None:
            self.addResolvedProcessNamesFromLog(log_file_path)
        else:
            print("Skipping resolved process names addition: ResolvedProcessNamesTableManager or ProcessesTableManager is not initialized.")

        # Add process executions from the HTML file
        if (
            self.process_executions_manager is not None
            and self.resolved_process_manager is not None
            and self.trace_manager is not None
        ):
            self.addProcessExecutionsFromHtml(html_file_path, tId)
        else:
            print("Skipping process executions addition: ProcessExecutionTableManager, ResolvedProcessNamesTableManager, or TraceTableManager is not initialized.")

        # Add process inputs from the log file
        if self.process_inputs_manager is not None and self.resolved_process_manager is not None:
            self.addProcessInputsFromLog(log_file_path)
        else:
            print("Skipping process inputs addition: ProcessInputsTableManager or ResolvedProcessNamesTableManager is not initialized.")

        # Add process execution parameters from the log file
        if (
            self.process_exec_params_manager is not None
            and self.resolved_process_manager is not None
            and self.process_executions_manager is not None
            and self.trace_manager is not None
        ):
            self.addProcessExecParamsFromLog(log_file_path)
        else:
            print("Skipping process execution parameters addition: ProcessExecParamsTableManager, ResolvedProcessNamesTableManager, ProcessExecutionTableManager, or TraceTableManager is not initialized.")

        # Add pipeline parameters from the log file
        if self.pipeline_params_manager is not None:
            self.addPipelineParamsFromLog(log_file_path)
        else:
            print("Skipping pipeline parameters addition: PipelineParamsTableManager is not initialized.")

        # Add pipeline parameter values from the log file
        if (
            self.pipeline_param_values_manager is not None
            and self.pipeline_params_manager is not None
            and self.trace_manager is not None
        ):
            self.addPipelineParamValuesFromLog(log_file_path)
        else:
            print("Skipping pipeline parameter values addition: PipelineParamValuesTableManager, PipelineParamsTableManager, or TraceTableManager is not initialized.")

    def printDBInfo(self):
        """
        Print information about the database, including counts of various entries.
        """
        # Retrieve and print the user_version of the database
        user_version = self.getUserVersion()
        print(f"Database user_version: {user_version}")

        # Retrieve and print the number of trace entries
        if self.trace_manager is not None:
            trace_count = len(self.trace_manager.getAllTraces())
            print(f"Number of trace entries: {trace_count}")

        # Retrieve and print the number of process entries
        if self.process_manager is not None:
            process_count = len(self.process_manager.getAllProcesses())
            print(f"Number of process entries: {process_count}")

        # Retrieve and print the number of resolved process names
        if self.resolved_process_manager is not None:
            process_name_count = len(self.resolved_process_manager.getAllResolvedProcessNames())
            print(f"Number of resolved process names: {process_name_count}")

        # Retrieve and print the number of process executions
        if self.process_executions_manager is not None:
            process_execution_count = len(self.process_executions_manager.getAllProcessExecutions())
            print(f"Number of process executions: {process_execution_count}")

        # Retrieve and print the number of process inputs
        if self.process_inputs_manager is not None:
            process_input_count = len(self.process_inputs_manager.getAllProcessInputs())
            print(f"Number of process inputs: {process_input_count}")

        # Retrieve and print the number of process execution parameters
        if self.process_exec_params_manager is not None:
            process_exec_param_count = len(self.process_exec_params_manager.getAllProcessExecParams())
            print(f"Number of process execution parameters: {process_exec_param_count}")

        # Retrieve and print the number of pipeline parameters
        if self.pipeline_params_manager is not None:
            pipeline_param_count = len(self.pipeline_params_manager.getAllPipelineParams())
            print(f"Number of pipeline parameters: {pipeline_param_count}")

        # Retrieve and print the number of pipeline parameter values
        if self.pipeline_param_values_manager is not None:
            pipeline_param_value_count = len(self.pipeline_param_values_manager.getAllPipelineParamValues())
            print(f"Number of pipeline parameter values: {pipeline_param_value_count}")

        # Retrieve and print the number of process parameter hints
        if self.process_params_hints_manager is not None:
            process_param_hint_count = len(self.process_params_hints_manager.getAllProcessParamHints())
            print(f"Number of process parameter hints: {process_param_hint_count}")


# Main prog
if __name__ == "__main__":
    # Initialize the database manager with the path to the SQLite database
    db_manager = NextflowTraceDBManager("./dat/nf_trace_db.sqlite")

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
    html_file_paths = [
        "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_report.html",
        "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_report.html",
        "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_report.html",
        "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_report.html",
        "./dat/250515_241226_CELEBI/karol_241226_ult_2025-05-15_13_41_42_report.html",
        "./dat/250522_241027_CELEBI/karol_241027_ult_2025-05-22_10_05_56_report.html",
        "./dat/250523_241014_CELEBI/karol_241014_ult_2025-05-23_12_12_20_report.html",
        "./dat/250616_241226_CELEBI_mold/karol_241226_ult_2025-06-16_10_22_54_report.html",
        "./dat/250616_210912_CELEBI_mold/karol_210912_ult_2025-06-16_13_46_53_report.html"
    ]

    # Add process definitions
    log_files = [
        "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_log.log",
        "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_log.log",
        "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_log.log",
        "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_log.log",
        "./dat/250515_241226_CELEBI/karol_241226_ult_2025-05-15_13_41_42_log.log",
        "./dat/250522_241027_CELEBI/karol_241027_ult_2025-05-22_10_05_56_log.log",
        "./dat/250523_241014_CELEBI/karol_241014_ult_2025-05-23_12_12_20_log.log",
        "./dat/250616_241226_CELEBI_mold/karol_241226_ult_2025-06-16_10_22_54_log.log",
        "./dat/250616_210912_CELEBI_mold/karol_210912_ult_2025-06-16_13_46_53_log.log"]

    for html_file_path, log_file in zip(html_file_paths, log_files):
        print(f"Loading files: {html_file_path} and {log_file}")
        db_manager.addAllFromFiles(html_file_path, log_file)

    # Add process parameter hints from code files
    # db_manager.addProcessParamHintsFromCode(
    #     root_folder="C:/Git/",
    #     prefix="/fred/oz313/src/users/kdesnos/"
    # )

    # Print database information
    db_manager.printDBInfo()

    db_manager.close()
    print("Connection closed.")
