import re
from extract_trace_from_html import extract_trace_data


class ProcessExecutionTableManager:
    def __init__(self, connection):
        """
        Initialize the ProcessExecutionTableManager with a database connection.
        """
        self.connection = connection

    def addProcessExecution(self, execution_entry):
        """
        Add a process execution entry to the ProcessExecutions table in the SQLite database.

        :param execution_entry: A ProcessExecutionEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO ProcessExecutions (tId, rId, instance, hash, time) VALUES (?, ?, ?, ?, ?);",
            (execution_entry.tId, execution_entry.rId, execution_entry.instance, execution_entry.hash, execution_entry.time),
        )
        self.connection.commit()

        # Retrieve the last inserted row ID and update the execution_entry's eId
        execution_entry.eId = cursor.lastrowid

    def addAllProcessExecutions(self, execution_entries):
        """
        Add multiple process execution entries to the ProcessExecutions table in the SQLite database.

        :param execution_entries: A collection of ProcessExecutionEntry instances.
        """
        for execution_entry in execution_entries:
            self.addProcessExecution(execution_entry)

    def addProcessExecutionsFromFile(self, trace_db_manager, file_path, trace_id=None):
        """
        Parse the content of a file using extract_trace_data and add the content to the ProcessExecutions table.
        Trace ID is automatically retrieved from the file and added to the appropriate table.
        In case the Table ID already existed in the database, nothing is added to the ProcessExecutions Table.

        :param file_path: The path to the trace HTML file.
        :param trace_db_manager: An instance of NextflowTraceDBManager to resolve rId.
        :param trace_it: Optionnaly, a trace_id can be given, thus skipping the update of the Trace table.
        """
        trace_id = trace_db_manager.addMetadataFromHtml(file_path) if trace_id is None else trace_id

        # Extract trace data from the file
        trace_data = extract_trace_data(file_path)
        if trace_data is None:
            raise Exception("Failed to extract trace data from the file.")

        # Iterate over the rows of the DataFrame and add each execution to the database
        for _, row in trace_data.iterrows():
            # Extract the instance number from the 'name' column (e.g., "process_name (1)")
            instance_match = re.search(r"\((\d+)\)$", row["name"])
            # in case no name was matches, there is a single instance of the process.
            instance = int(instance_match.group(1)) if instance_match else 1

            # Resolve the rId using the resolved process manager
            resolved_entry = trace_db_manager.resolved_process_manager.getResolvedProcessByName(row["process"])
            if resolved_entry is None:
                raise Exception(f"Resolved process name '{row['process']}' not found in ResolvedProcessNames table.")

            # Create a ProcessExecutionEntry and add it to the database
            execution_entry = ProcessExecutionEntry(
                eId=0,
                tId=trace_id,
                rId=resolved_entry.rId,
                instance=instance,
                hash=row["hash"],
                time=row["realtime"].total_seconds() * 1000.0
            )
            self.addProcessExecution(execution_entry)

    def getProcessExecutionByHash(self, hash_value):
        """
        Retrieve a process execution entry from the database by its hash.

        :param hash_value: The hash of the process execution to retrieve.
        :return: A ProcessExecutionEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT eId, tId, rId, instance, hash, time FROM ProcessExecutions WHERE hash = ?;", (hash_value,))
        row = cursor.fetchone()

        if row:
            return ProcessExecutionEntry(eId=row[0], tId=row[1], rId=row[2], instance=row[3], hash=row[4], time=row[5])
        return None

    def getAllProcessExecutions(self):
        """
        Retrieve all process execution entries from the database.

        :return: A list of ProcessExecutionEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT eId, tId, rId, instance, hash, time FROM ProcessExecutions;")
        rows = cursor.fetchall()

        return [
            ProcessExecutionEntry(eId=row[0], tId=row[1], rId=row[2], instance=row[3], hash=row[4], time=row[5])
            for row in rows
        ]

    def removeProcessExecutionByHash(self, hash_value):
        """
        Remove a process execution entry from the database by its hash.

        :param hash_value: The hash of the process execution to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM ProcessExecutions WHERE hash = ?;", (hash_value,))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0


class ProcessExecutionEntry:
    def __init__(self, eId, tId, rId, instance, hash, time):
        """
        Initialize a ProcessExecutionEntry instance.

        :param eId: Execution ID (integer).
        :param tId: Trace ID (integer).
        :param rId: Resolved Process ID (integer).
        :param instance: Instance number (integer).
        :param hash: Hash of the execution (string).
        :param time: Execution time in milliseconds (float).
        """
        self.eId = eId
        self.tId = tId
        self.rId = rId
        self.instance = instance
        self.hash = hash
        self.time = time

    def __repr__(self):
        return (
            f"ProcessExecutionEntry(eId={self.eId}, tId={self.tId}, rId={self.rId}, "
            f"instance={self.instance}, hash='{self.hash}', time={self.time})"
        )
