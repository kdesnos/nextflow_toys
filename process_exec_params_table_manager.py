from extract_from_nf_log import extractExecutionParameters, extractRunName


class ProcessExecParamsTableManager:
    def __init__(self, connection):
        """
        Initialize the ProcessExecParamsTableManager with a database connection.
        """
        self.connection = connection

    def addProcessExecParam(self, exec_param_entry):
        """
        Add a process execution parameter entry to the ProcessExecParams table in the SQLite database.

        :param exec_param_entry: A ProcessExecParamEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO ProcessExecParams (eId, rank, value) VALUES (?, ?, ?);",
            (exec_param_entry.eId, exec_param_entry.rank, exec_param_entry.value),
        )
        self.connection.commit()

    def addAllProcessExecParams(self, exec_param_entries):
        """
        Add multiple process execution parameter entries to the ProcessExecParams table.

        :param exec_param_entries: A collection of ProcessExecParamEntry instances.
        """
        for exec_param_entry in exec_param_entries:
            self.addProcessExecParam(exec_param_entry)

    def getProcessExecParamsByExecutionId(self, execution_id):
        """
        Retrieve all process execution parameter entries for a specific execution ID.

        :param execution_id: The execution ID to retrieve parameters for.
        :return: A list of ProcessExecParamEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT eId, rank, value FROM ProcessExecParams WHERE eId = ?;", (execution_id,))
        rows = cursor.fetchall()

        return [ProcessExecParamEntry(eId=row[0], rank=row[1], value=row[2]) for row in rows]

    def getAllProcessExecParams(self):
        """
        Retrieve all process execution parameter entries from the database.

        :return: A list of ProcessExecParamEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT eId, rank, value FROM ProcessExecParams;")
        rows = cursor.fetchall()

        return [ProcessExecParamEntry(eId=row[0], rank=row[1], value=row[2]) for row in rows]

    def removeProcessExecParam(self, execution_id, rank):
        """
        Remove a process execution parameter entry from the database by its execution ID and rank.

        :param execution_id: The execution ID of the parameter to remove.
        :param rank: The rank of the parameter to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM ProcessExecParams WHERE eId = ? AND rank = ?;", (execution_id, rank))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0

    def addExecutionParamsFromLog(self, trace_db_manager, file_path):
        """
        Extract execution parameters from a log file and add them to the ProcessExecParams table.

        :param trace_db_manager: An instance of NextflowTraceDBManager to resolve execution IDs.
        :param file_path: The path to the Nextflow log file.
        """

        # Extract trace run name from the log file
        run_name = extractRunName(file_path)
        if run_name is None:
            raise Exception(f"Trace run name not found in the log file '{file_path}'.")

        # Get the trace ID from the database using the run name
        trace_entry = trace_db_manager.trace_manager.getTraceEntry(run_name)
        if trace_entry is None:
            raise Exception(f"Trace '{run_name}' not found in the database.")

        # Extract execution parameters from the log file
        execution_params = extractExecutionParameters(file_path)

        def add_nested_params(eId, params, base_rank):
            """
            Recursively add execution parameters (both top-level and nested) to the database.

            :param eId: Execution ID.
            :param params: List of parameters (can be top-level or nested).
            :param base_rank: Base rank for the parameters.
            """
            for i, param in enumerate(params):
                rank = f"{base_rank}.{i}" if base_rank else str(i)
                if isinstance(param, list):  # Handle nested tuples
                    add_nested_params(eId, param, rank)
                else:
                    exec_param_entry = ProcessExecParamEntry(
                        eId=eId,
                        rank=rank,
                        value=param
                    )
                    self.addProcessExecParam(exec_param_entry)

        # Iterate over the rows of the DataFrame and add each parameter to the database
        for _, row in execution_params.iterrows():
            # Resolve the execution ID (eId) using the ResolvedProcessNamesTableManager
            resolved_process_entry = trace_db_manager.resolved_process_manager.getResolvedProcessByName(row["resolved_process_name"])
            if resolved_process_entry is None:
                raise Exception(f"Process '{row['resolved_process_name']}' not found in ResolvedProcessNames table.")

            # Retrieve the execution entry using the resolved process ID and the trace_entry
            execution_entry = trace_db_manager.process_executions_manager.getExecutionByResolvedIdAndInstanceAndTraceId(
                resolved_process_entry.rId, row["instance_number"], trace_entry.tId)
            if execution_entry is None:
                raise Exception(f"Execution for resolved process '{row['resolved_process_name']}' not found in ProcessExecutions table for run {trace_entry.name}.")

            # Add all execution parameters (top-level and nested) to the database
            add_nested_params(execution_entry.eId, row["input_values"], base_rank="")


class ProcessExecParamEntry:
    def __init__(self, eId, rank, value):
        """
        Initialize a ProcessExecParamEntry instance.

        :param eId: Execution ID (integer).
        :param rank: Rank of the parameter (string).
        :param value: Value of the parameter (string).
        """
        self.eId = eId
        self.rank = rank
        self.value = value

    def __repr__(self):
        return f"ProcessExecParamEntry(eId={self.eId}, rank='{self.rank}', value='{self.value}')"
