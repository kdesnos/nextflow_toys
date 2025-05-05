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