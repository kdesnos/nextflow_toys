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