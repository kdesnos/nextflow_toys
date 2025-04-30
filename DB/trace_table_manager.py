class TraceTableManager:
    def __init__(self, connection):
        """
        Initialize the TraceTableManager with a database connection.
        """
        self.connection = connection

    def addTraceEntry(self, trace_entry):
        """
        Add a TraceEntry instance to the Traces table in the SQLite database and update the tId of the input trace_entry.

        :param trace_entry: An instance of TraceEntry.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO Traces (day, name) VALUES (?, ?);",
            (trace_entry.day, trace_entry.name),
        )
        self.connection.commit()

        # Retrieve the last inserted row ID and update the trace_entry's tId
        trace_entry.tId = cursor.lastrowid

    def getTraceEntry(self, name):
        """
        Retrieve a trace entry from the database by its name.

        :param name: The name of the trace entry to retrieve.
        :return: A TraceEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT tId, day, name FROM Traces WHERE name = ?;", (name,))
        row = cursor.fetchone()

        if row:
            return TraceEntry(tId=row[0], day=row[1], name=row[2])
        return None

    def getAllTraces(self):
        """
        Retrieve all trace entries from the database.

        :return: A list of TraceEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT tId, day, name FROM Traces;")
        rows = cursor.fetchall()

        return [TraceEntry(tId=row[0], day=row[1], name=row[2]) for row in rows]

    def removeTraceEntry(self, name):
        """
        Remove a trace entry from the database by its name.

        :param name: The name of the trace entry to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM Traces WHERE name = ?;", (name,))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0

    def addTraces(self, trace_entries):
        """
        Add multiple TraceEntry instances to the Traces table in the SQLite database.

        :param trace_entries: A collection of TraceEntry instances.
        """
        for trace_entry in trace_entries:
            self.addTraceEntry(trace_entry)


class TraceEntry:
    def __init__(self, tId, day, name):
        """
        Initialize a TraceEntry instance.

        :param tId: Trace ID (integer).
        :param day: Day (string).
        :param name: Name (string).
        """
        self.tId = tId
        self.day = day
        self.name = name

    def __repr__(self):
        return f"TraceEntry(tId={self.tId}, day='{self.day}', name='{self.name}')"