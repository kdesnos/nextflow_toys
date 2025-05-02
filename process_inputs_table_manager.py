class ProcessInputsTableManager:
    def __init__(self, connection):
        """
        Initialize the ProcessInputsTableManager with a database connection.
        """
        self.connection = connection

    def addProcessInput(self, input_entry):
        """
        Add a process input entry to the ProcessInputs table in the SQLite database.

        :param input_entry: A ProcessInputEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO ProcessInputs (pId, rank, type, name) VALUES (?, ?, ?, ?);",
            (input_entry.pId, input_entry.rank, input_entry.type, input_entry.name),
        )
        self.connection.commit()

    def addAllProcessInputs(self, input_entries):
        """
        Add multiple process input entries to the ProcessInputs table in the SQLite database.

        :param input_entries: A collection of ProcessInputEntry instances.
        """
        for input_entry in input_entries:
            self.addProcessInput(input_entry)

    def getProcessInputsByProcessId(self, process_id):
        """
        Retrieve all process input entries for a specific process ID.

        :param process_id: The process ID to retrieve inputs for.
        :return: A list of ProcessInputEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT pId, rank, type, name FROM ProcessInputs WHERE pId = ?;", (process_id,))
        rows = cursor.fetchall()

        return [ProcessInputEntry(pId=row[0], rank=row[1], type=row[2], name=row[3]) for row in rows]

    def getAllProcessInputs(self):
        """
        Retrieve all process input entries from the database.

        :return: A list of ProcessInputEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT pId, rank, type, name FROM ProcessInputs;")
        rows = cursor.fetchall()

        return [ProcessInputEntry(pId=row[0], rank=row[1], type=row[2], name=row[3]) for row in rows]

    def removeProcessInput(self, process_id, rank):
        """
        Remove a process input entry from the database by its process ID and rank.

        :param process_id: The process ID of the input to remove.
        :param rank: The rank of the input to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM ProcessInputs WHERE pId = ? AND rank = ?;", (process_id, rank))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0


class ProcessInputEntry:
    def __init__(self, pId, rank, type, name):
        """
        Initialize a ProcessInputEntry instance.

        :param pId: Process ID (integer).
        :param rank: Rank of the input (integer).
        :param type: Type of the input (string).
        :param name: Name of the input (string).
        """
        self.pId = pId
        self.rank = rank
        self.type = type
        self.name = name

    def __repr__(self):
        return f"ProcessInputEntry(pId={self.pId}, rank={self.rank}, type='{self.type}', name='{self.name}')"
