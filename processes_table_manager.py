from extract_from_nf_log import extractProcessDefinitions

class ProcessesTableManager:
    def __init__(self, connection):
        """
        Initialize the ProcessesTableManager with a database connection.
        """
        self.connection = connection

    def addProcess(self, process_entry):
        """
        Add a process entry to the Processes table in the SQLite database.

        :param process_entry: A ProcessEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO Processes (name, path) VALUES (?, ?);",
            (process_entry.name, process_entry.path),
        )
        self.connection.commit()

        # Retrieve the last inserted row ID and update the process_entry's pId
        process_entry.pId = cursor.lastrowid

    def addProcessDefinitionsToTable(self, file_path):
        """
        Extract process definitions from a Nextflow log file and add them to the Processes table.

        :param file_path: The path to the Nextflow log file.
        """
        # Extract process definitions using the extractProcessDefinitions function
        process_definitions = extractProcessDefinitions(file_path)

        # Add each process to the database
        for _, row in process_definitions.iterrows():
            process_entry = ProcessEntry(pId=0, name=row['process_name'], path=row['path'])
            self.addProcess(process_entry)

    def getProcessByName(self, name):
        """
        Retrieve a process entry from the database by its name.

        :param name: The name of the process to retrieve.
        :return: A ProcessEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT pId, name, path FROM Processes WHERE name = ?;", (name,))
        row = cursor.fetchone()

        if row:
            return ProcessEntry(pId=row[0], name=row[1], path=row[2])
        return None

    def getAllProcesses(self):
        """
        Retrieve all process entries from the database.

        :return: A list of ProcessEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT pId, name, path FROM Processes;")
        rows = cursor.fetchall()

        return [ProcessEntry(pId=row[0], name=row[1], path=row[2]) for row in rows]

    def removeProcessByName(self, name):
        """
        Remove a process entry from the database by its name.

        :param name: The name of the process to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM Processes WHERE name = ?;", (name,))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0

    def addProcesses(self, processes):
        """
        Add multiple process entries to the Processes table in the SQLite database.

        :param processes: A list of ProcessEntry instances.
        """
        for process in processes:
            self.addProcess(process)


class ProcessEntry:
    def __init__(self, pId, name, path):
        """
        Initialize a ProcessEntry instance.

        :param pId: Process ID (integer).
        :param name: Name of the process (string).
        :param path: Path to the process definition (string).
        """
        self.pId = pId
        self.name = name
        self.path = path

    def __repr__(self):
        return f"ProcessEntry(pId={self.pId}, name='{self.name}', path='{self.path}')"