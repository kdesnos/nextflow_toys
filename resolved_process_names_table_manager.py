from extract_from_nf_log import extractResolvedProcessNames
from processes_table_manager import ProcessEntry


class ResolvedProcessNamesTableManager:
    def __init__(self, connection):
        """
        Initialize the ResolvedProcessNamesTableManager with a database connection.
        """
        self.connection = connection

    def addResolvedProcessName(self, resolved_entry):
        """
        Add a resolved process name entry to the ResolvedProcessNames table in the SQLite database.

        :param resolved_entry: A ResolvedProcessEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO ResolvedProcessNames (pId, name) VALUES (?, ?);",
            (resolved_entry.pId, resolved_entry.name),
        )
        self.connection.commit()

        # Retrieve the last inserted row ID and update the resolved_entry's rId
        resolved_entry.rId = cursor.lastrowid

    def addAllResolvedProcessNames(self, resolved_entries):
        """
        Add multiple resolved process name entries to the ResolvedProcessNames table.

        :param resolved_entries: A collection of ResolvedProcessEntry instances.
        """
        for resolved_entry in resolved_entries:
            self.addResolvedProcessName(resolved_entry)

    def addResolvedProcessNamesToTable(self, trace_db_manager, file_path):
        """
        Extract resolved process names from a Nextflow log file and add them to the ResolvedProcessNames table.

        :param file_path: The path to the Nextflow log file.
        :param trace_db_manager: An instance of NextflowTraceDBManager to resolve pId.
        """

        # Extract resolved process names using the extractResolvedProcessNames function
        resolved_names = extractResolvedProcessNames(file_path)

        # Add each resolved name to the database
        for _, row in resolved_names.iterrows():
            # Check if the resolved name already exists in the database
            existing_entry = self.getResolvedProcessByName(row["resolved_name"])
            if not existing_entry:
                # Retrieve the pId for the process name and path
                process = trace_db_manager.process_manager.getProcessByNameAndPath(row["process_name"], row["path"])
                if process is None:
                    raise Exception(f"Process '{row['process_name']}' not found in Processes table.")

                resolved_entry = ResolvedProcessEntry(
                    rId=0, pId=process.pId, name=row["resolved_name"]
                )
                self.addResolvedProcessName(resolved_entry)

    def getResolvedProcessByName(self, name):
        """
        Retrieve a resolved process entry from the database by its name.

        :param name: The resolved name to retrieve.
        :return: A ResolvedProcessEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT rId, pId, name FROM ResolvedProcessNames WHERE name = ?;", (name,))
        row = cursor.fetchone()

        if row:
            return ResolvedProcessEntry(rId=row[0], pId=row[1], name=row[2])
        return None

    def getAllResolvedProcessNames(self):
        """
        Retrieve all resolved process entries from the database.

        :return: A list of ResolvedProcessEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT rId, pId, name FROM ResolvedProcessNames;")
        rows = cursor.fetchall()

        return [ResolvedProcessEntry(rId=row[0], pId=row[1], name=row[2]) for row in rows]

    def getResolvedProcessesOfProcess(self, name):
        """
        Retrieve all resolved process entries associated with a specific process name.

        :param name: The process name to search for.
        :return: A list of ResolvedProcessEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT rId, pId, name FROM ResolvedProcessNames WHERE pId = (SELECT pId FROM Processes WHERE name = ?);", (name,))
        rows = cursor.fetchall()

        return [ResolvedProcessEntry(rId=row[0], pId=row[1], name=row[2]) for row in rows]

    def removeResolvedProcessByName(self, name):
        """
        Remove a resolved process entry from the database by its name.

        :param name: The resolved name to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM ResolvedProcessNames WHERE name = ?;", (name,))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0

    def getProcessByResolvedName(self, resolved_name):
        """
        Retrieve the process entry corresponding to a given resolved process name.

        :param resolved_name: The resolved process name to search for.
        :return: A ProcessEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        query = """
            SELECT p.pId, p.name, p.path
            FROM Processes p
            JOIN ResolvedProcessNames rpn ON p.pId = rpn.pId
            WHERE rpn.name = ?;
        """
        cursor.execute(query, (resolved_name,))
        row = cursor.fetchone()

        if row:
            return ProcessEntry(pId=row[0], name=row[1], path=row[2])
        return None


class ResolvedProcessEntry:
    def __init__(self, rId, pId, name):
        """
        Initialize a ResolvedProcessEntry instance.

        :param rId: Resolved ID (integer).
        :param pId: Process ID (integer).
        :param name: Resolved name (string).
        """
        self.rId = rId
        self.pId = pId
        self.name = name

    def __repr__(self):
        return f"ResolvedProcessEntry(rId={self.rId}, pId={self.pId}, name='{self.name}')"
