from extract_from_nf_log import extractProcessInputs


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

    def addInputsFromLog(self, trace_db_manager, file_path):
        """
        Extract process inputs from a log file and add them to the ProcessInputs table.

        :param trace_db_manager: An instance of NextflowTraceDBManager to resolve process IDs.
        :param file_path: The path to the Nextflow log file.
        """
        # Extract process inputs from the log file
        process_inputs = extractProcessInputs(file_path)

        def add_nested_inputs(pId, args, base_rank):
            """
            Recursively add inputs (both top-level and nested) to the database.

            :param resolved_process_entry: The associated ResolvedProcessEntry
            :param args: List of arguments (can be top-level or nested).
            :param base_rank: Base rank for the arguments.
            """
            for i, arg in enumerate(args):
                rank = f"{base_rank}.{i}" if base_rank else str(i)
                if isinstance(arg["value"], list):  # Handle nested tuples
                    add_nested_inputs(resolved_process_entry, arg["value"], rank)
                else:
                    input_entry = ProcessInputEntry(
                        pId=resolved_process_entry.pId,
                        rank=rank,
                        type=arg["type"],
                        name=arg["value"]
                    )
                    # Check if the process input exists first
                    existing_entry = self.getProcessInputByProcessIdAndRank(resolved_process_entry.pId, rank)
                    if (existing_entry is not None):
                        if (existing_entry != input_entry):
                            raise Exception(
                                f"Process with pId:{
                                    resolved_process_entry.pId} resolved as {
                                    resolved_process_entry.name} is aliased multiple times with different input arguments.")
                    else:
                        self.addProcessInput(input_entry)

        # Iterate over the rows of the DataFrame and add each input to the database
        for _, row in process_inputs.iterrows():
            # Resolve the process ID (pId) using the ResolvedProcessNamesTableManager
            resolved_process_entry = trace_db_manager.resolved_process_manager.getResolvedProcessByName(row["resolved_process_name"])
            if resolved_process_entry is None:
                raise Exception(f"Process '{row['resolved_process_name']}' not found in Processes table.")

            # Add all input arguments (top-level and nested) to the database
            add_nested_inputs(resolved_process_entry, row["inputs"], base_rank="")

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

    def getProcessInputByProcessIdAndRank(self, process_id, rank):
        """
        Retrieve a specific process input entry by process ID and rank.

        :param process_id: The process ID of the input to retrieve.
        :param rank: The rank of the input to retrieve.
        :return: A ProcessInputEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT pId, rank, type, name FROM ProcessInputs WHERE pId = ? AND rank = ?;", (process_id, rank))
        row = cursor.fetchone()

        if row:
            return ProcessInputEntry(pId=row[0], rank=row[1], type=row[2], name=row[3])
        return None

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
        :param rank: Rank of the input (string).
        :param type: Type of the input (string).
        :param name: Name of the input (string).
        """
        self.pId = pId
        self.rank = rank
        self.type = type
        self.name = name

    def __repr__(self):
        return f"ProcessInputEntry(pId={self.pId}, rank='{self.rank}', type='{self.type}', name='{self.name}')"

    def __eq__(self, other):
        if not isinstance(other, ProcessInputEntry):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.pId == other.pId and self.rank == other.rank and self.type == other.type and self.name == other.name
