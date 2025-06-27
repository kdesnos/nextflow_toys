import re

import pandas
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
            "INSERT INTO ProcessExecutions (tId, rId, instance, hash, time, cpu, nbCores, memory, allocated_mem) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
            (execution_entry.tId,
             execution_entry.rId,
             execution_entry.instance,
             execution_entry.hash,
             execution_entry.time,
             execution_entry.cpu,
             execution_entry.nbCores,
             execution_entry.memory,
             execution_entry.allocated_mem)  # Include allocated_mem in the insert statement,
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
                time=row["realtime"].total_seconds() * 1000.0,
                cpu=row["cpu_model"],
                nbCores=row["cpus"],
                memory=row["peak_rss"] if not pandas.isna(row["memory"]) else None,
                allocated_mem= row["memory"] if not pandas.isna(row["memory"]) else None
            )
            self.addProcessExecution(execution_entry)

    def getProcessExecutionByHash(self, hash_value):
        """
        Retrieve a process execution entry from the database by its hash.

        :param hash_value: The hash of the process execution to retrieve.
        :return: A ProcessExecutionEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT eId, tId, rId, instance, hash, time, cpu, nbCores, memory, allocated_mem FROM ProcessExecutions WHERE hash = ?;", (hash_value,))
        row = cursor.fetchone()

        if row:
            return ProcessExecutionEntry(eId=row[0], tId=row[1], rId=row[2], instance=row[3], hash=row[4],
                                         time=row[5], cpu=row[6], nbCores=row[7], memory=row[8], allocated_mem=row[9])
        return None

    def getExecutionByResolvedIdAndInstanceAndTraceId(self, rId, instance, tId):
        """
        Retrieve a process execution entry from the database by its resolved ID, instance number, and trace ID.

        :param rId: The resolved process ID to search for.
        :param instance: The instance number to search for.
        :param tId: the trace ID to search for.
        :return: A ProcessExecutionEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT eId, tId, rId, instance, hash, time, cpu, nbCores, memory, allocated_mem FROM ProcessExecutions WHERE rId = ? AND instance = ? AND tId = ?;",
            (rId, instance, tId),
        )
        row = cursor.fetchone()

        if row:
            return ProcessExecutionEntry(eId=row[0], tId=row[1], rId=row[2], instance=row[3], hash=row[4],
                                         time=row[5], cpu=row[6], nbCores=row[7], memory=row[8], allocated_mem=row[9])
        return None

    def getAllProcessExecutions(self):
        """
        Retrieve all process execution entries from the database.

        :return: A list of ProcessExecutionEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT eId, tId, rId, instance, hash, time, cpu, nbCores, memory, allocated_mem FROM ProcessExecutions;")
        rows = cursor.fetchall()

        return [
            ProcessExecutionEntry(
                eId=row[0],
                tId=row[1],
                rId=row[2],
                instance=row[3],
                hash=row[4],
                time=row[5],
                cpu=row[6],
                nbCores=row[7],
                memory=row[8],
                allocated_mem=row[9])
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

    def getExecutionMetricsForProcessAndTraces(self, process_name, trace_names=None, is_resolved_name=False):
        """
        Get execution times for a specific process and trace(s).

        :param process_name: Name of the process to retrieve execution times for.
        :param trace_names: List of trace names or a single trace name to filter the results. If None, fetch for all traces.
        :param is_resolved_name: Whether the process_name is a resolved name (True) or a basic process name (False).
        :return: A DataFrame containing execution times and related information for the specified process and traces.
        """
        import pandas as pd

        if isinstance(trace_names, str):
            trace_names = [trace_names]

        all_results = []
        query = f"""
            SELECT
                pe.eId,
                pe.time AS execution_time,
                pe.instance,
                pe.nbCores,
                pe.memory,
                pe.allocated_mem,
                t.name AS trace_name,
                rpn.name AS resolved_process_name
            FROM
                ProcessExecutions pe
            JOIN
                ResolvedProcessNames rpn ON pe.rId = rpn.rId
            JOIN
                Traces t ON pe.tId = t.tId
            JOIN
                Processes p ON rpn.pId = p.pId
            WHERE
                {"p" if not is_resolved_name else "rpn"}.name = ?
        """
        params = [process_name]

        if trace_names:
            query += " AND t.name IN ({})".format(",".join("?" for _ in trace_names))
            params.extend(trace_names)

        query += " ORDER BY pe.instance"

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        execution_data = cursor.fetchall()

        if execution_data:
            df = pd.DataFrame(execution_data,
                              columns=["eId", "execution_time", "instance",
                                       "nbCores", "memory", "allocated_mem", "trace_name", "resolved_process_name"])
            all_results.append(df)
        else:
            print(f"No execution times found for process '{process_name}'")

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def getProcessToCpuMapping(self, db_manager=None):
        """
        Builds a mapping between CPU core counts and the resolved process names that ran on them.

        :param db_manager: An instance of NextflowTraceDBManager (not used in this implementation).
        :return: A dictionary where keys are core counts and values are lists of resolved process names.
        """
        # Build a mapping of cpu types and associated processes
        process_to_cpu_mapping = {}

        # Direct SQL query to get the mapping
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT pe.nbCores, rpn.name
            FROM ProcessExecutions pe
            JOIN ResolvedProcessNames rpn ON pe.rId = rpn.rId
            GROUP BY pe.nbCores, rpn.name
            ORDER BY pe.nbCores
        """)

        rows = cursor.fetchall()

        # Transform query results into the desired dictionary format
        for core_id, actor_name in rows:
            if core_id not in process_to_cpu_mapping:
                process_to_cpu_mapping[core_id] = []
            process_to_cpu_mapping[core_id].append(actor_name)

        return process_to_cpu_mapping

class ProcessExecutionEntry:
    def __init__(self, eId, tId, rId, instance, hash, time, cpu, nbCores, memory, allocated_mem):
        """
        Initialize a ProcessExecutionEntry instance.

        :param eId: Execution ID (integer).
        :param tId: Trace ID (integer).
        :param rId: Resolved Process ID (integer).
        :param instance: Instance number (integer).
        :param hash: Hash of the execution (string).
        :param time: Execution time in milliseconds (float).
        :param cpu: Name of of CPUs used (string).
        :param nbCores: Number of CPU core used (integer).
        :param memory: Memory usage in bytes (float or None). Set to None if a per-core requirement was used.
        :param allocated_mem: Memory allocated to the process (float or None). Set to None if a per-core requirement was used.
        """
        self.eId = eId
        self.tId = tId
        self.rId = rId
        self.instance = instance
        self.hash = hash
        self.time = time
        self.cpu = cpu
        self.nbCores = nbCores
        self.memory = memory
        self.allocated_mem = allocated_mem

    def __repr__(self):
        return (
            f"ProcessExecutionEntry(eId={self.eId}, tId={self.tId}, rId={self.rId}, "
            f"instance={self.instance}, hash='{self.hash}', time={self.time}, "
            f"cpu='{self.cpu}', nbCores={self.nbCores}, memory={self.memory},"
            f"allocated_mem={self.allocated_mem})"
        )
