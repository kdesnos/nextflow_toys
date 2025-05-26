import os
from extract_from_process import extract_process_parameters_hints

class ProcessParamsHintsTableManager:
    def __init__(self, connection):
        """
        Initialize the ProcessParamsHintsTableManager with a database connection.
        """
        self.connection = connection

    def addProcessParamHint(self, hint_entry):
        """
        Add a process parameter hint entry to the ProcessParamHints table.

        :param hint_entry: A ProcessParamHintEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO ProcessParamHints (pId, paramId) VALUES (?, ?);",
            (hint_entry.pId, hint_entry.paramId),
        )
        self.connection.commit()

    def addAllProcessParamHints(self, hint_entries):
        """
        Add multiple process parameter hint entries to the ProcessParamHints table.

        :param hint_entries: A collection of ProcessParamHintEntry instances.
        """
        for hint_entry in hint_entries:
            self.addProcessParamHint(hint_entry)

    def getProcessParamHintsByProcessId(self, process_id):
        """
        Retrieve all process parameter hint entries for a specific process ID.

        :param process_id: The process ID to retrieve hints for.
        :return: A list of ProcessParamHintEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT pId, paramId FROM ProcessParamHints WHERE pId = ?;", (process_id,))
        rows = cursor.fetchall()

        return [ProcessParamHintEntry(pId=row[0], paramId=row[1]) for row in rows]

    def getAllProcessParamHints(self):
        """
        Retrieve all process parameter hint entries from the database.

        :return: A list of ProcessParamHintEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT pId, paramId FROM ProcessParamHints;")
        rows = cursor.fetchall()

        return [ProcessParamHintEntry(pId=row[0], paramId=row[1]) for row in rows]

    def removeProcessParamHint(self, process_id, param_id):
        """
        Remove a process parameter hint entry from the database by its process ID and parameter ID.

        :param process_id: The process ID of the hint to remove.
        :param param_id: The parameter ID of the hint to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM ProcessParamHints WHERE pId = ? AND paramId = ?;", (process_id, param_id))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0
    
    def addProcessParamHintsFromCode(self, db_manager, root_folder=None, prefix=None):
        """
        Extract process parameter hints from Nextflow files and store them in the ProcessParamsHints table.

        :param db_manager: An instance of NextflowTraceDBManager.
        :param root_folder: (Optional) A path to a root folder where process files are located.
        :param prefix: (Optional) A prefix path to replace with the root folder.
        """
        # Retrieve all process entries from the Processes table
        processes = db_manager.process_manager.getAllProcesses()

        for process in processes:
            process_name = process.name
            process_path = process.path

            # Adjust the process path if a root folder and prefix are provided
            if root_folder and prefix:
                if process_path.startswith(prefix):
                    process_path = os.path.join(root_folder, process_path[len(prefix):].lstrip(os.sep))

            # Check if the file exists
            if not os.path.isfile(process_path):
                print(f"Warning: File not found for process '{process_name}' at path '{process_path}'. Skipping.")
                continue

            # Extract parameter hints using the extract_process_parameters_hints function
            try:
                param_hints = extract_process_parameters_hints(process_path, process_name)
            except Exception as e:
                print(f"Error while extracting parameter hints for process '{process_name}': {e}")
                continue

            # Add parameter hints to the ProcessParamsHints table
            for param_name in param_hints:
                # Retrieve the paramId from the PipelineParams table
                pipeline_param = db_manager.pipeline_params_manager.getPipelineParamByName(param_name)
                if pipeline_param is None:
                    print(f"Warning: Parameter '{param_name}' not found in the PipelineParams table. Skipping.")
                    continue

                # Add the parameter hint to the ProcessParamsHints table
                hint_entry = ProcessParamHintEntry(pId=process.pId, paramId=pipeline_param.paramId)
                self.addProcessParamHint(hint_entry)

        print("Process parameter hints extraction and storage completed.")


class ProcessParamHintEntry:
    def __init__(self, pId, paramId):
        """
        Initialize a ProcessParamHintEntry instance.

        :param pId: Process ID (integer).
        :param paramId: Parameter ID (integer).
        """
        self.pId = pId
        self.paramId = paramId

    def __repr__(self):
        return f"ProcessParamHintEntry(pId={self.pId}, paramId={self.paramId})"
