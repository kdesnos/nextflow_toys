from extract_from_nf_log import extractPipelineParameters


class PipelineParamsTableManager:
    def __init__(self, connection):
        """
        Initialize the PipelineParamsTableManager with a database connection.
        """
        self.connection = connection

    def addPipelineParam(self, param_entry):
        """
        Add a pipeline parameter entry to the PipelineParams table.

        :param param_entry: A PipelineParamEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO PipelineParams (name, type) VALUES (?, ?);",
            (param_entry.name, param_entry.type),
        )
        self.connection.commit()

        # Retrieve the last inserted row ID and update the param_entry's paramId
        param_entry.paramId = cursor.lastrowid

    def getPipelineParamByName(self, name):
        """
        Retrieve a pipeline parameter entry from the database by its name.

        :param name: The name of the parameter to retrieve.
        :return: A PipelineParamEntry instance if found, None otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT paramId, name, type FROM PipelineParams WHERE name = ?;", (name,))
        row = cursor.fetchone()

        if row:
            return PipelineParamEntry(paramId=row[0], name=row[1], type=row[2])
        return None

    def getAllPipelineParams(self):
        """
        Retrieve all pipeline parameter entries from the database.

        :return: A list of PipelineParamEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT paramId, name, type FROM PipelineParams;")
        rows = cursor.fetchall()

        return [PipelineParamEntry(paramId=row[0], name=row[1], type=row[2]) for row in rows]

    def removePipelineParam(self, name):
        """
        Remove a pipeline parameter entry from the database by its name.

        :param name: The name of the parameter to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM PipelineParams WHERE name = ?;", (name,))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0

    def addPipelineParamsFromLog(self, db_manager, file_path):
        """
        Extract pipeline parameters from a log file and add them to the PipelineParams table.

        :param db_manager: The database manager instance.
        :param file_path: The path to the Nextflow log file.
        """
        # Extract pipeline parameters from the log file
        pipeline_params = extractPipelineParameters(file_path)

        # Iterate over the rows of the DataFrame and add each parameter to the database
        for _, row in pipeline_params.iterrows():
            # Check if the parameter already exists in the table
            existing_param = self.getPipelineParamByName(row["param_name"])
            if existing_param is None:
                # Add the parameter if it doesn't exist
                param_entry = PipelineParamEntry(
                    paramId=0,
                    name=row["param_name"],
                    type=row["type"]
                )
                self.addPipelineParam(param_entry)


class PipelineParamEntry:

    def __init__(self, paramId, name, type):
        """
        Initialize a PipelineParamEntry instance.

        :param paramId: Parameter ID (integer).
        :param name: Name of the parameter (string).
        :param type: Type of the parameter (string).
        """
        self.paramId = paramId
        self.name = name
        self.type = type


    def __repr__(self):
        return f"PipelineParamEntry(paramId={self.paramId}, name='{self.name}', type='{self.type}')"
