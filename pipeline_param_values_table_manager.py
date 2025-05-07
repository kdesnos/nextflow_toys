from extract_from_nf_log import extractPipelineParameters, extractRunName


class PipelineParamValuesTableManager:
    def __init__(self, connection):
        """
        Initialize the PipelineParamValuesTableManager with a database connection.
        """
        self.connection = connection

    def addPipelineParamValue(self, value_entry):
        """
        Add a pipeline parameter value entry to the PipelineParamValues table.

        :param value_entry: A PipelineParamValueEntry instance.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO PipelineParamValues (paramId, tId, value) VALUES (?, ?, ?);",
            (value_entry.paramId, value_entry.tId, value_entry.value),
        )
        self.connection.commit()

    def addPipelineParamValuesFromLog(self, trace_db_manager, file_path):
        """
        Extract pipeline parameter values from a log file and add them to the PipelineParamValues table.

        :param trace_db_manager: An instance of NextflowTraceDBManager to resolve trace IDs and parameter IDs.
        :param file_path: The path to the Nextflow log file.
        """
        # Extract trace run name from the log file
        run_name = extractRunName(file_path)
        if run_name is None:
            raise Exception(f"Trace run name not found in the log file '{file_path}'.")

        # Get the trace ID from the database using the run name
        trace_entry = trace_db_manager.trace_manager.getTraceEntry(run_name)
        if trace_entry is None:
            raise Exception(f"Trace '{run_name}' not found in the database.")

        # Extract pipeline parameters from the log file
        pipeline_params = extractPipelineParameters(file_path)

        # Iterate over the rows of the DataFrame and add each parameter value to the database
        for _, row in pipeline_params.iterrows():
            # Retrieve the parameter ID from the PipelineParams table
            param_entry = trace_db_manager.pipeline_params_manager.getPipelineParamByName(row["param_name"])
            if param_entry is None:
                raise Exception(f"Pipeline parameter '{row['param_name']}' not found in the PipelineParams table.")

            # Use the reformatted value if available, otherwise use the original value
            value = row["reformatted_value"] 

            # Add each value to the PipelineParamValues table
            value_entry = PipelineParamValueEntry(
                paramId=param_entry.paramId,
                tId=trace_entry.tId,
                value=f"{value}"
            )
            self.addPipelineParamValue(value_entry)

    def getPipelineParamValuesByTraceId(self, tId):
        """
        Retrieve all pipeline parameter values for a specific trace ID.

        :param tId: The trace ID to retrieve parameter values for.
        :return: A list of PipelineParamValueEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT paramId, tId, value FROM PipelineParamValues WHERE tId = ?;", (tId,))
        rows = cursor.fetchall()

        return [PipelineParamValueEntry(paramId=row[0], tId=row[1], value=row[2]) for row in rows]

    def getAllPipelineParamValues(self):
        """
        Retrieve all pipeline parameter values from the database.

        :return: A list of PipelineParamValueEntry instances.
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT paramId, tId, value FROM PipelineParamValues;")
        rows = cursor.fetchall()

        return [PipelineParamValueEntry(paramId=row[0], tId=row[1], value=row[2]) for row in rows]

    def removePipelineParamValue(self, paramId, tId):
        """
        Remove a pipeline parameter value entry from the database by its parameter ID and trace ID.

        :param paramId: The parameter ID of the value to remove.
        :param tId: The trace ID of the value to remove.
        :return: True if the entry existed and was successfully removed, False otherwise.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM PipelineParamValues WHERE paramId = ? AND tId = ?;", (paramId, tId))
        self.connection.commit()

        # Check if any rows were affected
        return cursor.rowcount > 0


class PipelineParamValueEntry:
    def __init__(self, paramId, tId, value):
        """
        Initialize a PipelineParamValueEntry instance.

        :param paramId: Parameter ID (integer).
        :param tId: Trace ID (integer).
        :param value: Value of the parameter (string).
        """
        self.paramId = paramId
        self.tId = tId
        self.value = value

    def __repr__(self):
        return f"PipelineParamValueEntry(paramId={self.paramId}, tId={self.tId}, value='{self.value}')"
