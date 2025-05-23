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

    def getParamValuesForTraces(self, trace_names=None):
        """
        Get parameter values for specific trace names from the pipeline_param_values table.

        :param trace_names: List of trace names or a single trace name to filter the results. If None, fetch for all traces.
        :return: A dictionary mapping trace names to their parameter dictionaries. Each parameter dictionary contains
                 parameter names as keys and their corresponding values, converted to appropriate types.
        """
        import pandas as pd

        if isinstance(trace_names, str):
            trace_names = [trace_names]

        params_by_trace = {}
        query = """
            SELECT
                pp.name,
                ppv.value,
                pp.type,
                t.name AS trace_name
            FROM
                PipelineParamValues ppv
            JOIN
                PipelineParams pp ON ppv.paramId = pp.paramId
            JOIN
                Traces t ON ppv.tId = t.tId
        """
        params = []

        if trace_names:
            query += " WHERE t.name IN ({})".format(",".join("?" for _ in trace_names))
            params.extend(trace_names)

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        param_data = cursor.fetchall()

        if param_data:
            df = pd.DataFrame(param_data, columns=["param_name", "value", "type", "trace_name"])
            for trace_name, group in df.groupby("trace_name"):
                params_dict = {}
                for _, row in group.iterrows():
                    if row["type"] == "Boolean":
                        params_dict[row["param_name"]] = 1 if row["value"] == "True" else 0
                    elif row["type"] == "Integer":
                        params_dict[row["param_name"]] = int(row["value"])
                    elif row["type"] == "Real":
                        params_dict[row["param_name"]] = float(row["value"])
                    else:
                        params_dict[row["param_name"]] = row["value"]
                params_by_trace[trace_name] = params_dict
        else:
            print("No parameter values found for the specified traces.")

        return params_by_trace


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
