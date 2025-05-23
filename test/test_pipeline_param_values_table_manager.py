import unittest
from unittest.mock import patch, MagicMock
from nf_trace_db_manager import NextflowTraceDBManager
from pipeline_param_values_table_manager import PipelineParamValuesTableManager, PipelineParamValueEntry
from pipeline_params_table_manager import PipelineParamEntry
from trace_table_manager import TraceEntry
import pandas as pd


class TestPipelineParamValuesTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Add required data into the dependent tables
        # Add a trace entry
        self.trace_entry1 = TraceEntry(tId=0, day="2025-01-01", name="trace_1")
        self.db_manager.trace_manager.addTraceEntry(self.trace_entry1)

        # Add pipeline parameters
        self.param1 = PipelineParamEntry(paramId=0, name="param1", type="String")
        self.param2 = PipelineParamEntry(paramId=0, name="param2", type="Integer")
        self.db_manager.pipeline_params_manager.addPipelineParam(self.param1)
        self.db_manager.pipeline_params_manager.addPipelineParam(self.param2)

        # Use the PipelineParamValuesTableManager from the database manager
        self.param_values_manager = self.db_manager.pipeline_param_values_manager

    def tearDown(self):
        self.db_manager.close()

    def test_addPipelineParamValue(self):
        # Add a pipeline parameter value
        value_entry = PipelineParamValueEntry(
            paramId=self.param1.paramId,
            tId=self.trace_entry1.tId,
            value="value1"
        )
        self.param_values_manager.addPipelineParamValue(value_entry)

        # Verify the parameter value was added
        values = self.param_values_manager.getPipelineParamValuesByTraceId(self.trace_entry1.tId)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].value, "value1")
        self.assertEqual(values[0].paramId, self.param1.paramId)

    def test_getAllPipelineParamValues(self):
        # Add multiple pipeline parameter values
        value_entry1 = PipelineParamValueEntry(
            paramId=self.param1.paramId,
            tId=self.trace_entry1.tId,
            value="value1"
        )
        value_entry2 = PipelineParamValueEntry(
            paramId=self.param2.paramId,
            tId=self.trace_entry1.tId,
            value="value2"
        )
        self.param_values_manager.addPipelineParamValue(value_entry1)
        self.param_values_manager.addPipelineParamValue(value_entry2)

        # Verify all parameter values were retrieved
        all_values = self.param_values_manager.getAllPipelineParamValues()
        self.assertEqual(len(all_values), 2)
        self.assertEqual(all_values[0].value, "value1")
        self.assertEqual(all_values[1].value, "value2")

    def test_removePipelineParamValue(self):
        # Add a pipeline parameter value
        value_entry = PipelineParamValueEntry(
            paramId=self.param1.paramId,
            tId=self.trace_entry1.tId,
            value="value1"
        )
        self.param_values_manager.addPipelineParamValue(value_entry)

        # Remove the parameter value
        result = self.param_values_manager.removePipelineParamValue(
            paramId=self.param1.paramId,
            tId=self.trace_entry1.tId
        )
        self.assertTrue(result)

        # Verify the parameter value was removed
        values = self.param_values_manager.getPipelineParamValuesByTraceId(self.trace_entry1.tId)
        self.assertEqual(len(values), 0)

    def test_removeNonExistentPipelineParamValue(self):
        # Attempt to remove a non-existent parameter value
        result = self.param_values_manager.removePipelineParamValue(
            paramId=999,
            tId=self.trace_entry1.tId
        )
        self.assertFalse(result)

    def test_uniqueParamIdAndTraceId(self):
        # Add a pipeline parameter value
        value_entry1 = PipelineParamValueEntry(
            paramId=self.param1.paramId,
            tId=self.trace_entry1.tId,
            value="value1"
        )
        self.param_values_manager.addPipelineParamValue(value_entry1)

        # Attempt to add another value for the same paramId and tId
        value_entry2 = PipelineParamValueEntry(
            paramId=self.param1.paramId,
            tId=self.trace_entry1.tId,
            value="value2"
        )
        with self.assertRaises(Exception) as context:
            self.param_values_manager.addPipelineParamValue(value_entry2)

        # Verify that the exception is related to a UNIQUE constraint
        self.assertIn("UNIQUE constraint failed", str(context.exception))

    @patch("pipeline_param_values_table_manager.extractPipelineParameters")
    @patch("pipeline_param_values_table_manager.extractRunName")
    def test_addPipelineParamValuesFromLog(self, mock_extract_run_name, mock_extract_pipeline_parameters):
        # Mock the extractRunName function
        mock_extract_run_name.return_value = "mock_run_name"

        # Mock the extractPipelineParameters function with a pandas DataFrame
        mock_pipeline_params_df = pd.DataFrame([
            {"param_name": "param1", "original_value": "true", "type": "Boolean", "reformatted_value": True},
            {"param_name": "param2", "original_value": "1 2 3", "type": "Integer", "reformatted_value": [1, 2, 3]},
        ])
        mock_extract_pipeline_parameters.return_value = mock_pipeline_params_df

        # Mock the TraceTableManager to return a trace entry
        mock_trace_entry = MagicMock(tId=1, name="mock_run_name")
        self.db_manager.trace_manager.getTraceEntry = MagicMock(return_value=mock_trace_entry)

        # Call the method to add pipeline parameter values from the log
        self.param_values_manager.addPipelineParamValuesFromLog(self.db_manager, "mock_log_file.log")

        # Verify that the parameter values were added to the database
        all_values = self.param_values_manager.getAllPipelineParamValues()
        self.assertEqual(len(all_values), 2)  # 1 value for param1, 1 values for param2
        self.assertEqual(all_values[0].paramId, self.param1.paramId)
        self.assertEqual(all_values[0].value, 'True')
        self.assertEqual(all_values[1].paramId, self.param2.paramId)
        self.assertEqual(all_values[1].value, "[1, 2, 3]")

    def test_getParamValuesForTraces(self):
        # Add pipeline parameter values
        value_entry1 = PipelineParamValueEntry(
            paramId=self.param1.paramId,
            tId=self.trace_entry1.tId,
            value="value1"
        )
        value_entry2 = PipelineParamValueEntry(
            paramId=self.param2.paramId,
            tId=self.trace_entry1.tId,
            value="42"
        )
        self.param_values_manager.addPipelineParamValue(value_entry1)
        self.param_values_manager.addPipelineParamValue(value_entry2)

        # Retrieve parameter values for the trace
        param_values = self.param_values_manager.getParamValuesForTraces([self.trace_entry1.name])
        self.assertIn(self.trace_entry1.name, param_values)
        self.assertEqual(param_values[self.trace_entry1.name]["param1"], "value1")
        self.assertEqual(param_values[self.trace_entry1.name]["param2"], 42)

        # Test with a non-existent trace
        param_values_empty = self.param_values_manager.getParamValuesForTraces(["non_existent_trace"])
        self.assertEqual(param_values_empty, {})


if __name__ == "__main__":
    unittest.main()