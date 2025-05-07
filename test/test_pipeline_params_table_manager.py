import unittest
from unittest.mock import patch
from nf_trace_db_manager import NextflowTraceDBManager
from pipeline_params_table_manager import PipelineParamsTableManager, PipelineParamEntry
import pandas as pd


class TestPipelineParamsTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Use the PipelineParamsTableManager from the database manager
        self.pipeline_params_manager = self.db_manager.pipeline_params_manager

    def tearDown(self):
        self.db_manager.close()

    def test_addPipelineParam(self):
        param_entry = PipelineParamEntry(paramId=0, name="param1", type="String")
        self.pipeline_params_manager.addPipelineParam(param_entry)

        self.assertNotEqual(param_entry.paramId, 0)  # paramId should be updated after insertion

        # Verify the parameter was added
        retrieved_param = self.pipeline_params_manager.getPipelineParamByName("param1")
        self.assertIsNotNone(retrieved_param)
        self.assertEqual(retrieved_param.name, "param1")
        self.assertEqual(retrieved_param.type, "String")

    def test_getAllPipelineParams(self):
        param1 = PipelineParamEntry(paramId=0, name="param1", type="String")
        param2 = PipelineParamEntry(paramId=0, name="param2", type="Integer")
        self.pipeline_params_manager.addPipelineParam(param1)
        self.pipeline_params_manager.addPipelineParam(param2)

        all_params = self.pipeline_params_manager.getAllPipelineParams()
        self.assertEqual(len(all_params), 2)
        self.assertEqual(all_params[0].name, "param1")
        self.assertEqual(all_params[1].name, "param2")

    def test_removePipelineParam(self):
        param_entry = PipelineParamEntry(paramId=0, name="param1", type="String")
        self.pipeline_params_manager.addPipelineParam(param_entry)

        # Remove the parameter
        result = self.pipeline_params_manager.removePipelineParam("param1")
        self.assertTrue(result)

        # Verify the parameter was removed
        retrieved_param = self.pipeline_params_manager.getPipelineParamByName("param1")
        self.assertIsNone(retrieved_param)

    def test_paramNameUnicity(self):
        # Add a parameter with a unique name
        param_entry = PipelineParamEntry(paramId=0, name="param1", type="String")
        self.pipeline_params_manager.addPipelineParam(param_entry)

        # Attempt to add another parameter with the same name
        duplicate_param = PipelineParamEntry(paramId=0, name="param1", type="Integer")
        with self.assertRaises(Exception) as context:
            self.pipeline_params_manager.addPipelineParam(duplicate_param)

        # Verify that the exception is related to a UNIQUE constraint
        self.assertIn("UNIQUE constraint failed", str(context.exception))

    @patch("pipeline_params_table_manager.extractPipelineParameters")
    def test_addPipelineParamsFromLog(self, mock_extract_pipeline_parameters):
        # Mock the extractPipelineParameters function
        mock_data = pd.DataFrame([
            {"param_name": "param1", "type": "String"},
            {"param_name": "param2", "type": "Integer"},
            {"param_name": "param3", "type": "Real"},
            {"param_name": "param4", "type": "List[Integer]"},
            {"param_name": "param5", "type": "list[int]"},
            {"param_name": "param6", "type": "Path"},
            {"param_name": "param7", "type": "Boolean"}
        ])
        mock_extract_pipeline_parameters.return_value = mock_data

        # Call the method to add pipeline parameters from the log
        self.pipeline_params_manager.addPipelineParamsFromLog(self.db_manager, "mock_log_file.log")

        # Verify that the parameters were added to the database
        all_params = self.pipeline_params_manager.getAllPipelineParams()
        self.assertEqual(len(all_params), 7)

        # Check the details of the added parameters
        self.assertEqual(all_params[0].name, "param1")
        self.assertEqual(all_params[0].type, "String")
        self.assertEqual(all_params[1].name, "param2")
        self.assertEqual(all_params[1].type, "Integer")
        self.assertEqual(all_params[2].name, "param3")
        self.assertEqual(all_params[2].type, "Real")


if __name__ == "__main__":
    unittest.main()
