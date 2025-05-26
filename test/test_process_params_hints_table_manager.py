import unittest
from unittest.mock import patch, MagicMock
from nf_trace_db_manager import NextflowTraceDBManager
from process_params_hints_table_manager import ProcessParamsHintsTableManager, ProcessParamHintEntry
from processes_table_manager import ProcessEntry
from pipeline_params_table_manager import PipelineParamEntry


class TestProcessParamsHintsTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Add a process entry for testing
        self.process_entry = ProcessEntry(pId=1, name="process_1", path="/path/to/process_1.nf")
        self.db_manager.process_manager.addProcess(self.process_entry)

        # Add pipeline parameter entries for testing
        self.pipeline_param_entry1 = PipelineParamEntry(paramId=1, name="param1", type="String")
        self.pipeline_param_entry2 = PipelineParamEntry(paramId=2, name="param2", type="Integer")
        self.db_manager.pipeline_params_manager.addPipelineParam(self.pipeline_param_entry1)
        self.db_manager.pipeline_params_manager.addPipelineParam(self.pipeline_param_entry2)

        # Use the ProcessParamsHintsTableManager from the database manager
        self.hints_manager = self.db_manager.process_params_hints_manager

    def tearDown(self):
        self.db_manager.close()

    def test_addProcessParamHint(self):
        # Add a process parameter hint
        hint_entry = ProcessParamHintEntry(pId=self.process_entry.pId, paramId=self.pipeline_param_entry1.paramId)
        self.hints_manager.addProcessParamHint(hint_entry)

        # Verify the hint was added
        hints = self.hints_manager.getProcessParamHintsByProcessId(self.process_entry.pId)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].pId, self.process_entry.pId)
        self.assertEqual(hints[0].paramId, self.pipeline_param_entry1.paramId)

    def test_addAllProcessParamHints(self):
        # Add multiple process parameter hints
        hint_entries = [
            ProcessParamHintEntry(pId=self.process_entry.pId, paramId=self.pipeline_param_entry1.paramId),
            ProcessParamHintEntry(pId=self.process_entry.pId, paramId=self.pipeline_param_entry2.paramId),
        ]
        self.hints_manager.addAllProcessParamHints(hint_entries)

        # Verify the hints were added
        hints = self.hints_manager.getProcessParamHintsByProcessId(self.process_entry.pId)
        self.assertEqual(len(hints), 2)

    def test_getAllProcessParamHints(self):
        # Add multiple process parameter hints
        hint_entries = [
            ProcessParamHintEntry(pId=self.process_entry.pId, paramId=self.pipeline_param_entry1.paramId),
            ProcessParamHintEntry(pId=self.process_entry.pId, paramId=self.pipeline_param_entry2.paramId),
        ]
        self.hints_manager.addAllProcessParamHints(hint_entries)

        # Retrieve all hints
        hints = self.hints_manager.getAllProcessParamHints()
        self.assertEqual(len(hints), 2)

    def test_removeProcessParamHint(self):
        # Add a process parameter hint
        hint_entry = ProcessParamHintEntry(pId=self.process_entry.pId, paramId=self.pipeline_param_entry1.paramId)
        self.hints_manager.addProcessParamHint(hint_entry)

        # Remove the hint
        result = self.hints_manager.removeProcessParamHint(self.process_entry.pId, self.pipeline_param_entry1.paramId)
        self.assertTrue(result)

        # Verify the hint was removed
        hints = self.hints_manager.getProcessParamHintsByProcessId(self.process_entry.pId)
        self.assertEqual(len(hints), 0)

    def test_removeNonExistentProcessParamHint(self):
        # Attempt to remove a non-existent hint
        result = self.hints_manager.removeProcessParamHint(999, 999)
        self.assertFalse(result)

    @patch("os.path.isfile")
    @patch("process_params_hints_table_manager.extract_process_parameters_hints")
    def test_addProcessParamHintsFromCode(self, mock_extract_hints, mock_isfile):
        # Mock file existence
        mock_isfile.side_effect = lambda path: path in ["/path/to/process_1.nf"]

        # Mock parameter hints extraction
        mock_extract_hints.return_value = ["param1", "param2"]

        # Call the function
        self.hints_manager.addProcessParamHintsFromCode(self.db_manager)

        # Verify the hints were added
        self.assertEqual(len(self.hints_manager.getAllProcessParamHints()), 2)


if __name__ == "__main__":
    unittest.main()
