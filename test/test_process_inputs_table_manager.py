import unittest
import sqlite3
from unittest.mock import patch
from nf_trace_db_manager import NextflowTraceDBManager
from process_inputs_table_manager import ProcessInputsTableManager, ProcessInputEntry
from processes_table_manager import ProcessEntry
import pandas as pd

from resolved_process_names_table_manager import ResolvedProcessEntry


class TestProcessInputsTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Add a process entry for testing
        self.process_entry = ProcessEntry(pId=0, name="process_1", path="/path/to/process.nf")
        self.db_manager.process_manager.addProcess(self.process_entry)

        # Add resolved process entry for testing
        resolved_process_entry1 = ResolvedProcessEntry(0, self.process_entry.pId, "path:process_1")
        resolved_process_entry2 = ResolvedProcessEntry(0, self.process_entry.pId, "path:process_2")

        self.db_manager.resolved_process_manager.addResolvedProcessName(resolved_process_entry1)
        self.db_manager.resolved_process_manager.addResolvedProcessName(resolved_process_entry2)

        self.inputs_manager = self.db_manager.process_inputs_manager

    def tearDown(self):
        self.db_manager.close()

    def test_addProcessInput(self):
        input_entry = ProcessInputEntry(pId=1, rank='0', type="input", name="input1")
        self.inputs_manager.addProcessInput(input_entry)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[0].type, "input")
        self.assertEqual(inputs[0].rank, '0')

    def test_addProcessInputWithNonExistentPid(self):
        # Attempt to add a process input with a non-existent pId
        input_entry = ProcessInputEntry(pId=999, rank=0, type="input", name="input1")
        with self.assertRaises(sqlite3.IntegrityError):
            self.inputs_manager.addProcessInput(input_entry)

    def test_addAllProcessInputs(self):
        input_entries = [
            ProcessInputEntry(pId=1, rank='0', type="input", name="input1"),
            ProcessInputEntry(pId=1, rank='1.1', type="input", name="input2"),
        ]
        self.inputs_manager.addAllProcessInputs(input_entries)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[1].name, "input2")
        self.assertEqual(inputs[1].rank, "1.1")

    def test_getProcessInputsByProcessId(self):
        input_entries = [
            ProcessInputEntry(pId=1, rank='0', type="input", name="input1"),
            ProcessInputEntry(pId=1, rank='1', type="input", name="input2"),
        ]
        self.inputs_manager.addAllProcessInputs(input_entries)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[1].name, "input2")

    def test_getAllProcessInputs(self):
        input_entries = [
            ProcessInputEntry(pId=1, rank='0', type="input", name="input1"),
            ProcessInputEntry(pId=1, rank='1.1', type="input", name="input2"),
        ]
        self.inputs_manager.addAllProcessInputs(input_entries)

        inputs = self.inputs_manager.getAllProcessInputs()
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[1].name, "input2")

    def test_removeProcessInput(self):
        input_entry = ProcessInputEntry(pId=1, rank="0", type="input", name="input1")
        self.inputs_manager.addProcessInput(input_entry)

        result = self.inputs_manager.removeProcessInput(1, 0)
        self.assertTrue(result)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 0)

    def test_removeNonExistentProcessInput(self):
        result = self.inputs_manager.removeProcessInput(1, 0)
        self.assertFalse(result)

    def test_getProcessInputByProcessIdAndRank(self):
        input_entry = ProcessInputEntry(pId=self.process_entry.pId, rank="0", type="input", name="input1")
        self.inputs_manager.addProcessInput(input_entry)

        retrieved_input = self.inputs_manager.getProcessInputByProcessIdAndRank(self.process_entry.pId, "0")
        self.assertIsNotNone(retrieved_input)
        self.assertEqual(retrieved_input.name, "input1")
        self.assertEqual(retrieved_input.type, "input")
        self.assertEqual(retrieved_input.rank, "0")

    @patch("process_inputs_table_manager.extractProcessInputs")
    def test_addInputsFromLog(self, mock_extract_process_inputs):
        # Mock the extractProcessInputs function
        mock_data = pd.DataFrame([
            {
                "resolved_process_name": "path:process_1",
                "inputs": [
                    {"type": "string", "value": "arg1"},
                    {"type": "tuple", "value": [
                        {"type": "int", "value": "arg2"},
                        {"type": "float", "value": "arg3"}
                    ]}
                ]
            }
        ])
        mock_extract_process_inputs.return_value = mock_data

    # Call the method to add inputs from the log
    @patch("process_inputs_table_manager.extractProcessInputs")
    def test_addInputsFromLog_withDuplicateAndModifiedInputs(self, mock_extract_process_inputs):
        # Mock the extractProcessInputs function for the first call
        mock_data_1 = pd.DataFrame([
            {
                "resolved_process_name": "path:process_1",
                "inputs": [
                    {"type": "string", "value": "arg1"},
                    {"type": "tuple", "value": [
                        {"type": "int", "value": "arg2"},
                        {"type": "float", "value": "arg3"}
                    ]}
                ]
            }
        ])
        mock_extract_process_inputs.return_value = mock_data_1

        # Add inputs for process_1
        self.inputs_manager.addInputsFromLog(self.db_manager, "mock_log_file_1.log")

        # Verify that the inputs were added to the database
        inputs = self.inputs_manager.getProcessInputsByProcessId(self.process_entry.pId)
        self.assertEqual(len(inputs), 3)
        self.assertEqual(inputs[0].name, "arg1")
        self.assertEqual(inputs[0].type, "string")
        self.assertEqual(inputs[0].rank, "0")
        self.assertEqual(inputs[1].name, "arg2")
        self.assertEqual(inputs[1].type, "int")
        self.assertEqual(inputs[1].rank, "1.0")
        self.assertEqual(inputs[2].name, "arg3")
        self.assertEqual(inputs[2].type, "float")
        self.assertEqual(inputs[2].rank, "1.1")

        # Mock the extractProcessInputs function for the second call
        mock_data_2 = pd.DataFrame([
            {
                "resolved_process_name": "path:process_2",
                "inputs": [
                    {"type": "string", "value": "arg1"},
                    {"type": "tuple", "value": [
                        {"type": "int", "value": "arg2"},
                        {"type": "float", "value": "arg3"}
                    ]}
                ]
            }
        ])
        mock_extract_process_inputs.return_value = mock_data_2

        # Add inputs for process_2
        self.inputs_manager.addInputsFromLog(self.db_manager, "mock_log_file_2.log")

        # Mock the extractProcessInputs function for the third call with modified data
        mock_data_3 = pd.DataFrame([
            {
                "resolved_process_name": "path:process_2",
                "inputs": [
                    {"type": "string", "value": "arg1"},
                    {"type": "tuple", "value": [
                        {"type": "string", "value": "arg2"},  # Modified type from int to string
                        {"type": "float", "value": "arg3"}
                    ]}
                ]
            }
        ])
        mock_extract_process_inputs.return_value = mock_data_3

        # Attempt to add modified inputs for process_2 and expect an exception
        with self.assertRaises(Exception) as context:
            self.inputs_manager.addInputsFromLog(self.db_manager, "mock_log_file_3.log")

        self.assertIn("is aliased multiple times with different input arguments", str(context.exception))


if __name__ == "__main__":
    unittest.main()
