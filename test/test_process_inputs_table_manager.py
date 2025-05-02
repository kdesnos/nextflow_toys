import unittest
import sqlite3
from nf_trace_db_manager import NextflowTraceDBManager
from process_inputs_table_manager import ProcessInputsTableManager, ProcessInputEntry
from processes_table_manager import ProcessEntry


class TestProcessInputsTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        p = ProcessEntry(0, "process_1", "/path/to/process.nf")
        self.db_manager.process_manager.addProcess(p)

        self.inputs_manager = self.db_manager.process_inputs_manager

    def tearDown(self):
        self.db_manager.close()

    def test_addProcessInput(self):
        input_entry = ProcessInputEntry(pId=1, rank=0, type="input", name="input1")
        self.inputs_manager.addProcessInput(input_entry)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[0].type, "input")

    def test_addProcessInputWithNonExistentPid(self):
        # Attempt to add a process input with a non-existent pId
        input_entry = ProcessInputEntry(pId=999, rank=0, type="input", name="input1")
        with self.assertRaises(sqlite3.IntegrityError):
            self.inputs_manager.addProcessInput(input_entry)

    def test_addAllProcessInputs(self):
        input_entries = [
            ProcessInputEntry(pId=1, rank=0, type="input", name="input1"),
            ProcessInputEntry(pId=1, rank=1, type="input", name="input2"),
        ]
        self.inputs_manager.addAllProcessInputs(input_entries)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[1].name, "input2")

    def test_getProcessInputsByProcessId(self):
        input_entries = [
            ProcessInputEntry(pId=1, rank=0, type="input", name="input1"),
            ProcessInputEntry(pId=1, rank=1, type="input", name="input2"),
        ]
        self.inputs_manager.addAllProcessInputs(input_entries)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[1].name, "input2")

    def test_getAllProcessInputs(self):
        input_entries = [
            ProcessInputEntry(pId=1, rank=0, type="input", name="input1"),
            ProcessInputEntry(pId=1, rank=1, type="input", name="input2"),
        ]
        self.inputs_manager.addAllProcessInputs(input_entries)

        inputs = self.inputs_manager.getAllProcessInputs()
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].name, "input1")
        self.assertEqual(inputs[1].name, "input2")

    def test_removeProcessInput(self):
        input_entry = ProcessInputEntry(pId=1, rank=0, type="input", name="input1")
        self.inputs_manager.addProcessInput(input_entry)

        result = self.inputs_manager.removeProcessInput(1, 0)
        self.assertTrue(result)

        inputs = self.inputs_manager.getProcessInputsByProcessId(1)
        self.assertEqual(len(inputs), 0)

    def test_removeNonExistentProcessInput(self):
        result = self.inputs_manager.removeProcessInput(1, 0)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
