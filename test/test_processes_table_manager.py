import unittest
import sqlite3
from unittest.mock import patch
from processes_table_manager import ProcessesTableManager, ProcessEntry
from nf_trace_db_manager import NextflowTraceDBManager
import pandas as pd


class TestProcessesTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Use the ProcessesTableManager from the database manager
        self.process_manager = self.db_manager.process_manager

    def tearDown(self):
        self.db_manager.close()

    def test_addProcess(self):
        process = ProcessEntry(pId=0, name="process1", path="/path/to/process1.nf")
        self.process_manager.addProcess(process)
        self.assertNotEqual(process.pId, 0)  # pId should be updated after insertion

    def test_getProcessByName(self):
        process = ProcessEntry(pId=0, name="process1", path="/path/to/process1.nf")
        self.process_manager.addProcess(process)

        retrieved_process = self.process_manager.getProcessByName("process1")
        self.assertIsNotNone(retrieved_process)
        self.assertEqual(retrieved_process.name, "process1")
        self.assertEqual(retrieved_process.path, "/path/to/process1.nf")

    def test_getAllProcesses(self):
        processes = [
            ProcessEntry(pId=0, name="process1", path="/path/to/process1.nf"),
            ProcessEntry(pId=0, name="process2", path="/path/to/process2.nf"),
        ]
        for process in processes:
            self.process_manager.addProcess(process)

        all_processes = self.process_manager.getAllProcesses()
        self.assertEqual(len(all_processes), 2)
        self.assertEqual(all_processes[0].name, "process1")
        self.assertEqual(all_processes[1].name, "process2")

    def test_removeProcessByName(self):
        process = ProcessEntry(pId=0, name="process1", path="/path/to/process1.nf")
        self.process_manager.addProcess(process)

        result = self.process_manager.removeProcessByName("process1")
        self.assertTrue(result)

        # Ensure the process is removed
        retrieved_process = self.process_manager.getProcessByName("process1")
        self.assertIsNone(retrieved_process)

    def test_removeNonExistentProcess(self):
        result = self.process_manager.removeProcessByName("non_existent_process")
        self.assertFalse(result)

    def test_addProcesses(self):
        processes = [
            ProcessEntry(pId=0, name="process1", path="/path/to/process1.nf"),
            ProcessEntry(pId=0, name="process2", path="/path/to/process2.nf"),
        ]
        self.process_manager.addProcesses(processes)

        all_processes = self.process_manager.getAllProcesses()
        self.assertEqual(len(all_processes), 2)
        self.assertEqual(all_processes[0].name, "process1")
        self.assertEqual(all_processes[1].name, "process2")

    @patch("processes_table_manager.extractProcessDefinitions")
    def test_addProcessDefinitionsToTable(self, mock_extract_process_definitions):
        # Mock the extractProcessDefinitions function
        mock_data = [
            {"path": "/home/user/sub.nf", "process_name": "x"},
            {"path": "/home/user/sub.nf", "process_name": "a"},
            {"path": "/home/user/sub.nf", "process_name": "b"},
            {"path": "/home/user/main.nf", "process_name": "a"},
            {"path": "/home/user/main.nf", "process_name": "x"},
        ]
        mock_extract_process_definitions.return_value = pd.DataFrame(mock_data)

        # Call the method to add process definitions to the table
        self.process_manager.addProcessDefinitionsToTable("mock_log_file.log")

        # Verify that the processes were added to the database
        all_processes = self.process_manager.getAllProcesses()
        self.assertEqual(len(all_processes), 5)
        self.assertEqual(all_processes[0].name, "x")
        self.assertEqual(all_processes[0].path, "/home/user/sub.nf")
        self.assertEqual(all_processes[4].name, "x")
        self.assertEqual(all_processes[4].path, "/home/user/main.nf")        


if __name__ == "__main__":
    unittest.main()