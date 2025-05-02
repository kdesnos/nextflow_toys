import unittest
import sqlite3
from unittest.mock import patch
from resolved_process_names_table_manager import ResolvedProcessNamesTableManager, ResolvedProcessEntry
from processes_table_manager import ProcessesTableManager, ProcessEntry
from nf_trace_db_manager import NextflowTraceDBManager
import pandas as pd


class TestResolvedProcessNamesTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Use the ResolvedProcessNamesTableManager and ProcessesTableManager from the database manager
        self.resolved_manager = self.db_manager.resolved_process_manager
        self.process_manager = self.db_manager.process_manager

        # Add a process to the Processes table for testing
        self.process = ProcessEntry(pId=0, name="process1", path="/path/to/process1.nf")
        self.process_manager.addProcess(self.process)

    def tearDown(self):
        self.db_manager.close()

    def test_addResolvedProcessName(self):
        resolved_entry = ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved1")
        self.resolved_manager.addResolvedProcessName(resolved_entry)
        self.assertNotEqual(resolved_entry.rId, 0)  # rId should be updated after insertion

    def test_addResolvedProcessNameWithNonExistentPid(self):
        # Attempt to add a resolved process name with a non-existent pId
        non_existent_pid = 999  # Arbitrary pId that doesn't exist in the Processes table
        resolved_entry = ResolvedProcessEntry(rId=0, pId=non_existent_pid, name="resolved_non_existent_pid")

        with self.assertRaises(Exception) as context:
            self.resolved_manager.addResolvedProcessName(resolved_entry)

        self.assertIn("FOREIGN KEY constraint failed", str(context.exception))

    def test_getResolvedProcessByName(self):
        resolved_entry = ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved1")
        self.resolved_manager.addResolvedProcessName(resolved_entry)

        retrieved_entry = self.resolved_manager.getResolvedProcessByName("resolved1")
        self.assertIsNotNone(retrieved_entry)
        self.assertEqual(retrieved_entry.name, "resolved1")
        self.assertEqual(retrieved_entry.pId, self.process.pId)

    def test_getAllResolvedProcessNames(self):
        resolved_entries = [
            ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved1"),
            ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved2"),
        ]
        for entry in resolved_entries:
            self.resolved_manager.addResolvedProcessName(entry)

        all_resolved = self.resolved_manager.getAllResolvedProcessNames()
        self.assertEqual(len(all_resolved), 2)
        self.assertEqual(all_resolved[0].name, "resolved1")
        self.assertEqual(all_resolved[1].name, "resolved2")

    def test_removeResolvedProcessByName(self):
        resolved_entry = ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved1")
        self.resolved_manager.addResolvedProcessName(resolved_entry)

        result = self.resolved_manager.removeResolvedProcessByName("resolved1")
        self.assertTrue(result)

        # Ensure the resolved process is removed
        retrieved_entry = self.resolved_manager.getResolvedProcessByName("resolved1")
        self.assertIsNone(retrieved_entry)

    def test_removeNonExistentResolvedProcess(self):
        result = self.resolved_manager.removeResolvedProcessByName("non_existent_resolved")
        self.assertFalse(result)

    @patch("resolved_process_names_table_manager.extractResolvedProcessNames")
    def test_addResolvedProcessNamesToTable(self, mock_extract_resolved_names):
        # Mock the extractResolvedProcessNames function
        mock_data = [
            {"path": "/path/to/process1.nf", "process_name": "process1", "resolved_name": "resolved1"},
            {"path": "/path/to/process1.nf", "process_name": "process1", "resolved_name": "resolved2"},
        ]
        mock_extract_resolved_names.return_value = pd.DataFrame(mock_data)

        # Call the method to add resolved process names to the table
        self.resolved_manager.addResolvedProcessNamesToTable("mock_log_file.log", self.process_manager)

        # Verify that the resolved names were added to the database
        all_resolved = self.resolved_manager.getAllResolvedProcessNames()
        self.assertEqual(len(all_resolved), 2)
        self.assertEqual(all_resolved[0].name, "resolved1")
        self.assertEqual(all_resolved[1].name, "resolved2")

    def test_addAllResolvedProcessNames(self):
        # Create multiple resolved process entries
        resolved_entries = [
            ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved1"),
            ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved2"),
            ResolvedProcessEntry(rId=0, pId=self.process.pId, name="resolved3"),
        ]

        # Add all resolved process names to the table
        self.resolved_manager.addAllResolvedProcessNames(resolved_entries)

        # Verify that all resolved names were added to the database
        all_resolved = self.resolved_manager.getAllResolvedProcessNames()
        self.assertEqual(len(all_resolved), 3)
        self.assertEqual(all_resolved[0].name, "resolved1")
        self.assertEqual(all_resolved[1].name, "resolved2")
        self.assertEqual(all_resolved[2].name, "resolved3")


if __name__ == "__main__":
    unittest.main()