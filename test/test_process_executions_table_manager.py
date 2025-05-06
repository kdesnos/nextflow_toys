import unittest
import sqlite3
from unittest.mock import patch

import pandas as pd
from process_executions_table_manager import ProcessExecutionTableManager, ProcessExecutionEntry
from nf_trace_db_manager import NextflowTraceDBManager
from processes_table_manager import ProcessEntry
from trace_table_manager import TraceEntry
from resolved_process_names_table_manager import ResolvedProcessEntry


class TestProcessExecutionTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Use the ProcessExecutionTableManager from the database manager
        self.execution_manager = ProcessExecutionTableManager(self.db_manager.connection)

        # Add a trace entry for testing
        self.trace_entry = TraceEntry(tId=0, day="2023-01-01", name="test_trace")
        self.db_manager.trace_manager.addTraceEntry(self.trace_entry)

        # Add a process entry for testing
        self.process_entry = ProcessEntry(pId=0, name="process", path="/path/to/file.nf")
        self.db_manager.process_manager.addProcess(self.process_entry)

        # Add a resolved process entry for testing
        resolved_entries = [
            ResolvedProcessEntry(rId=0, pId=1, name="resolved_process"),
            ResolvedProcessEntry(rId=0, pId=1, name="main:a"),
            ResolvedProcessEntry(rId=0, pId=1, name="proc"),
        ]
        self.db_manager.resolved_process_manager.addAllResolvedProcessNames(resolved_entries)
        self.resolved_entry = resolved_entries[0]  # Use the first entry for testing

    def tearDown(self):
        self.db_manager.close()

    def test_addProcessExecution(self):
        execution_entry = ProcessExecutionEntry(
            eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=1, hash="hash1", time=123.45
        )
        self.execution_manager.addProcessExecution(execution_entry)
        self.assertNotEqual(execution_entry.eId, 0)  # eId should be updated after insertion

    def test_addAllProcessExecutions(self):
        executions = [
            ProcessExecutionEntry(eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=1, hash="hash1", time=123.45),
            ProcessExecutionEntry(eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=2, hash="hash2", time=456.78),
        ]
        self.execution_manager.addAllProcessExecutions(executions)

        # Verify that all executions were added to the database
        all_executions = self.execution_manager.getAllProcessExecutions()
        self.assertEqual(len(all_executions), 2)
        self.assertEqual(all_executions[0].hash, "hash1")
        self.assertEqual(all_executions[0].time, 123.45)
        self.assertEqual(all_executions[1].hash, "hash2")
        self.assertEqual(all_executions[1].time, 456.78)

    def test_getProcessExecutionByHash(self):
        execution_entry = ProcessExecutionEntry(
            eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=1, hash="hash1", time=123.45
        )
        self.execution_manager.addProcessExecution(execution_entry)

        retrieved_execution = self.execution_manager.getProcessExecutionByHash("hash1")
        self.assertIsNotNone(retrieved_execution)
        self.assertEqual(retrieved_execution.hash, "hash1")
        self.assertEqual(retrieved_execution.time, 123.45)

    def test_getExecutionByResolvedIdAndInstanceAndTraceId(self):
        # Add process executions to the database
        execution_entry_1 = ProcessExecutionEntry(
            eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=1, hash="hash1", time=123.45
        )
        execution_entry_2 = ProcessExecutionEntry(
            eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=2, hash="hash2", time=456.78
        )
        self.execution_manager.addProcessExecution(execution_entry_1)
        self.execution_manager.addProcessExecution(execution_entry_2)

        # Retrieve the first execution by resolved ID and instance
        retrieved_execution_1 = self.execution_manager.getExecutionByResolvedIdAndInstanceAndTraceId(
            self.resolved_entry.rId, 1, self.trace_entry.tId
        )
        self.assertIsNotNone(retrieved_execution_1)
        self.assertEqual(retrieved_execution_1.hash, "hash1")
        self.assertEqual(retrieved_execution_1.instance, 1)

        # Retrieve the second execution by resolved ID and instance
        retrieved_execution_2 = self.execution_manager.getExecutionByResolvedIdAndInstanceAndTraceId(
            self.resolved_entry.rId, 2, self.trace_entry.tId
        )
        self.assertIsNotNone(retrieved_execution_2)
        self.assertEqual(retrieved_execution_2.hash, "hash2")
        self.assertEqual(retrieved_execution_2.instance, 2)

        # Attempt to retrieve a non-existent execution
        non_existent_execution = self.execution_manager.getExecutionByResolvedIdAndInstanceAndTraceId(
            self.resolved_entry.rId, 3, self.trace_entry.tId
        )
        self.assertIsNone(non_existent_execution)

    def test_addProcessExecutionWithInvalidTraceOrResolvedProcess(self):
        # Attempt to add a process execution with an invalid tId
        invalid_trace_execution = ProcessExecutionEntry(
            eId=0, tId=999, rId=self.resolved_entry.rId, instance=1, hash="invalid_trace_hash", time=123.45
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.execution_manager.addProcessExecution(invalid_trace_execution)

        # Attempt to add a process execution with an invalid rId
        invalid_resolved_execution = ProcessExecutionEntry(
            eId=0, tId=self.trace_entry.tId, rId=999, instance=1, hash="invalid_resolved_hash", time=123.45
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.execution_manager.addProcessExecution(invalid_resolved_execution)

    def test_getAllProcessExecutions(self):
        executions = [
            ProcessExecutionEntry(eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=1, hash="hash1", time=123.45),
            ProcessExecutionEntry(eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=2, hash="hash2", time=456.78),
        ]
        for execution in executions:
            self.execution_manager.addProcessExecution(execution)

        all_executions = self.execution_manager.getAllProcessExecutions()
        self.assertEqual(len(all_executions), 2)
        self.assertEqual(all_executions[0].hash, "hash1")
        self.assertEqual(all_executions[1].hash, "hash2")

    def test_removeProcessExecutionByHash(self):
        execution_entry = ProcessExecutionEntry(
            eId=0, tId=self.trace_entry.tId, rId=self.resolved_entry.rId, instance=1, hash="hash1", time=123.45
        )
        self.execution_manager.addProcessExecution(execution_entry)

        result = self.execution_manager.removeProcessExecutionByHash("hash1")
        self.assertTrue(result)

        # Ensure the execution is removed
        retrieved_execution = self.execution_manager.getProcessExecutionByHash("hash1")
        self.assertIsNone(retrieved_execution)

    def test_removeNonExistentProcessExecution(self):
        result = self.execution_manager.removeProcessExecutionByHash("non_existent_hash")
        self.assertFalse(result)

    @patch("process_executions_table_manager.extract_trace_data")
    def test_addProcessExecutionsFromFile(self, mock_extract_trace_data):
        # Mock the extract_trace_data function
        mock_data = pd.DataFrame(
            [
                {"name": "sub:a (1)", "hash": "hash1", "realtime": pd.Timedelta("123.45s"), "process": "main:a"},
                {"name": "sub:a (2)", "hash": "hash2", "realtime": pd.Timedelta("456.78s"), "process": "main:a"},
                {"name": "proc", "hash": "hash3", "realtime": pd.Timedelta("987.65s"), "process": "proc"},
            ]
        )
        mock_extract_trace_data.return_value = mock_data

        # Call the method to add process executions to the table
        self.execution_manager.addProcessExecutionsFromFile(
            self.db_manager,
            "mock_trace_file.html",
            self.trace_entry.tId,
        )

        # Verify that the process executions were added to the database
        all_executions = self.execution_manager.getAllProcessExecutions()
        self.assertEqual(len(all_executions), 3)

        self.assertEqual(all_executions[0].hash, "hash1")
        self.assertEqual(all_executions[0].instance, 1)
        self.assertEqual(all_executions[0].time, 123450)
        self.assertEqual(all_executions[1].hash, "hash2")
        self.assertEqual(all_executions[1].instance, 2)
        self.assertEqual(all_executions[1].time, 456780)


if __name__ == "__main__":
    unittest.main()
