import unittest
import sqlite3
from trace_table_manager import TraceTableManager, TraceEntry
from nf_trace_db_manager import NextflowTraceDBManager


class TestTraceTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.connection = sqlite3.connect(":memory:")

        # Use the NextflowTraceDBManager to create the tables
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connection = self.connection
        self.db_manager.createTables()

        # Initialize the TraceTableManager with the same connection
        self.trace_manager = TraceTableManager(self.connection)

    def tearDown(self):
        self.connection.close()

    def test_addTraceEntry(self):
        trace = TraceEntry(tId=0, day="2023-01-01", name="test_trace")
        self.trace_manager.addTraceEntry(trace)
        self.assertNotEqual(trace.tId, 0)  # tId should be updated after insertion

    def test_getTraceEntry(self):
        trace = TraceEntry(tId=0, day="2023-01-01", name="test_trace")
        self.trace_manager.addTraceEntry(trace)

        retrieved_trace = self.trace_manager.getTraceEntry("test_trace")
        self.assertIsNotNone(retrieved_trace)
        self.assertEqual(retrieved_trace.name, "test_trace")

    def test_getAllTraces(self):
        traces = [
            TraceEntry(tId=0, day="2023-01-01", name="trace1"),
            TraceEntry(tId=0, day="2023-01-02", name="trace2"),
        ]
        for trace in traces:
            self.trace_manager.addTraceEntry(trace)

        all_traces = self.trace_manager.getAllTraces()
        self.assertEqual(len(all_traces), 2)

    def test_removeTraceEntry(self):
        trace = TraceEntry(tId=0, day="2023-01-01", name="test_trace")
        self.trace_manager.addTraceEntry(trace)

        result = self.trace_manager.removeTraceEntry("test_trace")
        self.assertTrue(result)

        # Ensure the trace is removed
        retrieved_trace = self.trace_manager.getTraceEntry("test_trace")
        self.assertIsNone(retrieved_trace)

    def test_removeNonExistentTraceEntry(self):
        result = self.trace_manager.removeTraceEntry("non_existent_trace")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()