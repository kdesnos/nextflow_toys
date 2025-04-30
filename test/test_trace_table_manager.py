import unittest
import sqlite3
from unittest.mock import patch
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
    
    def test_addDuplicateTraceFails(self):
        trace1 = TraceEntry(tId=0, day="2023-01-01", name="duplicate_trace")
        trace2 = TraceEntry(tId=0, day="2023-01-02", name="duplicate_trace")  # Same name

        # Add the first trace
        self.trace_manager.addTraceEntry(trace1)

        # Attempt to add the duplicate trace and expect an exception
        with self.assertRaises(sqlite3.IntegrityError):
            self.trace_manager.addTraceEntry(trace2)

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

    @patch('trace_table_manager.extract_pipeline_metadata')
    def test_addMetadataToTraceTable(self, mock_extract_pipeline_metadata):
        # Mock the metadata extraction function
        mock_metadata = {
            'start_time': '22-Apr-2025 14:03:40',
            'run_name': 'hungry_lamarr'
        }
        mock_extract_pipeline_metadata.return_value = mock_metadata

        # Call the method to add metadata to the trace table
        self.trace_manager.addMetadataToTraceTable("mock_file_path.html")

        # Verify that the metadata was added to the database
        all_traces = self.trace_manager.getAllTraces()
        self.assertEqual(len(all_traces), 1)
        self.assertEqual(all_traces[0].day, '22-Apr-2025 14:03:40')
        self.assertEqual(all_traces[0].name, 'hungry_lamarr')

    @patch('trace_table_manager.extract_pipeline_metadata')
    def test_addMetadataToTraceTable_invalid_metadata(self, mock_extract_pipeline_metadata):
        # Mock the metadata extraction function to return invalid metadata
        mock_extract_pipeline_metadata.return_value = None

        # Ensure the method raises an exception for invalid metadata
        with self.assertRaises(Exception) as context:
            self.trace_manager.addMetadataToTraceTable("mock_file_path.html")
        self.assertEqual(str(context.exception), "Failed to extract metadata from the file.")

if __name__ == "__main__":
    unittest.main()