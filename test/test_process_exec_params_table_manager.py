import unittest
from unittest.mock import patch
from nf_trace_db_manager import NextflowTraceDBManager
from process_exec_params_table_manager import ProcessExecParamsTableManager, ProcessExecParamEntry
from processes_table_manager import ProcessEntry
from resolved_process_names_table_manager import ResolvedProcessEntry
from trace_table_manager import TraceEntry
from process_executions_table_manager import ProcessExecutionEntry


class TestProcessExecParamsTableManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()
        self.db_manager.createTables()

        # Add required data into the dependent tables
        # Add a process entry
        self.db_manager.process_manager.addProcess(
            ProcessEntry(pId=0, name="process_1", path="/path/to/process_1.nf")
        )

        # Add a resolved process name
        self.db_manager.resolved_process_manager.addResolvedProcessName(
            ResolvedProcessEntry(rId=0, pId=1, name="resolved_process_1")
        )

        # Add a trace entry
        self.db_manager.trace_manager.addTraceEntry(
            TraceEntry(tId=0, day="2023-01-01", name="trace_1")
        )

        # Add process execution entries
        self.db_manager.process_executions_manager.addAllProcessExecutions(
            [
                ProcessExecutionEntry(eId=0, tId=1, rId=1, instance=1, hash="hash_1", time=123.45),
                ProcessExecutionEntry(eId=0, tId=1, rId=1, instance=2, hash="hash_2", time=111.11)
            ]
        )

        # Use the ProcessExecParamsTableManager from the database manager
        self.exec_params_manager = self.db_manager.process_exec_params_manager

    def tearDown(self):
        self.db_manager.close()

    def test_addProcessExecParam(self):
        exec_param_entry = ProcessExecParamEntry(eId=1, rank="0", value="param1")
        self.exec_params_manager.addProcessExecParam(exec_param_entry)

        params = self.exec_params_manager.getProcessExecParamsByExecutionId(1)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].value, "param1")
        self.assertEqual(params[0].rank, "0")

    def test_addAllProcessExecParams(self):
        exec_param_entries = [
            ProcessExecParamEntry(eId=1, rank="0", value="param1"),
            ProcessExecParamEntry(eId=1, rank="1.1", value="param2"),
        ]
        self.exec_params_manager.addAllProcessExecParams(exec_param_entries)

        params = self.exec_params_manager.getProcessExecParamsByExecutionId(1)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].value, "param1")
        self.assertEqual(params[1].value, "param2")
        self.assertEqual(params[1].rank, "1.1")

    def test_getProcessExecParamsByExecutionId(self):
        exec_param_entries = [
            ProcessExecParamEntry(eId=1, rank="0", value="param1"),
            ProcessExecParamEntry(eId=1, rank="1", value="param2"),
        ]
        self.exec_params_manager.addAllProcessExecParams(exec_param_entries)

        params = self.exec_params_manager.getProcessExecParamsByExecutionId(1)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].value, "param1")
        self.assertEqual(params[1].value, "param2")

    def test_getAllProcessExecParams(self):
        exec_param_entries = [
            ProcessExecParamEntry(eId=1, rank="0", value="param1"),
            ProcessExecParamEntry(eId=2, rank="1.1", value="param2"),
        ]
        self.exec_params_manager.addAllProcessExecParams(exec_param_entries)

        params = self.exec_params_manager.getAllProcessExecParams()
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].value, "param1")
        self.assertEqual(params[1].value, "param2")

    def test_removeProcessExecParam(self):
        exec_param_entry = ProcessExecParamEntry(eId=1, rank="0", value="param1")
        self.exec_params_manager.addProcessExecParam(exec_param_entry)

        result = self.exec_params_manager.removeProcessExecParam(1, "0")
        self.assertTrue(result)

        params = self.exec_params_manager.getProcessExecParamsByExecutionId(1)
        self.assertEqual(len(params), 0)

    def test_removeNonExistentProcessExecParam(self):
        result = self.exec_params_manager.removeProcessExecParam(1, "0")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
