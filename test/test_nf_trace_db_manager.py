import unittest
from nf_trace_db_manager import NextflowTraceDBManager


class TestNextflowTraceDBManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = NextflowTraceDBManager(":memory:")
        self.db_manager.connect()

        self.html_file_paths = [
            "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_report.html",
            "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_report.html",
            "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_report.html",
            "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_report.html",]

        # Add process definitions
        self.log_files = [
            "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_log.log",
            "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_log.log",
            "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_log.log",
            "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_log.log",]

        # Create tables for testing
        self.db_manager.createTables()

    def tearDown(self):
        self.db_manager.close()

    def test_isConnected(self):
        self.assertTrue(self.db_manager.isConnected())

    def test_isDatabaseEmpty(self):
        self.assertFalse(self.db_manager.isDatabaseEmpty())  # Tables exist after creation

    def test_getUserVersion(self):
        version = self.db_manager.getUserVersion()
        self.assertEqual(version, 0)  # Default user_version is 0

    def test_createTables_force(self):
        self.db_manager.createTables(force=True)  # Recreate tables
        self.assertFalse(self.db_manager.isDatabaseEmpty())

    def test_close(self):
        self.db_manager.close()
        self.assertFalse(self.db_manager.isConnected())

    def test_addMetadataFromHtml(self):
        self.db_manager.addMetadataFromHtml(self.html_file_paths[0])
        self.assertEqual(len(self.db_manager.trace_manager.getAllTraces()), 1)

    def test_addProcessDefinitionsFromLog(self):
        self.db_manager.addMetadataFromHtml(self.html_file_paths[0])
        self.db_manager.addProcessDefinitionsFromLog(self.log_files[0])
        self.assertEqual(len(self.db_manager.process_manager.getAllProcesses()), 55)

    def test_addResolvedProcessNamesFromLog(self):
        self.db_manager.addMetadataFromHtml(self.html_file_paths[0])
        self.db_manager.addProcessDefinitionsFromLog(self.log_files[0])
        self.db_manager.addResolvedProcessNamesFromLog(self.log_files[0])
        self.assertEqual(len(self.db_manager.resolved_process_manager.getAllResolvedProcessNames()), 56)

    def test_addProcessExecutionsFromHtml(self):
        self.db_manager.addProcessDefinitionsFromLog(self.log_files[0])
        self.db_manager.addResolvedProcessNamesFromLog(self.log_files[0])
        self.db_manager.addProcessExecutionsFromHtml(self.html_file_paths[0])
        self.assertEqual(len(self.db_manager.process_executions_manager.getAllProcessExecutions()), 292)

    def test_addProcessInputsFromLog(self):
        self.db_manager.addProcessDefinitionsFromLog(self.log_files[0])
        self.db_manager.addResolvedProcessNamesFromLog(self.log_files[0])
        self.db_manager.addProcessInputsFromLog(self.log_files[0])
        self.assertEqual(len(self.db_manager.process_inputs_manager.getAllProcessInputs()), 84)

    def test_addProcessExecParamsFromLog(self):
        self.db_manager.addProcessDefinitionsFromLog(self.log_files[0])
        self.db_manager.addResolvedProcessNamesFromLog(self.log_files[0])
        self.db_manager.addProcessExecutionsFromHtml(self.html_file_paths[0])
        self.db_manager.addProcessExecParamsFromLog(self.log_files[0])
        self.assertEqual(len(self.db_manager.process_exec_params_manager.getAllProcessExecParams()), 3972)

    def test_addPipelineParamsFromLog(self):
        self.db_manager.addMetadataFromHtml(self.html_file_paths[0])
        self.db_manager.addPipelineParamsFromLog(self.log_files[0])
        self.assertEqual(len(self.db_manager.pipeline_params_manager.getAllPipelineParams()), 184)
    
    def test_addPipelineParamValuesFromLog(self):
        self.db_manager.addMetadataFromHtml(self.html_file_paths[0])
        self.db_manager.addPipelineParamsFromLog(self.log_files[0])
        self.db_manager.addPipelineParamValuesFromLog(self.log_files[0])
        self.assertEqual(len(self.db_manager.pipeline_param_values_manager.getAllPipelineParamValues()), 184)

    def test_addAllFromFiles(self):
        for html_file_path, log_file in zip(self.html_file_paths, self.log_files):
            self.db_manager.addAllFromFiles(html_file_path, log_file)

        self.assertEqual(len(self.db_manager.trace_manager.getAllTraces()), 4)
        self.assertEqual(len(self.db_manager.process_manager.getAllProcesses()), 55)
        self.assertEqual(len(self.db_manager.resolved_process_manager.getAllResolvedProcessNames()), 57)
        self.assertEqual(len(self.db_manager.process_executions_manager.getAllProcessExecutions()), 1092)
        self.assertEqual(len(self.db_manager.process_inputs_manager.getAllProcessInputs()), 84)
        self.assertEqual(len(self.db_manager.process_exec_params_manager.getAllProcessExecParams()), 14693)
        self.assertEqual(len(self.db_manager.pipeline_params_manager.getAllPipelineParams()), 186)
        self.assertEqual(len(self.db_manager.pipeline_param_values_manager.getAllPipelineParamValues()), 738)

if __name__ == "__main__":
    unittest.main()