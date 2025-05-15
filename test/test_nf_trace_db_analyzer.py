import unittest

import pandas as pd
from nf_trace_db_analyzer import analyze_process_execution_correlation, analyze_process_execution_time_consistency, anova_on_process_execution_times, identify_process_execution_time_consistency, identify_variable_pipeline_numerical_parameters, print_process_execution_time_consistency, summarize_consistency_analysis
from nf_trace_db_manager import NextflowTraceDBManager


class TestNextflowTraceDBAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an in-memory SQLite database for testing
        cls.db_manager = NextflowTraceDBManager(":memory:")
        cls.db_manager.connect()

        cls.html_file_paths = [
            "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_report.html",
            "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_report.html",
            "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_report.html",
            "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_report.html",
        ]

        cls.log_files = [
            "./dat/250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_log.log",
            "./dat/250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_log.log",
            "./dat/250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_log.log",
            "./dat/250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_log.log",
        ]

        # Create tables for testing
        cls.db_manager.createTables()

        # Ignore ParamInputs and ExecParams for now to speed up tests
        cls.db_manager.process_inputs_manager = None
        cls.db_manager.process_exec_params_manager = None

        # Load all files
        for html_file_path, log_file in zip(cls.html_file_paths, cls.log_files):
            cls.db_manager.addAllFromFiles(html_file_path, log_file)

        # Setup pandas display params
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_rows", None)

    @classmethod
    def tearDownClass(cls):
        cls.db_manager.close()

    def test_analyze_process_execution_time_consistency(self):
        results = analyze_process_execution_time_consistency(self.db_manager)

        print(results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 55)
        self.assertEqual(results["is_constant"].sum(), 10)

    def test_analyze_process_execution_time_consistency_with_process_filter(self):
        process_name = "do_correlation"
        results = analyze_process_execution_time_consistency(self.db_manager, process_names=[process_name], group_by_resolved_name=True)

        # Print result
        print('## Results grouped by resolved names:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 6)
        self.assertEqual(results["is_constant"].sum(), 0)

        results = analyze_process_execution_time_consistency(self.db_manager,
                                                             process_names=[process_name],
                                                             group_by_resolved_name=False,
                                                             group_by_trace_name=True)

        print('## Results grouped by trace:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 4)
        self.assertEqual(results["is_constant"].sum(), 1)

        results = analyze_process_execution_time_consistency(self.db_manager,
                                                             process_names=[process_name],
                                                             group_by_resolved_name=True,
                                                             group_by_trace_name=True)

        print('## Results grouped by resolved names and trace:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 22)
        self.assertEqual(results["is_constant"].sum(), 22)

    def test_identify_process_execution_time_consistency(self):
        process_consistency_analysis, per_trace_analysis, per_resolved_analysis, per_resolved_per_trace_analysis = identify_process_execution_time_consistency(
            self.db_manager, tolerance=0.1, std_dev_threshold=15000, quantile=0.80)

        print_process_execution_time_consistency(
            process_consistency_analysis,
            per_trace_analysis,
            per_resolved_analysis,
            per_resolved_per_trace_analysis)

        self.assertEqual(len(process_consistency_analysis), 55)

        summary = summarize_consistency_analysis(
            process_consistency_analysis,
            per_trace_analysis,
            per_resolved_analysis,
            per_resolved_per_trace_analysis
        )

        # Define the custom order for the consistency_level column
        consistency_order = [
            "Constant", "Per trace", "Per resolved", "Per resolved and trace", "Inconstant", "Not Executed"
        ]
        summary["consistency_level"] = pd.Categorical(
            summary["consistency_level"], categories=consistency_order, ordered=True
        )

        # Sort the summary by the custom order of consistency_level, then by process_name and resolved_name
        sorted_summary = summary.sort_values(
            by=["consistency_level", "process_name", "resolved_name"],
            ascending=[True, True, True]
        )

        print("\n## Summary of consistency analysis:")
        print(sorted_summary)

        self.assertEqual((summary["consistency_level"] == "Constant").sum(), 10)
        self.assertEqual((summary["consistency_level"] == "Per trace").sum(), 4)
        self.assertEqual((summary["consistency_level"] == "Per resolved").sum(), 3)
        self.assertEqual((summary["consistency_level"] == "Per resolved and trace").sum(), 22)
        self.assertEqual((summary["consistency_level"] == "Inconstant").sum(), 70)

    def test_identify_variable_pipeline_numerical_parameters(self):
        variable_pipeline_params = identify_variable_pipeline_numerical_parameters(self.db_manager)

        variable_pipeline_params = variable_pipeline_params.sort_values(by=["param_name", "trace_name", "value"], ascending=[True, True, True])
        print(variable_pipeline_params)

        self.assertEqual(variable_pipeline_params.shape[0], 180)
        self.assertEqual(variable_pipeline_params.shape[1], 4)
        self.assertEqual(variable_pipeline_params["param_name"].nunique(), 45)
        self.assertEqual(variable_pipeline_params["trace_name"].nunique(), 4)

    def test_analyze_process_execution_correlation(self):
        # For processes that are categorized as "Per trace", analyze the correlation
        # between the execution times and the varying pipeline parameters
        per_trace_analysis = ["determine_flux_cal_solns"]
        correlations = {}
        for process_name in per_trace_analysis:
            correlations[process_name] = analyze_process_execution_correlation(self.db_manager, process_name)

        # Print the correlation results
        for process_name, correlation_df in correlations.items():
            print(f"\n## Correlation analysis for process '{process_name}':")
            print(correlation_df)

        # Check the correlation results
        self.assertEqual(len(correlations), 1)
        self.assertEqual(correlations[per_trace_analysis[0]].shape[0], 45)
        self.assertEqual(correlations[per_trace_analysis[0]].shape[1], 7)

    def test_anova_on_process_execution_times(self):
        anova_results = anova_on_process_execution_times(self.db_manager)

        anova_results = anova_results.sort_values(by=["process_name"], ascending=[True])
        print(anova_results)

        self.assertEqual(anova_results.shape[0], 19)
        self.assertEqual(anova_results.shape[1], 11)


if __name__ == "__main__":
    unittest.main()
