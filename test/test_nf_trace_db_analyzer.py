from nf_trace_db_manager import NextflowTraceDBManager
import unittest

import pandas as pd
from nf_trace_db_analyzer import analyze_process_execution_metric_correlation, analyze_process_execution_metric_consistency, anova_on_process_execution_metrics, get_metric_distribution_characteristics, identify_process_execution_metric_consistency, identify_variable_pipeline_numerical_parameters, print_process_execution_metric_consistency, summarize_metric_consistency_analysis, extract_metric_linear_reg


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

    def test_analyze_process_execution_metric_consistency_time(self):
        results = analyze_process_execution_metric_consistency(self.db_manager, metric="time")

        print(results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 55)
        self.assertEqual(results["is_constant"].sum(), 10)

    def test_analyze_process_execution_metric_consistency_with_process_filter(self):
        process_name = "do_correlation"
        results = analyze_process_execution_metric_consistency(
            self.db_manager,
            metric="time",
            process_names=[process_name],
            group_by_resolved_name=True
        )

        # Print result
        print('## Results grouped by resolved names:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 6)
        self.assertEqual(results["is_constant"].sum(), 0)

        results = analyze_process_execution_metric_consistency(
            self.db_manager,
            metric="time",
            process_names=[process_name],
            group_by_resolved_name=False,
            group_by_trace_name=True
        )

        print('## Results grouped by trace:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 4)
        self.assertEqual(results["is_constant"].sum(), 1)

        results = analyze_process_execution_metric_consistency(
            self.db_manager,
            metric="time",
            process_names=[process_name],
            group_by_resolved_name=True,
            group_by_trace_name=True
        )

        print('## Results grouped by resolved names and trace:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)
        self.assertEqual(results.shape[0], 22)
        self.assertEqual(results["is_constant"].sum(), 22)

    def test_analyze_process_execution_metric_consistency_memory(self):
        results = analyze_process_execution_metric_consistency(self.db_manager, metric="memory")

        print('## Memory consistency results:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)

        process_name = "do_correlation"
        results = analyze_process_execution_metric_consistency(
            self.db_manager,
            metric="memory",
            process_names=[process_name],
            group_by_resolved_name=False,
            group_by_trace_name=True
        )

        print('## Memory results grouped by trace:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)

        results = analyze_process_execution_metric_consistency(
            self.db_manager,
            metric="memory",
            process_names=[process_name],
            group_by_resolved_name=True,
            group_by_trace_name=True
        )

        print('## Memory results grouped by resolved names and trace:\n', results)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results.shape[1], 9)

    def test_identify_process_execution_metric_consistency_time(self):
        process_consistency_analysis, per_trace_analysis, per_resolved_analysis, per_resolved_per_trace_analysis = identify_process_execution_metric_consistency(
            self.db_manager, metric="time", tolerance=0.1, threshold=15000, quantile=0.80)

        print_process_execution_metric_consistency(
            process_consistency_analysis,
            per_trace_analysis,
            per_resolved_analysis,
            per_resolved_per_trace_analysis,
            metric="time"
        )

        self.assertEqual(len(process_consistency_analysis), 55)

        summary = summarize_metric_consistency_analysis(
            process_consistency_analysis,
            per_trace_analysis,
            per_resolved_analysis,
            per_resolved_per_trace_analysis,
            metric="time"
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

    def test_identify_process_execution_metric_consistency_memory(self):
        process_consistency_analysis, per_trace_analysis, per_resolved_analysis, per_resolved_per_trace_analysis = identify_process_execution_metric_consistency(
            self.db_manager, metric="memory", tolerance=0.2, threshold=100 * 1024 * 1024, quantile=0.80)

        print_process_execution_metric_consistency(
            process_consistency_analysis,
            per_trace_analysis,
            per_resolved_analysis,
            per_resolved_per_trace_analysis,
            metric="memory"
        )

        self.assertGreaterEqual(len(process_consistency_analysis), 0)

        summary = summarize_metric_consistency_analysis(
            process_consistency_analysis,
            per_trace_analysis,
            per_resolved_analysis,
            per_resolved_per_trace_analysis,
            metric="memory"
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

        print("\n## Summary of memory consistency analysis:")
        print(sorted_summary)

    def test_identify_variable_pipeline_numerical_parameters(self):
        variable_pipeline_params = identify_variable_pipeline_numerical_parameters(self.db_manager)

        variable_pipeline_params = variable_pipeline_params.sort_values(by=["param_name", "trace_name", "value"], ascending=[True, True, True])
        print(variable_pipeline_params)

        self.assertEqual(variable_pipeline_params.shape[0], 180)
        self.assertEqual(variable_pipeline_params.shape[1], 4)
        self.assertEqual(variable_pipeline_params["param_name"].nunique(), 45)
        self.assertEqual(variable_pipeline_params["trace_name"].nunique(), 4)

    def test_analyze_process_execution_metric_correlation_time(self):
        # For processes that are categorized as "Per trace", analyze the correlation
        # between the execution times and the varying pipeline parameters
        per_trace_analysis = ["determine_flux_cal_solns"]
        correlations = {}
        for process_name in per_trace_analysis:
            correlations[process_name] = analyze_process_execution_metric_correlation(
                self.db_manager, process_name, metric="time"
            )

        # Print the correlation results
        for process_name, correlation_df in correlations.items():
            print(f"\n## Time correlation analysis for process '{process_name}':")
            print(correlation_df)

        # Check the correlation results
        self.assertEqual(len(correlations), 1)
        self.assertEqual(correlations[per_trace_analysis[0]].shape[0], 45)
        self.assertEqual(correlations[per_trace_analysis[0]].shape[1], 7)

    def test_analyze_process_execution_metric_correlation_memory(self):
        # For processes that are categorized as "Per trace", analyze the correlation
        # between the memory usage and the varying pipeline parameters
        per_trace_analysis = ["determine_flux_cal_solns"]
        correlations = {}
        for process_name in per_trace_analysis:
            correlations[process_name] = analyze_process_execution_metric_correlation(
                self.db_manager, process_name, metric="memory"
            )

        # Print the correlation results
        for process_name, correlation_df in correlations.items():
            print(f"\n## Memory correlation analysis for process '{process_name}':")
            print(correlation_df)

        # Check the correlation results
        self.assertEqual(len(correlations), 1)
        self.assertEqual(correlations[per_trace_analysis[0]].shape[1], 7)

    def test_anova_on_process_execution_metrics_time(self):
        anova_results = anova_on_process_execution_metrics(self.db_manager, metric="time")

        anova_results = anova_results.sort_values(by=["process_name"], ascending=[True])
        print("\n## ANOVA results for time metrics:")
        print(anova_results)

        self.assertGreaterEqual(anova_results.shape[0], 0)
        self.assertEqual(anova_results.shape[1], 11)

    def test_anova_on_process_execution_metrics_memory(self):
        anova_results = anova_on_process_execution_metrics(self.db_manager, metric="memory")

        anova_results = anova_results.sort_values(by=["process_name"], ascending=[True])
        print("\n## ANOVA results for memory metrics:")
        print(anova_results)

        self.assertGreaterEqual(anova_results.shape[0], 0)
        self.assertEqual(anova_results.shape[1], 11)

    def test_get_metric_distribution_characteristics_time(self):
        stats = get_metric_distribution_characteristics(
            self.db_manager, "fcal1:corr_fcal:difx_to_fits", metric="time", is_resolved_name=True
        )
        print("\n## Time distribution characteristics:")
        print(stats)
        self.assertEqual(stats["mean_time"][0], 112115.75)
        self.assertEqual(stats["std_dev_time"][0], 10778.339815922493)

    def test_get_metric_distribution_characteristics_memory(self):
        stats = get_metric_distribution_characteristics(
            self.db_manager, "fcal1:corr_fcal:difx_to_fits", metric="memory", is_resolved_name=True
        )
        print("\n## Memory distribution characteristics:")
        print(stats)
        self.assertIn("mean_memory", stats.columns)
        self.assertIn("std_dev_memory", stats.columns)

    def test_extract_metric_linear_reg_process_time(self):
        """
        Test the extract_metric_linear_reg method with a regular process name for time metric.
        """
        # Test with regular process name
        process_name = "generate_binconfig"
        result = extract_metric_linear_reg(
            self.db_manager, process_name, metric="time", top_n=2, rmse_threshold=100000
        )

        # Check that the result has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("expression", result)
        self.assertIn("rmse", result)
        self.assertIn("selected_parameters", result)

        # Check that the expression is a non-empty string
        self.assertIsInstance(result["expression"], str)
        self.assertTrue(len(result["expression"]) > 0)

        # Check that at least one parameter was selected
        self.assertIsInstance(result["selected_parameters"], dict)
        self.assertTrue(len(result["selected_parameters"]) > 0)

        # RMSE should be a positive float
        self.assertIsInstance(result["rmse"], float)
        self.assertGreater(result["rmse"], 0)

        # Check that the expression contains the selected parameters
        for param in result["selected_parameters"]:
            if not param.startswith("previous_model_"):  # Skip model parameters in checking
                self.assertIn(param, result["expression"])

        # Print some information for manual inspection
        print(f"\nTime process expression: {result['expression']}")
        print(f"Time process RMSE: {result['rmse']:.2f}")
        print(f"Time process parameters: {result['selected_parameters']}")

    def test_extract_metric_linear_reg_process_memory(self):
        """
        Test the extract_metric_linear_reg method with a regular process name for memory metric.
        """
        # Test with regular process name
        process_name = "generate_binconfig"
        result = extract_metric_linear_reg(
            self.db_manager, process_name, metric="memory", top_n=2, rmse_threshold=100 * 1024 * 1024
        )

        # Check that the result has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("expression", result)
        self.assertIn("rmse", result)
        self.assertIn("selected_parameters", result)
        self.assertEqual(result["metric"], "memory")

        # Check that the expression is a non-empty string
        self.assertIsInstance(result["expression"], str)
        self.assertTrue(len(result["expression"]) > 0)

        # Print some information for manual inspection
        print(f"\nMemory process expression: {result['expression']}")
        print(f"Memory process RMSE: {result['rmse']:.2f}")
        print(f"Memory process parameters: {result['selected_parameters']}")

    def test_extract_metric_linear_reg_resolved_process(self):
        """
        Test the extract_metric_linear_reg method with a resolved process name for time metric.
        """
        # Test with resolved process name
        resolved_process_name = "fcal1:corr_fcal:do_correlation"
        result = extract_metric_linear_reg(
            self.db_manager,
            resolved_process_name,
            metric="time",
            top_n=2,
            rmse_threshold=100000,
            is_resolved_name=True
        )

        # Check that the result has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("expression", result)
        self.assertIn("rmse", result)
        self.assertIn("selected_parameters", result)

        # Check that the expression is a non-empty string
        self.assertIsInstance(result["expression"], str)
        self.assertTrue(len(result["expression"]) > 0)

        # Check that at least one parameter was selected
        self.assertIsInstance(result["selected_parameters"], dict)
        self.assertTrue(len(result["selected_parameters"]) > 0)

        # RMSE should be a positive float
        self.assertIsInstance(result["rmse"], float)
        self.assertGreater(result["rmse"], 0)

        # Check that the expression contains the selected parameters
        for param in result["selected_parameters"]:
            if not param.startswith("previous_model_"):  # Skip model parameters in checking
                self.assertIn(param, result["expression"])

        # Print some information for manual inspection
        print(f"\nResolved process expression: {result['expression']}")
        print(f"Resolved process RMSE: {result['rmse']:.2f}")
        print(f"Resolved process parameters: {result['selected_parameters']}")


if __name__ == "__main__":
    unittest.main()
