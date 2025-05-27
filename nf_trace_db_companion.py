import time
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
from datetime import timedelta

from extract_from_nf_log import extract_task_info_from_lines, extractPipelineParameters, extractProcessInputs
from nf_trace_db_analyzer import build_execution_predictors
from nf_trace_db_manager import NextflowTraceDBManager
from extract_trace_from_html import extract_trace_data


class NextflowTraceDBCompanion(FileSystemEventHandler):
    def __init__(self, stub_report_path, stub_dag_path, db_path, file_path, observer):
        self.file_path = file_path
        self.observer = observer  # Store the observer instance
        self.last_position = 0  # To keep track of the last read position
        self.last_size = 0      # To keep track of the last file size
        self.param_values = None
        self.db_manager = NextflowTraceDBManager(db_path)
        self.db_manager.connect()
        if self.db_manager.isDatabaseEmpty():
            print("Database is empty. Exiting...")
            self.db_manager.close()
            self.stop_observer()

        # Parse given stub report and dag files
        print("Parsing stub report file...")
        stub_df = extract_trace_data(stub_report_path)
        # From stub_df, extract the number of times each process was executed
        # group by resolved process name and count occurrences
        self.process_exec_counts = stub_df['process'].value_counts().reset_index()
        # Add an extra column to the df to keep track of completed executions, initialized to 0
        self.process_exec_counts['completed_executions'] = 0

        # Build the model used to predict execution time
        print("Building execution time prediction model from DB...")
        self.stats_based_config, self.model_based_config = build_execution_predictors(self.db_manager)

        # Use a dictionary to store model entries for each process
        self.process_model_map = {}

        # Build a lookup map to associate each resolved process name to its prediction model
        print("Building process name to prediction model lookup map...")
        for _, row in self.process_exec_counts.iterrows():
            process_name = row['process']
            model_entry = self.lookup_process_model(process_name)
            if model_entry is not None:
                # Store the model entry in the dictionary
                self.process_model_map[process_name] = model_entry
            else:
                print(f"Warning: No time prediction model found for process '{process_name}'.")

        # Close the database connection
        self.db_manager.close()

    def on_modified(self, event):
        # Normalize both paths for consistent comparison
        normalized_event_path = os.path.abspath(event.src_path)
        normalized_file_path = os.path.abspath(self.file_path)
        if normalized_event_path == normalized_file_path:
            self.check_file_changes()

    def check_file_changes(self):
        current_size = self.get_file_size()

        if current_size < self.last_size:
            print("Error: Some content has been removed from the file.")
            # Stop the observer and exit
            self.stop_observer()
            sys.exit(1)
        else:
            self.parse_new_content()
            self.last_size = current_size

    def get_file_size(self):
        return os.path.getsize(self.file_path)

    def parse_new_content(self):
        with open(self.file_path, 'r') as file:
            # 1. Look for parameters in the log if not already extracted
            if self.param_values is None:
                # Extract the parameter values from the beginning of the file
                self.param_values = extractPipelineParameters(self.file_path)
                if self.param_values is None:
                    self.param_values = None
                    return
                else:
                    # 2. Predict the total processing time of the pipeline from the parameters and the prediction model
                    predicted_times = []
                    for _, row in self.process_exec_counts.iterrows():
                        process_name = row['process']
                        model_entry = self.process_model_map.get(process_name)
                        if model_entry is not None:
                            if model_entry['type'] == 'stats_based':
                                # Use the stats-based model entry to predict the execution time
                                predicted_time = model_entry['mean_time']

                            elif model_entry['type'] == 'linear_reg_based':
                                # Use the linear regression-based model entry to predict the execution time
                                model = model_entry['model']
                                selected_params = model['selected_parameters']

                                # Check if all required parameters are available in self.param_values
                                missing_params = [param for param in selected_params if param not in self.param_values['param_name'].values]
                                if missing_params:
                                    print(f"Error: Missing required parameters for process '{process_name}': {missing_params}")
                                    self.stop_observer()
                                    return

                                # Construct the input array for prediction
                                X_pred = np.array([[self.param_values.loc[self.param_values['param_name'] == param,
                                                  'reformatted_value'].iloc[0] for param in selected_params]])
                                predicted_time = model['model'].predict(X_pred)[0]  # Use the predict method
                                if predicted_time < 0:
                                    print(f"Error: Negative predicted execution time for process '{process_name}'. Setting to 0.")
                                    self.stop_observer()
                                    return

                            else:
                                predicted_time = None
                                print(f"Error: Unknown model type for process '{process_name}'.")
                                self.stop_observer()
                                return
                        else:
                            predicted_time = None
                            print(f"Error: No model found for process '{process_name}'.")
                            self.stop_observer()

                        # Append the predicted time to the list
                        predicted_times.append(predicted_time)

                    # Add the predicted execution times to the DataFrame
                    self.process_exec_counts['predicted_execution_time'] = predicted_times

                    # 3. Compute and record predicted total execution times
                    # Compute the total predicted execution time using the "count" column
                    self.process_exec_counts['total_execution_time'] = (
                        self.process_exec_counts['predicted_execution_time'] * self.process_exec_counts['count']
                    )

                    self.total_predicted_time = self.process_exec_counts['total_execution_time'].sum()
                    # Convert total predicted time from seconds to a timedelta object for human-readable format
                    print(f"Total predicted execution time: {str(timedelta(seconds=self.total_predicted_time / 1000))}")

            # 4. Read the file from the last read position
            # Move to the last read position
            file.seek(self.last_position)

            # Read new content
            new_content = file.readlines()

            # Look for finished processes in the new content
            finished_process = extract_task_info_from_lines(new_content)
            
            for process_name in finished_process:
                self.process_exec_counts.loc[self.process_exec_counts['process'] == process_name, 'completed_executions'] += 1

            # 5. Print total elapsed time and remaining time
            self.process_exec_counts['elapsed_time'] = (
                        self.process_exec_counts['predicted_execution_time'] * self.process_exec_counts['completed_executions']
                    )

            total_elapsed_time = self.process_exec_counts['elapsed_time'].sum()
            print(f"Progress {100.0 * total_elapsed_time / self.total_predicted_time:.2f}%: {str(timedelta(seconds= total_elapsed_time/1000.0))} / {str(timedelta(seconds=self.total_predicted_time / 1000))}")

            
            
            # Update the last read position
            self.last_position = file.tell()

    def stop_observer(self):
        """
        Stop the observer gracefully.
        """
        self.observer.stop()

    def lookup_process_model(self, process_name):
        """
        Look up the corresponding entry for a process name in the stats_based_config or model_based_config dataframes.

        :param process_name: The name of the process to look up.
        :return: The row from stats_based_config or model_based_config with an added "type" column, otherwise None.
        """
        # First, look for the process name "as is" in the stats_based_config dataframe
        stats_row = self.stats_based_config[self.stats_based_config["process_name"] == process_name]
        if not stats_row.empty:
            stats_row = stats_row.iloc[0].copy()  # Copy the row to modify it
            stats_row["type"] = "stats_based"  # Add the "type" column
            return stats_row

        # Next, look for the process name "as is" in the model_based_config dataframe
        model_row = self.model_based_config[self.model_based_config["process_name"] == process_name]
        if not model_row.empty:
            model_row = model_row.iloc[0].copy()  # Copy the row to modify it
            model_row["type"] = "linear_reg_based"  # Add the "type" column
            return model_row

        # If not found, assume it is a resolved process name and retrieve the actual process name
        actual_process_name = self.db_manager.resolved_process_manager.getProcessByResolvedName(process_name)
        if actual_process_name is None:
            print(f"Warning: No process found for resolved process name '{process_name}'.")
            return None

        actual_process_name = actual_process_name.name  # Get the actual process name

        # Look for the actual process name in the stats_based_config dataframe
        stats_row = self.stats_based_config[self.stats_based_config["process_name"] == actual_process_name]
        if not stats_row.empty:
            stats_row = stats_row.iloc[0].copy()  # Copy the row to modify it
            stats_row["type"] = "stats_based"  # Add the "type" column
            return stats_row

        # Look for the actual process name in the model_based_config dataframe
        model_row = self.model_based_config[self.model_based_config["process_name"] == actual_process_name]
        if not model_row.empty:
            model_row = model_row.iloc[0].copy()  # Copy the row to modify it
            model_row["type"] = "linear_reg_based"  # Add the "type" column
            return model_row

        # If no entry is found, return None
        print(f"Warning: No model found for process name '{process_name}' or its resolved name '{actual_process_name}'.")
        return None


def watch_file(stub_report_path, stub_dag_path, db_path, log_path):
    observer = Observer()
    event_handler = NextflowTraceDBCompanion(stub_report_path, stub_dag_path, db_path, log_path, observer)
    # Pass the directory containing the file to observer.schedule
    observer.schedule(event_handler, path=os.path.dirname(log_path) or ".", recursive=False)
    # Start the observer
    print(f"Watching for changes in {log_path}...")
    observer.start()
    try:
        while event_handler.observer.is_alive():
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping observer...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    print("Nextflow Trace DB Companion is running...")
    # if len(sys.argv) != 3:
    # print("Usage: python script.py <sqlite_database path> <.nextflow.log file path>")
    # sys.exit(1)
    # db_path = sys.argv[1]
    # if not os.path.isfile(db_path):
    # print(f"Error: File '{db_path}' does not exist.")
    # sys.exit(1)
    # file_path = sys.argv[2]
    # if not os.path.isfile(file_path):
    # print(f"Error: File '{file_path}' does not exist.")
    # sys.exit(1)

    stub_report_path = "./dat/250516_241226_CELEBI_stub/karol_241226_ult_2025-05-16_13_48_58_report.html"
    stub_dag_path = "./dat/250516_241226_CELEBI_stub/karol_241226_ult_2025-05-16_13_48_58_dag.html"

    db_path = "./dat/nf_trace_db.sqlite"
    log_path = "./dat/.nextflow.log"
    watch_file(stub_report_path, stub_dag_path , db_path, log_path)
