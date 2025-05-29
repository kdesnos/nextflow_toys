import time
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
from datetime import timedelta

from extract_from_nf_log import contains_execution_complete_message, extract_task_info_from_lines, extract_trace_file_path_from_lines, extractPipelineParameters, extractProcessInputs
from extract_from_trace import extract_trace_data_from_lines
from nf_trace_db_analyzer import build_execution_predictors
from nf_trace_db_manager import NextflowTraceDBManager
from extract_trace_from_html import extract_trace_data


class NextflowTraceDBCompanion(FileSystemEventHandler):
    def __init__(self, stub_report_path, db_path, file_path, observer):
        self.file_path = file_path
        self.observer = observer  # Store the observer instance
        self.last_position_log = 0  # To keep track of the last read position for the log file
        self.last_size_log = 0      # To keep track of the last file size for the log file
        self.last_position_trace = 0  # To keep track of the last read position for the trace file
        self.last_size_trace = 0      # To keep track of the last file size for the trace file
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
        self.process_tracker = stub_df['process'].value_counts().reset_index()
        # Add an extra column to the df to keep track of completed executions, initialized to 0
        self.process_tracker['completed_executions'] = 0
        self.process_tracker['elapsed_time'] = 0

        # Build the model used to predict execution time
        print("Building execution time prediction model from DB...")
        self.stats_based_config, self.model_based_config = build_execution_predictors(self.db_manager)

        # Use a dictionary to store model entries for each process
        self.process_model_map = {}

        # Build a lookup map to associate each resolved process name to its prediction model
        for _, row in self.process_tracker.iterrows():
            process_name = row['process']
            model_entry = self.lookup_process_model(process_name)
            if model_entry is not None:
                # Store the model entry in the dictionary
                self.process_model_map[process_name] = model_entry
            else:
                print(f"Error: No time prediction model found for process '{process_name}'.")
                sys.exit(1)

        # Close the database connection
        self.db_manager.close()

    def on_modified(self, event):
        """
        Handles file modification events and calls the appropriate parsing method
        based on the file that triggered the event.
        """
        # Normalize both paths for consistent comparison
        normalized_event_path = os.path.abspath(event.src_path)
        normalized_log_path = os.path.abspath(self.file_path)
        normalized_trace_path = os.path.abspath(self.trace_path) if hasattr(self, 'trace_path') and self.trace_path else None

        if normalized_event_path == normalized_log_path:
            # If the log file is modified, parse its new content
            self.parse_new_log_content()
        elif normalized_trace_path and normalized_event_path == normalized_trace_path:
            # If the trace file is modified, parse its new content
            self.parse_new_trace_content()

    def initialize_process_tracker(self):
        """
        Initializes the process_tracker table with predicted execution times and standard deviations.

        :return: None
        """
        predicted_times = []
        predicted_stddevs = []  # New list to store stddev or rmse values
        for _, row in self.process_tracker.iterrows():
            process_name = row['process']
            model_entry = self.process_model_map.get(process_name)
            if model_entry is not None:
                if model_entry['type'] == 'stats_based':
                    # Use the stats-based model entry to predict the execution time
                    predicted_time = model_entry['mean_time']
                    predicted_stddev = model_entry['std_dev_time']  # Extract stddev

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
                    predicted_stddev = model_entry['model']['rmse']  # Extract rmse
                    if predicted_time < 0:
                        print(f"Error: Negative predicted execution time for process '{process_name}'. Setting to 0.")
                        self.stop_observer()
                        return

                else:
                    predicted_time = None
                    predicted_stddev = None
                    print(f"Error: Unknown model type for process '{process_name}'.")
                    self.stop_observer()
                    return
            else:
                predicted_time = None
                predicted_stddev = None
                print(f"Error: No model found for process '{process_name}'.")
                self.stop_observer()

            # Append the predicted time and stddev/rmse to the lists
            predicted_times.append(predicted_time)
            predicted_stddevs.append(predicted_stddev)

        # Add the predicted execution times and stddev/rmse to the DataFrame
        self.process_tracker['predicted_execution_time'] = predicted_times
        self.process_tracker['predicted_stddev'] = predicted_stddevs  # Add the new column

        # Compute and record predicted total execution times
        self.compute_total_execution_time()

        # Display the initial progress with delta
        self.display_progress_with_delta()

    def parse_new_log_content(self):
        """
        Parses new content from the log file and initializes the process tracker if needed.

        :return: None
        """
        with open(self.file_path, 'r') as file:
            # 1. Look for parameters in the log if not already extracted
            if self.param_values is None:
                # Extract the parameter values from the beginning of the file
                self.param_values = extractPipelineParameters(self.file_path)
                if self.param_values is None:
                    self.param_values = None
                    return
                else:
                    # Initialize the process tracker
                    self.initialize_process_tracker()

                    # Compute and record predicted total execution times
                    self.compute_total_execution_time()

            # 4. Read the file from the last read position
            # Move to the last read position
            file.seek(self.last_position_log)

            # Read new content
            new_content = file.readlines()

            # 5. Find the trace file path in the new content if not already set
            if not hasattr(self, 'trace_path') or self.trace_path is None:
                self.trace_path = extract_trace_file_path_from_lines(new_content)
                if self.trace_path is not None:
                    print(f"Start monitoring trace file: {self.trace_path}")

                    # Immediately parse the trace file to initialize the process tracker
                    self.parse_new_trace_content()

                    # Schedule monitoring for the trace file
                    self.observer.schedule(self, path=os.path.dirname(self.trace_path), recursive=False)

            # 6. Wait for completion of log
            if contains_execution_complete_message(new_content):
                print("Execution complete message found in the log. Stopping observer.")
                self.stop_observer()
                return

            # Update the last read position
            self.last_position_log = file.tell()

    def update_process_tracker(self, process_trace):
        """
        Updates the process tracker with the completed executions and elapsed time for a given process.

        :param process_trace: A dictionary containing process execution details.
        :return: None
        """
        # Update the tracker for completed executions
        self.process_tracker.loc[self.process_tracker['process'] == process_trace['name'], 'completed_executions'] += 1

        # Update the tracker for elapsed time (convert to milliseconds)
        self.process_tracker.loc[self.process_tracker['process'] == process_trace['name'],
                                 'elapsed_time'] += process_trace['realtime'].total_seconds() * 1000

    def display_progress_with_delta(self):
        """
        Displays the progress of execution with the predicted time and 99% probabilistic delta.

        :return: None
        """
        # Compute total elapsed time directly within the function
        total_elapsed_time = self.process_tracker['elapsed_time'].sum()

        print(f"Progress {100.0 * total_elapsed_time / self.total_predicted_time:.2f}%:"
              + f" {format_timedelta_as_hours(timedelta(seconds=total_elapsed_time / 1000.0))}"
              + f" / {format_timedelta_as_hours(timedelta(seconds=self.total_predicted_time / 1000))}"
              + f" +/- {format_timedelta_as_hours(timedelta(seconds=(2.576 * self.total_relative_delta) / 1000))}")

    def parse_new_trace_content(self):
        """
        Parse new content from the trace file and update the process tracker.

        :return: None
        """
        with open(self.trace_path, 'r') as file:
            # Read the file from the last read position
            file.seek(self.last_position_trace)
            new_content = file.readlines()
            # Update the last read position
            self.last_position_trace = file.tell()

            # Process the new content (e.g., update execution statistics)
            trace_df = extract_trace_data_from_lines(new_content)
            if trace_df is not None and not trace_df.empty:
                for _, process_trace in trace_df.iterrows():
                    # Skip processes that did not complete successfully
                    if process_trace['exit'] != 0:
                        continue

                    # Call the method to check execution time
                    self.check_execution_time_within_probability_range(process_trace)

                    # Call the new method to update the tracker
                    self.update_process_tracker(process_trace)

                # Compute total execution time and display progress
                self.compute_total_execution_time()
                self.display_progress_with_delta()

                # If all processes are completed, stop the observer
                if self.process_tracker['completed_executions'].sum() == self.process_tracker['count'].sum():
                    print("All processes completed. Stopping observer.")
                    self.stop_observer()
                    return

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
            print(f"Error: No process found for resolved process name '{process_name}'.")
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
        print(f"Error: No model found for process name '{process_name}' or its resolved name '{actual_process_name}'.")
        return None

    def compute_total_execution_time(self):
        """
        Computes the total predicted execution time using the "count" column
        and updates the total_predicted_time attribute. Also computes the relative deviation (delta).

        :return: None
        """
        # Compute the total execution time
        self.process_tracker['total_execution_time'] = (
            self.process_tracker['elapsed_time']
            + (self.process_tracker['count'] - self.process_tracker['completed_executions']) * self.process_tracker['predicted_execution_time']
        )

        # Compute the accumulated standard deviation (delta) for not-yet-executed processes
        self.process_tracker['relative_delta'] = (
            np.sqrt(
                (self.process_tracker['count'] - self.process_tracker['completed_executions'])
                * (self.process_tracker['predicted_stddev'] ** 2)
            )
        )

        self.total_predicted_time = self.process_tracker['total_execution_time'].sum()
        self.total_relative_delta = np.sqrt((self.process_tracker['relative_delta'] ** 2).sum())

    def check_execution_time_within_probability_range(self, process_trace):
        """
        Checks if the execution time of a process is within the 99% probability range.
        Issues a warning if it is outside the range.

        :param process_trace: A dictionary containing process execution details.
        :return: None
        """
        predicted_time = self.process_tracker.loc[self.process_tracker['process'] == process_trace['name'], 'predicted_execution_time'].iloc[0]
        delta = self.process_tracker.loc[self.process_tracker['process'] == process_trace['name'], 'predicted_stddev'].iloc[0]
        lower_bound = predicted_time - 2.576 * delta
        upper_bound = predicted_time + 2.576 * delta

        if not (lower_bound <= process_trace['realtime'].total_seconds() * 1000 <= upper_bound):
            print(
                f"Warning: Execution time for process '{process_trace['name']}' ({process_trace['realtime'].total_seconds():.2f} s) "
                f"is outside the 99% probability range ({lower_bound / 1000.0:.2f} s to {upper_bound / 1000:.2f} s)."
            )


def format_timedelta_as_hours(td):
    """
    Format a timedelta object as "hours:minutes:seconds".

    :param td: timedelta object
    :return: formatted string
    """
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}min {seconds}s"


def watch_file(stub_report_path, db_path, log_path):
    observer = Observer()
    event_handler = NextflowTraceDBCompanion(stub_report_path, db_path, log_path, observer)
    print(f"Watching for changes in {log_path}...")
    # Immediately execute file observation
    event_handler.parse_new_log_content()
    # Pass the directory containing the file to observer.schedule
    observer.schedule(event_handler, path=os.path.dirname(log_path) or ".", recursive=False)
    # Start the observer
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
    if len(sys.argv) > 1 and len(sys.argv) != 4:
        print("Usage: python script.py <sqlite_database path> <.nextflow.log file path> <stub report file path>")
        sys.exit(1)

    if len(sys.argv) == 4:
        db_path = sys.argv[1]
        if not os.path.isfile(db_path):
            print(f"Error: File '{db_path}' does not exist.")
            sys.exit(1)
        log_path = sys.argv[2]
        if not os.path.isfile(log_path):
            print(f"Error: File '{log_path}' does not exist.")
            sys.exit(1)
        stub_report_path = sys.argv[3]
        if not os.path.isfile(stub_report_path):
                print(f"Error: File '{stub_report_path}' does not exist.")
                sys.exit(1)
    else:
        stub_report_path = "./dat/250516_241226_CELEBI_stub/karol_241226_ult_2025-05-16_13_48_58_report.html"
        db_path = "./dat/nf_trace_db.sqlite"
        log_path = "./dat/.nextflow.log"

    print(f"DB Path: {db_path}")
    print(f"Log Path: {log_path}")
    print(f"Stub Report Path: {stub_report_path}")

    watch_file(stub_report_path, db_path, log_path)
