import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from nf_trace_db_analyzer import extract_execution_time_linear_reg
from nf_trace_db_manager import NextflowTraceDBManager


def plot_realtime_boxplot(trace_df, filter_string=None):
    """
    Creates a boxplot of realtime for each process and full path, with optional filtering.

    Parameters:
    - trace_df (pd.DataFrame): DataFrame containing 'process', 'process_name', and 'realtime' columns.
    - filter_string (str, optional): String used to filter the displayed processes and full paths.

    Returns:
    - None: Displays the plot.
    """

    # Convert 'realtime' to minutes
    trace_df['realtime_minutes'] = trace_df['realtime'].dt.total_seconds() / 60

    # Filter the DataFrame based on the filter string, if provided
    if filter_string:
        filtered_df = trace_df[trace_df.apply(lambda row: filter_string.lower() in row['process'].lower()
                                              or filter_string.lower() in row['process_name'].lower(), axis=1)]
    else:
        filtered_df = trace_df

    # Create subplots with tabs
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "box"}]])

    # Add boxplot traces for each unique process
    for process in filtered_df['process'].unique():
        fig.add_trace(go.Box(
            x=[process] * len(filtered_df[filtered_df['process'] == process]),
            y=filtered_df[filtered_df['process'] == process]['realtime_minutes'],
            name=f"Full Path: {process}",
            visible='legendonly'  # Initially hidden, can be toggled from the legend
        ), row=1, col=1)

    # Add boxplot traces for each unique process_name with the prefix "Process"
    for process_name in filtered_df['process_name'].unique():
        fig.add_trace(go.Box(
            x=[process_name] * len(filtered_df[filtered_df['process_name'] == process_name]),
            y=filtered_df[filtered_df['process_name'] == process_name]['realtime_minutes'],
            name=f"Process: {process_name}",
            visible=True  # Initially visible
        ), row=1, col=1)

    # Update layout to include buttons for tabs
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="Process",
                         method="update",
                         args=[{"visible": [True if 'Process' in t.name else 'legendonly' for t in fig.data]}]),
                    dict(label="Full Path",
                         method="update",
                         args=[{"visible": [True if 'Full Path' in t.name else 'legendonly' for t in fig.data]}]),
                    dict(label="All",
                         method="update",
                         args=[{"visible": True}]),
                    dict(label="None",
                         method="update",
                         args=[{"visible": 'legendonly'}])
                ]),
            )
        ],
        title="Boxplot of Realtime",
        xaxis_title="Process",
        yaxis_title="Realtime (minutes)"
    )

    # Show the plot
    fig.show()


def plot_wait_times(trace_df):
    """
    Plots the waiting time for jobs as a function of the number of allocated CPUs.

    This function calculates the average waiting time for jobs and creates a boxplot
    to visualize the waiting time distribution based on the number of allocated CPUs.

    Parameters:
    - trace_df (pd.DataFrame): DataFrame containing 'start', 'submit', and 'cpus' columns.

    Returns:
    - None: Displays the plot.

    The function performs the following steps:
    1. Initializes an empty DataFrame `wait_time`.
    2. Calculates the waiting time for each job by subtracting the 'submit' time from the 'start' time.
    3. Computes the average waiting time and prints it.
    4. Adds the number of CPUs allocated for each job to the `wait_time` DataFrame.
    5. Converts the waiting time to minutes.
    6. Creates a boxplot using Plotly Express to visualize the waiting time distribution based on the number of CPUs.
    7. Sets the x-axis tick values to ensure all distinct CPU values are displayed.
    8. Displays the plot.
    """
    wait_time = pd.DataFrame()

    # Average wait time
    wait_time['wait'] = trace_df['start'] - trace_df['submit']
    avg_wait = (wait_time['wait']).mean()

    # Per number of CPU wait time
    wait_time['nb_cpu'] = trace_df['cpus']

    print(f'Average job waiting time: {avg_wait}.')

    # Convert 'wait' column to total seconds
    wait_time['wait_minutes'] = wait_time['wait'].dt.total_seconds() / 60

    # Create the boxplot
    fig = px.box(wait_time, y='wait_minutes', x='nb_cpu', points="all",
                 labels={'wait_minutes': 'Waiting Time (minutes)', 'nb_cpu': 'Number of CPUs'},
                 title="Waiting time = f(Number of allocated CPUs)")

    # Set x-axis tick values to ensure all distinct CPU values are displayed
    fig.update_xaxes(tickvals=sorted(wait_time['nb_cpu'].unique()))

    # Show the plot
    fig.show()


def plot_icicle_chart(trace_df, include_names=False):
    """
    Creates and displays an icicle chart using the hierarchical data in the 'process' column,
    with values representing the sum of realtime durations in minutes. Optionally handles duplicate processes
    by adding entries from the 'name' column.

    Parameters:
    - trace_df (pd.DataFrame): DataFrame containing 'process', 'realtime', and 'name' columns.
    - include_names (bool): Flag to include per-process name handling.

    Returns:
    - None: Displays the chart directly.
    """

    # Initialize the hierarchy array with the root element
    hierarchy = [["Workflow", "Workflow", "", 0]]

    # Calculate the total sum of realtime durations for the Workflow node in minutes
    total_realtime_sum = trace_df['realtime'].sum().total_seconds() / 60

    # Update the Workflow node with the total sum in minutes
    hierarchy[0][3] = total_realtime_sum

    # Function to add unique entries to the hierarchy and calculate the sum of realtime durations in minutes
    def add_to_hierarchy(process, realtime_sum):
        levels = process.split(':')
        parent = "Workflow"
        for i, level in enumerate(levels):
            current_id = ":".join(levels[:i + 1])
            # Find the existing entry or create a new one
            existing_entry = next((entry for entry in hierarchy if entry[0] == current_id), None)
            if existing_entry:
                existing_entry[3] += realtime_sum  # Add to the existing sum
            else:
                hierarchy.append([current_id, level, parent, realtime_sum])
            parent = current_id

    # Iterate over each row in the trace_df to handle duplicates
    for index, row in trace_df.iterrows():
        process = row['process']
        realtime_sum = row['realtime'].total_seconds() / 60  # Convert to minutes
        add_to_hierarchy(process, realtime_sum)

    # Subfunction to analyze the 'name' column and add additional hierarchy levels for duplicates
    def analyze_names():
        process_counts = trace_df['process'].value_counts()
        for process, count in process_counts.items():
            if count > 1:  # Only process if there are duplicates
                relevant_rows = trace_df[trace_df['process'] == process]
                parent_id = process
                # Add entries for each duplicate process with names
                for _, row in relevant_rows.iterrows():
                    name = row['name']
                    realtime_sum = row['realtime'].total_seconds() / 60  # Convert to minutes
                    unique_id = f"{parent_id} ({name})"
                    hierarchy.append([unique_id, name, parent_id, realtime_sum])

    # Call the subfunction if include_names is True
    if include_names:
        analyze_names()

    # Convert the hierarchy list to a DataFrame for easier manipulation
    hierarchy_df = pd.DataFrame(hierarchy, columns=['ids', 'labels', 'parents', 'values'])

    # Ensure all entries in the hierarchy are unique
    hierarchy_df = hierarchy_df.drop_duplicates()

    # Create the icicle chart with top-down orientation and branchvalues set to "total"
    fig = go.Figure(go.Icicle(
        labels=hierarchy_df['labels'],
        parents=hierarchy_df['parents'],
        ids=hierarchy_df['ids'],
        values=hierarchy_df['values'],
        root_color="lightgrey",
        tiling=dict(
            orientation='v'  # Vertical orientation for top-down
        ),
        branchvalues="total"  # Ensure parent widths are proportional to the sum of their children's values
    ))

    # Update layout for better readability
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        title="Icicle Chart of Process Hierarchy with Realtime Sums (in minutes)"
    )

    # Display the chart
    fig.show()


def plot_execution_time_predictions(db_manager: NextflowTraceDBManager, process_name, trace_names=None, linear_model=None, traces_for_model=None, is_resolved_name=True):
    """
    Plots the actual vs predicted execution times for a process across multiple traces.

    :param db_manager: An instance of NextflowTraceDBManager.
    :param process_name: The name of the process being analyzed.
    :param trace_names: List of trace names used in the analysis. If None, all traces are used.
    :param linear_model: The linear regression model. If None, it will be computed.
    :param traces_for_model: List of traces to use for building the model. If None, all traces are used.
    """
    # Use all traces if trace_names is not provided
    if trace_names is None:
        trace_names = db_manager.process_executions_manager.getAllTraceNames()

    # Use all traces for the model if traces_for_model is not provided
    if traces_for_model is None:
        traces_for_model = trace_names

    # Compute the linear model if not provided
    if linear_model is None:
        linear_model = extract_execution_time_linear_reg(
            db_manager,
            process_name,
            print_info=True,
            is_resolved_name=is_resolved_name,
            trace_names=traces_for_model
        )
        print(f"\nLinear model expression: {linear_model['expression']}")

    # Retrieve execution times for the given traces
    execution_times = db_manager.process_executions_manager.getExecutionTimesForProcessAndTraces(
        process_name, trace_names, is_resolved_name=is_resolved_name
    )
    if execution_times.empty:
        print("No execution data found. Exiting.")
        return

    # Retrieve parameter values for the given traces
    params_by_trace = db_manager.pipeline_param_values_manager.getParamValuesForTraces(trace_names)

    # Create a plot for linear regression
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Actual vs Predicted Execution Times for {process_name}\nAcross Multiple Traces', fontsize=16)

    colors = [hsv_to_rgb((i / len(trace_names), 0.8, 0.9)) for i in range(len(trace_names))]

    all_predictions_linear = []
    all_actuals = []
    selected_params_linear = linear_model["selected_parameters"]

    for i, trace_name in enumerate(trace_names):
        trace_execution_times = execution_times[execution_times["trace_name"] == trace_name]
        if trace_execution_times.empty:
            continue
        param_values = params_by_trace[trace_name]
        X_pred = np.array([[param_values.get(param, 0) for param in selected_params_linear]] * len(trace_execution_times))
        predictions = linear_model["model"].predict(X_pred)
        all_predictions_linear.extend(predictions)
        all_actuals.extend(trace_execution_times["execution_time"].values)
        ax.scatter(predictions, trace_execution_times["execution_time"], alpha=0.7,
                   color=colors[i], s=40, label=f"{trace_name} (n={len(trace_execution_times)})")

    # Calculate RMSE and additional lines
    rmse = linear_model["rmse"]
    min_actual = min(all_actuals)
    max_actual = max(all_actuals)
    x_range = [min_actual, max_actual]

    # Line for predicted value + 2 * RMSE
    y_pred_plus_2rmse = [val + 2 * rmse for val in x_range]
    ax.plot(x_range, y_pred_plus_2rmse, 'g--', label='Predicted + 2 * RMSE')

    # Line for predicted value + 2 * RMSE + 60 seconds (60000 ms)
    y_pred_plus_2rmse_60s = [val + 2 * rmse + 60000 for val in x_range]
    ax.plot(x_range, y_pred_plus_2rmse_60s, 'b--', label='Predicted + 2 * RMSE + 60s')

    # Perfect prediction line
    ax.plot(x_range, x_range, 'r--', label='Perfect prediction (y = x)')

    # Add the linear model expression to the plot
    ax.text(
        0.05, 0.90, f"Linear Model: {linear_model['expression']}",
        transform=ax.transAxes,
        fontsize=12,
        color="red",
        verticalalignment="top"
    )

    # Set labels, legend, and grid
    ax.set_xlabel('Predicted Execution Time (ms)')
    ax.set_ylabel('Actual Execution Time (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
