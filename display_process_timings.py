import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

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
        filtered_df = trace_df[trace_df.apply(lambda row: filter_string.lower() in row['process'].lower() or filter_string.lower() in row['process_name'].lower(), axis=1)]
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

# Example usage
# plot_realtime_boxplot(trace_df, filter_string="example")

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


