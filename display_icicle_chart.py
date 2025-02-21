import plotly.graph_objects as go
import pandas as pd

def create_icicle_chart(trace_df, include_names=False):
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
            current_id = ":".join(levels[:i+1])
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