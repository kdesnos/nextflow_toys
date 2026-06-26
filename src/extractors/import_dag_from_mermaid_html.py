import re
import networkx as nx
from pathlib import Path

# Operators that fire once or have variable consumption/production rates
unique_firing_operators = [
    "branch",        # May not emit on some output channels for certain inputs
    "buffer",        # Accumulates before producing (may not output for every input)
    "collect",       # Collects all items
    "collectFile",   # Collects all items into a file
    "collate",       # Groups items by size (doesn't emit for every input)
    "combine",       # Cartesian product
    "concat",        # Concatenates channels with variable timing
    "count",         # Counts all items
    "countFasta",    # Counts all items
    "countFastq",    # Counts all items
    "countJson",     # Counts all items
    "countLines",    # Counts all items
    "cross",         # Cartesian product
    "distinct",      # Filters duplicates (may not emit for duplicate inputs)
    "dump",          # Debug output
    "filter",        # May not emit anything if condition isn't met
    "first",         # Only emits first item
    "flatMap",       # Variable number of outputs
    "flatten",       # Flattens collections (variable output)
    "groupTuple",    # Groups by key
    "ifEmpty",       # Conditional output
    "join",          # Joins channels
    "last",          # Only emits last item
    "max",           # Finds maximum
    "merge",         # Interleaves streams with variable timing
    "min",           # Finds minimum
    "mix",           # Randomizes merged streams
    "multiMap",      # Multiple variable outputs
    "randomSample",  # May not select some items
    "reduce",        # Reduces to single value
    "splitCsv",      # Variable output based on content
    "splitFasta",    # Variable output based on content
    "splitFastq",    # Variable output based on content
    "splitJson",     # Variable output based on content
    "splitText",     # Variable output based on content
    "subscribe",     # Side effect only
    "sum",           # Sums all values
    "take",          # Takes first N items
    "tap",           # Side branch with variable effects
    "toList",        # Collects all items
    "toSortedList",  # Collects and sorts all items
    "transpose",     # Transposes collections
    "unique",        # Filters duplicates (may not emit for duplicate inputs)
    "until"          # Terminates after condition
]

# Operators with strict SDF behavior: reliable 1-to-1 input to output ratio
multi_firing_operators = [
    "map",           # Always transforms each input to exactly one output
    "set",           # Always sets variable and passes through exactly one item
    "toInteger",     # Always converts each input to an integer
    "view"           # Side effect only
]


def extract_mermaid_graph(dag_report: Path) -> nx.DiGraph:
    """
    Extracts a Mermaid DAG from an HTML file and returns it as a NetworkX graph.

    Parameters:
    - dag_report (Path): The path to the HTML file containing the Mermaid DAG.

    Returns:
    - nx.DiGraph: The NetworkX directed graph representing the DAG.
    """
    # List of known operator names
    known_operators = unique_firing_operators + multi_firing_operators

    # Read the HTML file
    with open(dag_report, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Regular expression to extract the Mermaid graph definition
    mermaid_pattern = re.compile(r'<pre class="mermaid" style="text-align: center;">(.*?)</pre>', re.DOTALL)
    mermaid_match = mermaid_pattern.search(html_content)

    if not mermaid_match:
        raise ValueError("Mermaid graph definition not found in the HTML file.")

    mermaid_graph = mermaid_match.group(1).strip()

    # Regular expressions to extract nodes, edges, and subgraphs
    node_pattern = re.compile(r'(\w+)[\[|\()|\"]+(.*?)[\"|\]|\)]+')
    edge_pattern = re.compile(r'(\w+) -->(?:\|(.*?)\|)? (\w+)')
    subgraph_pattern = re.compile(r'subgraph\s+"(([\w\:]+)?\s(\[.*\])?)"', re.DOTALL)
    factory_pattern = re.compile(r'Channel\..*')

    # Create a directed graph
    G = nx.MultiDiGraph()

    # Extract nodes and their names
    nodes = node_pattern.findall(mermaid_graph)
    for node, name in nodes:
        node_type = "operator" if name in known_operators else "factory" if factory_pattern.match(name) else "process"
        G.add_node(node, name=name, type=node_type)

    # Extract edges
    edges = edge_pattern.findall(mermaid_graph)
    for source, label, target in edges:
        label = label if label else None
        G.add_edge(source, target, label=label)

    # Extract subgraphs
    subgraph_lines = mermaid_graph.splitlines()
    subgraph_stack = []
    anonymous_subgraph_count = 0

    for line in subgraph_lines:
        line = line.strip()
        if line.startswith('subgraph'):
            subgraph_name_match = subgraph_pattern.match(line)
            if subgraph_name_match:
                subgraph_name = subgraph_name_match.group(1).strip('"')
                if subgraph_name == " ":
                    subgraph_name = f'unnamed_{anonymous_subgraph_count}'
                    anonymous_subgraph_count += 1
                else:
                    subgraph_name = subgraph_name_match.group(2)
            subgraph_stack.append(subgraph_name)
        elif line == 'end' and len(subgraph_stack) > 0:
            subgraph_stack.pop()
        elif len(subgraph_stack) > 0:
            node_match = node_pattern.match(line)
            if node_match:
                node, name = node_match.groups()
                G.nodes[node]["subgraph"] = subgraph_stack[-1]

    return G


def add_execution_counts_to_graph(dag: nx.MultiDiGraph, trace_df) -> None:
    """
    Adds the number of executions of each process to the graph based on the trace data.

    :param dag: The directed graph (DAG) to update.
    :param trace_df: A DataFrame containing trace data with a 'process' column.
    """
    for node, data in dag.nodes(data=True):
        # For process nodes, we count the number of executions
        if data['type'] == 'process':
            process_full_name = (
                data.get('name', '') if not data.get('subgraph')
                else data.get('subgraph', '') + ":" + data.get('name', '')
            )
            nb_exec = trace_df['process'].str.contains(process_full_name).sum()
            dag.nodes[node]['nb_exec'] = nb_exec if nb_exec > 0 else 1  # Default to 1 if not found
        elif data['type'] == 'factory':
            dag.nodes[node]['nb_exec'] = 1
        elif data['type'] == 'operator':

            if data['name'] in unique_firing_operators:
                dag.nodes[node]['nb_exec'] = 1
            elif data['name'] in multi_firing_operators:
                # For multi-firing operators, set nb_exec to -1 to indicate multiple executions
                dag.nodes[node]['nb_exec'] = -1
            else:
                dag.nodes[node]['nb_exec'] = 1  # Default for unknown operators
        else:
            # Unknown node type, set default execution count
            dag.nodes[node]['nb_exec'] = 1
            print(f"Warning: Unknown node type for {node}: {data['type']}. nb_exec set to 1.")
