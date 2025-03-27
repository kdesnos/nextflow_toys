import re
import networkx as nx
from pathlib import Path

def extract_mermaid_graph(dag_report: Path) -> nx.DiGraph:
    """
    Extracts a Mermaid DAG from an HTML file and returns it as a NetworkX graph.

    Parameters:
    - dag_report (Path): The path to the HTML file containing the Mermaid DAG.

    Returns:
    - nx.DiGraph: The NetworkX directed graph representing the DAG.
    """
    # List of known operator names
    known_operators = [
        "combine", "first", "collect", "groupTuple", "join", "merge", "mix", "branch",
        "buffer", "compose", "concat", "count", "cross", "cycle", "debug", "debounce",
        "demultiplex", "distribute", "emit", "filter", "flatMap", "flatten", "groupBy",
        "groupKey", "map", "multiplex", "partition", "paste", "print", "publish",
        "reduce", "sample", "set", "setVal", "shuffle", "sort", "splitCsv", "splitText",
        "stageIn", "stageOut", "stdin", "stdout", "subscribe", "switch", "tee", "toList",
        "transpose", "unique", "until", "watchPath", "wrap", "zip", "min", "max", "sum",
        "mean", "median", "stddev", "variance"
    ]

    # List of known Channel factories
    known_factories = [
        "Channel.fromList", "Channel.fromPath", "Channel.fromFilePairs", "Channel.fromSRA",
        "Channel.fromConfigs", "Channel.fromWatch", "Channel.fromEnv", "Channel.fromParams",
        "Channel.fromTuple", "Channel.fromMap"
    ]

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
    subgraph_pattern = re.compile(r'subgraph\s+([\w\:]+|".*?")', re.DOTALL)

    # Create a directed graph
    G = nx.DiGraph()

    # Extract nodes and their names
    nodes = node_pattern.findall(mermaid_graph)
    for node, name in nodes:
        node_type = "operator" if name in known_operators else "factory" if name in known_factories else "process"
        G.add_node(node, name=name, type=node_type)

    # Extract edges
    edges = edge_pattern.findall(mermaid_graph)
    for source, label, target in edges:
        label = label if label else None
        G.add_edge(source, target, label=label)

    # Extract subgraphs
    subgraph_lines = mermaid_graph.splitlines()
    subgraph_name = None
    in_subgraph = False
    anonymous_subgraph_count = 0

    for line in subgraph_lines:
        line = line.strip()
        if line.startswith('subgraph'):
            if in_subgraph:
                # Process the previous subgraph
                subgraph_name = None
            subgraph_name_match = subgraph_pattern.match(line)
            if subgraph_name_match:
                subgraph_name = subgraph_name_match.group(1).strip('"')
                if subgraph_name == " ":
                    subgraph_name = f'unnamed_{anonymous_subgraph_count}'
                    anonymous_subgraph_count += 1
            in_subgraph = True
        elif line == 'end' and in_subgraph:
            in_subgraph = False
        elif in_subgraph:
            node_match = node_pattern.match(line)
            if node_match:
                node, name = node_match.groups()
                node_type = "operator" if name in known_operators else "factory" if name in known_factories else "process"
                G.add_node(node, name=name, subgraph=subgraph_name, type=node_type)

    return G
