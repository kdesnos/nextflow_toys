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
    G = nx.DiGraph()

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
