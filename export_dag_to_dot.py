import networkx as nx
from pathlib import Path

def export_to_dot(G: nx.DiGraph, output_path: Path):
    """
    Exports a NetworkX graph to a DOT file with the specified format.

    Parameters:
    - G (nx.DiGraph): The NetworkX directed graph to export.
    - output_path (Path): The path to the output DOT file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("digraph {\n")
        file.write("\tnode [shape=record];\n")
        file.write("\trankdir=LR;\n")
        file.write("\tnode [witdh=0.0, height=0.0];\n")
        # Write nodes
        for node, data in G.nodes(data=True):
            actor_name = data['name']
            inputs = [f"<i{i}>" for i in range(1, G.in_degree(node) + 1)]
            outputs = [f"<o{i}>" for i in range(1, G.out_degree(node) + 1)]
            inputs_str = " | ".join(inputs)
            outputs_str = " | ".join(outputs)
            file.write(f'\t{node}[label="{{{actor_name}}} | {{{{ {inputs_str} }} | | {{ {outputs_str} }}}} "];\n')

        # Write edges
        for source, target, data in G.edges(data=True):
            label = data.get('label', '')
            source_port = f"o{list(G.out_edges(source)).index((source, target)) + 1}"
            target_port = f"i{list(G.in_edges(target)).index((source, target)) + 1}"
            file.write(f'\t{source}:{source_port} -> {target}:{target_port} [label="{label}"];\n')

        file.write("}\n")