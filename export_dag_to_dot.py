import networkx as nx
from pathlib import Path

def export_to_dot(G: nx.DiGraph, output_path: Path):
    """
    Exports a NetworkX graph to a DOT file with the specified format.

    Parameters:
    - G (nx.DiGraph): The NetworkX directed graph to export.
    - output_path (Path): The path to the output DOT file.
    """
    top_graph_name = "main"

    def write_node(file, node, data, level=0):
        actor_name = data['name']
        inputs = [f"<i{i}>" for i in range(1, G.in_degree(node) + 1)]
        outputs = [f"<o{i}>" for i in range(1, G.out_degree(node) + 1)]
        inputs_str = " | ".join(inputs)
        outputs_str = " | ".join(outputs)
        file.write(f'\t{"  " * level}{node}[label="{{{actor_name}}} | {{{{ {inputs_str} }} | | {{ {outputs_str} }}}}"')
        file.write(f', tooltip="x{data["nb_exec"]}"')
        if(data['type'] == "operator"):
            file.write(f', xlabel="{actor_name}", shape=point, height=0.1, width=0.1')
        if(data['type'] == "factory"):
            file.write(f', xlabel="{actor_name}", peripheries=2, shape=point, height=0.15, width=0.15')
        file.write('];\n')

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("digraph {\n")
        file.write("\tnode [shape=record, width=0.0, height=0.0];\n")
        file.write("\trankdir=LR;\n")
        file.write("\tnewrank=true;\n")
        file.write("\tranksep=1.5;\n")

        # Collect subgraphs
        # Nodes not belonging to any subgraph are placed in the "main" subgraph.
        subgraphs = {}
        for node, data in G.nodes(data=True):
            subgraph_name = data.get('subgraph')
            if subgraph_name:
                if not subgraph_name.startswith("unnamed_"):
                    subgraph_name = f"{top_graph_name}:" + subgraph_name
                if subgraph_name not in subgraphs:
                    subgraphs[subgraph_name] = []
                subgraphs[subgraph_name].append(node)
            else:
                if top_graph_name not in subgraphs:
                    subgraphs[top_graph_name] = []
                subgraphs[top_graph_name].append(node)

        # Write subgraphs
        def write_subgraph(file, subgraph_name, nodes, level=0):
            if not subgraph_name.startswith("unnamed_") and subgraph_name != top_graph_name:
                file.write(f'\t{"  " * level}subgraph cluster_{subgraph_name.replace(":", "_")} {{\n')
                file.write(f'\t{"  " * (level + 1)}label="{subgraph_name.split(":")[-1]}";\n')
            elif subgraph_name == top_graph_name:
                file.write(f'\t{"  " * level}subgraph {subgraph_name} {{\n')
            else:
                file.write(f'\t{"  " * level}{{\n')
                file.write(f'\t{"  " * (level + 1)}rank=same;\n')
            for node in nodes:
                write_node(file, node, G.nodes[node], level + 1)
            file.write(f'\t{"  " * level}}}\n')

        # Write nested subgraphs
        def write_nested_subgraphs(file, subgraphs):
            for subgraph_name, nodes in subgraphs.items():
                level = 0
                nested_subgraphs = subgraph_name.split(":")
                current_nodes = nodes
                for i, subgraph in enumerate(nested_subgraphs):
                    current_subgraph_name = ":".join(nested_subgraphs[:i+1])
                    if current_subgraph_name == subgraph_name:
                        write_subgraph(file, current_subgraph_name, current_nodes, level=level)
                    else:
                        if current_subgraph_name != top_graph_name:
                            file.write(f'\t{"  " * level}subgraph cluster_{current_subgraph_name.replace(":", "_")} {{\n')
                        else:
                            file.write(f'\t{"  " * level}subgraph {top_graph_name} {{\n')
                        level += 1
                for l in range(level, 0, -1):
                    file.write(f'\t{"  " * (l-1)}}}\n')
                level = 0

        write_nested_subgraphs(file, subgraphs)

        # Write edges
        for source, target, data in G.edges(data=True):
            label = data.get('label', '')
            source_port = f"o{list(G.out_edges(source)).index((source, target)) + 1}"
            target_port = f"i{list(G.in_edges(target)).index((source, target)) + 1}"
            file.write(f'\t{source}:{source_port} -> {target}:{target_port} [label="{label}"];\n')

        file.write("}\n")