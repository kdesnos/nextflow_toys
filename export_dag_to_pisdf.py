from pathlib import Path
import networkx as nx
import export_dag_to_dot
import extract_trace_from_html
import import_dag_from_mermaid_html


class PisdfExporter:
    """
    A class to export a networkx.DiGraph to a PiSDF representation.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initializes the PisdfExporter with a directed graph.

        :param graph: A networkx.DiGraph instance representing the graph to export.
        """
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("The graph parameter must be an instance of networkx.DiGraph.")
        self.graph = graph
        self.interface_counter = 0  # Counter for interface nodes

    def export(self) -> str:
        """
        Exports the directed graph to a PiSDF representation.
        :return: A string representing the PiSDF format of the graph.
        """
        results = []

        for node, data in self.graph.nodes(data=True):
            results.append(f"{node}")

        return "\n".join(results)

    def transform_graph(self):
        """
        Transforms the graph to ensure it is in a suitable format for PiSDF export.
        This method can be extended to include any necessary transformations.
        """
        # Collect subgraphs
        # Nodes not belonging to any subgraph are placed in the "main" subgraph.
        subgraph_nodes = self.collect_subgraph_nodes()
        subgraph_edges = self.collect_subgraph_edges(subgraph_nodes)

    def collect_subgraph_nodes(self) -> dict:
        """
        Collects subgraphs from the directed graph. Nodes not belonging to any subgraph
        are placed in the "main" subgraph.

        :return: A dictionary where keys are subgraph names and values are lists of nodes.
        """
        subgraphs_nodes = {}
        top_graph_name = "main"
        for node, data in self.graph.nodes(data=True):
            subgraph_name = data.get('subgraph')
            if subgraph_name:
                if not subgraph_name.startswith("unnamed_"):
                    subgraph_name = f"{top_graph_name}:" + subgraph_name
                else:
                    subgraph_name = top_graph_name

                # Check if the subgraph name is already in the dictionary
                # If not, initialize it with an empty list
                if subgraph_name not in subgraphs_nodes:
                    subgraphs_nodes[subgraph_name] = []

                # Append the node to the appropriate subgraph list
                subgraphs_nodes[subgraph_name].append(node)
            else:
                # If the node does not belong to any subgraph, add it to the main subgraph
                if top_graph_name not in subgraphs_nodes:
                    subgraphs_nodes[top_graph_name] = []
                subgraphs_nodes[top_graph_name].append(node)
        return subgraphs_nodes

    def lookup_node_subgraph(self, node, subgraph_nodes: dict) -> str:
        """
        Finds the subgraph name for a given node.

        :param node: The node to look up.
        :param subgraph_nodes: A dictionary where keys are subgraph names and values are lists of nodes.
        :return: The name of the subgraph the node belongs to, or None if not found.
        """
        return next((name for name, nodes in subgraph_nodes.items() if node in nodes), None)

    def collect_subgraph_edges(self, subgraph_nodes: dict) -> dict:
        """
        Collects edges for each subgraph and handles hierarchical relationships by adding interface nodes
        and splitting edges as needed.

        :param subgraph_nodes: A dictionary where keys are subgraph names and values are lists of nodes.
        :return: A dictionary where keys are subgraph names and values are lists of edges.
        """
        subgraph_edges = {subgraph: [] for subgraph in subgraph_nodes.keys()}

        # Iterate over a copy of the edges to avoid modifying the graph during iteration
        for source, target, data in list(self.graph.edges(data=True)):
            # Use the lookup_node_subgraph function to find the subgraph names
            source_subgraph = self.lookup_node_subgraph(source, subgraph_nodes)
            target_subgraph = self.lookup_node_subgraph(target, subgraph_nodes)

            if source_subgraph == target_subgraph:
                # Case 1: Both producer and consumer are in the same subgraph
                subgraph_edges[source_subgraph].append((source, target, data))
            else:
                # Generalized Case: Producer and consumer are in distinct subgraphs
                source_path = (source_subgraph or "main").split(":")
                target_path = (target_subgraph or "main").split(":")
                self.build_path_and_edges(source_path, target_path, source, target, data, subgraph_edges, subgraph_nodes)

                # Safely remove the original edge (there can be only one edge between two nodes in a DiGraph)
                self.graph.remove_edge(source, target)

        return subgraph_edges

    def build_path_and_edges(self, source_path, target_path, source, target, data, subgraph_edges, subgraphs_nodes):
        """
        Builds the path between the source and target subgraphs, adding interface nodes and edges.
        :param source_path: The hierarchical path of the source subgraph.
        :param target_path: The hierarchical path of the target subgraph.
        :param source: The source node.
        :param target: The target node.
        :param data: The edge data.
        :param subgraph_edges: The dictionary of subgraph edges.
        :param subgraphs: The dictionary of subgraph nodes.
        """
        self.interface_counter  # Use the outer scope's counter
        # Find the common prefix of the source and target paths
        common_prefix = []
        for a, b in zip(source_path, target_path):
            if a == b:
                common_prefix.append(a)
            else:
                break

        # Build the path from the source to the common prefix
        current_node = source
        for i in range(len(source_path), len(common_prefix), -1):
            interface_name = f"i{self.interface_counter:03d}"  # Unique name for the interface
            self.interface_counter += 1
            # Add the interface node
            self.graph.add_node(interface_name, type="interface", name=data['label'], nb_exec=1, subgraph=":".join(source_path[1:i]))
            subgraphs_nodes[":".join(source_path[:i])].append(interface_name)
            # Add the edge berween the internal node and the interface toward upper level
            self.graph.add_edge(current_node, interface_name, **data)  # Add edge to the graph
            subgraph_edges[":".join(source_path[:i])].append((current_node, interface_name, data))
            current_node = interface_name

        # Build the path from the common prefix to the target
        for i in range(len(common_prefix), len(target_path)):
            interface_name = f"i{self.interface_counter:03d}"  # Unique name for the interface
            self.interface_counter += 1
            # Add the interface node
            subgraphs_nodes[":".join(target_path[:i + 1])].append(interface_name)
            self.graph.add_node(interface_name, type="interface", name=data['label'], nb_exec=1, subgraph=":".join(target_path[1:i + 1]))
            # Add the edge between the external node and the interface toward lower level
            self.graph.add_edge(current_node, interface_name, **data)  # Add edge to the graph
            subgraph_edges[":".join(target_path[:i])].append((current_node, interface_name, data))
            current_node = interface_name

        # Add the final edge to the target
        subgraph_edges[":".join(target_path)].append((current_node, target, data))
        self.graph.add_edge(current_node, target, **data)  # Add edge to the graph

    def export_to_file(self, file_path: str):
        """
        Exports the directed graph to a file in PiSDF format.

        :param file_path: The path to the file where the PiSDF representation will be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(self.export())


if __name__ == "__main__":
    nf_report_path = Path("./dat/250515_241226_CELEBI/", "karol_241226_ult_2025-05-15_13_41_42")

    # Create the HTML and DAG files
    html_report = nf_report_path.with_name(nf_report_path.name + "_report.html")
    dag_report = nf_report_path.with_name(nf_report_path.name + "_dag.html")

    # Get the trace data from the HTML report
    trace_df = extract_trace_from_html.extract_trace_data(html_report)

    # Import the DAG from the Mermaid report
    dag = import_dag_from_mermaid_html.extract_mermaid_graph(dag_report)

    # Add the number of executions of each process to the graph
    for node, data in dag.nodes(data=True):
        process_full_name = data.get('name', '') if not data.get('subgraph') else data.get('subgraph', '') + ":" + data.get('name', '')
        nb_exec = trace_df['process'].str.contains(process_full_name).sum()
        dag.nodes[node]['nb_exec'] = nb_exec if nb_exec > 0 else 1  # If the process is not in the trace, we assume it was executed once

    # Export the DAG to PiSDF format and to a DOT file
    pisdf_exporter = PisdfExporter(dag)
    pisdf_exporter.transform_graph()
    export_dag_to_dot.export_to_dot(pisdf_exporter.graph, nf_report_path.with_name(nf_report_path.name + "_dag.dot"))
    pisdf_exporter.export_to_file(nf_report_path.with_name(nf_report_path.name + "_dag.pi"))
