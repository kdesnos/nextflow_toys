from pathlib import Path
import networkx as nx
import export_dag_to_dot
import extract_trace_from_html
import import_dag_from_mermaid_html
from lxml import etree


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
        self.graph = nx.MultiDiGraph(graph)
        self.interface_counter = 0  # Counter for interface nodes
        self.hierarchical_counter = 0  # Counter for hierarchical actors

    def transform_graph(self):
        """
        Transforms the graph to ensure it is in a suitable format for PiSDF export.
        This method can be extended to include any necessary transformations.
        """
        # Collect subgraphs
        # Nodes not belonging to any subgraph are placed in the "main" subgraph.
        subgraph_nodes = self.collect_subgraph_nodes()

        # Create a unique source and sink for the top-level graph
        # Don't not keep merged source, as they represent parameters passed to processes without processing.
        # Keep sink, as they represent the result of the processing.
        self.merge_source_sink(subgraph_nodes)

        # Collect edges for each subgraph and handle hierarchical relationships
        # by adding interface nodes and splitting edges as needed.
        # This will also modify the graph in place to add interface nodes.
        subgraph_edges = self.collect_subgraph_edges(subgraph_nodes)

        # Split the main subgraph into subgraphs based on the provided nodes and edges
        # This will also modify the graph in place to add hierarchical actors.
        # This is done to ensure that the hierarchical actors are created correctly
        # and that the edges are split correctly.
        self.split_subgraphs(subgraph_nodes, subgraph_edges)

        # Give a name to all input and output "ports" of edges
        self.name_ports(subgraph_nodes, subgraph_edges)

    def name_ports(self, subgraph_nodes: dict, subgraph_edges: dict) -> None:
        # Browse through the subgraphs and their nodes and name the ports
        for subgraph_name, nodes in subgraph_nodes.items():
            for node in nodes:
                # Get all edges connected to the node
                in_edges = list(self.graph.in_edges(node, data=True, keys=True))
                out_edges = list(self.graph.out_edges(node, data=True, keys=True))

                # Create a dictionary of edge names
                edge_names = {}
                for edge in in_edges + out_edges:
                    data = edge[3]
                    edge_name = data.get('label') if data.get('label') is not None else 'unnamed'
                    if edge_name not in edge_names:
                        edge_names[edge_name] = []
                    edge_names[edge_name].append(edge)

                # Name the ports based on the edge names, with unique identifiers
                for name, edges in edge_names.items():
                    if len(edges) > 1:
                        # If there are multiple edges, we need to create unique names for the ports
                        for i, edge in enumerate(edges):
                            source, target, key, data = edge
                            port_name = f"{name}_{i:02d}"
                            self.graph.edges[source, target, key]['snk_port' if edge in in_edges else 'src_port'] = port_name
                    else:
                        # If there is only one edge, we can use the edge name directly
                        source, target, key, data = edges[0]
                        self.graph.edges[source, target, key]['snk_port' if edges[0] in in_edges else 'src_port'] = name

                    for edge in edges:
                        source, target, key, data = edge
                        # If the edge is linked to an hierarchical actor, update the name of the interface_source or interface_target
                        if edge in in_edges and data.get('interface_source'):
                            self.graph.nodes[data.get('interface_source')]['name'] = self.graph.edges[source, target, key]['snk_port']
                        if edge in out_edges and data.get('interface_target'):
                            self.graph.nodes[data.get('interface_target')]['name'] = self.graph.edges[source, target, key]['src_port']

    def split_subgraphs(self, subgraph_nodes: list, subgraph_edges) -> None:
        """
        Splits the main subgraph into subgraphs based on the provided nodes and edges.
        This method modifies the graph in place.

        :param subgraph_nodes: A list of nodes that belong to the main subgraph.
        :param subgraph_edges: A dictionary where keys are subgraph names and values are lists of edges.
        """
        # Process each subgraph (work on a copy of the subgraph_nodes to avoid modifying it during iteration)
        for subgraph_name, nodes in list(subgraph_nodes.items()):
            # Find the hierarchical actors within the subgraph
            hierarchical_nodes = {}
            for edge in list(subgraph_edges[subgraph_name]):
                replacement_edge = [None, None]
                # If the target or the source is an interface to a subgraph, we need to find the path
                for i in [0, 1]:  # Ensure both source and target are looked up
                    path = self.lookup_node_subgraph(edge[i], subgraph_nodes)
                    if (len(subgraph_name) < len(path)):
                        # The edge connects to a subgraph, create the hierarchical actor unless it already exists
                        if path not in hierarchical_nodes:
                            hierarchical_actor_name = path[len(subgraph_name):].lstrip(':')
                            hierarchical_actor_node = "h" + f"{self.hierarchical_counter:03d}"
                            self.hierarchical_counter += 1
                            self.graph.add_node(hierarchical_actor_node, type="hierarchical", name=hierarchical_actor_name, nb_exec=1, subgraph=path[:len(subgraph_name)].strip('main').lstrip(':'),
                                                hierarchical_subgraph=path)
                            subgraph_nodes[subgraph_name].append(hierarchical_actor_node)
                            hierarchical_nodes[path] = hierarchical_actor_node

                        replacement_edge[i] = hierarchical_nodes[path]

                if (replacement_edge[0] is not None or replacement_edge[1] is not None):
                    # Fetch the source & target of replacement edge
                    source = replacement_edge[0] if replacement_edge[0] else edge[0]
                    target = replacement_edge[1] if replacement_edge[1] else edge[1]
                    # Add the edge to the graph
                    new_key = self.graph.add_edge(source, target, None, **self.graph.get_edge_data(edge[0], edge[1], edge[2]))
                    subgraph_edges[subgraph_name].append((source, target, new_key))
                    # Keep a ref to the interfaces in the edge data
                    if replacement_edge[0] is not None:
                        self.graph.edges[source, target, new_key]['interface_target'] = edge[0]
                    if replacement_edge[1] is not None:
                        self.graph.edges[source, target, new_key]['interface_source'] = edge[1]

                    # Remove the original edges that were replaced by hierarchical actors
                    self.graph.remove_edge(edge[0], edge[1], edge[2])
                    subgraph_edges[subgraph_name].remove((edge[0], edge[1], edge[2]))

    def merge_source_sink(self, subgraph_nodes: dict, keep_merged_source=False, keep_merged_sink=True) -> None:
        """
        Merges source and sink nodes in the main subgraph that have a space as a name and no incoming or outgoing edges.
        If keep_merged_source is True, a unique source node is created if there are multiple source nodes.
        If keep_merged_sink is True, a unique sink node is created if there are multiple sink nodes.
        This method modifies the graph in place.

        :param subgraph_nodes: A dictionary where keys are subgraph names and values are lists of nodes.
        :param keep_merged_source: If True, a unique source node will be created if there are multiple source nodes.
        :param keep_merged_sink: If True, a unique sink node will be created if there are multiple sink nodes.
        """
        # Identify nodes in main subgraph that have a space as a name and no incoming edges
        main_subgraph = subgraph_nodes.get("main", [])
        source_nodes = [node for node in main_subgraph if self.graph.in_degree(node) == 0 and self.graph.nodes[node].get('name', '').strip() == '']

        # If keep_merged_source is True, we do not create a unique source node
        if keep_merged_source:
            # Create a unique source node if there are multiple source nodes
            if len(source_nodes) > 1:
                unique_source = "source"
                self.graph.add_node(unique_source, type="process", name="Source", nb_exec=1, subgraph="unnamed_src")
                main_subgraph.append(unique_source)
                # connect all outgoing edges from source nodes to the unique source
                for source in source_nodes:
                    for target in self.graph.successors(source):
                        data = self.graph.get_edge_data(source, target)
                        self.graph.add_edge(unique_source, target, None, **data)  # There can be only one edge at this point from DAG import use key=0
                    # remove the original source node
                    self.graph.remove_node(source)
                    subgraph_nodes["main"].remove(source)
        else:
            # Remove all source nodes that have a space as a name and no incoming edges
            for source in source_nodes:
                self.graph.remove_node(source)
                subgraph_nodes["main"].remove(source)

        # Identify nodes in main subgraph that have a space as a name and no outgoing edges
        sink_nodes = [node for node in main_subgraph if self.graph.out_degree(node) == 0 and self.graph.nodes[node].get('name', '').strip() == '']

        # If keep_merged_sink is True, we do not create a unique sink node
        if keep_merged_sink:
            # Create a unique sink node if there are multiple sink nodes
            if len(sink_nodes) > 1:
                unique_sink = "sink"
                self.graph.add_node(unique_sink, type="operator", name="Sink", nb_exec=1, subgraph="unnamed_sink")
                main_subgraph.append(unique_sink)
                # connect all incoming edges to the unique sink
                for sink in sink_nodes:
                    for source in self.graph.predecessors(sink):
                        data = self.graph.get_edge_data(source, sink, 0)  # There can be only one edge at this point from DAG import use key=0
                        self.graph.add_edge(source, unique_sink, None, **data)
                    # remove the original sink node
                    self.graph.remove_node(sink)
                    subgraph_nodes["main"].remove(sink)
        else:
            # Remove all sink nodes that have a space as a name and no outgoing edges
            for sink in sink_nodes:
                self.graph.remove_node(sink)
                subgraph_nodes["main"].remove(sink)

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
        for source, target, key, data in list(self.graph.edges(data=True, keys=True)):
            # Use the lookup_node_subgraph function to find the subgraph names
            source_subgraph = self.lookup_node_subgraph(source, subgraph_nodes)
            target_subgraph = self.lookup_node_subgraph(target, subgraph_nodes)

            if source_subgraph == target_subgraph:
                # Case 1: Both producer and consumer are in the same subgraph
                subgraph_edges[source_subgraph].append((source, target, key))
            else:
                # Generalized Case: Producer and consumer are in distinct subgraphs
                source_path = (source_subgraph or "main").split(":")
                target_path = (target_subgraph or "main").split(":")
                self.build_path_and_edges(source_path, target_path, source, target, key, data, subgraph_edges, subgraph_nodes)

                # Safely remove the original edge
                self.graph.remove_edge(source, target, key)

        return subgraph_edges

    def build_path_and_edges(self, source_path, target_path, source, target, key, data, subgraph_edges, subgraphs_nodes):
        """
        Builds the path between the source and target subgraphs, adding interface nodes and edges.
        :param source_path: The hierarchical path of the source subgraph.
        :param target_path: The hierarchical path of the target subgraph.
        :param source: The source node.
        :param target: The target node.
        :param key: The edge key (if applicable).
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
            self.graph.add_node(interface_name,
                                type="interface",
                                direction="out",
                                name=data['label'] if data['label'] is not None else "unnamed",
                                nb_exec=1,
                                subgraph=":".join(source_path[1:i]))
            subgraphs_nodes[":".join(source_path[:i])].append(interface_name)
            # Add the edge berween the internal node and the interface toward upper level
            new_key = self.graph.add_edge(current_node, interface_name, None , **data)  # Add edge to the graph
            subgraph_edges[":".join(source_path[:i])].append((current_node, interface_name, new_key))
            current_node = interface_name

        # Build the path from the common prefix to the target
        for i in range(len(common_prefix), len(target_path)):
            interface_name = f"i{self.interface_counter:03d}"  # Unique name for the interface
            self.interface_counter += 1
            # Add the interface node
            subgraphs_nodes[":".join(target_path[:i + 1])].append(interface_name)
            self.graph.add_node(interface_name,
                                type="interface",
                                direction="in",
                                name=data['label'] if data['label'] is not None else "unnamed",
                                nb_exec=1,
                                subgraph=":".join(target_path[1:i + 1]))
            # Add the edge between the external node and the interface toward lower level
            new_key = self.graph.add_edge(current_node, interface_name, None, **data)  # Add edge to the graph
            subgraph_edges[":".join(target_path[:i])].append((current_node, interface_name, new_key))
            current_node = interface_name

        # Add the final edge to the target
        new_key = self.graph.add_edge(current_node, target, None, **data)  # Add edge to the graph
        subgraph_edges[":".join(target_path)].append((current_node, target, new_key))

    def print_subgraph_header(self, subgraph_name: str):
        """
        Prints the header for a subgraph in PiSDF format.

        :param subgraph_name: The name of the subgraph.
        """
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
        <key attr.name="parameters" for="graph" id="parameters"/>
        <key attr.name="variables" for="graph" id="variables"/>
        <key attr.name="arguments" for="node" id="arguments"/>
        <key attr.name="name" attr.type="string" for="graph"/>
        <key attr.name="graph_desc" attr.type="string" for="node"/>
        <graph edgedefault="directed">
            <data key="name">{subgraph_name}</data>
        """

    def print_subgraph_footer(self):
        """
        Prints the footer for a subgraph in PiSDF format.
        """
        return """
            </graph>
        </graphml>"""

    def print_actor_node(self, node: str, data: dict):
        """
        Prints a node in PiSDF format.

        :param node: The name of the node.
        :param data: A dictionary containing the node's attributes.
        """
        return f"""<node id="{data.get('name').replace('.', '_').replace(',', '_')}" kind="actor">
            </node>"""

    def print_hierarchical_actor_node(self, node: str, data: dict):
        return f"""<node id="{data.get('name').replace('.', '_').replace(',', '_')}" kind="actor">
            <data key="graph_desc">Algo/{data.get('hierarchical_subgraph', '').replace(':', '_')}.pi</data>
            </node>"""

    def print_interface_node(self, node: str, data: dict):
        """
        Prints an interface node in PiSDF format.

        :param node: The name of the node.
        :param data: A dictionary containing the node's attributes.
        """
        return f"""<node id="{data.get('name').replace('.', '_').replace(',', '_')}" kind="{"src" if data.get('direction') == 'in' else "snk"}">
            </node>"""

    def export_to_files(self, folder_path: str):
        """
        Exports the directed graph to a file in PiSDF format.

        :param folder_path: The path to the folder where the PiSDF representation will be saved.
        """
        for subgraph_name, nodes in self.collect_subgraph_nodes().items():
            subgraph_path = Path(folder_path, f"{subgraph_name.replace(':', '_')}.pi")

            # Collect the XML content in a string
            xml_content = []
            xml_content.append(self.print_subgraph_header(subgraph_name))
            for node in nodes:
                data = self.graph.nodes[node]
                if data.get('type') == 'hierarchical':
                    xml_content.append(self.print_hierarchical_actor_node(node, data))
                elif data.get('type') == 'interface':
                    # Interface nodes are printed as interface nodes
                    xml_content.append(self.print_interface_node(node, data))
                else:
                    xml_content.append(self.print_actor_node(node, data))
            xml_content.append(self.print_subgraph_footer())

            # Join the XML content and remove unnecessary whitespace
            raw_xml = "\n".join(xml_content)
            raw_xml = "".join(line.strip() for line in raw_xml.splitlines())  # Remove all newlines and indentation

            # Parse and format the XML using lxml.etree
            try:
                root = etree.fromstring(raw_xml.encode("utf-8"))
                formatted_xml = etree.tostring(root, pretty_print=True, encoding="unicode")
            except etree.XMLSyntaxError as e:
                raise ValueError(f"Error parsing XML: {e}")

            # Write the formatted XML to the file
            with open(subgraph_path, 'w', encoding='utf-8') as file:
                file.write(formatted_xml)


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
    pisdf_exporter.export_to_files(nf_report_path.with_name("Algo"))
