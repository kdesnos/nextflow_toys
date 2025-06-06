from math import gcd, lcm
from pathlib import Path
import networkx as nx
import pandas
import export_dag_to_dot
import extract_trace_from_html
import import_dag_from_mermaid_html
from lxml import etree


class PreesmExporter:
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

        # Fill the actor_name attributes of the nodes so that each node has a unique name within its subgraph
        self.name_actors(subgraph_nodes)

        # Give a name to all input and output "ports" of edges
        self.name_ports(subgraph_nodes, subgraph_edges)

        # Set an SDF production and consumption rate for each edge
        # according to the number of executions of the process
        self.set_sdf_rates(subgraph_nodes, subgraph_edges)

    def set_sdf_rates(self, subgraph_nodes: dict, subgraph_edges: dict) -> None:
        """
        Sets the SDF production and consumption rates for each edge according to the number of executions of the process.
        This method modifies the graph in place.

        :param subgraph_nodes: A dictionary where keys are subgraph names and values are lists of nodes.
        :param subgraph_edges: A dictionary where keys are subgraph names and values are lists of edges.
        """
        # Dictionnary to hold the computed number of executions of hierarchical actors
        hierarchical_actor_nb_exec = {}

        # Iterate over the subgraphs, in a bottom-up manner
        subgraph_names = list(subgraph_nodes.keys())
        subgraph_names.sort(key=lambda x: len(x.split(':')), reverse=True)
        for subgraph_name in subgraph_names:
            nodes = subgraph_nodes[subgraph_name]

            # Find the common divisor of the number of executions of the processes in the subgraph
            # exclude interface nodes as they do not have a number of executions
            all_nb_exec = [self.graph.nodes[node].get('nb_exec') for node in nodes if self.graph.nodes[node].get('type') != 'interface']
            # Exclude -1 values to avoid division by zero. They correspond to channel operators whose number of executions is not defined
            # and will be inferred from the number of executions of the processes they connect.
            all_nb_exec = [nb_exec for nb_exec in all_nb_exec if nb_exec > 0]
            # Calculate the greatest common divisor (GCD) of the number of executions
            subgraph_gcd = gcd(*all_nb_exec)
            hierarchical_actor_nb_exec[subgraph_name] = subgraph_gcd

            # If the subgraph GCD is greater than 1, we need to normalize the number of execution of nodes
            if subgraph_gcd > 1:
                # Normalize the number of executions of each node in the subgraph
                for node in nodes:
                    nb_exec = self.graph.nodes[node].get('nb_exec')
                    if self.graph.nodes[node].get('type') != 'interface' and nb_exec > 0:
                        self.graph.nodes[node]['nb_exec'] = nb_exec / subgraph_gcd

            # For now, print an error if a channel operator is found with a number of executions not defined
            if any(self.graph.nodes[node].get('nb_exec') < 0 for node in nodes):
                # TODO: Infer the number of execution of channel operators.
                # Hint: Find the connected subgraphs linking one or more channel operators to
                # other nodes with a defined number of executions. For each identified
                # connected subgraph, set the number of executions of the channel operator
                # to the maximum number of executions of the connected nodes.
                raise ValueError(f"Channel operators in subgraph {subgraph_name} have a number of executions not defined. "
                                 "This is not supported yet. See todo in the code for more details.")

            # Function to get the number of executions of a node, considering hierarchical actors
            def get_nb_exec(node):
                if self.graph.nodes[node].get('type') == 'hierarchical':
                    subgraph = self.graph.nodes[node].get('hierarchical_subgraph')
                    return hierarchical_actor_nb_exec[subgraph]
                return self.graph.nodes[node].get('nb_exec')

            # Iterate over the edges in the subgraph and set the SDF production and consumption rates.
            for source, target, key in subgraph_edges[subgraph_name]:
                source_nb_exec = get_nb_exec(source)
                target_nb_exec = get_nb_exec(target)

                target_nb_exec = self.graph.nodes[target]['nb_exec']

                # Set the rates of each node to the LCM of the two execution couts / number of executions of the process
                edge_lcm = lcm(source_nb_exec, target_nb_exec)
                self.graph.edges[source, target, key]['sdf_prod'] = edge_lcm / source_nb_exec
                self.graph.edges[source, target, key]['sdf_cons'] = edge_lcm / target_nb_exec

        # Interface nodes need special post-processing to make sure their inner and external rates are equal.
        # Iterate as long as needed to ensure all interface nodes are processed correctly.
        has_done_change = True
        while has_done_change:
            has_done_change = False

            # Use bottom-up approach to ensure that all hierarchical actors are processed
            # and their internal rates are set correctly before processing the interface nodes.
            for subgraph_name in subgraph_names:
                for source, target, key in subgraph_edges[subgraph_name]:
                    data = self.graph.edges[source, target, key]

                    if self.graph.nodes[source]['type'] == 'hierarchical':
                        external_prod = data['sdf_prod']
                        interface = data['interface_target']
                        internal_edge = list(self.graph.in_edges(interface, data=True, keys=True))[
                            0]  # There should be only one incoming edge to the interface node
                        internal_cons = internal_edge[3]['sdf_cons']
                        if (external_prod != internal_cons):
                            has_done_change = True
                            # Set the rate to the LCM of the external and internal rates
                            lcm_rate = lcm(int(external_prod), int(internal_cons))
                            data['sdf_prod'] = lcm_rate
                            self.graph.edges[internal_edge[0], internal_edge[1], internal_edge[2]]['sdf_cons'] = lcm_rate
                            # Update other rates of the edges to match the new production and consumption rates
                            data['sdf_cons'] *= (lcm_rate / external_prod)
                            self.graph.edges[internal_edge[0], internal_edge[1], internal_edge[2]]['sdf_prod'] *= (lcm_rate / internal_cons)

                    if self.graph.nodes[target]['type'] == 'hierarchical':
                        external_cons = data['sdf_cons']
                        interface = data['interface_source']
                        internal_edge = list(self.graph.out_edges(interface, data=True, keys=True))[
                            0]  # There should be only one outgoing edge from the interface node
                        internal_prod = internal_edge[3]['sdf_prod']
                        if (external_cons != internal_prod):
                            has_done_change = True
                            # Set the rate to the LCM of the external and internal rates
                            lcm_rate = lcm(int(external_cons), int(internal_prod))
                            data['sdf_cons'] = lcm_rate
                            self.graph.edges[internal_edge[0], internal_edge[1], internal_edge[2]]['sdf_prod'] = lcm_rate
                            # Update other rates of the edges to match the new production and consumption rates
                            data['sdf_prod'] *= (lcm_rate / external_cons)
                            self.graph.edges[internal_edge[0], internal_edge[1], internal_edge[2]]['sdf_cons'] *= (lcm_rate / internal_prod)

    def name_actors(self, subgraph_nodes: dict) -> None:
        """
        Names the actors in the subgraphs based on their attributes.
        This method modifies the graph in place.

        :param subgraph_nodes: A dictionary where keys are subgraph names and values are lists of nodes.
        """
        for _, nodes in subgraph_nodes.items():
            node_names = {}
            for node in nodes:
                # skip interface nodes, they will be named when naming the ports of hierarchical actors
                if self.graph.nodes[node].get('type') == 'interface':
                    continue
                # Get the name of the node, replacing dots and commas with underscores for valid names
                name = self.graph.nodes[node].get('name').replace('.', '_').replace(',', '_')
                if not name in node_names:
                    node_names[name] = []
                node_names[name].append(node)

            for name, nodes in node_names.items():
                if (len(nodes) > 1):
                    # If there are multiple nodes with the same name, we need to rename them
                    for i, node in enumerate(nodes):
                        new_name = f"{name}_{i:02d}"
                        self.graph.nodes[node]['actor_name'] = new_name
                else:
                    # If there is only one node with the name, we can use the name directly
                    self.graph.nodes[nodes[0]]['actor_name'] = name

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
                    edge_name = data.get('label') if data.get('label') not in [None, "", "-"] else 'unnamed'
                    if edge_name not in edge_names:
                        edge_names[edge_name] = []
                    edge_names[edge_name].append(edge)

                # Name the ports based on the edge names, with unique identifiers
                for name, edges in edge_names.items():
                    port_name = name.replace(",", "_")  # Replace commas with underscores for valid port names
                    if len(edges) > 1:
                        # If there are multiple edges, we need to create unique names for the ports
                        for i, edge in enumerate(edges):
                            source, target, key, data = edge
                            port_name_idx = f"{port_name}_{i:02d}"
                            self.graph.edges[source, target, key]['snk_port' if edge in in_edges else 'src_port'] = port_name_idx
                    else:
                        # If there is only one edge, we can use the edge name directly
                        source, target, key, data = edges[0]
                        self.graph.edges[source, target, key]['snk_port' if edges[0] in in_edges else 'src_port'] = port_name

                    for edge in edges:
                        source, target, key, data = edge
                        # If the edge is linked to an hierarchical actor, update the name of the interface_source or interface_target
                        if edge in in_edges and data.get('interface_source'):
                            interface_data = self.graph.nodes[data.get('interface_source')]
                            snk_port_name = self.graph.edges[source, target, key]['snk_port']
                            interface_data['name'] = snk_port_name
                            interface_data['actor_name'] = snk_port_name
                        if edge in out_edges and data.get('interface_target'):
                            interface_data = self.graph.nodes[data.get('interface_target')]
                            src_port_name = self.graph.edges[source, target, key]['src_port']
                            interface_data['name'] = src_port_name
                            interface_data['actor_name'] = src_port_name

        # Find all edges connected to an interface node, and rename theirs port to be the same as the interface node name
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            if self.graph.nodes[source].get('type') == 'interface':
                # If the source is an interface node, rename the target port to the interface node name
                self.graph.edges[source, target, key]['src_port'] = self.graph.nodes[source]['name']
            if self.graph.nodes[target].get('type') == 'interface':
                # If the target is an interface node, rename the source port to the interface node name
                self.graph.edges[source, target, key]['snk_port'] = self.graph.nodes[target]['name']

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
        return f"""<node id="{data.get('actor_name')}" kind="actor">
                {self.print_node_ports(node)}
            </node>"""

    def print_hierarchical_actor_node(self, node: str, data: dict):
        """
        Prints a hierarchical actor node in PiSDF format.

        :param node: The name of the node.
        :param data: A dictionary containing the node's attributes.
        """
        return f"""<node id="{data.get('actor_name')}" kind="actor">
                <data key="graph_desc">Algo/{data.get('hierarchical_subgraph', '').replace(':', '_')}.pi</data>
                {self.print_node_ports(node)}
            </node>"""

    def print_node_ports(self, node):
        result = []
        for source, target, key, data in self.graph.in_edges(node, data=True, keys=True):
            result.append(f'<port annotation="NONE" expr="{data['sdf_cons']:.0f}" kind="input" name="{data['snk_port']}"/>')
        for source, target, key, data in self.graph.out_edges(node, data=True, keys=True):
            result.append(f'<port annotation="NONE" expr="{data['sdf_prod']:.0f}" kind="output" name="{data['src_port']}"/>')

        return "\n".join(result)

    def print_interface_node(self, node: str, data: dict):
        """
        Prints an interface node in PiSDF format.

        :param node: The name of the node.
        :param data: A dictionary containing the node's attributes.
        """
        return f"""<node id="{data.get('actor_name')}" kind="{"src" if data.get('direction') == 'in' else "snk"}">
                {self.print_node_ports(node)}
            </node>"""

    def print_edge(self, source: str, target: str, key: int, data: dict):
        source_name = self.graph.nodes[source]["actor_name"]
        target_name = self.graph.nodes[target]["actor_name"]
        src_port = data.get('src_port', 'u')
        snk_port = data.get('snk_port', 'u')
        return f"""<edge kind="fifo" source="{source_name}" sourceport="{src_port}" target="{target_name}" targetport="{snk_port}" type="uchar"/>"""

    def export_to_files(self, folder_path: str):
        """
        Exports the directed graph to a file in PiSDF format.

        :param folder_path: The path to the folder where the PiSDF representation will be saved.
        """
        subgraph_nodes = self.collect_subgraph_nodes()
        subgraph_edges = self.collect_subgraph_edges(subgraph_nodes)
        for subgraph_name, nodes in subgraph_nodes.items():
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

            # Add edges to the XML content
            for source, target, key in subgraph_edges[subgraph_name]:
                data = self.graph.get_edge_data(source, target, key)
                xml_content.append(self.print_edge(source, target, key, data))

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

    DEFAULT_NB_CORES = 4

    def print_archi_header(self):
        """
        Prints the header for the architecture file in SLAM format.
        """
        return """<?xml version="1.0" encoding="UTF-8"?>
            <spirit:design xmlns:spirit="http://www.spiritconsortium.org/XMLSchema/SPIRIT/1.4">
            <spirit:vendor>ietr</spirit:vendor>
            <spirit:library>preesm</spirit:library>
            <spirit:name>nf_archi</spirit:name>
            <spirit:version>1</spirit:version>"""

    def print_archi_footer(self):
        """
        Prints the footer for the architecture file in SLAM format.
        """
        return """</spirit:design>"""

    def print_archi_nodes(self, nb_cores: dict) -> str:
        """
        Prints the nodes in the architecture file in SLAM format.

        :param nb_cores: A dictionary mapping node types to the number of cores.
        :return: A string containing the XML representation of the nodes.
        """
        xml_content = []
        #  header for component instances
        xml_content.append("""<spirit:componentInstances>""")

        total_id = 0  # Initialize a total ID counter for unique hardware IDs
        # Nodes
        for node_type, cores in nb_cores.items():
            for i in range(cores):
                xml_content.append(f"""
                <spirit:componentInstance>
                    <spirit:instanceName>node_{node_type}_{i}</spirit:instanceName>
                    <spirit:hardwareId>{total_id}</spirit:hardwareId>
                    <spirit:componentRef spirit:library="" spirit:name="node_{node_type}" spirit:vendor="" spirit:version=""/>
                    <spirit:configurableElementValues/>
                </spirit:componentInstance>
                """)
                total_id += 1  # Increment the total ID counter for the next hardware ID

        # Print Generic communication node
        xml_content.append("""<spirit:componentInstance>
            <spirit:instanceName>shared_mem</spirit:instanceName>
            <spirit:hardwareId>0</spirit:hardwareId>
            <spirit:componentRef spirit:library="" spirit:name="SHARED_MEM" spirit:vendor="" spirit:version=""/>
            <spirit:configurableElementValues/>
        </spirit:componentInstance>""")

        # footer for component instances
        xml_content.append("""</spirit:componentInstances>""")

        # Vendor extensions for node types
        xml_content.append("""<spirit:vendorExtensions>
            <slam:componentDescriptions xmlns:slam="http://sourceforge.net/projects/dftools/slam">""")

        for node_type, _ in nb_cores.items():
            xml_content.append(f"""<slam:componentDescription slam:componentRef="node_{node_type}" slam:componentType="CPU" slam:refinement=""/>""")

        xml_content.append("""<slam:componentDescription slam:componentRef="SHARED_MEM" slam:componentType="parallelComNode" slam:refinement="" slam:speed="1000000000"/>
                </slam:componentDescriptions>
                <slam:designDescription xmlns:slam="http://sourceforge.net/projects/dftools/slam">
                    <slam:parameters/>
                </slam:designDescription>
            </spirit:vendorExtensions>""")

        return "\n".join(xml_content)

    def print_archi_links(self, nb_cores: dict) -> str:
        xml_content = []

        # Header for communication bus
        xml_content.append("""<spirit:interconnections>""")

        total_id = 0  # Initialize a total ID counter for unique bus IDs
        for node_type, cores in nb_cores.items():
            for i in range(cores):
                xml_content.append(f"""
                <spirit:interconnection>
                    <spirit:name>{total_id}</spirit:name>
                    <spirit:activeInterface spirit:busRef="shared_mem" spirit:componentRef="shared_mem"/>
                    <spirit:activeInterface spirit:busRef="shared_mem" spirit:componentRef="node_{node_type}_{i}"/>
                </spirit:interconnection>
                """)
                total_id += 1  # Increment the total ID counter for the next bus ID

        xml_content.append(f"""</spirit:interconnections>
            <spirit:vendorExtensions>
                <slam:linkDescriptions xmlns:slam="http://sourceforge.net/projects/dftools/slam">""")

        for i in range(total_id):
            xml_content.append(f"""<slam:linkDescription slam:directedLink="undirected" slam:linkType="DataLink" slam:referenceId="{i}"/>""")

        # Footer
        xml_content.append("""</slam:linkDescriptions>
            </spirit:vendorExtensions>""")

        return "\n".join(xml_content)

    def export_archi_to_files(self, folder_path: str, trace_df: pandas.DataFrame, nb_cores: dict | int = DEFAULT_NB_CORES) -> None:
        """
        Exports the architecture description to a file in SLAM format.

        :param folder_path: The path to the folder where the architecture representation will be saved.
        :param trace_df: The trace data as a pandas DataFrame.
        :param nb_cores: The number of cores generated in the archi, or a dictionary mapping node types to core counts.
        """
        # Find the different type of HW nodes (for now, number of cores of the job) used in the trace data
        node_types = trace_df['cpus'].unique()

        # Prepare the number of cores for each node type
        nb_cores_to_print = {}
        if isinstance(nb_cores, int):
            # If nb_cores is an integer, use it for all node types
            nb_cores_to_print = {node_type: nb_cores for node_type in node_types}
        elif isinstance(nb_cores, dict):
            # If nb_cores is a dictionary, ensure it contains all node types
            for node_type in node_types:
                if node_type in nb_cores:
                    nb_cores_to_print[node_type] = nb_cores[node_type]
                else:
                    nb_cores_to_print[node_type] = self.DEFAULT_NB_CORES

        # Collect the XML content in a list
        xml_content = []
        xml_content.append(self.print_archi_header())

        # Add component instances here if needed
        xml_content.append(self.print_archi_nodes(nb_cores_to_print))

        # Add communication bus
        xml_content.append(self.print_archi_links(nb_cores_to_print))

        # Print footer for the architecture file
        xml_content.append(self.print_archi_footer())

        # Join the XML content and remove unnecessary whitespace
        raw_xml = "\n".join(xml_content)
        raw_xml = "".join(line.strip() for line in raw_xml.splitlines())  # Remove all newlines and indentation

        # Parse and format the XML using lxml.etree
        archi_path = Path(folder_path, "archi.slam")
        try:
            root = etree.fromstring(raw_xml.encode("utf-8"))
            formatted_xml = etree.tostring(root, pretty_print=True, encoding="unicode")
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Error parsing architecture XML: {e}")

        # Write the formatted XML to the file
        with open(archi_path, 'w', encoding='utf-8') as file:
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
    import_dag_from_mermaid_html.add_execution_counts_to_graph(dag, trace_df)

    # Export the DAG to PiSDF format and to a DOT file
    pisdf_exporter = PreesmExporter(dag)
    pisdf_exporter.transform_graph()
    export_dag_to_dot.export_to_dot(pisdf_exporter.graph, nf_report_path.with_name(nf_report_path.name + "_dag.dot"))
    pisdf_exporter.export_to_files(nf_report_path.with_name("Algo"))
    pisdf_exporter.export_archi_to_files(nf_report_path.with_name("Archi"), trace_df, {32: 4, 16: 8, 1: 16, 4: 8})
