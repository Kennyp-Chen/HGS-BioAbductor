import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
import re
import itertools

def get_BINN_Pathways(data,n_levels=4,cox_num=400,
    fn_pathways = "./data/PriorKnow/PNET/PNET_pathways.tsv",
    fn_mapping = "./data/PriorKnow/PNET/ENSG2HSA.csv"):
    data_ = data.iloc[:,:-2].T.reset_index(drop=False).rename(columns={"index": "Protein"})

    pathways = pd.read_csv(fn_pathways, sep="\t")
    mapping = pd.read_csv(fn_mapping, index_col=0).iloc[:,:2]
    mapping.rename(columns={"Gene": "input","Identifier":"translation"}, inplace=True)

    network = BINN_Network(
        input_data=data_,
        pathways=pathways,
        mapping=mapping,
        source_column="parent",
        target_column="child",
    )
    connectivity_matrices = network.get_connectivity_matrices(n_levels = n_levels)
    connectivity_matrices[0] = connectivity_matrices[0].iloc[:cox_num,:]
    connectivity_matrices[0] = connectivity_matrices[0].loc[:, connectivity_matrices[0].sum(axis=0) != 0]

    for i in range(1,len(connectivity_matrices)):
        connectivity_matrices[i] = connectivity_matrices[i].loc[connectivity_matrices[i-1].columns,:]
        connectivity_matrices[i] = connectivity_matrices[i].loc[:,connectivity_matrices[i].sum(axis=0) != 0]
         
    data = data.loc[:,list(connectivity_matrices[0].index)+list(data.columns[-2:])]
    return connectivity_matrices,data

class BINN_Network:
    """
    Coding by BINN
    A class for building and analyzing a directed graph network of biological pathways.

    Args:
        input_data (pandas.DataFrame): A DataFrame containing the input data for the pathways.
        pathways (pandas.DataFrame): A DataFrame containing information on the pathways.
        mapping (pandas.DataFrame or None, optional): A DataFrame containing mapping information.
            If None, then a DataFrame will be constructed from the `input_data` argument.
            Default is None.
        input_data_column (str, optional): The name of the column in `input_data` that contains
            the input data. Default is 'Protein'.
        subset_pathways (bool, optional): Whether to subset the pathways DataFrame to include
            only those pathways that are relevant to the input data. Default is True.

    Attributes:
        mapping (pandas.DataFrame): A DataFrame containing the mapping information.
        pathways (pandas.DataFrame): A DataFrame containing information on the pathways.
        input_data (pandas.DataFrame): A DataFrame containing the input data for the pathways.
        inputs (list): A list of the unique inputs in the mapping DataFrame.
        netx (networkx.DiGraph): A directed graph network of the pathways.

    """

    def __init__(
        self,
        input_data: pd.DataFrame,
        pathways: pd.DataFrame,
        mapping: pd.DataFrame = None,
        input_data_column: str = "Protein",
        subset_pathways: bool = True,
        source_column: str = "source",
        target_column: str = "target",
    ):
        self.input_data_column = input_data_column
        pathways = pathways.rename(
            columns={source_column: "source", target_column: "target"}
        )

        if isinstance(mapping, pd.DataFrame):
            self.mapping = mapping
            self.unaltered_mapping = mapping

        else:
            self.mapping = pd.DataFrame(
                {
                    "input": input_data[input_data_column].values,
                    "translation": input_data[input_data_column].values,
                }
            )
            self.unaltered_mapping = mapping

        if subset_pathways:
            self.mapping = _subset_input(input_data, self.mapping, input_data_column)

            self.pathways = _subset_pathways_on_idx(pathways, self.mapping)

        else:
            self.pathways = pathways

        self.mapping = _get_mapping_to_all_layers(self.pathways, self.mapping)

        self.input_data = input_data

        self.inputs = self.mapping["input"].unique()

        self.netx = self.build_network()

    def build_network(self):
        """
        Constructs a networkx DiGraph from the edges in the 'pathways' attribute of the object, with a root node added to the graph to connect all root nodes together.

        Returns:
            A networkx DiGraph object representing the constructed network.
        """
        if hasattr(self, "netx"):
            return self.netx

        net = nx.from_pandas_edgelist(
            self.pathways, source="target", target="source", create_using=nx.DiGraph()
        )
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = "root"
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def get_layers(self, n_levels, direction="root_to_leaf") -> list:
        """
        Returns a list of dictionaries where each dictionary contains the pathways at a certain level of the completed network and their inputs.

        Args:
            n_levels: The number of levels below the root node to complete the network to.
            direction: The direction of the layers to return. Must be either "root_to_leaf" or "leaf_to_root". Defaults to "root_to_leaf".

        Returns:
            A list of dictionaries, where each dictionary contains pathway names as keys and input lists as values.
        """
        if direction == "root_to_leaf":
            net = _complete_network(self.netx, n_levels=n_levels)
            layers = _get_layers_from_net(net, n_levels)

        terminal_nodes = [n for n, d in net.out_degree() if d == 0]

        mapping_df = self.mapping
        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            inputs = (
                mapping_df[mapping_df["connections"] == pathway_name]["input"]
                .unique()
                .tolist()
            )
            if len(inputs) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = inputs
        layers.append(dict)
        return layers

    def get_connectivity_matrices(self, n_levels, direction="root_to_leaf") -> list:
        """
        Returns a list of connectivity matrices for each layer of the completed network, ordered from leaf nodes to root node.

        Args:
            n_levels: The number of levels below the root node to complete the network to.
            direction: The direction of the layers to return. Must be either "root_to_leaf" or "leaf_to_root". Defaults to "root_to_leaf".

        Returns:
            A list of pandas DataFrames representing the connectivity matrices for each layer of the completed network.
        """
        connectivity_matrices = []
        layers = self.get_layers(n_levels, direction)
        for i, layer in enumerate(layers[::-1]):
            layer_map = _get_map_from_layer(layer)
            if i == 0:
                inputs = list(layer_map.index)
                self.inputs = sorted(inputs)
            filter_df = pd.DataFrame(index=inputs)
            all = filter_df.merge(
                layer_map, right_index=True, left_index=True, how="inner"
            )
            all = all.reindex(sorted(all.columns), axis=1)
            all = all.sort_index()
            inputs = list(layer_map.columns)
            connectivity_matrices.append(all)
        return connectivity_matrices


def _get_mapping_to_all_layers(pathways, mapping):
    graph = nx.from_pandas_edgelist(
        pathways, source="source", target="target", create_using=nx.DiGraph()
    )
    components = {"input": [], "connections": []}
    for translation in mapping["input"]:
        ids = mapping[mapping["input"] == translation]["translation"]
        for id in ids:
            if graph.has_node(id):
                connections = graph.subgraph(
                    nx.single_source_shortest_path(graph, id).keys()
                ).nodes
                for connection in connections:
                    components["input"].append(translation)
                    components["connections"].append(connection)
    components = pd.DataFrame(components)
    components.drop_duplicates(inplace=True)
    return components


def _get_map_from_layer(layer_dict):
    pathways = layer_dict.keys()
    inputs = list(itertools.chain.from_iterable(layer_dict.values()))
    inputs = list(np.unique(inputs))
    df = pd.DataFrame(index=pathways, columns=inputs)
    for k, v in layer_dict.items():
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T


def _add_edges(G, node, n_levels):
    edges = []
    source = node
    for level in range(n_levels):
        target = node + "_copy" + str(level + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def _complete_network(G, n_levels=4):
    nr_copies = 0
    sub_graph = nx.ego_graph(G, "root", radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="root", target=node))
        if distance <= n_levels:
            nr_copies = nr_copies + n_levels - distance
            diff = n_levels - distance + 1
            sub_graph = _add_edges(sub_graph, node, diff)
    return sub_graph


def _get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, "root", radius=distance))
    if distance >= 1.0:
        nodes -= set(nx.ego_graph(net, "root", radius=distance - 1))
    return list(nodes)


def _get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = _get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub("_copy.*", "", n)
            next = net.successors(n)
            dict[n_name] = [re.sub("_copy.*", "", nex) for nex in next]
        layers.append(dict)
    return layers


def _subset_input(
    input_df,
    translation,
    input_data_column,
):
    keys_in_data = input_df[input_data_column].unique()
    translation = translation[translation["input"].isin(keys_in_data)]
    return translation


def _subset_pathways_on_idx(pathways, translation):
    def add_pathways(idx_list, target):
        if len(target) == 0:
            return idx_list
        else:
            idx_list = idx_list + target
            subsetted_pathway = pathways[pathways["source"].isin(target)]
            new_target = list(subsetted_pathway["target"].unique())
            return add_pathways(idx_list, new_target)

    original_target = list(translation["translation"].unique())
    idx_list = []
    idx_list = add_pathways(idx_list, original_target)
    pathways = pathways[pathways["source"].isin(idx_list)]
    return pathways


class ReactomeNet:
    def __init__(self, pathways_relation_data, gene_pathway_data):
        """
        Create the directed graph from pathways relation data.
        pathways_relation_data indicates the relationship between child and parent
        gene_pathway_data indicates the mapping between gene and pathway.
        """
        self.pathways_relation_data = pathways_relation_data
        self.gene_pathway_data = gene_pathway_data
        self.G = nx.DiGraph()
        self.add_pathways()
        self.add_genes()
        self.depths = {node: self._get_depth(node) for node in list(self.G.nodes())}
        self.max_depth_dict = self.node_multi_layers_handler()
        nx.set_node_attributes(self.G, self.max_depth_dict, 'layers')
        self.max_depth = max(self.max_depth_dict.values())
        self.state_dict = {}
        self.added_pathways = {}
        self.removed_pathways = {}
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    def add_pathways(self):
        """
        Add pathways to the graph, including relations (edges) between them.
        """
         
        for parent, child in self.pathways_relation_data:
            # the nodes parent and child will be automatically add to the graph if it does not exist
            self.G.add_edge(parent, child)
        

    def add_genes(self):
        """
        Add genes to the graph, linking them to their respective pathways.
        """
        
        for gene, pathway, orthologous_events in self.gene_pathway_data:
            self.G.add_node(gene, type='gene')
            # add event orthologous for gene's parent identifier
            self.G.add_node(pathway,description=orthologous_events)
            self.G.add_edge(pathway, gene)

    def _get_depth(self, node, current_depth=0):
        """
        Recursively find the depth of the given node, where gene have depth 0.
        """
        if len(list(self.G.predecessors(node))) == 0:  # If the node has no predecessors, it's a root.
            return [current_depth]
        depth_list = []
        for parent in list(self.G.predecessors(node)):
            depth = self._get_depth(parent, current_depth + 1)
            depth_list.extend(depth)
        
        return list(set(depth_list))


    def node_multi_layers_handler(self):
        relabel_mapping = {}
        nodes_layer = {}
        for node in self.depths:
            depth_list = self.depths[node]
            max_depth = max(depth_list)

            if len(depth_list) > 1 and 'type' not in self.G.nodes[node]:
                logging.info(f"node {node} has {len(depth_list)+1} different layers")
                depth_list.remove(max_depth)
                for i in depth_list:
                    clone_name = f'{node}_{i}'
                    if not self.G.has_node(clone_name):
                        self.clone_node(node, clone_name, i)
                if not self.G.has_node(f'{node}_{max_depth}'):
                    relabel_mapping[node] = f'{node}_{max_depth}'
                    nodes_layer[f'{node}_{max_depth}'] = max_depth
            else:
                nodes_layer[node] = max(depth_list)
        nx.relabel_nodes(self.G, relabel_mapping, copy=False)
        return nodes_layer

    def clone_node(self, original_node, new_node, at_layers):
        """
        Clone a node in the graph G.
        """
        # Copy node attributes
        self.G.add_node(new_node, **self.G.nodes[original_node])
        self.G.nodes[new_node]['layers'] = at_layers
        # Copy all outgoing and incoming edges
        for successor in self.G.successors(original_node):
            self.G.add_edge(new_node, successor, **self.G.get_edge_data(original_node, successor))
            
        for predecessor in self.G.predecessors(original_node):
            self.G.add_edge(predecessor, new_node, **self.G.get_edge_data(predecessor, original_node))



    def get_genes(self, node):
        """
        Recursively get all genes in the sub-tree rooted at the given node.
        """
         
        if self.G.nodes[node].get('type') == 'gene':  # If the node is a gene, return it.
            return {node}
        genes = set()
        for child in self.G.successors(node):
            genes.update(self.get_genes(child))
        return genes

    def prune_to_genes(self, genes_to_keep:set):
        """
        Prune the graph to keep only the branches that include the specified genes.
        """
        nodes_to_retain = set()
        all_nodes = set(self.G.nodes)
        for node in self.G:
            if node in genes_to_keep:
                nodes_to_retain |= set([node])
                nodes_to_retain |= set(self.get_all_parent(node))
                 
        nodes_to_remove = all_nodes - nodes_to_retain
        
        # nodes_to_remove = [node for node in self.G if self.G.nodes[node].get('type') != 'gene' and not self.get_genes(node) & genes_to_keep]
        self.G.remove_nodes_from(nodes_to_remove)

    def prune_to_level(self, max_level:int):
        """
        Prune the graph to retain only N levels from the root nodes.

        """
        removed_gene_dict = {}
        
        
        for i in range(1, self.max_depth+1):
            removed_gene_dict.setdefault(i, 0)
            self.removed_pathways.setdefault(i, 0)
            self.added_pathways.setdefault(i, 0)
        nodes_to_prune = set()
        nodes_to_clone = set()
        for node in list(self.G.nodes):
            depth = self.G.nodes.get(node)['layers']            
            if depth >= max_level:
                if self.G.nodes[node].get('type') == 'gene':
                    self.gene_merge_higher_layers(node, max_level - 1)
                    self.G.nodes.get(node)['layers'] = max_level
                else:
                    nodes_to_prune.add(node)
                    self.removed_pathways[depth] += 1
            else:
                if self.G.nodes[node].get('type') == 'gene':
                    pred_has_path = []
                    for pred in list(self.G.predecessors(node)):
                        
                        if self.has_no_pathway(pred):
                            pred_has_path.append(False)
                            # clone nodes to the lower layer
                            pred_layer = self.G.nodes.get(pred)['layers']
                            for layers in range(pred_layer  + 1, max_level):
                                new_pred_name = f'{pred}_{layers}'
                                if not self.G.has_node(new_pred_name):
                                    nodes_to_clone.add((pred, new_pred_name, layers))
                        else:
                            pred_has_path.append(True)

                    if all(pred_has_path):
                        nodes_to_prune.add(node)
                        removed_gene_dict[depth] += 1
                    else:        
                        self.G.nodes[node]['layers'] = max_level
                    pred_has_path = []
                else:
                    if self.has_no_pathway(node):
                        for layers in range(depth + 1, max_level):
                                new_node_name = f'{node}_{layers}'
                                if not self.G.has_node(new_node_name):
                                    nodes_to_clone.add((node, new_node_name, layers))
        for node, new_node, layers in nodes_to_clone:
            self.clone_node(node, new_node, layers)
            original_node_layers =  self.G.nodes.get(node)['layers']
            new_node_layers = self.G.nodes.get(new_node)['layers']
            layer_difference = new_node_layers - original_node_layers
            if layer_difference > 1:
                for i in range(1, layer_difference):
                    self.G.add_edge(f'{node}_{new_node_layers-i}', f'{node}_{new_node_layers}')
                    
            self.G.add_edge(node, new_node)
                    
            self.added_pathways[layers] += 1
                    
        for i in range(1, max_level+1):
            logging.info(f'Removing {self.removed_pathways[i]} pathways from the {i}th layer...')
            logging.info(f'Added {self.added_pathways[i]} pathways from the {i}th layer...')
            logging.info(f'Removing {removed_gene_dict[i]} pathways from the {i}th layer...')
            removed_gene_dict[i] = 0
        self.G.remove_nodes_from(nodes_to_prune)
        
        status = True
        while status:
            terminal_pathway_without_gene = set()
            n = 0
            for nodes in self.get_all_terminal():
                terminal_node = list(nodes.keys())[0]
                if 'type' not in self.G.nodes[terminal_node]:
                    terminal_pathway_without_gene.add(terminal_node)
                    self.removed_pathways[self.G.nodes[terminal_node]['layers']] += 1
                    n += 1
            if len(terminal_pathway_without_gene) == 0:
                status = False
            logging.info(f"Removing {n} pathways with zero out degree")
            self.G.remove_nodes_from(terminal_pathway_without_gene)

    def has_no_pathway(self, nodes):
        """
        return True if sucessors has pathway, else False
        """
        succs = [node for node in self.G.successors(nodes)]
        for succ in succs:
            if 'type' not in self.G.nodes[succ]:
                return False
        return True

        

    def get_network(self) -> nx.digraph:
        """
        Return the underlying NetworkX graph.
        """
        return self.G

    def get_all_root_ids(self) -> list:
        """
        Return all root nodes 
        """
        roots = [node for node, in_degree in self.G.in_degree() if in_degree == 0]
        return roots
    
    def get_all_terminal(self) -> list:
        """
        Return all genes layers
        """
        terminals = [{node:self.G.nodes.get(node)['layers']} for node, out_degree in self.G.out_degree() if out_degree == 0]
        return terminals
    
    def get_direct_parent(self, node)-> list:
        """
        Return direct parent from a node
        """
        return list(self.G.predecessors(node))

    def get_all_parent(self, node) -> list:
        """
        recursively get parents from a node.
        """
        parents = []

        stack = [node]

        while stack:  
            current_node = stack.pop()  
            predecessors = list(self.G.predecessors(current_node))  
            
            for predecessor in predecessors:
                if predecessor not in parents:  
                    parents.append(predecessor)  
                    stack.append(predecessor)  

        return parents
    
    def save_reactome_net(self, path:str):
        """
        save net to graphml
        """
        nx.write_graphml(self.get_network(), path)
    

    def get_nodes_by_attribute(self, attribute, value):
        """
        Get all nodes with a specific attribute value.
        """
        matching_nodes = [node for node, attrs in self.G.nodes(data=True) if attrs.get(attribute) == value]
        return matching_nodes
    
    def gene_merge_higher_layers(self, gene_nodes, layers:int):
        nodes_at_target_depth = self.get_nodes_by_attribute('layers', layers)
        for nodes in nodes_at_target_depth:
            if nx.has_path(self.G, nodes, gene_nodes):
                self.G.add_edge(nodes, gene_nodes)
            


    def incidence_mat(self, h1_depth:int, h2_depth:int):
        """
        return incidence matrix for two incidence layers 
        h1_depth: higher level layer
        h2_depth: lower level layer
        """
        if abs(h1_depth - h2_depth) != 1:
            raise ValueError("To calculate the incidence matrix, the difference of the layers' depth has to be 1")
        
        # 获取h1 layer的所有node
        h1_node = self.get_nodes_by_attribute('layers', h1_depth)
        # 获取h2 layer的所有node
        h2_node = self.get_nodes_by_attribute('layers', h2_depth)
        # 建立incidence matrix
        H = pd.DataFrame(0, index=h1_node, columns=h2_node)
        for vertex in h1_node:
            for edge in h2_node:
                if self.G.has_predecessor(vertex, edge):
                    H.loc[vertex, edge] = 1
        return H

def dataFrame_gen(df:pd.DataFrame, columns:list):
    for i, row in df.iterrows():
        yield tuple(row[col] for col in columns)
        