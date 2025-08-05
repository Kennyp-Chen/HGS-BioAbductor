import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import logging



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
        