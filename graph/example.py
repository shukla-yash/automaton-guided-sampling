from Parse import *
from Graph import *
from ltlf2dfa.parser.ltlf import LTLfParser

def main():
    parser = LTLfParser()
    formula_str = "G(a -> X b)"

    # returns an LTLfFormula
    formula = parser(formula_str)       
    print(formula) 

    dfa = formula.to_dfa()
    print(dfa)

    # list of nodes from dfa string 
    initial_node, node_list = Parse.parse_dfa_string_list(dfa)

    # list of [starting node, ending node, edge label]
    graph_list = Parse.parse_dfa_nodes_list(node_list)
    print("Initial node is ")
    print(initial_node)
    print(graph_list)

    graph_instance = Graph(graph_list, initial_node)
    print("Graph dict")
    print(graph_instance.returnGraph())

    print("Edge dict ")
    print(graph_instance.returnEdgeGraph())

    print("Vertices are ")
    vertices = graph_instance.getVertices()
    print(vertices)

    for key in vertices:
        connecting_nodes = graph_instance.getConnectingNodes(key)
        print("Key is " + key + " Connecting nodes are ")
        print(connecting_nodes)

        for n in connecting_nodes:
            print("Edge between node " + key + " and " + n + " is " + graph_instance.getEdge(key, n))
    
    print(graph_instance.removeCycle())
    print("Pruned edge graph is ")
    print(graph_instance.pruned)
    visited = [False] * 3
    print(graph_instance.dfs('2', '3', visited, 3))
        
# make sure right directed acyclic graph is outputted
# check if u need the init node 
# update the sets 
# first get the path from starting node to ending node 
    
if __name__ == "__main__":
    main()


