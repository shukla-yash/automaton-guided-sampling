class Graph:
    def __init__(self, graph_nodes_list, init):
        # init is type string of the starting node of graph 
        # dictionary where key is string of starting and ending node and value is edge label
        self.edges = {}
        # dictionary of starting node as key and value is list of ending nodes
        self.gdict = self.createGraph(graph_nodes_list)
        self.V = len(self.gdict)
        self.init = init
        self.pruned = self.removeCycle()
        # self.active is list of edges (task names) that are currently being learned 
        self.active = self.initializeSets()
        self.learned = []
        self.discarded = []
    
    def createGraph(self, graph_nodes_list):
        '''
            Input: graph_nodes_list is [[starting_node, ending_node, label]]
        '''
        gdict = {}
        for val in graph_nodes_list:
            self.AddNode(val, gdict)
        return gdict
    
    def getVertices(self):
        return list(self.gdict.keys())

    def AddNode(self, nodes_list, gdict):
        '''
            nodes_list is [starting_node, ending_node, edge]
        '''
        starting_node = nodes_list[0]
        ending_node = nodes_list[1]
        edge = nodes_list[2]
        # if the starting node already exists in the graph 
        if starting_node in gdict:
            value = gdict[starting_node]
            value.append(ending_node)
        else:
            gdict[starting_node] = list(ending_node)
        
        edge_key = starting_node + ending_node
        self.edges[edge_key] = edge
    
    def getConnectingNodes(self, node):
        '''
            Returns list of nodes that node is connected to 
        '''
        return self.gdict[node]

    def getEdge(self, node_one, node_two):
        '''
            Returns Edge between node_one and node_two
        '''
        node_to_find = node_one + node_two 
        return self.edges[node_to_find]

    def returnGraph(self):
        return self.gdict

    def returnEdgeGraph(self):
        return self.edges
    
    def returnPrunedGraph(self):
        return self.pruned

    def isCyclicUtil(self, v, visited, recStack, pruned):
        # Mark current node as visited and
        # adds to recursion stack
        visited[int(v)] = True
        recStack[int(v)] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.gdict[str(v)]:
            if visited[int(neighbour)] == False:
                if self.isCyclicUtil(neighbour, visited, recStack, pruned) == True:
                    return True
                else:
                    pruned[str(v)].append(str(neighbour))

    # https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
    def removeCycle(self):
        pruned = {}
        visited = [False] * (self.V + 1)
        recStack = [False] * (self.V + 1)
        for node in range(1, self.V+1):
            pruned[str(node)] = []
        for node in range(1, self.V+1):
            if visited[node] == False:
                self.isCyclicUtil(node,visited,recStack, pruned)
        return pruned 
    
    def initializeSets(self):
        """
            Adds all edges from init node to active tasks list
        """
        active_tasks = []
        for edge in self.pruned[self.init]:
            neighbor_edge = self.init + edge
            task_name = self.edges[neighbor_edge]
            active_tasks.append(task_name)
        return active_tasks
    
    def checkForPath(self, start_node, end_node):
        """
            Checks if there is a path from the start_node to the end_node 
            Returns edge names in path
        """
    # https://www.geeksforgeeks.org/find-if-there-is-a-path-between-two-vertices-in-a-given-graph/
    def dfs(self, start, end, visited, V):
        """
        start: str
        end: str
        visited: List[bool]
        V: str (number of vertices in the graph)
        output: bool
        Returns whether or not there is a path from starting node to ending node 
        """
        if start == end:
            return True
        visited[int(start)-1] = True
        for x in self.pruned[start]:
            print(x)
            if not visited[int(x)-1]:
                if self.dfs(x, end, visited, V):
                    return True
        return False

    def updateSets(self, learned):
        """
            Updates the active, learned and discarded sets given that learned is ['edge_name'] of tasks that have been learned 
        """
        # add the learned task to the learned set and remove it from the active set 
        for task in learned:
            self.learned.append(task)
            self.active.remove(task)

        

    
  


