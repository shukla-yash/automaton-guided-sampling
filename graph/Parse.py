class Parse:
    def parse_dfa_string_list(dfa_string_list):
        '''
            Input: list of dfa string
            Output: list of strings of starting node to ending node with label 
        '''
        dfa_string_list = dfa_string_list.split('\n')
        init_index = -1
        # find index of " init -> "
        str_check = " init -> "
        for ind,val in enumerate(dfa_string_list):
            if str_check in val:
                initial_node = val.split()[2].split(';')[0]
                init_index = ind
                break
        # next index has the starting node 
        start = init_index + 1
        # we don't want the '}'
        end = len(dfa_string_list) - 1
        new_list = dfa_string_list[start:end]

        return initial_node, new_list

    def parse_dfa_nodes_list(dfa_nodes_list):
        '''
            Input: list of strings of all the starting, ending and label of all nodes in the graph 
            Parses each line to get list of starting node, ending node and label 
            Output: list of [starting_node, ending_node, edge_label]
        
        '''
        return_list = []
        for val in dfa_nodes_list:
            nodes_list = Parse.parse_line(val)
            return_list.append(nodes_list)
        
        return return_list


    def parse_line(line):
        '''
            Input: string of starting node, ending node and label 
                Example: ' 1 -> 1 [label="~a"];'
            Output: list of the starting node, ending node and the label
                Example: ['1', '1', '~a']
        
        '''
        split_line = line.split()
        starting_node = split_line[0]
        ending_node = split_line[2]
        unstripped_label = line.split("=")[1].split("\"")
        label = unstripped_label[1]

        return [starting_node, ending_node, label]

    



