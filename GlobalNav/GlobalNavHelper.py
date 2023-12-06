import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import convolve
from scipy.signal import convolve2d


def create_empty_plot(max_val, ax=None):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :param max_val: dimension of the map along the x and y dimensions
    :return: the fig and ax objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7))
    
    ax.clear()
    ax.set_title('D* Plan')
    major_ticks = np.arange(0, max_val+1, 10)
    minor_ticks = np.arange(0, max_val+1, 5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,max_val])
    ax.set_xlim([-1,max_val])
    ax.grid(True)
    
    return ax




def _get_movements_4n():
    """
    Get all possible 4-connectivity movements (up, down, left right).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]




def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]




def Key(current_node,start_node):
    """
    The key is used to prioritize nodes in the priority queue. It gives an idea on which is the next best node to explore
    :param current_node: current node (x,y)
    :param start_node: starting node
    :return: Key of the node. Based on the gScore, the rhs, the heuristic and the key modifier km
    """
    current_node=tuple(current_node)
    Key_s=((min(gScore[current_node], rhs[current_node])) + h(current_node,start_node) + km, min(gScore[current_node], rhs[current_node]))
    Key_s=tuple(Key_s)
    return Key_s

def h(current_node,start_node):
    """
    The heuristic functiton. Here it is the euclidean distance from the start_node to the current_node, ignoring obstacles
    :param current_node: current node (x,y)
    :param start_node: starting node
    :return: Euclidean distance between start node and current node
    """
    current_node = np.array(current_node)
    start_node = np.array(start_node)
    return np.linalg.norm(current_node - start_node, axis=-1)
    
def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately 
                     preceding it on the cheapest path from start to n 
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.append(cameFrom[current]) 
        current=cameFrom[current]
    return total_path

def convolution_map(grid):
    """
    This function convolves the map of the environment with the circle in which lies the robot. 
    This way, it prevent the robot from getting too close to the obstacles
    :param grid: This is the gridded map of the environment
    :return: A convoluted map which has its obstacles made larger
    """
    #Create the mask, its approximatively the circle in which the thymio can lie without tuching the border
    r=13
    mask=np.ones((r,r))
    mask[0,0]=0
    mask[0,1]=0
    mask[1,0]=0
    mask[0,r-1]=0
    mask[1,r-1]=0
    mask[0,r-2]=0
    mask[r-1,0]=0
    mask[r-1,1]=0
    mask[r-2,0]=0
    mask[r-1,r-1]=0
    mask[r-2,r-1]=0
    mask[r-1,r-2]=0

    convolved_grid=convolve2d(grid, mask,mode='same') #make the convolution between the grid and the mask,
    
    limit = 0
    convolved_grid[convolved_grid>limit] = 1
    convolved_grid[convolved_grid<=limit] = 0

    return convolved_grid



def neighborhood(s):
    return [((s[0]-1),(s[1]+1)),((s[0]),(s[1]+1)),((s[0]+1),(s[1]+1)),((s[0]-1),(s[1])),(s),((s[0]+1),(s[1])),((s[0]-1),(s[1]-1)),((s[0]),(s[1]-1)),((s[0]+1),(s[1]-1))]




def D_Star_lite(start, goal, coords, occupancy_grid_actual, occupancy_grid_initial, movement_type="8N", max_val=120):
    """
    D*Lite for 2D occupancy grid. Finds a path from start to goal and can efficiently handle obstacle changes.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal: goal node (x, y)
    :param coords: set of all coordinates of the grid
    :param occupancy_grid: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """
    
    # -----------------------------------------
    # DO NOT EDIT THIS PORTION OF CODE
    # -----------------------------------------
    
    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:
        for coord in point:
            assert coord>=0 and coord<max_val, "start or end goal not contained in the map"
    
    # check if start and goal nodes correspond to free spaces
    if occupancy_grid_initial[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid_initial[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
    
    # get the possible movements corresponding to the selected connectivity
    if movement_type == '4N':
        movements = _get_movements_4n()
    elif movement_type == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')
    
    # --------------------------------------------------------------------------------------------
    # A* Algorithm implementation - feel free to change the structure / use another pseudo-code
    # --------------------------------------------------------------------------------------------
    
    state=0 #1 is the initialization, 2 is the update
    closedSetIter=[] #show the visited nodes during the iteration
    priority=[]

    if np.array_equal(occupancy_grid_actual,occupancy_grid_initial): #initialization
        state=1
        print("initialization")
        previous_start=start


        # The set of visited nodes that no longer need to be expanded.
        global closedSet
        closedSet = []

        # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
        global cameFrom
        cameFrom = dict()

        global km # Km stands for "key modifier". It corrects the heuristic function when recomputing new optimal path
        km=0

        # For node n, gScore[n] is the cost of the cheapest path from start to n knowing the initial states of the research.
        global gScore
        gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
        gScore[goal] = 0

        # Right hand side. Rhs[n] is the theoretical cheapest path from start to n given the current state of the search and the environment changes
        global rhs
        rhs = dict(zip(coords, [np.inf for x in range(len(coords))]))
        rhs[goal] = 0


        # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
        # Initially, only the goal node is known (in contrary to the A* algorithm)
        # This set is also refered to as the "priority queue" or simply the "queue"
        global openSet
        openSet = dict(zip([(goal)],Key(goal,start)))

        # The D*Lite algorithm starts it research from the goal back to the start
        #This is because the goal stays the same whereas the start node changes each time the robot moves. 
        current=goal


    if not np.array_equal(occupancy_grid_actual,occupancy_grid_initial):
        if state==1:
            print("Warning: initialization and update occured in the same run")
        state=2
        
        #km=h(start,previous_start)
        previous_start=start
        print("update")

        modified_nodes=[]
        for i in range(len(coords)):    
            if (occupancy_grid_actual[coords[i]]-occupancy_grid_initial[coords[i]]):
                current=coords[i]
                rhs[current]=np.inf
                openSet[current] = Key(current,start)
                modified_nodes.append(current)
                priority.append(current)


        for (i,j) in modified_nodes:
            for dx, dy, deltacost in movements:
                neighbor = (i+dx, j+dy)
                # if the node is not in the map, skip
                if (neighbor[0] >= occupancy_grid_initial.shape[0]) or (neighbor[1] >= occupancy_grid_initial.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                        continue
                # if the node is occupied, skip
                if (occupancy_grid_initial[neighbor[0], neighbor[1]]): 
                        continue
                if neighbor not in openSet:
                    openSet[neighbor]=Key(neighbor,start)


    print("state = ",state)

    # while there are still elements to investigate
    while openSet != {}:
        #the node in openSet having the lowest Key[] value (it replaces the f function of the A* algorithm)
        current=min(openSet,key=openSet.get)
        if current in priority:
            priority.remove(current)

        openSet.pop(current)

        if gScore[current] >= rhs[current]:
            gScore[current]=rhs[current]
            if state==2:
                closedSetIter.append(current)
            closedSet.append(current)
        else: #gScore[current] < rhs[current]:
            gScore[current]=np.inf
            openSet[current]=Key(current,start)
            for dx, dy, deltacost in movements:
                neighbor = (current[0]+dx, current[1]+dy)
                # if the node is not in the map, skip
                if (neighbor[0] >= occupancy_grid_actual.shape[0]) or (neighbor[1] >= occupancy_grid_actual.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                    continue
                # if the node is occupied or has already been visited, skip
                if (occupancy_grid_initial[neighbor[0], neighbor[1]]): #or (gScore[neighbor] == rhs[neighbor]<1000.0): 
                    continue
                if cameFrom[neighbor]==current:
                    rhs[neighbor] = np.inf
        

        #If the goal is reached, reconstruct and return the obtained path
        if gScore[start]==rhs[start]<1000 and priority==[]:# and current == new_start :
            neighborhood_g=[]
            neighborhood_rhs=[]
            for (i,j) in neighborhood(start):
                neighborhood_g.append(gScore[i,j])
                neighborhood_rhs.append(rhs[i,j])
            if neighborhood_g==neighborhood_rhs:
                del neighborhood_g,neighborhood_rhs
                print("path found !")
                return reconstruct_path(cameFrom, start), closedSet, closedSetIter
            

        

        #for each neighbor of current, put them in openSet compute the rhs value of neighbor: this will help tell which node to chose next
        for dx, dy, deltacost in movements:
            neighbor = (current[0]+dx, current[1]+dy)
            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid_actual.shape[0]) or (neighbor[1] >= occupancy_grid_actual.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            # if the node is occupied or has already been visited, skip
            if (occupancy_grid_initial[neighbor[0], neighbor[1]]): #or (gScore[neighbor] == rhs[neighbor]<1000.0): 
                continue
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_rhs = gScore[current] + deltacost
            if neighbor in cameFrom:
                if occupancy_grid_actual[cameFrom[neighbor]]:
                    rhs[neighbor]=np.inf
            if (occupancy_grid_actual[neighbor]):
                tentative_rhs=np.inf
            if tentative_rhs < rhs[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                rhs[neighbor] = tentative_rhs
            if gScore[neighbor] != rhs[neighbor]:
                openSet[neighbor]=Key(neighbor,start)
        

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet, closedSetIter


def FindGlobalPath(start, goal, global_map, previous_map, ax_astar=None):
    # List of all coordinates in the grid
    max_val = global_map.shape[0]
    x,y = np.mgrid[0:max_val:1, 0:max_val:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    for i in range(max_val):
        global_map[i,0] = 1
        global_map[i,max_val-1] = 1
        global_map[0,i] = 1
        global_map[max_val-1,i] = 1
        previous_map[i,0] = 1
        previous_map[i,max_val-1] = 1
        previous_map[0,i] = 1
        previous_map[max_val-1,i] = 1


    cmap = colors.ListedColormap(['white', 'red'])

    # Convolve the map
    #Make the obstacles larger to compensate thymio size (the thymio lies in a radius of ~8cm = 4 grid boxes) 
    occupancy_grid_conv=convolution_map(global_map)
    previous_grid_conv=convolution_map(previous_map)

    limit = 0
    occupancy_grid_conv[occupancy_grid_conv>limit] = 1
    occupancy_grid_conv[occupancy_grid_conv<=limit] = 0

    limit = 0
    previous_grid_conv[previous_grid_conv>limit] = 1
    previous_grid_conv[previous_grid_conv<=limit] = 0


    # Run the D* algorithm
    path, visitedNodes, visitedNodesIter = D_Star_lite(start, goal, coords, occupancy_grid_conv,previous_grid_conv, movement_type="8N",max_val=max_val)
    if len(path) == 0:
        return []
    path = np.array(path).reshape(-1, 2).transpose()
    visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()
    visitedNodesIter = np.array(visitedNodesIter).reshape(-1, 2).transpose()

    # Displaying the map
    ax_astar = create_empty_plot(max_val, ax_astar)

    # Plot the best path found and the list of visited nodes
    ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange');
    ax_astar.scatter(visitedNodesIter[0], visitedNodesIter[1], marker="o", color = 'green');
    ax_astar.imshow(global_map.transpose(), cmap=cmap)
    ax_astar.plot(path[0], path[1], marker="o", color = 'blue');
    ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200);
    ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200);
    for i in range(len(path[0])-1):
        plt.arrow(path[0,i], path[1,i], path[0,i+1]-path[0,i], path[1,i+1]-path[1,i], width=0.5)

    return path
