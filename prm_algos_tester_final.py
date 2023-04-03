#!/usr/bin/env python3
#
#   rrttriangles.py
#
#   Use RRT to find a path around triangular obstacles.
#
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import random
import bisect
import time
import math
import pickle

from math          import pi, sin, cos, atan2, sqrt, ceil
from scipy.spatial import KDTree
from planarutils   import *


######################################################################
#
#   Parameters
#
#   Define the step size.  Also set a maximum number of nodes...
#
# RRT, EST parameters
dstep = 1
Nmax  = 120

# DLazyPRM parameters
N_dlazy = 80
K_dlazy = 7

# GLazyPRM parameters
N_glazy = 80
K_glazy = 7

# Global counters
connectsTo_used = 0
inFreespace_used = 0

######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)

(startx, starty) = ( 2, 3)
(goalx,  goaly)  = (12, 3)
(goalx2, goaly2) = (13, 8)

# Original RRT map
def triangles_original():
    return ((( 2, 6), ( 3, 2), ( 4, 6)),
             (( 6, 5), ( 7, 7), ( 8, 5)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))

# Original RRT map with big triangle
def triangles_with_big():
    return ((( 2, 6), ( 3, 2), ( 4, 6)),
             (( 5, 2), ( 9, 1), ( 7, 7)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))

# Original RRT map with one triangle removed
def triangles_with_one_removed():
    return ((( 2, 6), ( 3, 2), ( 4, 6)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))

# Move triangle down
def triangles_move_1():
    return ((( 2, 8), ( 3, 5), ( 4, 8)),
             (( 6, 9), ( 8, 9), ( 8, 6)),
             ((10, 3), (11, 6), (12, 3)))

def triangles_move_2():
    return ((( 2, 7), ( 3, 4), ( 4, 7)),
             (( 6, 9), ( 8, 9), ( 8, 6)),
             ((10, 3), (11, 6), (12, 3)))

def triangles_move_3():
    return ((( 2, 6), ( 3, 3), ( 4, 6)),
             (( 6, 9), ( 8, 9), ( 8, 6)),
             ((10, 3), (11, 6), (12, 3)))

def triangles_move_4():
    return ((( 2, 5), ( 3, 2), ( 4, 5)),
             (( 6, 9), ( 8, 9), ( 8, 6)),
             ((10, 3), (11, 6), (12, 3)))

def triangles_move_5():
    return ((( 2, 4), ( 3, 1), ( 4, 4)),
             (( 6, 9), ( 8, 9), ( 8, 6)),
             ((10, 3), (11, 6), (12, 3)))

def triangles_concave():
    return ((( 4, 5), ( 2, 8), ( 6, 5)),
             (( 4, 5), ( 1, 2), ( 6, 5)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))

# Original RRT map
def rectangles_original():
    return ((( 7, 4), ( 8, 4), ( 7, 5)),
             (( 8, 4), ( 7, 5), ( 8, 5)),
             (( 8, 0), ( 7, 0), ( 7, 2)),
             (( 8, 0), ( 7, 2), ( 8, 2)),
             (( 4.5, 5), ( 4.5, 6), ( 7, 5)),
             (( 4.5, 6), ( 7, 5), ( 7, 6)),
             (( 0, 5), ( 0, 6), ( 2.5, 5)),
             (( 0, 6), ( 2.5, 5), ( 2.5, 6)),
             (( 7, 6), ( 7, 7), ( 8, 6)),
             (( 7, 7), ( 8, 6), ( 8, 7)),
             (( 7, 9), ( 7, 10), ( 8, 9)),
             (( 7, 10), ( 8, 9), ( 8, 10)),
             (( 8, 5), ( 8, 6), ( 10.5, 5)),
             (( 8, 6), ( 10.5, 5), ( 10.5, 6)),
             (( 12.5, 5), ( 12.5, 6), ( 14, 5)),
             (( 12.5, 6), ( 14, 5), ( 14, 6)))

# Map with closed gap in rooms
def rectangles_close_gap():
    return ((( 8, 0), ( 7, 0), ( 7, 5)),
             (( 8, 0), ( 7, 5), ( 8, 5)),
             (( 4.5, 5), ( 4.5, 6), ( 7, 5)),
             (( 4.5, 6), ( 7, 5), ( 7, 6)),
             (( 0, 5), ( 0, 6), ( 2.5, 5)),
             (( 0, 6), ( 2.5, 5), ( 2.5, 6)),
             (( 7, 6), ( 7, 7), ( 8, 6)),
             (( 7, 7), ( 8, 6), ( 8, 7)),
             (( 7, 9), ( 7, 10), ( 8, 9)),
             (( 7, 10), ( 8, 9), ( 8, 10)),
             (( 8, 5), ( 8, 6), ( 10.5, 5)),
             (( 8, 6), ( 10.5, 5), ( 10.5, 6)),
             (( 12.5, 5), ( 12.5, 6), ( 14, 5)),
             (( 12.5, 6), ( 14, 5), ( 14, 6)))

ALL_MAPS = [
    ((startx, starty), (goalx, goaly), rectangles_original()),
    ((startx, starty), (goalx, goaly), rectangles_close_gap())
    # ((startx, starty), (goalx, goaly), triangles_with_big()),
    # ((startx, starty), (goalx, goaly), triangles_with_one_removed()),
    # ((startx, starty), (goalx, goaly), triangles_original())
    # ((startx, starty), (goalx, goaly), triangles_move_1()),
    # ((startx, starty), (goalx, goaly), triangles_move_2()),
    # ((startx, starty), (goalx, goaly), triangles_move_3()),
    # ((startx, starty), (goalx, goaly), triangles_move_4()),
    # ((startx, starty), (goalx, goaly), triangles_move_5()),
    # ((startx, starty), (goalx, goaly), triangles_with_one_removed())
    # ((startx, starty), (goalx, goaly), triangles_with_big()),
    # ((startx, starty), (goalx, goaly), triangles_original()),
    # ((startx, starty), (goalx, goaly), triangles_with_one_removed()),
    # ((startx, starty), (goalx, goaly), triangles_with_big()),
    # ((startx, starty), (goalx, goaly), triangles_original()),
    # ((startx, starty), (goalx, goaly), triangles_with_one_removed()),
    # ((startx, starty), (goalx, goaly), triangles_with_big()),
    # ((startx, starty), (goalx, goaly), triangles_original()),
    # ((startx, starty), (goalx, goaly), triangles_with_one_removed()),
    # ((startx, starty), (goalx, goaly), triangles_with_big())
]

# Set the first map we are going to use, so that the visualization can start up correctly
triangles = ALL_MAPS[0][2]



######################################################################
#
#   Node Definition
#
class Node:
    # Initialize with coordinates.
    def __init__(self, x, y):
        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

        # Clear any parent information.
        self.parent = None

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    ######################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self):
        global inFreespace_used
        inFreespace_used += 1

        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        for triangle in triangles:
            if PointInTriangle((self.x, self.y), triangle):
                return False
        return True
        # for rect in triangles:
        #     if PointInBox((self.x, self.y), rect):
        #         return False
        # return True

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        global connectsTo_used
        connectsTo_used += 1

        for triangle in triangles:
            if SegmentCrossTriangle(((self.x, self.y), (other.x, other.y)),
                                    triangle):
                return False
        return True
        # for rect in triangles:
        #     if SegmentCrossBox(((self.x, self.y), (other.x, other.y)), rect):
        #         return False
        # return True

class LazyPRMNode:
    def __init__(self, x, y):
        # Edges = set of neighbors.  You need to fill in.
        self.neighbors = set()

        # Reset the A* search tree information
        self.reset()
        self.x = x
        self.y = y

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return LazyPRMNode(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Compute the relative distance to another node.
    def distance(self, other):
        # TODO: Compute and return the distance.
        return ((self.x - other.x) ** 2 + (self.y - other.y) **2) ** 0.5

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    ###############
    # A* functions:
    # Actual and Estimated costs.
    def costToConnect(self, other):
        return self.distance(other)

    def costToGoEst(self, other):
        return self.distance(other)

    ################
    # PRM functions:
    # Check whether in free space.
    def inFreespace(self):
        global inFreespace_used
        inFreespace_used += 1

        # TODO: Determine with the point is inside any of the triangles
        for triangle in triangles:
            if PointInTriangle(self.coordinates(), triangle):
                return False
        return True
        # for rect in triangles:
        #     if PointInBox(self.coordinates(), rect):
        #         return False
        # return True

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        global connectsTo_used
        connectsTo_used += 1

        # TODO: Determine whether the path (self to other) crosses any triangles
        segment = (self.coordinates(), other.coordinates())
        for triangle in triangles:
            if SegmentCrossTriangle(segment, triangle):
                return False
        return True
        # for rect in triangles:
        #     if SegmentCrossBox(segment, rect):
        #         return False
        # return True


    def reset(self):
        # Clear the status, connection, and costs for the A* search tree.
        #   TRUNK:  done = True
        #   LEAF:   done = False, seen = True
        #   AIR:    done = False, seen = False
        self.done   = False
        self.seen   = False
        self.parent = []
        self.creach = 0
        self.ctogo  = math.inf

    # Define the "less-than" to enable sorting in A*. Use total cost estimate.
    def __lt__(self, other):
        return (self.creach + self.ctogo) < (other.creach + other.ctogo)


######################################################################
#
#   Visualization
#
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the triangles.
        for tr in triangles:
            plt.plot((tr[0][0], tr[1][0], tr[2][0], tr[0][0]),
                     (tr[0][1], tr[1][1], tr[2][1], tr[0][1]),
                     'k-', linewidth=2)

        # # Show.
        # if SHOW_PLOT:
        #     self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)

# Post Process the Path
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1

# Get length of the path
def length(path):
    length = 0
    for i in range(len(path) - 1):
        length += path[i].distance(path[i + 1])
    return length

########################################################################
#
#   DLazy PRM Planning Algorithm
#
def DLazyPRM(nodes, start_coords, goal_coords, visual, show_plot):
    # Add the start/goal nodes.
    start = LazyPRMNode(start_coords[0], start_coords[1])
    goal = LazyPRMNode(goal_coords[0], goal_coords[1])
    nodes.append(start)
    nodes.append(goal)
    connectOneNodeNearestNeighbors(start, nodes, K_dlazy)
    connectOneNodeNearestNeighbors(goal, nodes, K_dlazy)

    nodes_considered = 0

    # Visualize
    if show_plot:
        for node in nodes:
            visual.drawNode(node, color='k', marker='x')
            # Show the neighbor connections.  Yes, each edge is drawn twice...
            for neighbor in node.neighbors:
                visual.drawEdge(node, neighbor, color='g', linewidth=0.5)

    # Clear the A* search tree information.
    for node in nodes:
        node.reset()

    # Prepare the still empty *sorted* on-deck queue.
    onDeck = []

    # Begin with the start node on-deck.
    start.done   = False
    start.seen   = True
    start.parent = None
    start.creach = 0
    start.ctogo  = start.costToGoEst(goal)
    bisect.insort(onDeck, start)

    # Continually expand/build the search tree.
    while True:
        # Grab the next node (first on deck).
        node = onDeck.pop(0)
        nodes_considered += 1

        # Add the neighbors to the on-deck queue (or update)
        for neighbor in node.neighbors:
            # Skip if already done.
            if neighbor.done:
                continue

            # Compute the cost to reach the neighbor via this new path.
            creach = node.creach + node.costToConnect(neighbor)

            # Just add to on-deck if not yet seen (in correct order).
            if not neighbor.seen and node.connectsTo(neighbor) and neighbor.inFreespace():
                neighbor.seen   = True
                neighbor.parent = node
                neighbor.creach = creach
                neighbor.ctogo  = neighbor.costToGoEst(goal)
                bisect.insort(onDeck, neighbor)
                continue

            # Skip if the previous path to reach (cost) was better!
            if neighbor.creach <= creach:
                continue

            # Update the neighbor's connection and resort the on-deck queue.
            neighbor.parent = node
            neighbor.creach = creach
            onDeck.remove(neighbor)
            bisect.insort(onDeck, neighbor)

        # Declare this node done.
        node.done = True

        # Check whether we have processed the goal (now done).
        if (goal.done):
            break

        # If we are out of nodes and haven't found a path, we failed
        if not (len(onDeck) > 0):
            # Draw the nodes we tried
            if show_plot:
                for node in nodes:
                    if node.done:
                        visual.drawNode(node, color='b', marker='o')

            # Return empty path
            return [], nodes_considered

    # Build the path.
    path = [goal]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Return the path.
    return path, nodes_considered



#######################################################################################
#   GLazy PRM Planning Algorithm
#
def GLazyPRM(nodes, start_coords, goal_coords, visual, show_plot):
    # Add the start/goal nodes.
    start = LazyPRMNode(start_coords[0], start_coords[1])
    goal = LazyPRMNode(goal_coords[0], goal_coords[1])
    nodes.append(start)
    nodes.append(goal)
    connectOneNodeNearestNeighbors(start, nodes, K_glazy)
    connectOneNodeNearestNeighbors(goal, nodes, K_glazy)

    # Visualize
    if show_plot:
        for node in nodes:
            visual.drawNode(node, color='k', marker='x')
            # Show the neighbor connections.  Yes, each edge is drawn twice...
            for neighbor in node.neighbors:
                visual.drawEdge(node, neighbor, color='g', linewidth=0.5)


    removed_edges = set()
    removed_nodes = set()

    # Run the A* planner.
    valid_path = False
    trial = 1
    total_nodes_considered = 0
    # prev_path = []
    while not valid_path:
        path, n_considered = GLazyPRM_helper(nodes, start, goal, removed_nodes, removed_edges)
        total_nodes_considered += n_considered

        if show_plot:
            # print(f"Running Lazy PRM attempt {trial}...")
            # print(path == prev_path)
            visual.drawPath(path, color='m', linewidth=2)

        # prev_path = path

        # If unable to connect, show the part explored.
        if not path:
            if show_plot:
                for node in nodes:
                    if node.done:
                        visual.drawNode(node, color='b', marker='o')
            return [], total_nodes_considered

        valid_path = True
        for i in range(len(path)):
            # Invalid edge in the path, have to retry
            if i + 1 < len(path) and not path[i].connectsTo(path[i + 1]):
                valid_path = False
                node = path[i]
                neighbor = path[i + 1]
                removed_edges.add((node, neighbor))
                removed_edges.add((neighbor, node))
                if show_plot:
                    visual.drawEdge(node, neighbor, color='r', linewidth=0.5)

            # Invalid node in the path, have to retry
            if not path[i].inFreespace():
                valid_path = False
                removed_nodes.add(path[i])
                if show_plot:
                    visual.drawNode(path[i], color='r', marker='x')

        trial +=1

    # Return the successful path
    return path, total_nodes_considered


def GLazyPRM_helper(nodes, start, goal, removed_nodes, removed_edges):
    # Clear the A* search tree information.
    for node in nodes:
        node.reset()

    # Prepare the still empty *sorted* on-deck queue.
    onDeck = []

    nodes_considered = 0

    # Begin with the start node on-deck.
    start.done   = False
    start.seen   = True
    start.parent = None
    start.creach = 0
    start.ctogo  = start.costToGoEst(goal)
    bisect.insort(onDeck, start)


    # Continually expand/build the search tree.
    while True:
        # Grab the next node (first on deck).
        node = onDeck.pop(0)
        nodes_considered += 1

        # Add the neighbors to the on-deck queue (or update)
        for neighbor in node.neighbors:
            # Skip if already done.
            if neighbor.done:
                continue

            # Compute the cost to reach the neighbor via this new path.
            creach = node.creach + node.costToConnect(neighbor)

            # Just add to on-deck if not yet seen (in correct order).
            if not neighbor.seen and not (neighbor in removed_nodes or (node, neighbor) in removed_edges):
            # if not neighbor.seen:
                neighbor.seen   = True
                neighbor.parent = node
                neighbor.creach = creach
                neighbor.ctogo  = neighbor.costToGoEst(goal)
                bisect.insort(onDeck, neighbor)
                continue

            # Skip if the previous path to reach (cost) was better!
            if neighbor.creach <= creach or neighbor in removed_nodes or (node, neighbor) in removed_edges:
                continue

            # Update the neighbor's connection and resort the on-deck queue.
            neighbor.parent = node
            neighbor.creach = creach
            onDeck.remove(neighbor)
            bisect.insort(onDeck, neighbor)


        # Declare this node done.
        node.done = True

        # Check whether we have processed the goal (now done).
        if (goal.done):
            break

        # Also make sure we still have something to look at!
        if not (len(onDeck) > 0):
            return [], nodes_considered

    # Build the path.
    path = [goal]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Return the path.
    return path, nodes_considered

######################################################################
#
#   PRM Functions
#
# Create the list of nodes.
def createNodes(N):
    # Add nodes sampled uniformly across the space.
    nodes = []
    while len(nodes) < N:
        node = LazyPRMNode(random.uniform(xmin, xmax),
                    random.uniform(ymin, ymax))
        nodes.append(node)
    return nodes

# Connect the nearest neighbors
def connectNearestNeighbors(nodes, K):
    # Clear any existing neighbors.  Use a set to add below.
    for node in nodes:
        node.neighbors = set()

    # Determine the indices for the K nearest neighbors.  Distance is
    # computed as the Euclidean distance of the coordinates.  This
    # also reports the node itself as the closest neighbor, so add one
    # extra here and ignore the first element below.
    X = np.array([node.coordinates() for node in nodes])
    [dist, idx] = KDTree(X).query(X, k=(K+1))

    # Add the edges.  Ignore the first neighbor (being itself).
    for i, nbrs in enumerate(idx):
        for n in nbrs[1:]:
            nodes[i].neighbors.add(nodes[n])
            nodes[n].neighbors.add(nodes[i])

def connectOneNodeNearestNeighbors(node, nodes, K):
    node.neighbors = set()

    closest_nodes = sorted(nodes, key=lambda x: x.distance(node))
    for i in range(K):
        # Add +1 to ignore first neighbor, which is the node itself
        node.neighbors.add(closest_nodes[i+1])
        closest_nodes[i+1].neighbors.add(node)

def sample_lazy_prm(N, K):
    nodes = createNodes(N)
    connectNearestNeighbors(nodes, K)

    return nodes

######################################################################
#
#   Lazy RRT Functions
#
def lazy_rrt_helper(startnode, goalnode, visual, show_plot):
    percent = 0.05

    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if show_plot:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            # visual.show()

    # Loop - keep growing the tree.
    while True:
        # Determine the target state.
        if random.random() < percent:
            targetnode = goalnode
        else:
            targetnode = Node(random.uniform(xmin, xmax), random.uniform(ymin, ymax))

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index]
        d         = distances[index]

        # Determine the next node.
        x_diff = targetnode.x - nearnode.x
        y_diff = targetnode.y - nearnode.y
        x = dstep * x_diff / d
        y = dstep * y_diff / d
        nextnode = Node(nearnode.x + x, nearnode.y + y)

        # Add node to tree ignoring whether or not there exists a collision
        addtotree(nearnode, nextnode)

        # If within dstep, also try connecting to the goal.  If
        # the connection is made, break the loop to stop growing.
        if nextnode.distance(goalnode) <= dstep:
            addtotree(nextnode, goalnode)
            break

        # Check whether we should abort (tree has gotten too large).
        # if (len(tree) >= Nmax):
        #     return (None, len(tree))

    # Build and return the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
    return (path, len(tree))

def lazy_rrt(path, startnode, goalnode, visual, show_plot):
    N = 0
    if not path:
        (path, N) = lazy_rrt_helper(startnode, goalnode, visual, show_plot)
    new_path = []
    curr_idx = len(path) - 1
    curr_node = path[curr_idx]
    is_removed = False
    while curr_node.x != startnode.x or curr_node.y != startnode.y:
        if not curr_node.inFreespace():
            curr_idx -= 1
        elif not curr_node.connectsTo(path[curr_idx-1]):
            new_goal_idx = curr_idx
            is_removed = True
            curr_idx -= 1
        elif curr_node.inFreespace() and is_removed:
            is_removed = False
            # Run RRT again
            (new_tree, N1) = lazy_rrt_helper(curr_node, path[new_goal_idx], visual, show_plot)
            N += N1
            # Fill in the path with the new tree
            path = path[:curr_idx] + new_tree + path[new_goal_idx+1:]
            curr_idx = len(path) - 1
        else:
            if curr_node not in new_path:
                new_path.append(curr_node)
            curr_idx -= 1
        curr_node = path[curr_idx]
    new_path.append(startnode)

    return new_path, N

######################################################################
#
#   RRT Functions
#
def rrt(startnode, goalnode, visual, show_plot):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if show_plot:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            # visual.show()

    # Loop - keep growing the tree.
    while True:
        # Determine the target state.
        # TODO
        if np.random.uniform() < 0.05:
            targetnode = goalnode
        else:
            targetnode = Node(random.uniform(xmin, xmax),random.uniform(ymin, ymax))

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index]
        d         = distances[index]

        # Determine the next node.
        # TODO
        nextnode = Node(nearnode.x + dstep / d * (targetnode.x - nearnode.x),
                        nearnode.y + dstep / d *  (targetnode.y - nearnode.y))

        # Check whether to attach.
        if nearnode.connectsTo(nextnode) and nextnode.inFreespace() and nextnode.x >= xmin and nextnode.x <= xmax and nextnode.y >= ymin and nextnode.y <= ymax:
            addtotree(nearnode, nextnode)

            # If within dstep, also try connecting to the goal.  If
            # the connection is made, break the loop to stop growing.
            # TODO
            if nextnode.distance(goalnode) <= dstep and nextnode.connectsTo(goalnode):
                addtotree(nextnode, goalnode)
                break

        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            return (None, len(tree))

    # Build and return the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
    return (path, len(tree))


######################################################################
#
#   Lazy EST Functions
#
def lazy_est_helper(startnode, goalnode, visual, show_plot):
    scale = 5

    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if show_plot:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            # visual.show()

    # Loop - keep growing the tree.
    while True:
        # Determine the local density by the number of nodes nearby.
        # KDTree uses the coordinates to compute the Euclidean distance.
        # It returns a NumPy array, same length as nodes in the tree.
        X = np.array([node.coordinates() for node in tree])
        kdtree  = KDTree(X)
        numnear = kdtree.query_ball_point(X, r=1.5*dstep, return_length=True)

        # Directly determine the distances to the goal node.
        distances = np.array([node.distance(goalnode) for node in tree])

        # Select the node from which to grow, which minimizes some metric.
        # lst = numnear
        lst = np.add(numnear, scale * distances)
        idxs = np.array([i for i in range(len(lst)) if lst[i] == min(lst)])
        index = random.choice(idxs)
        grownode = tree[index]

        # Check the incoming heading, potentially to bias the next node.
        if grownode.parent is None:
            heading = 0
        else:
            heading = atan2(grownode.y - grownode.parent.y,
                            grownode.x - grownode.parent.x)

        # Find something nearby: keep looping until the tree grows.
        while True:
            # Pick the next node randomly with bias towards the goal node
            # angle = random.uniform(-pi, pi)
            angle = np.random.normal(heading, pi/2)
            nextnode = Node(grownode.x + cos(angle) * dstep, grownode.y + sin(angle) * dstep)

            # Add node to tree ignoring whether or not there exists a collision
            addtotree(grownode, nextnode)
            break

        if nextnode.distance(goalnode) <= dstep:
            addtotree(nextnode, goalnode)
            break

        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            return (None, len(tree))

    # Build and return the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
    return (path, len(tree))


def lazy_est(path, startnode, goalnode, visual, show_plot):
    N = 0
    if not path:
        (path, N) = lazy_est_helper(startnode, goalnode, visual, show_plot)
    new_path = []
    curr_idx = len(path) - 1
    curr_node = path[curr_idx]
    is_removed = False
    while curr_node.x != startnode.x or curr_node.y != startnode.y:
        if not curr_node.inFreespace():
            curr_idx -= 1
        elif not curr_node.connectsTo(path[curr_idx-1]):
            new_goal_idx = curr_idx
            is_removed = True
            curr_idx -= 1
        elif curr_node.inFreespace() and is_removed:
            is_removed = False
            # Run lazy EST again
            (new_tree, N1) = lazy_est_helper(curr_node, path[new_goal_idx], visual, show_plot)
            N += N1
            # Fill in the path with the new tree
            path = path[:curr_idx]+ new_tree + path[new_goal_idx+1:]
            curr_idx = len(path) - 1
        else:
            if curr_node not in new_path:
                new_path.append(curr_node)
            curr_idx -= 1
        curr_node = path[curr_idx]
    new_path.append(startnode)
    if new_path is None:
        print('HELP')

    return new_path, N


######################################################################
#
#   EST Functions
#
def est(startnode, goalnode, visual, show_plot):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if show_plot:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            # visual.show()

    # Loop - keep growing the tree.
    break_outer_loop = False
    while not break_outer_loop:
        # Determine the local density by the number of nodes nearby.
        # KDTree uses the coordinates to compute the Euclidean distance.
        # It returns a NumPy array, same length as nodes in the tree.
        X = np.array([node.coordinates() for node in tree])
        kdtree  = KDTree(X)
        numnear = kdtree.query_ball_point(X, r=1.5*dstep, return_length=True)

        # Directly determine the distances to the goal node.
        distances = np.array([node.distance(goalnode) for node in tree])

        # Select the node from which to grow, which minimizes some metric.
        # TODO
        # indices_of_closest = [i for i, v in enumerate(numnear) if v == min(numnear)]

        values = [numnear[i] + 10.0 * distances[i] for i in range(len(numnear))]
        indices_of_best = [i for i, v in enumerate(values) if v == min(values)]
        grownode = tree[random.choice(indices_of_best)]

        # Check the incoming heading, potentially to bias the next node.
        if grownode.parent is None:
            heading = 0
        else:
            heading = atan2(grownode.y - grownode.parent.y,
                            grownode.x - grownode.parent.x)

        # Find something nearby: keep looping until the tree grows.
        while True:
            # Pick the next node randomly.
            # TODO
            # direction = np.random.uniform(-pi, pi)
            direction = np.random.normal(heading, pi / 3)
            nextnode = Node(grownode.x + dstep * cos(direction), grownode.y + dstep * sin(direction))

            # TODO: Try to connect. If connected, try to connect to the goal.
            if grownode.connectsTo(nextnode) and nextnode.inFreespace() and nextnode.x >= xmin and nextnode.x <= xmax and nextnode.y >= ymin and nextnode.y <= ymax:
                addtotree(grownode, nextnode)
                if nextnode.distance(goalnode) <= dstep and nextnode.connectsTo(goalnode):
                    addtotree(nextnode, goalnode)
                    break_outer_loop = True
                break

        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            return (None, len(tree))

    # Build and return the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
    return (path, len(tree))


######################################################################
#
#  Main Code
#
def main():
    SHOW_PLOT = False
    NUM_TRIALS = 100 if not SHOW_PLOT else 1
    ALGORITHMS = ['RRT', 'EST', 'Semi-lazy PRM', 'Fully-lazy PRM', 'Lazy RRT', 'Lazy EST']
    DO_ALGO = [True, True, True, True, False, False]
    # STOP_MAP_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
    STOP_MAP_INDICES = [0, 1]

    for map_try_number, up_to_map in enumerate(STOP_MAP_INDICES):
        print(f"\n\nUP TO MAP NUMBER {up_to_map}")

        all_stats = {}
        for a, algo in enumerate(ALGORITHMS):
            if not DO_ALGO[a]:
                continue
            # Report the parameters.
            print(f'Running {NUM_TRIALS} trials of {algo}...')

            # Initialize statistics array
            # [planning_time, connectsTo_used, inFreeSpace_used, number_of_nodes, before_processing_cost, post_processing_cost]
            stats = np.empty((0, 6))

            for i in range(NUM_TRIALS):
                # Create the figure.
                visual = Visualization()

                # Initialize metrics
                global connectsTo_used
                global inFreespace_used
                connectsTo_used = 0
                inFreespace_used = 0
                planning_time = 0
                nodes_checked = 0
                found_path = True

                # Sample lazy PRMs and update planning time
                if algo == 'Fully-lazy PRM':
                    start = time.time()
                    nodes = sample_lazy_prm(N_glazy, K_glazy)
                    planning_time += (time.time() - start)
                elif algo == 'Semi-lazy PRM':
                    start = time.time()
                    nodes = sample_lazy_prm(N_dlazy, K_dlazy)
                    planning_time += (time.time() - start)

                # Setup the lazy path for lazy_rrt, lazy_est
                if algo == 'Lazy RRT' or algo == 'Lazy EST':
                    path = []

                # Loop over all map configurations
                for m in range(up_to_map + 1):
                    current_map = ALL_MAPS[m]

                    # Set the map used by connectsTo and inFreespace
                    global triangles
                    triangles = current_map[2]

                    # Set the start and goal coordinates
                    startx = current_map[0][0]
                    starty = current_map[0][1]
                    goalx = current_map[1][0]
                    goaly = current_map[1][1]

                    # Reset the figure.
                    visual = Visualization()

                    # Set start/goal nodes.
                    startnode = Node(startx, starty)
                    goalnode = Node(goalx, goaly)

                    if SHOW_PLOT: # Show the start/goal nodes.
                        visual.drawNode(startnode, color='r', marker='o')
                        visual.drawNode(goalnode,  color='r', marker='o')

                    start = time.time()

                    if algo == 'Lazy RRT':
                        path.reverse()
                        (path, N) = lazy_rrt(path, startnode, goalnode, visual, SHOW_PLOT)
                    elif algo == 'Lazy EST':
                        path.reverse()
                        (path, N) = lazy_est(path, startnode, goalnode, visual, SHOW_PLOT)
                    elif algo == 'Fully-lazy PRM':
                        (path, N) = GLazyPRM(nodes, (startx, starty), (goalx, goaly), visual, SHOW_PLOT)
                    elif algo == 'Semi-lazy PRM':
                        (path, N) = DLazyPRM(nodes, (startx, starty), (goalx, goaly), visual, SHOW_PLOT)
                    elif algo == 'RRT':
                        (path, N) = rrt(startnode, goalnode, visual, SHOW_PLOT)
                    elif algo == 'EST':
                        (path, N) = est(startnode, goalnode, visual, SHOW_PLOT)
                    else:
                        print('Unrecognized algo name')

                    planning_time += (time.time() - start)
                    nodes_checked += N

                    # If unable to connect, show what we have.
                    if not path:
                        if SHOW_PLOT:
                            visual.show("UNABLE TO FIND A PATH. Showing current status of computation.")

                        # Do not add statistics for a failed path
                        found_path = False
                        continue

                    # Show the pre-processed path.
                    if SHOW_PLOT:
                        print("PATH found after %d nodes" % N)
                        visual.drawPath(path, color='r', linewidth=2)
                        # visual.show("Showing the raw path")

                    # Post Process the path.
                    pre_length = length(path)
                    PostProcess(path)
                    post_length = length(path)

                    # Show the post-processed path.
                    if SHOW_PLOT:
                        visual.drawPath(path, color='b', linewidth=2)
                        visual.show("Showing the post-processed path")

                    # all_stats[algo].append([planning_time, connectsTo_used, inFreespace_used, nodes_checked, pre_length, post_length])

                # Append statistics for this trial after all maps have been run
                if found_path:
                    stats = np.append(stats, np.array([[planning_time, connectsTo_used, inFreespace_used, nodes_checked, pre_length, post_length]]), axis=0)

            # Compute results for this algo
            num_successes = len(stats)
            averages = np.mean(stats, axis=0).tolist()
            averages.append(num_successes / NUM_TRIALS)
            table = np.vstack((np.array(['Planning Time (secs)', 'connectsTo calls', 'inFreespace calls', '# Nodes Checked', 'Pre-processed Path Cost', 'Post-processed Path Cost', 'Success Rate']), averages))

            # Normalize per query
            for each_index in (0, 1, 2, 3):
                averages[each_index] /= (map_try_number + 1)
            all_stats[algo] = averages

            # Print results
            print()
            print(tabulate(table, headers='firstrow', tablefmt='presto'))
            print()

        # Save results to a file
        # np.save(f'133b_data_{time.strftime("%m-%d_%H:%M:%S", time.localtime())}', np.array(all_stats))
        # np.save(f'133b_data', np.array(all_stats))
        f = open(f"133b_data{map_try_number}.pickle", "wb")
        pickle.dump(all_stats, f)
        f.close()

if __name__== "__main__":
    main()
