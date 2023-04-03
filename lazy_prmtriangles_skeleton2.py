#!/usr/bin/env python3
#
#   prmtriangles.py
#
#   Use PRM to find a path around triangular obstacles.
#
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import bisect
import math

from math          import pi, sin, cos, sqrt, ceil
from scipy.spatial import KDTree
from planarutils   import *


######################################################################
#
#   Parameters
#
#   Define the N/K...
#
N = 250
K = 5

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

triangles = ((( 2, 6), ( 3, 2), ( 4, 6)),
             (( 6, 5), ( 7, 7), ( 8, 5)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))

(startx, starty) = ( 1, 5)
(goalx,  goaly)  = (13, 5)


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


#
#   Lazy PRM Planning Algorithm
#
def lazyPRM(nodes, start, goal):
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

        # Also make sure we still have something to look at!
        if not (len(onDeck) > 0):
            return []

    # Build the path.
    path = [goal]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Return the path.
    return path



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

        # Show.
        self.show()

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

# Post Process the Path
def PostProcess(path):
    # TODO: Remove nodes in the path than can be skipped without collisions
    idx = 1
    while idx < len(path) - 1:
        if path[idx - 1].connectsTo(path[idx + 1]):
            path = path[:idx] + path[idx + 1:]
            idx = 1
        else:
            idx += 1
    return path

def whole_lazy_prm(start_coords, goal_coords, visual, show_plot):
    global connectsTo_used
    connectsTo_used = 0
    global inFreespace_used
    inFreespace_used = 0

    nodes = createNodes(N)

    if show_plot:
        for node in nodes:
            visual.drawNode(node, color='k', marker='x')
    
    # Add the start/goal nodes.
    startnode = LazyPRMNode(start_coords[0], start_coords[1])
    goalnode = LazyPRMNode(goal_coords[0], goal_coords[1])
    nodes.append(startnode)
    nodes.append(goalnode)

    connectNearestNeighbors(nodes, K)

    if show_plot:
        # Show the neighbor connections.  Yes, each edge is drawn twice...
        for node in nodes:
            for neighbor in node.neighbors:
                visual.drawEdge(node, neighbor, color='g', linewidth=0.5)
    
    path = lazyPRM(nodes, startnode, goalnode)

    # If unable to connect, show the part explored.
    if not path:
        if show_plot:
            for node in nodes:
                if node.done:
                    visual.drawNode(node, color='b', marker='o')
            visual.show("Failed to find path. Showing DONE nodes")
    
    return (path, N, connectsTo_used, inFreespace_used)



######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with', N, 'nodes and', K, 'neighbors.')

    # Create the figure.
    visual = Visualization()

    # Create the start/goal nodes.
    startnode = LazyPRMNode(startx, starty)
    goalnode  = LazyPRMNode(goalx,  goaly)

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='r', marker='o')
    visual.drawNode(goalnode,  color='r', marker='o')
    visual.show("Showing basic world")


    # Create the list of nodes.
    print("Sampling the nodes...")
    nodes = createNodes(N)

    # Show the sample nodes.
    for node in nodes:
        visual.drawNode(node, color='k', marker='x')
    visual.show("Showing the nodes")

    # Add the start/goal nodes.
    nodes.append(startnode)
    nodes.append(goalnode)


    # Connect to the nearest neighbors.
    print("Connecting the nodes...")
    connectNearestNeighbors(nodes, K)

    # Show the neighbor connections.  Yes, each edge is drawn twice...
    for node in nodes:
        for neighbor in node.neighbors:
            visual.drawEdge(node, neighbor, color='g', linewidth=0.5)
    visual.show("Showing the full graph")


    # Run the A* planner.
    print("Running Lazy PRM...")
    path = lazyPRM(nodes, startnode, goalnode)

    # If unable to connect, show the part explored.
    if not path:
        print("UNABLE TO FIND A PATH")
        for node in nodes:
            if node.done:
                visual.drawNode(node, color='b', marker='o')
        visual.show("Showing DONE nodes")
        return

    # Show the path.
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path")


    # Post Process the path.
    path = PostProcess(path)

    # Show the post-processed path.
    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing the post-processed path")


if __name__== "__main__":
    main()
