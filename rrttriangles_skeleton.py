#!/usr/bin/env python3
#
#   rrttriangles.py
#
#   Use RRT to find a path around triangular obstacles.
#
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math          import pi, sin, cos, sqrt, ceil
from planarutils   import *


######################################################################
#
#   Parameters
#
#   Define the step size.  Also set a maximum number of nodes...
#
dstep = 1
Nmax  = 1000


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
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        for triangle in triangles:
            if PointInTriangle((self.x, self.y), triangle):
                return False
        return True

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        for triangle in triangles:
            if SegmentCrossTriangle(((self.x, self.y), (other.x, other.y)),
                                    triangle):
                return False
        return True


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
            visual.show()

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
        if nearnode.connectsTo(nextnode) and nextnode.x >= xmin and nextnode.x <= xmax and nextnode.y >= ymin and nextnode.y <= ymax:
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


# Post Process the Path
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1


######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with step size ', dstep, ' and up to ', Nmax, ' nodes.')

    # Create the figure.
    visual = Visualization()

    # Create the start/goal nodes.
    startnode = Node(startx, starty)
    goalnode  = Node(goalx,  goaly)

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='r', marker='o')
    visual.drawNode(goalnode,  color='r', marker='o')
    visual.show("Showing basic world")


    # Run the RRT planner.
    print("Running RRT...")
    (path, N) = rrt(startnode, goalnode, visual, True)

    # If unable to connect, show what we have.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    print("PATH found after %d nodes" % N)
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path")


    # Post Process the path.
    PostProcess(path)

    # Show the post-processed path.
    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing the post-processed path")


if __name__== "__main__":
    main()
