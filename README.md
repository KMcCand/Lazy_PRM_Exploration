# Lazy_PRM_Exploration
Probabilistic Roadmaps sample points in a space, use them to create a graph, and return a path through the graph consisting of completely valid points and edges that have no collisions with obstacles. Traditional roadmaps check the validity of nodes and edges during the graph construction, ensuring that any node or edge in the graph is valid. However, checking the validity of a point or an edge is computationally intensive, and in dynamic environments where nodes and edges are not guaranteed to stay valid forever, a lazy approach to validity checking can save computation time. This project evaluates the performance of 4 algorithms that delay validity checking until the path-planning (Semi-lazy PRM) or path-processing (Fully-lazy PRM, Lazy RRT, Lazy EST) phases, finding that lazy evaluation can have significant benefits in dynamic environments.

See the [project report](https://drive.google.com/file/d/1X05W6xjaKobLmBNoxW9U5DoaC0lNGMZp/view?usp=sharing) for algorithm overviews and specific results.

## Usage
1. Configure prm_algos_tester_final.py parameters:
    * ALL_MAPS - maps to test algorithms on, one map per query
    * SHOW_PLOT - set true to see the outcome of each test, false to run tests for numerical results
    * DO_ALGO - true if executing the corresponding algorithm, false otherwise
    * Algorithm parameters specified at the top (keep consistent for fair testing)
2. Run ```python3 prm_algos_tester_final.py```
3. Configure mode, METRIC_TO_PLOT and file names, and run ```python3 plotter.py```

## Results
1. Overall, Fully-lazy PRM adapts well to dynamic environments and consistently has the fewest
validity checks. With the current results, Fully-lazy PRM would be our go to in real dynamic environments.
2. In an environment with a large, impeding shape, Semi-lazy PRM narrows down A* options
quickly and may outperform Fully-lazy PRM.
3. While they have high numbers of validity checks in a static environment, Lazy RRT and Lazy EST learn dynamic environments best, improving the most per query with increasing numbers of queries in realistic, slowly changing environments.

See [project report](https://drive.google.com/file/d/1X05W6xjaKobLmBNoxW9U5DoaC0lNGMZp/view?usp=sharing) and [data visualization video](https://drive.google.com/file/d/1foXZOEMhVgtpQhKP5o8E5yAD6b-jcua0/view?usp=sharing).

