# Lazy_PRM_Exploration
Exploring the efficiency of lazy Probabilistic Roadmaps for robotic path planning. Lazy PRMs delay the performing of costly path validity checks in order to decrease computation time in dynamic environments.

## Usage
1. Configure prm_algos_tester_final.py parameters:
  * ALL_MAPS - maps to test algorithms on, one map per query
  * Algorithm parameters specified at the top (keep consistent for fair testing)
  * SHOW_PLOT - set true to see the outcome of each test, false to run tests for numerical results
  * DO_ALGO - true if executing the corresponding algorithm, false otherwise
2. Run ```python3 prm_algos_tester_final.py```
3. Configure mode, METRIC_TO_PLOT and file names, and run ```python3 plotter.py```

## Results
See [project report](https://drive.google.com/file/d/1X05W6xjaKobLmBNoxW9U5DoaC0lNGMZp/view?usp=sharing) or [video](https://drive.google.com/file/d/1foXZOEMhVgtpQhKP5o8E5yAD6b-jcua0/view?usp=sharing).

