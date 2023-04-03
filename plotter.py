import matplotlib.pyplot as plt
import numpy as np
import pickle

# ALGOS = ['RRT', 'EST', 'Semi-lazy PRM', 'Fully-lazy PRM', 'Lazy RRT', 'Lazy EST']

def gen_bar_graph(dict_vals):
    courses = list(dict_vals.keys())
    values = np.array(list(dict_vals.values()))[:, METRIC_TO_PLOT]
    
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(courses, values, color ='green', width = 0.4)
    
    plt.yscale('log')
    plt.xlabel("Algorithm")
    plt.ylabel("connectsTo calls (log scale)")
    plt.title("ConnectsTo calls by Algorithm for static concave shape environment")
    plt.show()

def gen_double_bar_graph(dict_vals, dict_vals2):
    algorithms = list(dict_vals.keys())
    values1 = np.array(list(dict_vals.values()))[:, METRIC_TO_PLOT]
    values2 = np.array(list(dict_vals2.values()))[:, METRIC_TO_PLOT]
    # values1 = list(i[0] for i in dict_vals.values())
    # values2 = list(i[1] for i in dict_vals.values())

    fig = plt.figure(figsize = (10, 5))
    X_axis = np.arange(len(algorithms))
    
    plt.bar(X_axis - 0.2, values1, 0.4, label = 'Environment 1')
    plt.bar(X_axis + 0.2, values2, 0.4, label = 'Environment 1')
    plt.xticks(X_axis, algorithms)

    plt.yscale('log')
    plt.xlabel("Algorithm")
    plt.ylabel("connectsTo calls (log scale)")
    plt.title("ConnectsTo calls for PRMs in closing door environment")
    plt.show()

def gen_line_graph(dict_vals):
    iterations = [i for i in range(1, len(dict_vals['RRT']) + 1)]
    
    fig = plt.figure(figsize = (10, 5))

    for key in dict_vals:
        plt.plot(iterations, dict_vals[key], label = key)

    plt.yscale('log')
    plt.xlabel("Number of obstacle changes in Dynamic Environment")
    plt.ylabel("connectsTo calls (log scale)")
    plt.title("ConnectsTo calls versus number of Dynamic Environment moves for various PRMs")
    plt.legend()
    plt.show()

# Set the code to plot connectsTo calls
METRIC_TO_PLOT = 1
METRIC_TO_PLOT2 = 2
mode = 'double_bar'

if mode == 'bar':
    with open(f'133b_data_static2.pickle', 'rb') as f:
        dict_vals = pickle.load(f)
    with open(f'133b_data_static3.pickle', 'rb') as f:
        dict_vals2 = pickle.load(f)
    dict_vals.update(dict_vals2)
    gen_bar_graph(dict_vals)
    f.close()
elif mode == 'double_bar':
    with open(f'133b_data0.pickle', 'rb') as f:
        dict_vals = pickle.load(f)
    with open(f'133b_data1.pickle', 'rb') as f:
        dict_vals2 = pickle.load(f)
    gen_double_bar_graph(dict_vals, dict_vals2)
    f.close()
elif mode == 'line':
    final_dict = {'RRT':[], 'EST':[], 'Semi-lazy PRM':[], 'Fully-lazy PRM':[], 'Lazy RRT':[], 'Lazy EST':[]}
    for i in range(6):
        with open(f'133b_data{i}.pickle', 'rb') as f:
            dict_vals = pickle.load(f)
            for key in dict_vals:
                final_dict[key].append(dict_vals[key][METRIC_TO_PLOT])
        f.close()
    gen_line_graph(final_dict)
