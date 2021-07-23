# k-armed-bandits

A modified version of the k-armed testbed in which all the q*(a) start out equal and then take
independent random walks (by adding a normally distributed increment with mean
zero and standard deviation 0.01 to all the q*(a) on each step). 
Try different action-value methods by modifying the dictionary "methods", 
providing the name, stepsize and the epsilon of each method 
and get Average performance of each action-value method on the k-armed testbed.
The data are averages over RUNS number of runs with different bandit problems.
