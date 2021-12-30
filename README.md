# k-armed-bandits

A modified version of the k-armed testbed in which each q*(a) starts with an initial value and then takes
independent random walks (by adding a normally distributed increment with mean
zero and standard deviation 0.01 to all the q*(a) on each step). 

Create the Armed Bandits with the desirable initial q*(a) values using the ArmedBandits class.

Try different action-value methods by modifying the list "methods" and using it in an AverageStudy instance for an average performance study of the methods or by modifying the list "parametric_methods" and using it in an ParameterStudy instance for a parameter study of the methods.



