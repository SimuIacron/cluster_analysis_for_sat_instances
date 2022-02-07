import os

import matplotlib.pyplot as plt
import numpy as np

# 100 linearly spaced numbers
x = np.linspace(0, 5000, 5000)

# the function, which is y = x^2 here
y1 = x
offset = 3
grade = 1/10
y2 = (x + offset) * (1 + grade)

# setting the axes at the centre
fig = plt.figure()

# plot the function
plt.plot(x, y1, 'b', label='SBS')
plt.plot(x, y2, 'r', label='Upper bound determined by SBS')

plt.legend()

# show the plot
# plt.show()
plt.savefig(os.environ['TEXPATH'] + '/strip/strip.svg')

