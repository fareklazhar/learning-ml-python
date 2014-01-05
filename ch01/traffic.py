import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

data = sp.genfromtxt('data/web_traffic.tsv', delimiter='\t')

x = data[:,0]
y = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
# plt.show()

def error(f, x, y):
	return sp.sum((f(x)-y)**2) # Seems to be MSE error.

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
# What is returned in fp1 are the best parameters for the straight line
# to model the data
print "Model parameters: {0} ".format(fp1)
print "So the best straight line fit is: f(x) = {0} * x + {1}".format(fp1[0], fp1[1])

f1 = sp.poly1d(fp1)
print "Error:", error(f1, x, y)

fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), linewidth=4)



f2p = sp.polyfit(x, y, 2)
print f2p
f2 = sp.poly1d(f2p)
print "Error: ", error(f2, x, y)
print "Model parameters: {0}".format(f2p)

fx2 = sp.linspace(0, x[-1], 1000)
plt.plot(fx2, f2(fx2), linewidth=4)

reached_max = fsolve(f2 - 100000, 800) / (7 * 24)
print "100k hits/hour expected at week {0}".format(reached_max[0])

plt.legend(["d={0}".format(f1.order), "d={0}".format(f2.order)], loc="upper left")
plt.show()