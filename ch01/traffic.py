import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def error(f, x, y):
	return sp.sum((f(x)-y)**2) # Seems to be MSE error.

data = sp.genfromtxt('data/web_traffic.tsv', delimiter='\t')
# split the data, training and test set from inflection point

x = data[:,0]
y = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


inflection = 3.5 * 7 * 24 # calc in hours
xa = x[:inflection] #before
ya = y[:inflection] 

xb = x[inflection:] #after
yb = y[inflection:]

# create the model using degree of 1
fa = sp.poly1d(sp.polyfit(xa, ya, 2))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)

# Try it with a training set
plt.scatter(xa,ya)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()

fxa1 = sp.linspace(0, xa[-1], 1000)
plt.plot(fxa1, fa(fxa1), linewidth=3)
print "Error: ", error(fa, xa, ya)
plt.show()

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()

# d = 1, degree of 1 - line
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
# What is returned in fp1 are the best parameters for the straight line
# to model the data
print "Model parameters: {0} ".format(fp1)
print "So the best straight line fit is: f(x) = {0} * x + {1}".format(fp1[0], fp1[1])

f1 = sp.poly1d(fp1)
print "Error:", error(f1, x, y)

fx = sp.linspace(0, x[-1], 1000)
# plt.plot(fx, f1(fx), linewidth=4)


# d = 2, degree of 2
f2p = sp.polyfit(x, y, 2)
print f2p
f2 = sp.poly1d(f2p)
print "Error: ", error(f2, x, y)
print "Model parameters: {0}".format(f2p)

fx2 = sp.linspace(0, x[-1], 1000)
# plt.plot(fx2, f2(fx2), linewidth=4)

reached_max = fsolve(f2 - 100000, 800) / (7 * 24)
print "100k hits/hour expected at week {0}".format(reached_max[0])

plt.legend(["d={0}".format(f1.order), "d={0}".format(f2.order)], loc="upper left")
plt.show()