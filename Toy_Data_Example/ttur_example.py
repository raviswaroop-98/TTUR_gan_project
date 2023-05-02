import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = plt.axes(projection='3d')

# Make data.
x_lim = [-3.0, 3.0]
y_lim = [i * 10.0 for i in x_lim]
x = np.arange(x_lim[0], x_lim[1], 0.25)
y = np.arange(y_lim[0], y_lim[1], 0.25 * 10.0)
X, Y = np.meshgrid(x, y)
Z = (1 + X**2) * (100 - Y**2)

# Plot the surface.
cmap=[cm.inferno, cm.magma]
surf = ax.plot_surface(X, Y, Z, cmap=cmap[1],linewidth=0, antialiased=False)

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))



def run(lr_x, lr_y, n_iter, sigma):
    x = 0.5 #.509124898
    y = -0.4 #-.402918798624
    lrx_hist = []
    lry_hist = []
    x_hist = [x]
    y_hist = [y]
    obj_list = []
    norm_list = []
    for i in range(n_iter):
        x -= lr_x * (2*x*(100-y*y) + np.random.normal(scale=sigma))
        y += lr_y * (-2*y*(1+x*x) + np.random.normal(scale=sigma))

        norm = np.sqrt(x ** 2 + y ** 2)
        obj = (1+x*x)*(100-y*y)
        obj_list.append(obj)
        norm_list.append(norm)
        x_hist.append(x)
        y_hist.append(y)
    print (x, y, obj, norm)
    return x_hist, y_hist, obj_list, norm_list

n_iter = 5000
sigma = 1
base_lr = 0.01
res_otur_1 = run(base_lr, base_lr, n_iter, sigma)
res_otur_2 = run(base_lr/10, base_lr/10, n_iter, sigma)
res_ttur_1 = run(base_lr / 100, base_lr, n_iter, sigma)
res_ttur_2 = run(base_lr, base_lr / 100, n_iter, sigma)

res = [res_otur_1, res_ttur_1, res_ttur_2]

fig2, ax2 = plt.subplots(3, 3, figsize=(15, 6))
for i, r in enumerate(res):
    ax2[i, 0].plot(r[2])
    ax2[i, 1].plot(r[3])
    ax2[i, 2].plot(r[0], r[1], '->', alpha=0.1)
    # ax2[i, 2].scatter(r[0], r[1], s=5.0, alpha=1.0, c=np.arange(0, len(r[0]), 1), cmap=cm.viridis)

ax2[0, 0].set_title("objective")
ax2[0, 1].set_title("norms")
ax2[0, 2].set_title("x vs y")
ax2[0, 0].set_ylabel("orig fast")
# ax2[1, 0].set_ylabel("orig slow")
ax2[1, 0].set_ylabel("TTUR x")
ax2[2, 0].set_ylabel("TTUR y")

plt.show()