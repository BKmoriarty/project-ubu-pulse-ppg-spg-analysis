import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize a line object
line, = ax.plot([], [], lw=2)

# Set the axis limits
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1, 1)

# Initialize the data
xdata, ydata = [], []

# Define the initialization function
def init():
    line.set_data([], [])
    return line,

# Define the update function
def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    line.set_data(xdata, ydata)
    return line,

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                              init_func=init, blit=True)

# Display the plot
plt.show()