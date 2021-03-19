import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

a_hidden_vals = [100, 200, 300]
a_out_vals = [50, 60, 500]

data = pd.DataFrame(np.array([[0.1, 0.2, 0.3], [0.0, 1.0, 0.5], [np.nan, 0.3, 0.8]]))

ylabels = a_hidden_vals
xlabels = a_out_vals

fig, ax = plt.subplots()
im = ax.imshow(data.to_numpy())

# We want to show all ticks...
ax.set_xticks(np.arange(len(a_hidden_vals)))
ax.set_yticks(np.arange(len(a_out_vals)))
# ... and label them with the respective list entries
ax.set_xticklabels(a_hidden_vals)
ax.set_yticklabels(a_out_vals)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(a_hidden_vals)):
    for j in range(len(a_out_vals)):
        text = ax.text(j, i, data.to_numpy()[i, j],
                       ha="center", va="center", color="w")

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

ax.set_title("LABEL HERE")
fig.tight_layout()
plt.savefig('heatmap.png')