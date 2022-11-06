import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# matplotlib.rcParams["mathtext.fontset"] = "cm"
# matplotlib.rcParams["font.family"] = "STIXGeneral"
# plt.rcParams.update({
# "text.usetex": True,
# "font.family": "sans-serif",
# "font.sans-serif": ["Helvetica"],
# })
# matplotlib.pyplot.title(r"ABC123 vs $\mathrm{ABC123}^{123}$")
# plt.style.use('science')
task_dict = {
    "Place in slider": 7.96,
    "Open drawer": 7.58,
    "Move slider right": 6.34,
    "Move slider left": 5.68,
    "Close drawer": 5.00,
    "Stack block": 4.40,
    "Place in drawer": 4.16,
    "Lift blue block table": 4.10,
    "Turn on lightbulb": 4.04,
    "Lift red block table": 3.98,
    "Turn on led": 3.90,
    "Lift pink block table": 3.74,
    "Turn off led": 3.70,
    "Turn off lightbulb": 3.42,
    "Lift red block slider": 3.14,
    "Lift pink block slider": 3.12,
    "Lift blue block slider": 3.02,
    "Push into drawer": 2.76,
    "Unstack block": 1.62,
    "Push red block left": 1.58,
    "Rotate blue block right": 1.56,
    "Push pink block left": 1.54,
    "Rotate red block right": 1.50,
    "Push blue block right": 1.44,
    "Rotate pink block right": 1.44,
    "Push red block right": 1.44,
    "Push blue block left": 1.40,
    "Push pink block right": 1.36,
    "Rotate blue block left": 1.36,
    "Rotate red block left": 1.32,
    "Rotate pink block left": 1.14,
    "Lift blue block drawer": 0.46,
    "Lift red block drawer": 0.46,
    "Lift pink block drawer": 0.34,
}

task_classes = {
    "Rotate red block right": 1,
    "Rotate red block left": 1,
    "Rotate blue block right": 1,
    "Rotate blue block left": 1,
    "Rotate pink block right": 1,
    "Rotate pink block left": 1,
    "Push red block right": 2,
    "Push red block left": 2,
    "Push blue block right": 2,
    "Push blue block left": 2,
    "Push pink block right": 2,
    "Push pink block left": 2,
    "Move slider left": 3,
    "Move slider right": 3,
    "Open drawer": 4,
    "Close drawer": 4,
    "Lift red block table": 5,
    "Lift blue block table": 5,
    "Lift pink block table": 5,
    "Lift red block slider": 5,
    "Lift blue block slider": 5,
    "Lift pink block slider": 5,
    "Lift red block drawer": 5,
    "Lift blue block drawer": 5,
    "Lift pink block drawer": 5,
    "Place in slider": 6,
    "Place in drawer": 6,
    "Turn on lightbulb": 7,
    "Turn off lightbulb": 7,
    "Turn on led": 8,
    "Turn off led": 8,
    "Push into drawer": 6,
    "Stack block": 9,
    "Unstack block": 9,
}

task_legend = {
    "Rotate blocks": 1,
    "Push blocks": 2,
    "Move slider": 3,
    "Open/Close drawer": 4,
    "Lift blocks": 5,
    "Place in slider/drawer": 6,
    "Turn lightbulb on/off": 7,
    "Turn led on/off": 8,
    "Stack/Unstack blocks": 9,
}
plt.rcParams["font.size"] = "18"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
fig, ax = plt.subplots(figsize=(16, 10))
y_pos = np.arange(len(task_dict.keys()))
unique_categories = list(set(task_classes.values()))
palette = sns.color_palette("muted", len(unique_categories))
color_dict = {key: palette[unique_categories.index(value)] for key, value in task_classes.items()}
color_list = [color_dict[key] for key, value in task_dict.items()]
ax.barh(y_pos, width=task_dict.values(), color=color_list)
plt.ylim([-1, len(task_dict.keys())])
plt.xlim([0, max(task_dict.values()) + 0.1])
ax.set_yticks(y_pos, labels=task_dict.keys())
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel("Probability")
ax.set_title("Task distribution across LH-MTLC")


color_dict = {key: palette[unique_categories.index(value)] for key, value in task_legend.items()}
labels = list(task_legend.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[label]) for label in labels]
plt.legend(handles, labels)
# plt.legend(bbox_to_anchor=(1.01, 4.6), loc=2, borderaxespad=0.0, prop={"size": 18})
# plt.show()
fig.savefig("/tmp/eval_task_distribution.pdf", dpi=400, bbox_inches="tight")
