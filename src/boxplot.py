from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(df["MedHouseVal"], vert=True, widths=0.5)
ax.set_title("MedHouseVal")
ax.set_xticks([])
plt.suptitle("Boxplot for California Housing Dataset", fontsize=12)
plt.tight_layout()

plt.savefig("figs/boxplot.png")
plt.show()