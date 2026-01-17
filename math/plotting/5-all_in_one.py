#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)


layout = [
  ['A', 'B'],
  ['C', 'D'],
  ['E', 'E']
  ]

fig, axis = plt.subplot_mosaic(layout, figsize=(10, 10))
fig.suptitle("All in One")
# First plot
axis['A'].plot(y0, 'r-')
# Second plot
axis['B'].scatter(x1, y1, c='m', marker='.')
axis['B'].set_xlabel("Height (in)", fontsize="x-small")
axis['B'].set_ylabel("Weight (lbs)", fontsize="x-small")
axis['B'].set_title("Men's Height vs Weight", fontsize="x-small")
# Third plot
axis['C'].plot(x2, y2)
axis['C'].set_yscale('log')
axis['C'].set_xlabel("Time (years)", fontsize="x-small")
axis['C'].set_ylabel("Fraction Remaining", fontsize="x-small")
axis['C'].set_title("Exponential Decay of C-14", fontsize="x-small")
# Fourth plot
axis['D'].plot(x3, y31, 'r--', label="C-14")
axis['D'].plot(x3, y32, 'g', label="Ra-226")
axis['D'].set_xlabel("Time (years)", fontsize="x-small")
axis['D'].set_ylabel("Fraction Remaining", fontsize="x-small")
axis['D'].set_title("Exponential Decay of Radioactive Elements", fontsize="x-small")
axis['D'].legend()
# Fifth plot should span the entire row
axis['E'].hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
axis['E'].set_xticks(np.arange(0, 101, 10))
axis['E'].set_yticks(np.arange(0, 31, 5))
axis['E'].set_xlabel("Grades", fontsize="x-small")
axis['E'].set_ylabel("Number of Students", fontsize="x-small")
axis['E'].set_title("Project A", fontsize="x-small")

plt.tight_layout()
plt.show()

