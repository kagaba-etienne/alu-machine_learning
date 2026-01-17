#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ('Farah', 'Fred', 'Felicia')

plt.bar(people, fruit[0], label='apples', color='r',  width=0.5)
plt.bar(people, fruit[1], bottom=fruit[0], label='bananas', color='yellow', width=0.5)
plt.bar(people, fruit[2], bottom=fruit[0]+ fruit[1], label='oranges', color='orange', width=0.5)
plt.bar(people, fruit[3], bottom=fruit[0]+ fruit[1]+ fruit[2], label='peaches', color='#ffe5b4', width=0.5)
plt.ylim(0, 80)
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
