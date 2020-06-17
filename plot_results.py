import numpy as np
import matplotlib.pyplot as plt

files = [
    "case1_data.csv.results_sw1.csv",
    "case2_data.csv.results_sw1.csv",
    "case1_data.csv.results_sw0.csv",
    "case2_data.csv.results_sw0.csv",
    "case1_data.csv.results_sw2.csv",
    "case2_data.csv.results_sw2.csv"
]

labels = [
    "network 1 resilient switching",
    "network 2 resilient switching",
    "network 1 resilient weights",
    "network 2 resilient weights",
    "network 1 no resilience",
    "network 2 no resilience"
]

colors = [ 'blue', 'orange', 'green', 'red', 'purple', 'brown'  ]


N = 100

for i,filename in zip(range(len(files)), files):

    f = open(filename, 'r')
    if f.mode != 'r':
        raise Exception("Cant read data")

    n1 = int(f.readline())
    n2 = int(f.readline())
    assert(n1 == N and n2 == N)

    means = np.array(list(map(float, f.readline().split(',')[:-1])))
    variances = np.array(list(map(float, f.readline().split(',')[:-1])))

    f.close()

    x = np.linspace(1, N, N)

    plt.plot(x, means, color=colors[i])
    plt.fill_between(x, means - variances / 2, means + variances / 2, facecolor=colors[i], alpha=0.2)
    # plt.errorbar(np.linspace(1, N, N), means, variances, alpha=0.25)

# plt.legend(labels, loc=1)
# plt.ylim([0, 25])
plt.xlim([0, 100])
plt.show()
