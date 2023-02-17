import matplotlib.pyplot as plt

labels = ['Общее время', 'Время выполнения циклов']
parameters = [('CPU + float'), ('CPU + double'), ('GPU + float'), ('GPU + double')]
times = [(236, 40), (263, 50), (24, 0.164), (25, 0.220)]

for time in times:
    fig1, ax1 = plt.subplots()
    ax1.pie(time, labels=labels, autopct='%1.2f%%')
    ax1.axis('equal')
    plt.show()
