import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# if __name__ == "__main__":
#     csv_path = input("CSV Relative Path: ") 
#     data_runs = int(input("Number of Data Runs: "))
#     graph_title = input("Graph Title: ") 

#     row_names = []
#     results = []

#     col_sums = [0.0] * data_runs
#     with open(csv_path,'r') as csvfile:
#         plots = csv.reader(csvfile, delimiter=',')
#         for row in plots:
#             row_names.append(row[0])

#             temp_row = []
#             for idx in range(data_runs):
#                 col_value = float(row[idx+1]) if row[idx+1] != '' else 0.0
#                 temp_row.append(col_value)

#                 col_sums[idx] += col_value

#             results.append(temp_row)


#     # Normalize results
#     for idx, arr in enumerate(results):
#         for col in range(len(arr)):
#             results[idx][col] /= col_sums[col]
    
#     ind = np.arange(data_runs)
#     width = 0.6

#     y_heights = [0.0] * data_runs
#     bar_plots = []
#     for idx, arr in enumerate(results):
#         bar_plots.append(plt.bar(ind, arr, width, bottom=y_heights))
#         y_heights = [sum(x) for x in zip(y_heights, arr)]

#     plt.ylabel('Percent of Total CPU Time')
#     plt.title('CPU Time Breakdown Per Run')
#     plt.xticks(ind, labels=['Jaes Results'])
#     plt.yticks(np.arange(0, 1.01, .1))

#     plt.subplots_adjust(right=0.7)
#     plt.legend([x[0] for x in bar_plots[::-1]], row_names[::-1], bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
#     fig1 = plt.gcf()
#     fig1.savefig('Test.png')

#     plt.show()

if __name__ == "__main__":
    row_names = []
    start_times = []
    stop_times = []

    with open('test.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            row_names.append(row[0])
            start_times.append(float(row[1]))
            stop_times.append(float(row[2]))

    first_start_time = start_times[0]
    for time in start_times:
        if time < first_start_time:
            first_start_time = time

    last_stop_time = stop_times[0]
    for time in stop_times:
        if time > last_stop_time:
            last_stop_time = time
    
    y_pos = np.arange(len(row_names))
    plt.barh(y_pos, [x[1]-x[0] for x in zip(start_times, stop_times)], align='center', left=start_times)
    plt.yticks(y_pos, row_names)
    # plt.set_labels()
    
    plt.title('Jae Sponza Full Runtime Results')
    plt.xticks(np.arange(first_start_time-50, last_stop_time+1, 50))
    # plt.subplots_adjust(left=0.25)

    plt.show()