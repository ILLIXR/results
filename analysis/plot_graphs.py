import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_fps(trial_num):
    data = pd.read_csv('../output/fps.csv')
    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 2, 2]})

    def create_row(data, ax_row, trials):
        width = 0.35  # the width of the bars

        labels = ['Camera','OpenVINS Camera']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][0:2]
        platformer_fps = data[trials[1]][0:2]
        materials_fps = data[trials[2]][0:2]
        demo_fps = data[trials[3]][0:2]

        ax_row[0].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[0].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[0].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[0].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

        labels = ['IMU', 'IMU Integrator']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][2:4]
        platformer_fps = data[trials[1]][2:4]
        materials_fps = data[trials[2]][2:4]
        demo_fps = data[trials[3]][2:4]

        ax_row[1].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[1].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[1].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[1].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

        labels = ['Applicaton', 'Reprojection']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][4:6]
        platformer_fps = data[trials[1]][4:6]
        materials_fps = data[trials[2]][4:6]
        demo_fps = data[trials[3]][4:6]

        ax_row[2].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[2].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[2].bar(x + (width * 0.5), matetirials_fps, width, label='Materials')
        ax_row[2].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

        labels = ['Playback', 'Encoding']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][6:8]
        platformer_fps = data[trials[1]][6:8]
        materials_fps = data[trials[2]][6:8]
        demo_fps = data[trials[3] ][6:8]

        ax_row[3].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[3].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[3].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[3].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

    # Uncomment one of these at a time depending on if you want to gen Desktop/Jetson HP/Jetson LP FPS values
    if trial_num == 0:
        create_row(data, axes, ['sponza-desktop', 'platformer-desktop', 'materials-desktop', 'demo-desktop'])
    elif trial_num == 1: 
        create_row(data, axes, ['sponza-jetsonhp', 'platformer-jetsonhp', 'materials-jetsonhp', 'demo-jetsonhp'])
    else:
        create_row(data, axes, ['sponza-jetsonlp', 'platformer-jetsonlp', 'materials-jetsonlp', 'demo-jetsonlp'])

    axes[0].legend(loc='lower right')

    labels = ['Camera','OpenVINS Camera']
    axes[0].set_xticks(np.arange(len(labels)) * 2)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('Perception Pipeline (Camera)', fontsize=18)
    labels = ['IMU', 'IMU Integrator']
    axes[1].set_xticks(np.arange(len(labels)) * 2)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('Perception Pipeline (IMU)', fontsize=18)
    labels = ['Applicaton', 'Reprojection']
    axes[2].set_xticks(np.arange(len(labels)) * 2)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('Visual Pipeline', fontsize=18)
    labels = ['Playback', 'Encoding']
    axes[3].set_xticks(np.arange(len(labels)) * 2)
    axes[3].set_xticklabels(labels)
    axes[3].set_xlabel('Audio Pipeline', fontsize=18)

    fig.set_size_inches(16, 4)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.15, hspace=.1)

    plt.show()


def plot_cpu():
    data = pd.read_csv('../output/cpu.csv').transpose()
    fig, axs = plt.subplots()

    new_header = data.iloc[0] #grab the first row for the header
    data = data[1:] #take the data less the header row
    data.columns = new_header #set the header row as the df header
    x = np.arange(4) * 2
    width = 0.40

    # One application per group
    desktop_bars = []
    jetsonhp_bars = []
    jetsonlp_bars = []
    for idx in range(len(data['sponza-desktop'])):
        desktop_bars.append([data['sponza-desktop'][idx] / data['sponza-desktop'].sum(), data['materials-desktop'][idx] / data['materials-desktop'].sum(), data['platformer-desktop'][idx] / data['platformer-desktop'].sum(), data['demo-desktop'][idx] / data['demo-desktop'].sum()])
        jetsonhp_bars.append([data['sponza-jetsonhp'][idx] / data['sponza-jetsonhp'].sum(), data['materials-jetsonhp'][idx] / data['materials-jetsonhp'].sum(), data['platformer-jetsonhp'][idx] / data['platformer-jetsonhp'].sum(), data['demo-jetsonhp'][idx] / data['demo-jetsonhp'].sum()])
        jetsonlp_bars.append([data['sponza-jetsonlp'][idx] / data['sponza-jetsonlp'].sum(), data['materials-jetsonlp'][idx] / data['materials-jetsonlp'].sum(), data['platformer-jetsonlp'][idx] / data['platformer-jetsonlp'].sum(), data['demo-jetsonlp'][idx] / data['demo-jetsonlp'].sum()])

    desktop_sum = np.zeros(4)
    jetsonhp_sum = np.zeros(4)
    jetsonlp_sum = np.zeros(4)
    colors = ['tab:blue', 'lightcoral', 'tab:orange', 'tab:green', 'gold', 'tab:red', 'tab:brown', 'skyblue', 'tab:pink']

    temp_bar_list = []
    for idx in range(len(desktop_bars)):
        temp_bar = axs.bar(x - (width * 1.1), desktop_bars[idx], width, bottom=desktop_sum, color=colors[idx])
        axs.bar(x, jetsonhp_bars[idx], width, bottom=jetsonhp_sum, color=colors[idx])
        axs.bar(x + (width * 1.1), jetsonlp_bars[idx], width, bottom=jetsonlp_sum, color=colors[idx])

        temp_bar_list.append(temp_bar)
        desktop_sum += desktop_bars[idx]
        jetsonhp_sum += jetsonhp_bars[idx]
        jetsonlp_sum += jetsonlp_bars[idx]

    axs.set_xticks(x)
    axs.set_xticklabels(['D     HP    LP', 'D     HP    LP', 'D     HP    LP', 'D     HP    LP'])
    axs.legend(temp_bar_list, ['OpenVINS Camera', 'OpenVINS IMU', 'Playback', 'Encoding', 'IMU Integrator', 'Reprojection', 'Camera', 'IMU', 'Application'], loc='upper center', bbox_to_anchor=(0.5, -0.07),
            fancybox=False, shadow=False, ncol=5)
    fig.tight_layout()
    fig.subplots_adjust(left=.05, right=.95, bottom=.2)
    fig.set_size_inches(8, 4.5)

    plt.show()

def plot_gpu():
    data = pd.read_csv('../output/gpu.csv').transpose()
    fig, axs = plt.subplots()

    new_header = data.iloc[0] #grab the first row for the header
    data = data[1:] #take the data less the header row
    data.columns = new_header #set the header row as the df header
    x = np.arange(4) * 2
    width = 0.40

    # One application per group
    desktop_bars = []
    jetsonhp_bars = []
    jetsonlp_bars = []
    for idx in range(len(data['sponza-desktop'])):
        if idx == 0:
            desktop_bars.append([data['sponza-desktop'][0:2].sum() / data['sponza-desktop'].sum(), data['materials-desktop'][0:2].sum() / data['materials-desktop'].sum(), data['platformer-desktop'][0:2].sum() / data['platformer-desktop'].sum(), data['demo-desktop'][0:2].sum() / data['demo-desktop'].sum()])
            jetsonhp_bars.append([data['sponza-jetsonhp'][0:2].sum() / data['sponza-jetsonhp'].sum(), data['materials-jetsonhp'][0:2].sum() / data['materials-jetsonhp'].sum(), data['platformer-jetsonhp'][0:2].sum() / data['platformer-jetsonhp'].sum(), data['demo-jetsonhp'][0:2].sum() / data['demo-jetsonhp'].sum()])
            jetsonlp_bars.append([data['sponza-jetsonlp'][0:2].sum() / data['sponza-jetsonlp'].sum(), data['materials-jetsonlp'][0:2].sum() / data['materials-jetsonlp'].sum(), data['platformer-jetsonlp'][0:2].sum() / data['platformer-jetsonlp'].sum(), data['demo-jetsonlp'][0:2].sum() / data['demo-jetsonlp'].sum()])
        elif idx == 1:
            continue
        else:
            desktop_bars.append([data['sponza-desktop'][idx] / data['sponza-desktop'].sum(), data['materials-desktop'][idx] / data['materials-desktop'].sum(), data['platformer-desktop'][idx] / data['platformer-desktop'].sum(), data['demo-desktop'][idx] / data['demo-desktop'].sum()])
            jetsonhp_bars.append([data['sponza-jetsonhp'][idx] / data['sponza-jetsonhp'].sum(), data['materials-jetsonhp'][idx] / data['materials-jetsonhp'].sum(), data['platformer-jetsonhp'][idx] / data['platformer-jetsonhp'].sum(), data['demo-jetsonhp'][idx] / data['demo-jetsonhp'].sum()])
            jetsonlp_bars.append([data['sponza-jetsonlp'][idx] / data['sponza-jetsonlp'].sum(), data['materials-jetsonlp'][idx] / data['materials-jetsonlp'].sum(), data['platformer-jetsonlp'][idx] / data['platformer-jetsonlp'].sum(), data['demo-jetsonlp'][idx] / data['demo-jetsonlp'].sum()])

    desktop_sum = np.zeros(4)
    jetsonhp_sum = np.zeros(4)
    jetsonlp_sum = np.zeros(4)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:purple', 'tab:pink', 'skyblue']

    temp_bar_list = []
    for idx in range(len(desktop_bars)):
        temp_bar = axs.bar(x - (width * 1.1), desktop_bars[idx], width, bottom=desktop_sum, color=colors[idx])
        axs.bar(x, jetsonhp_bars[idx], width, bottom=jetsonhp_sum, color=colors[idx])
        axs.bar(x + (width * 1.1), jetsonlp_bars[idx], width, bottom=jetsonlp_sum, color=colors[idx])

        temp_bar_list.append(temp_bar)
        desktop_sum += desktop_bars[idx]
        jetsonhp_sum += jetsonhp_bars[idx]
        jetsonlp_sum += jetsonlp_bars[idx]

    axs.set_xticks(x)
    axs.set_xticklabels(['D     HP    LP', 'D     HP    LP', 'D     HP    LP', 'D     HP    LP'])
    axs.legend(temp_bar_list, ['Application', 'Hologram', 'Reprojection'], loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=False, shadow=False, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(left=.05, right=.95, bottom=.2)
    fig.set_size_inches(8, 4)

    plt.show()


def plot_power():
    data = pd.read_csv('../output/power.csv').transpose()
    fig, axs = plt.subplots()

    new_header = data.iloc[0] #grab the first row for the header
    data = data[1:] #take the data less the header row
    data.columns = new_header #set the header row as the df header
    x = np.arange(4) * 2
    width = 0.40

    # One application per group
    desktop_bars = []
    jetsonhp_bars = []
    jetsonlp_bars = []
    for idx in range(len(data['sponza-jetsonhp'])):
        if idx < 3:
            desktop_bars.append([data['sponza-desktop'][idx] / data['sponza-desktop'].sum(), data['materials-desktop'][idx] / data['materials-desktop'].sum(), data['platformer-desktop'][idx] / data['platformer-desktop'].sum(), data['demo-desktop'][idx] / data['demo-desktop'].sum()])
        jetsonhp_bars.append([data['sponza-jetsonhp'][idx] / data['sponza-jetsonhp'].sum(), data['materials-jetsonhp'][idx] / data['materials-jetsonhp'].sum(), data['platformer-jetsonhp'][idx] / data['platformer-jetsonhp'].sum(), data['demo-jetsonhp'][idx] / data['demo-jetsonhp'].sum()])
        jetsonlp_bars.append([data['sponza-jetsonlp'][idx] / data['sponza-jetsonlp'].sum(), data['materials-jetsonlp'][idx] / data['materials-jetsonlp'].sum(), data['platformer-jetsonlp'][idx] / data['platformer-jetsonlp'].sum(), data['demo-jetsonlp'][idx] / data['demo-jetsonlp'].sum()])

    desktop_sum = np.zeros(4)
    jetsonhp_sum = np.zeros(4)
    jetsonlp_sum = np.zeros(4)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:purple', 'tab:pink', 'skyblue']

    temp_bar_list = []
    for idx in range(len(jetsonhp_bars)):
        if idx < 3:
            axs.bar(x - (width * 1.1), desktop_bars[idx], width, bottom=desktop_sum, color=colors[idx])
            desktop_sum += desktop_bars[idx]

        temp_bar = axs.bar(x, jetsonhp_bars[idx], width, bottom=jetsonhp_sum, color=colors[idx])
        jetsonhp_sum += jetsonhp_bars[idx]

        axs.bar(x + (width * 1.1), jetsonlp_bars[idx], width, bottom=jetsonlp_sum, color=colors[idx])
        jetsonlp_sum += jetsonlp_bars[idx]
        
        temp_bar_list.append(temp_bar)

    axs.set_xticks(x)
    axs.set_xticklabels(['D     HP    LP', 'D     HP    LP', 'D     HP    LP', 'D     HP    LP'])
    axs.legend(temp_bar_list, ['CPU Power', 'GPU Power', 'DDR Power', 'SOC Power', 'SYS Power'], loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=False, shadow=False, ncol=5)
    fig.tight_layout()
    fig.subplots_adjust(left=.05, right=.95, bottom=.2)
    fig.set_size_inches(8, 4)

    plt.show()


def plot_mtp(trial_list, title, max_y):
    fig, axs = plt.subplots()
    data_d = pd.read_csv('../output/' + trial_list[0] + '/mtp.csv')
    data_hp = pd.read_csv('../output/' + trial_list[1] + '/mtp.csv')
    data_lp = pd.read_csv('../output/' + trial_list[2] + '/mtp.csv')

    max_val = max([len(data_d['render_to_display']), len(data_hp['render_to_display']), len(data_lp['render_to_display'])])

    axs.plot(np.arange(len(data_lp)) * (max_val / len(data_lp)), data_lp['render_to_display'], color='firebrick')
    axs.plot(np.arange(len(data_hp)) * (max_val / len(data_hp)), data_hp['render_to_display'], color='gold')
    axs.plot(np.arange(len(data_d)) * (max_val / len(data_d)), data_d['render_to_display'], color='limegreen')

    axs.legend(['Jetson LP', 'Jetson HP', 'Desktop'], ncol=3, loc='upper center')
    axs.axes.xaxis.set_visible(False)
    fig.set_size_inches(20, 4)
    fig.subplots_adjust(left=.02, right=.98, bottom=.05, top=.90)

    plt.xlim(left=0, right=max_val)
    plt.ylim(bottom=0, top=max_y)
    plt.title(title)
    plt.show()


def plot_frame_time(trial_num):
    # They are similar to the fps graphs you did in terms organization, but with per-frame execution time instead of fps. 
    # Plus at the top of each bar, we'd like to put the co-efficient of variation. Plus we'd like to draw a horizontal line 
    # across the bars showing the max exec time that each one should have taken (this is 1/target fps).
    data = pd.read_csv('../output/fps.csv')
    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 2, 2]})

    def autolabel(rects, ax, trial_num):
        for rect in rects:
            ax.annotate('{}'.format(1),
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def create_row(data, ax_row, trials):
        width = 0.35  # the width of the bars

        labels = ['Camera','OpenVINS Camera']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = 1 / data[trials[0]][0:2]
        platformer_fps = 1 / data[trials[1]][0:2]
        materials_fps = 1 / data[trials[2]][0:2]
        demo_fps = 1 / data[trials[3]][0:2]

        s_bar = ax_row[0].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        autolabel(s_bar, ax_row[0], 0)
        p_bar = ax_row[0].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        m_bar = ax_row[0].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        a_bar = ax_row[0].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

        labels = ['IMU', 'IMU Integrator']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = 1 / data[trials[0]][2:4]
        platformer_fps = 1 / data[trials[1]][2:4]
        materials_fps = 1 / data[trials[2]][2:4]
        demo_fps = 1 / data[trials[3]][2:4]

        ax_row[1].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[1].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[1].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[1].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

        labels = ['Applicaton', 'Reprojection']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = 1 / data[trials[0]][4:6]
        platformer_fps = 1 / data[trials[1]][4:6]
        materials_fps = 1 / data[trials[2]][4:6]
        demo_fps = 1 / data[trials[3]][4:6]

        ax_row[2].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[2].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[2].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[2].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

        labels = ['Playback', 'Encoding']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = 1 / data[trials[0]][6:8]
        platformer_fps = 1 / data[trials[1]][6:8]
        materials_fps = 1 / data[trials[2]][6:8]
        demo_fps = 1 / data[trials[3] ][6:8]

        ax_row[3].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[3].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[3].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[3].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

    # Uncomment one of these at a time depending on if you want to gen Desktop/Jetson HP/Jetson LP FPS values
    if trial_num == 0:
        create_row(data, axes, ['sponza-desktop', 'platformer-desktop', 'materials-desktop', 'demo-desktop'])
    elif trial_num == 1: 
        create_row(data, axes, ['sponza-jetsonhp', 'platformer-jetsonhp', 'materials-jetsonhp', 'demo-jetsonhp'])
    else:
        create_row(data, axes, ['sponza-jetsonlp', 'platformer-jetsonlp', 'materials-jetsonlp', 'demo-jetsonlp'])

    axes[0].legend(loc='lower right')

    labels = ['Camera','OpenVINS Camera']
    axes[0].set_xticks(np.arange(len(labels)) * 2)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('Perception Pipeline (Camera)', fontsize=18)
    labels = ['IMU', 'IMU Integrator']
    axes[1].set_xticks(np.arange(len(labels)) * 2)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('Perception Pipeline (IMU)', fontsize=18)
    labels = ['Applicaton', 'Reprojection']
    axes[2].set_xticks(np.arange(len(labels)) * 2)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('Visual Pipeline', fontsize=18)
    labels = ['Playback', 'Encoding']
    axes[3].set_xticks(np.arange(len(labels)) * 2)
    axes[3].set_xticklabels(labels)
    axes[3].set_xlabel('Audio Pipeline', fontsize=18)

    fig.set_size_inches(16, 4)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.15, hspace=.1)

    plt.show()
