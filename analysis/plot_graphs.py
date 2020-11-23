import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_fps(trial_num):
    plt.clf()
    data = pd.read_csv('../output/fps.csv')
    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 2, 2]})

    def create_row(data, ax_row, trials):
        width = 0.35  # the width of the bars

        labels = ['Camera','SLAM']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][0:2]
        platformer_fps = data[trials[1]][0:2]
        materials_fps = data[trials[2]][0:2]
        demo_fps = data[trials[3]][0:2]

        ax_row[0].bar(x - (width * 1.5), sponza_fps, width, label='Sponza', color='royalblue')
        ax_row[0].bar(x - (width * 0.5), platformer_fps, width, label='Platformer', color='mediumturquoise')
        ax_row[0].bar(x + (width * 0.5), materials_fps, width, label='Materials', color='paleturquoise')
        ax_row[0].bar(x + (width * 1.5), demo_fps, width, label='AR Demo', color='orange')

        labels = ['IMU', 'Integrator']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][2:4]
        platformer_fps = data[trials[1]][2:4]
        materials_fps = data[trials[2]][2:4]
        demo_fps = data[trials[3]][2:4]

        ax_row[1].bar(x - (width * 1.5), sponza_fps, width, label='Sponza', color='royalblue')
        ax_row[1].bar(x - (width * 0.5), platformer_fps, width, label='Platformer', color='mediumturquoise')
        ax_row[1].bar(x + (width * 0.5), materials_fps, width, label='Materials', color='paleturquoise')
        ax_row[1].bar(x + (width * 1.5), demo_fps, width, label='AR Demo', color='orange')

        labels = ['Application', 'Reprojection']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][4:6]
        platformer_fps = data[trials[1]][4:6]
        materials_fps = data[trials[2]][4:6]
        demo_fps = data[trials[3]][4:6]

        ax_row[2].bar(x - (width * 1.5), sponza_fps, width, label='Sponza', color='royalblue')
        ax_row[2].bar(x - (width * 0.5), platformer_fps, width, label='Platformer', color='mediumturquoise')
        ax_row[2].bar(x + (width * 0.5), materials_fps, width, label='Materials', color='paleturquoise')
        ax_row[2].bar(x + (width * 1.5), demo_fps, width, label='AR Demo', color='orange')

        labels = ['Playback', 'Encoding']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][6:8]
        platformer_fps = data[trials[1]][6:8]
        materials_fps = data[trials[2]][6:8]
        demo_fps = data[trials[3] ][6:8]

        ax_row[3].bar(x - (width * 1.5), sponza_fps, width, label='Sponza', color='royalblue')
        ax_row[3].bar(x - (width * 0.5), platformer_fps, width, label='Platformer', color='mediumturquoise')
        ax_row[3].bar(x + (width * 0.5), materials_fps, width, label='Materials', color='paleturquoise')
        ax_row[3].bar(x + (width * 1.5), demo_fps, width, label='AR Demo', color='orange')

    # Uncomment one of these at a time depending on if you want to gen Desktop/Jetson HP/Jetson LP FPS values
    if trial_num == 0:
        create_row(data, axes, ['sponza-desktop', 'platformer-desktop', 'materials-desktop', 'demo-desktop'])
    elif trial_num == 1: 
        create_row(data, axes, ['sponza-jetsonhp', 'platformer-jetsonhp', 'materials-jetsonhp', 'demo-jetsonhp'])
    else:
        create_row(data, axes, ['sponza-jetsonlp', 'platformer-jetsonlp', 'materials-jetsonlp', 'demo-jetsonlp'])

    axes[0].legend(loc='lower right')

    labels = ['Camera','SLAM']
    axes[0].set_xticks(np.arange(len(labels)) * 2)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('Perception Pipeline (Camera)', fontsize=18)
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_yticks(np.arange(6) * 3)
    axes[0].set_ylim(bottom=0, top=15.01)
    
    labels = ['IMU', 'Integrator']
    axes[1].set_xticks(np.arange(len(labels)) * 2)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('Perception Pipeline (IMU)', fontsize=18)
    axes[1].set_yticks(np.arange(6) * 100)
    axes[1].set_ylim(bottom=0, top=500.01)

    labels = ['Application', 'Reprojection']
    axes[2].set_xticks(np.arange(len(labels)) * 2)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('Visual Pipeline', fontsize=18)
    axes[2].set_yticks(np.arange(9) * 15)
    axes[2].set_ylim(bottom=0, top=120.01)

    labels = ['Playback', 'Encoding']
    axes[3].set_xticks(np.arange(len(labels)) * 2)
    axes[3].set_xticklabels(labels)
    axes[3].set_xlabel('Audio Pipeline', fontsize=18)
    axes[3].set_yticks(np.arange(9) * 6)
    axes[3].set_ylim(bottom=0, top=48.01)

    fig.set_size_inches(16, 4)
    fig.tight_layout()
    fig.subplots_adjust(left=.1, wspace=.15, hspace=.1)

    if trial_num == 0:
        plt.savefig('../Graphs/FPS_Desktop.pdf')
    elif trial_num == 1:
        plt.savefig('../Graphs/FPS_JHP.pdf')
    else:
        plt.savefig('../Graphs/FPS_JLP.pdf')



def plot_cpu():
    plt.clf()
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

    desktop_bars = [[y*100 for y in x] for x in desktop_bars]
    jetsonhp_bars = [[y*100 for y in x] for x in jetsonhp_bars]
    jetsonlp_bars = [[y*100 for y in x] for x in jetsonlp_bars]

    desktop_sum = np.zeros(4)
    jetsonhp_sum = np.zeros(4)
    jetsonlp_sum = np.zeros(4)
    colors = ['navy', 'steelblue', 'mediumturquoise', 'lightcoral', 'orange', 'saddlebrown', 'firebrick', 'yellowgreen']

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
    axs.set_xticklabels(['D     HP    LP\nSponza', 'D     HP    LP\nMaterials', 'D     HP    LP\nPlatformer', 'D     HP    LP\nDemo'])
    axs.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs.set_ylim(bottom=0, top=100)
    axs.legend(temp_bar_list, ['Camera', 'SLAM', 'IMU', 'Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding',], loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=False, shadow=False, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(left=.08, right=.95, bottom=.20)
    fig.set_size_inches(8, 4.5)

    plt.savefig('../Graphs/CPU.pdf')

def plot_gpu():
    plt.clf()
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

    desktop_bars = [[y*100 for y in x] for x in desktop_bars]
    jetsonhp_bars = [[y*100 for y in x] for x in jetsonhp_bars]
    jetsonlp_bars = [[y*100 for y in x] for x in jetsonlp_bars]

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
    axs.set_xticklabels(['D     HP    LP\nSponza', 'D     HP    LP\nMaterials', 'D     HP    LP\nPlatformer', 'D     HP    LP\nDemo'])
    axs.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs.set_ylim(bottom=0, top=100)
    axs.legend(temp_bar_list, ['Application', 'Hologram', 'Reprojection'], loc='upper center', bbox_to_anchor=(0.5, -0.12),
            fancybox=False, shadow=False, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(left=.08, right=.95, bottom=.20)
    fig.set_size_inches(8, 4)

    plt.savefig('../Graphs/GPU.pdf')


def plot_power():
    plt.clf()
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

    desktop_bars = [[y*100 for y in x] for x in desktop_bars]
    jetsonhp_bars = [[y*100 for y in x] for x in jetsonhp_bars]
    jetsonlp_bars = [[y*100 for y in x] for x in jetsonlp_bars]

    desktop_sum = np.zeros(4)
    jetsonhp_sum = np.zeros(4)
    jetsonlp_sum = np.zeros(4)
    colors = ['navy', 'steelblue', 'mediumturquoise', 'lightcoral', 'orange']

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
    axs.set_xticklabels(['D     HP    LP\nSponza', 'D     HP    LP\nMaterials', 'D     HP    LP\nPlatformer', 'D     HP    LP\nDemo'])
    axs.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs.set_ylim(bottom=0, top=100)
    axs.legend(temp_bar_list, ['CPU Power', 'GPU Power', 'DDR Power', 'SOC Power', 'SYS Power'], loc='upper center', bbox_to_anchor=(0.5, -0.12),
            fancybox=False, shadow=False, ncol=5)
    fig.tight_layout()
    fig.subplots_adjust(left=.08, right=.95, bottom=0.20)
    fig.set_size_inches(8, 4)

    plt.savefig('../Graphs/Power.pdf')


def plot_power_total():
    plt.clf()
    data = pd.read_csv('../output/power.csv').transpose()
    fig, axs = plt.subplots()

    new_header = data.iloc[0] #grab the first row for the header
    data = data[1:] #take the data less the header row
    data.columns = new_header #set the header row as the df header
    x = np.arange(4) * 2
    width = 0.40

    desktop_bars = [sum(data['sponza-desktop'][0:3]), sum(data['materials-desktop'][0:3]), sum(data['platformer-desktop'][0:3]), sum(data['demo-desktop'][0:3])]
    jetsonhp_bars = [sum(data['sponza-jetsonhp']), sum(data['materials-jetsonhp']), sum(data['platformer-jetsonhp']), sum(data['demo-jetsonhp'])]
    jetsonlp_bars = [sum(data['sponza-jetsonlp']), sum(data['materials-jetsonlp']), sum(data['platformer-jetsonlp']), sum(data['demo-jetsonlp'])]

    axs.bar(x - (width * 1.1), desktop_bars, width, color='dimgray')
    axs.bar(x, jetsonhp_bars, width , color='dimgray')
    axs.bar(x + (width * 1.1), jetsonlp_bars, width, color='dimgray')        

    axs.set_xticks(x)
    axs.set_xticklabels(['D     HP    LP\nSponza', 'D     HP    LP\nMaterials', 'D     HP    LP\nPlatformer', 'D     HP    LP\nDemo'])
    axs.set_yticklabels([10, 100, 1000])
    axs.set_yscale('log')
    axs.set_ylabel('Power (Watts)')
    axs.set_ylim(bottom=1, top=1000)

    fig.tight_layout()
    fig.subplots_adjust(left=.08, right=.95, bottom=.1)
    fig.set_size_inches(8, 4)

    plt.savefig('../Graphs/Power_Total.pdf')


def plot_mtp(trial_list, title, max_y):
    plt.clf()
    fig, axs = plt.subplots()
    data_d = pd.read_csv('../output/' + trial_list[0] + '/mtp.csv')
    data_hp = pd.read_csv('../output/' + trial_list[1] + '/mtp.csv')
    data_lp = pd.read_csv('../output/' + trial_list[2] + '/mtp.csv')

    max_val = max([len(data_d['imu_to_display']), len(data_hp['imu_to_display']), len(data_lp['imu_to_display'])])

    axs.plot(np.arange(len(data_lp)) * (max_val / len(data_lp)), data_lp['imu_to_display'], color='royalblue')
    axs.plot(np.arange(len(data_hp)) * (max_val / len(data_hp)), data_hp['imu_to_display'], color='paleturquoise')
    axs.plot(np.arange(len(data_d)) * (max_val / len(data_d)), data_d['imu_to_display'], color='darkorange')

    axs.legend(['Jetson LP', 'Jetson HP', 'Desktop'], ncol=3, loc='upper center')
    axs.axes.xaxis.set_visible(False)
    axs.set_ylabel('Time (ms)')

    fig.set_size_inches(20, 4)
    fig.subplots_adjust(left=.04, right=.98, bottom=.05, top=.90)

    plt.xlim(left=200, right=max_val)
    plt.ylim(bottom=0, top=max_y)
    plt.title(title)

    plt.savefig('../Graphs/MTP_' + title + '.pdf')


def plot_frame_time(trial_num):
    plt.clf()
    mean_data = pd.read_csv('../output/frame_time_mean.csv')
    std_data = pd.read_csv('../output/frame_time_std.csv')
    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 2, 2]})

    def autolabel(rects, ax, sd, mean):
        for idx, rect in enumerate(rects):
            ax.annotate(int((sd[idx] / mean[idx]) * 100 ),
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Yes this is gross i know and im sorry DX
    def create_row(data, ax_row, trials):
        width = 0.35  # the width of the bars

        labels = ['Camera','SLAM']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][0:2]
        platformer_fps = data[trials[1]][0:2]
        materials_fps = data[trials[2]][0:2]
        demo_fps = data[trials[3]][0:2]

        s_bar = ax_row[0].bar(x - (width * 1.65), sponza_fps, width, label='Sponza', color='royalblue')
        autolabel(s_bar, ax_row[0], std_data[trials[0]][0:2].to_numpy(), sponza_fps.to_numpy())
        p_bar = ax_row[0].bar(x - (width * 0.55), platformer_fps, width, label='Platformer', color='mediumturquoise')
        autolabel(p_bar, ax_row[0], std_data[trials[1]][0:2].to_numpy(), platformer_fps.to_numpy())
        m_bar = ax_row[0].bar(x + (width * 0.55), materials_fps, width, label='Materials', color='paleturquoise')
        autolabel(m_bar, ax_row[0], std_data[trials[2]][0:2].to_numpy(), materials_fps.to_numpy())
        a_bar = ax_row[0].bar(x + (width * 1.65), demo_fps, width, label='AR Demo', color='orange')
        autolabel(a_bar, ax_row[0], std_data[trials[3]][0:2].to_numpy(), demo_fps.to_numpy())

        labels = ['IMU', 'Integrator']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][2:4]
        platformer_fps = data[trials[1]][2:4]
        materials_fps = data[trials[2]][2:4]
        demo_fps = data[trials[3]][2:4]

        s_bar = ax_row[1].bar(x - (width * 1.65), sponza_fps, width, label='Sponza', color='royalblue')
        autolabel(s_bar, ax_row[1], std_data[trials[0]][2:4].to_numpy(), sponza_fps.to_numpy())
        p_bar = ax_row[1].bar(x - (width * 0.55), platformer_fps, width, label='Platformer', color='mediumturquoise')
        autolabel(p_bar, ax_row[1], std_data[trials[1]][2:4].to_numpy(), platformer_fps.to_numpy())
        m_bar = ax_row[1].bar(x + (width * 0.55), materials_fps, width, label='Materials', color='paleturquoise')
        autolabel(m_bar, ax_row[1], std_data[trials[2]][2:4].to_numpy(), materials_fps.to_numpy())
        a_bar = ax_row[1].bar(x + (width * 1.65), demo_fps, width, label='AR Demo', color='orange')
        autolabel(a_bar, ax_row[1], std_data[trials[3]][2:4].to_numpy(), demo_fps.to_numpy())

        labels = ['Application', 'Reprojection']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][4:6]
        platformer_fps = data[trials[1]][4:6]
        materials_fps = data[trials[2]][4:6]
        demo_fps =  data[trials[3]][4:6]

        s_bar = ax_row[2].bar(x - (width * 1.65), sponza_fps, width, label='Sponza', color='royalblue')
        autolabel(s_bar, ax_row[2], std_data[trials[0]][4:6].to_numpy(), sponza_fps.to_numpy())
        p_bar = ax_row[2].bar(x - (width * 0.55), platformer_fps, width, label='Platformer', color='mediumturquoise')
        autolabel(p_bar, ax_row[2], std_data[trials[1]][4:6].to_numpy(), platformer_fps.to_numpy())
        m_bar = ax_row[2].bar(x + (width * 0.55), materials_fps, width, label='Materials', color='paleturquoise')
        autolabel(m_bar, ax_row[2], std_data[trials[2]][4:6].to_numpy(), materials_fps.to_numpy())
        a_bar = ax_row[2].bar(x + (width * 1.65), demo_fps, width, label='AR Demo', color='orange')
        autolabel(a_bar, ax_row[2], std_data[trials[3]][4:6].to_numpy(), demo_fps.to_numpy())

        labels = ['Playback', 'Encoding']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][6:8]
        platformer_fps = data[trials[1]][6:8]
        materials_fps = data[trials[2]][6:8]
        demo_fps = data[trials[3] ][6:8]

        s_bar = ax_row[3].bar(x - (width * 1.65), sponza_fps, width, label='Sponza', color='royalblue')
        autolabel(s_bar, ax_row[3], std_data[trials[0]][6:8].to_numpy(), sponza_fps.to_numpy())
        p_bar = ax_row[3].bar(x - (width * 0.55), platformer_fps, width, label='Platformer', color='mediumturquoise')
        autolabel(p_bar, ax_row[3], std_data[trials[0]][6:8].to_numpy(), platformer_fps.to_numpy())
        m_bar = ax_row[3].bar(x + (width * 0.55), materials_fps, width, label='Materials', color='paleturquoise')
        autolabel(m_bar, ax_row[3], std_data[trials[0]][6:8].to_numpy(), materials_fps.to_numpy())
        a_bar = ax_row[3].bar(x + (width * 1.65), demo_fps, width, label='AR Demo', color='orange')
        autolabel(a_bar, ax_row[3], std_data[trials[0]][6:8].to_numpy(), demo_fps.to_numpy())

    # Uncomment one of these at a time depending on if you want to gen Desktop/Jetson HP/Jetson LP FPS values
    if trial_num == 0:
        create_row(mean_data, axes, ['sponza-desktop', 'platformer-desktop', 'materials-desktop', 'demo-desktop'])
    elif trial_num == 1: 
        create_row(mean_data, axes, ['sponza-jetsonhp', 'platformer-jetsonhp', 'materials-jetsonhp', 'demo-jetsonhp'])
    else:
        create_row(mean_data, axes, ['sponza-jetsonlp', 'platformer-jetsonlp', 'materials-jetsonlp', 'demo-jetsonlp'])

    axes[0].legend(loc='upper left')

    labels = ['Camera','SLAM']
    axes[0].set_xticks(np.arange(len(labels)) * 2)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('Perception Pipeline (Camera)', fontsize=18)
    ylim_val = axes[0].get_ylim()
    axes[0].set_ylim(top=ylim_val[1] * 1.1)
    axes[0].axhline(y=66.666, xmin=0.02, xmax=0.98, color='r')
    axes[0].set_ylabel('Time (ms)')

    labels = ['IMU', 'Integrator']
    axes[1].set_xticks(np.arange(len(labels)) * 2)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('Perception Pipeline (IMU)', fontsize=18)
    ylim_val = axes[1].get_ylim()
    axes[1].set_ylim(top=ylim_val[1] * 1.1)
    axes[1].axhline(y=2, xmin=0.02, xmax=0.98, color='r')

    labels = ['Application', 'Reprojection']
    axes[2].set_xticks(np.arange(len(labels)) * 2)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('Visual Pipeline', fontsize=18)
    ylim_val = axes[2].get_ylim()
    axes[2].set_ylim(top=ylim_val[1] * 1.1)
    axes[2].axhline(y=8.3333, xmin=0.02, xmax=0.98, color='r')

    labels = ['Playback', 'Encoding']
    axes[3].set_xticks(np.arange(len(labels)) * 2)
    axes[3].set_xticklabels(labels)
    axes[3].set_xlabel('Audio Pipeline', fontsize=18)
    ylim_val = axes[3].get_ylim()
    axes[3].set_ylim(top=ylim_val[1] * 1.1)
    axes[3].axhline(y=20.83, xmin=0.02, xmax=0.98, color='r')

    fig.set_size_inches(16, 4)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.15, hspace=.1)

    if trial_num == 0:
        plt.savefig('../Graphs/Frame_Time_Desktop.pdf')
    elif trial_num == 1:
        plt.savefig('../Graphs/Frame_Time_JHP.pdf')
    else:
        plt.savefig('../Graphs/Frame_Time_JLP.pdf')