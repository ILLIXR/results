import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util import PerTrialData
from typing import List, Dict
from tqdm import tqdm

def plot_fps(trial_num):
    plt.clf()
    data = pd.read_csv('../output/fps.csv')
    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 2, 2]})

    def create_row(data, ax_row, trials):
        width = 0.35  # the width of the bars

        labels = ['Camera','VIO']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][0:2]
        materials_fps = data[trials[1]][0:2]
        platformer_fps = data[trials[2]][0:2]
        demo_fps = data[trials[3]][0:2]

        ax_row[0].bar(x - (width * 1.80), sponza_fps, width, label='Sponza', color='darkred')
        ax_row[0].bar(x - (width * 0.60), materials_fps, width, label='Materials', color='indigo')
        ax_row[0].bar(x + (width * 0.60), platformer_fps, width, label='Platformer', color='royalblue')
        ax_row[0].bar(x + (width * 1.80), demo_fps, width, label='AR Demo', color='green')

        labels = ['IMU', 'Integrator']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][2:4]
        materials_fps = data[trials[1]][2:4]
        platformer_fps = data[trials[2]][2:4]
        demo_fps = data[trials[3]][2:4]

        ax_row[1].bar(x - (width * 1.80), sponza_fps, width, label='Sponza', color='darkred')
        ax_row[1].bar(x - (width * 0.60), materials_fps, width, label='Materials', color='indigo')
        ax_row[1].bar(x + (width * 0.60), platformer_fps, width, label='Platformer', color='royalblue')
        ax_row[1].bar(x + (width * 1.80), demo_fps, width, label='AR Demo', color='green')

        labels = ['Application', 'Reprojection']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][4:6]
        materials_fps = data[trials[1]][4:6]
        platformer_fps = data[trials[2]][4:6]
        demo_fps = data[trials[3]][4:6]

        ax_row[2].bar(x - (width * 1.80), sponza_fps, width, label='Sponza', color='darkred')
        ax_row[2].bar(x - (width * 0.60), materials_fps, width, label='Materials', color='indigo')
        ax_row[2].bar(x + (width * 0.60), platformer_fps, width, label='Platformer', color='royalblue')
        ax_row[2].bar(x + (width * 1.80), demo_fps, width, label='AR Demo', color='green')

        labels = ['Playback', 'Encoding']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][6:8]
        materials_fps = data[trials[1]][6:8]
        platformer_fps = data[trials[2]][6:8]
        demo_fps = data[trials[3] ][6:8]

        ax_row[3].bar(x - (width * 1.80), sponza_fps, width, label='Sponza', color='darkred')
        ax_row[3].bar(x - (width * 0.60), materials_fps, width, label='Materials', color='indigo')
        ax_row[3].bar(x + (width * 0.60), platformer_fps, width, label='Platformer', color='royalblue')
        ax_row[3].bar(x + (width * 1.80), demo_fps, width, label='AR Demo', color='green')

    # Uncomment one of these at a time depending on if you want to gen Desktop/Jetson HP/Jetson LP FPS values
    if trial_num == 0:
        create_row(data, axes, ['sponza-desktop', 'materials-desktop', 'platformer-desktop', 'demo-desktop'])
    elif trial_num == 1: 
        create_row(data, axes, ['sponza-jetsonhp', 'materials-jetsonhp', 'platformer-jetsonhp', 'demo-jetsonhp'])
    else:
        create_row(data, axes, ['sponza-jetsonlp', 'materials-jetsonlp', 'platformer-jetsonlp', 'demo-jetsonlp'])

    if trial_num == 0:
        axes[0].legend(loc='lower right', prop={'size': 12}, fontsize=14)

    labels = ['Camera','VIO']
    axes[0].set_xticks(np.arange(len(labels)) * 2)
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_xlabel('Perception (Camera)', fontsize=14)
    axes[0].set_ylabel('Rate (Hz)', fontsize=14)
    axes[0].set_yticks(np.arange(6) * 3)
    axes[0].set_ylim(bottom=0, top=15.01)
    
    labels = ['IMU', 'Integrator']
    axes[1].set_xticks(np.arange(len(labels)) * 2)
    axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_xlabel('Perception (IMU)', fontsize=14)
    axes[1].set_yticks(np.arange(6) * 100)
    axes[1].set_ylim(bottom=0, top=500.01)

    labels = ['App', 'Reprojection']
    axes[2].set_xticks(np.arange(len(labels)) * 2)
    axes[2].set_xticklabels(labels, fontsize=11)
    axes[2].set_xlabel('Visual', fontsize=14)
    axes[2].set_yticks(np.arange(9) * 15)
    axes[2].set_ylim(bottom=0, top=120.01)

    labels = ['Playback', 'Encoding']
    axes[3].set_xticks(np.arange(len(labels)) * 2)
    axes[3].set_xticklabels(labels, fontsize=11)
    axes[3].set_xlabel('Audio', fontsize=14)
    axes[3].set_yticks(np.arange(9) * 6)
    axes[3].set_ylim(bottom=0, top=48.01)

    fig.set_size_inches(8, 2.5)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.30, hspace=.18)

    if trial_num == 0:
        plt.savefig('../Graphs/fps-desktop.pdf')
    elif trial_num == 1:
        plt.savefig('../Graphs/fps-jhp.pdf')
    else:
        plt.savefig('../Graphs/fps-jlp.pdf')



def plot_cpu():
    plt.clf()
    data = pd.read_csv('../output/cpu.csv').transpose()
    fig, axs = plt.subplots()

    new_header = data.iloc[0] #grab the first row for the header
    data = data[1:] #take the data less the header row
    data.columns = new_header #set the header row as the df header
    x = np.arange(3) * 2
    width = 0.40

    # One application per group
    sponza_bars = []
    materials_bars = []
    platformer_bars = []
    demo_bars = []
    for idx in range(len(data['sponza-desktop'])):
        sponza_bars.append([data['sponza-desktop'][idx] / data['sponza-desktop'].sum(), data['sponza-jetsonhp'][idx] / data['sponza-jetsonhp'].sum(), data['sponza-jetsonlp'][idx] / data['sponza-jetsonlp'].sum()]) 
        materials_bars.append([data['materials-desktop'][idx] / data['materials-desktop'].sum(), data['materials-jetsonhp'][idx] / data['materials-jetsonhp'].sum(), data['materials-jetsonlp'][idx] / data['materials-jetsonlp'].sum()])
        platformer_bars.append([data['platformer-desktop'][idx] / data['platformer-desktop'].sum(), data['platformer-jetsonhp'][idx] / data['platformer-jetsonhp'].sum(), data['platformer-jetsonlp'][idx] / data['platformer-jetsonlp'].sum()])
        demo_bars.append([data['demo-desktop'][idx] / data['demo-desktop'].sum(), data['demo-jetsonhp'][idx] / data['demo-jetsonhp'].sum(), data['demo-jetsonlp'][idx] / data['demo-jetsonlp'].sum()])

    sponza_bars = [[y*100 for y in x] for x in sponza_bars]
    materials_bars = [[y*100 for y in x] for x in materials_bars]
    platformer_bars = [[y*100 for y in x] for x in platformer_bars]
    demo_bars = [[y*100 for y in x] for x in demo_bars]

    sponza_sum = np.zeros(3)
    materials_sum = np.zeros(3)
    platformer_sum = np.zeros(3)
    demo_sum = np.zeros(3)
    colors = ['saddlebrown', 'steelblue', 'indigo', 'lightcoral', 'orange', 'navy', 'firebrick', 'yellowgreen']

    temp_bar_list = []
    for idx in range(len(sponza_bars)):
        temp_bar = axs.bar(x - (width * 1.65), sponza_bars[idx], width, bottom=sponza_sum, color=colors[idx])
        axs.bar(x - (width * 0.55), materials_bars[idx], width, bottom=materials_sum, color=colors[idx])
        axs.bar(x + (width * 0.55), platformer_bars[idx], width, bottom=platformer_sum, color=colors[idx])
        axs.bar(x + (width * 1.65), demo_bars[idx], width, bottom=demo_sum, color=colors[idx])

        temp_bar_list.append(temp_bar)
        sponza_sum += sponza_bars[idx]
        materials_sum += materials_bars[idx]
        platformer_sum += platformer_bars[idx]
        demo_sum += demo_bars[idx]

    axs.set_xticks(x)
    axs.set_xticklabels([' S  M  P AR\nDesktop', ' S  M  P AR\nJetson LP', ' S  M  P AR\nJetson HP'], fontsize=14)
    axs.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=14)
    axs.set_ylim(bottom=0, top=100)
    axs.legend(temp_bar_list[::-1], ['Camera', 'VIO', 'IMU', 'Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding',], loc='upper left', bbox_to_anchor=(1, 1.05),
            fancybox=False, shadow=False, ncol=1, prop={'size': 12})
    fig.tight_layout()
    fig.subplots_adjust(left=.13, right=.70, bottom=.20)
    fig.set_size_inches(8, 2.5)

    plt.savefig('../Graphs/cpu-breakdown.pdf')

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
    axs.set_xticklabels(['D     HP    LP\nSponza', 'D     HP    LP\nMaterials', 'D     HP    LP\nPlatformer', 'D     HP    LP\nAR Demo'])
    axs.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs.set_ylim(bottom=0, top=100)
    axs.legend(temp_bar_list, ['Application', 'Hologram', 'Reprojection'], loc='upper center', bbox_to_anchor=(0.5, -0.12),
            fancybox=False, shadow=False, ncol=4, prop={'size': 14})
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
    x = np.arange(3) * 2
    width = 0.40

    # One application per group
    sponza_bars = []
    materials_bars = []
    platformer_bars = []
    demo_bars = []
    for idx in range(len(data['sponza-desktop'])):
        if idx < 3:
            sponza_bars.append([data['sponza-desktop'][idx] / data['sponza-desktop'].sum(), data['sponza-jetsonhp'][idx] / data['sponza-jetsonhp'].sum(), data['sponza-jetsonlp'][idx] / data['sponza-jetsonlp'].sum()]) 
            materials_bars.append([data['materials-desktop'][idx] / data['materials-desktop'].sum(), data['materials-jetsonhp'][idx] / data['materials-jetsonhp'].sum(), data['materials-jetsonlp'][idx] / data['materials-jetsonlp'].sum()])
            platformer_bars.append([data['platformer-desktop'][idx] / data['platformer-desktop'].sum(), data['platformer-jetsonhp'][idx] / data['platformer-jetsonhp'].sum(), data['platformer-jetsonlp'][idx] / data['platformer-jetsonlp'].sum()])
            demo_bars.append([data['demo-desktop'][idx] / data['demo-desktop'].sum(), data['demo-jetsonhp'][idx] / data['demo-jetsonhp'].sum(), data['demo-jetsonlp'][idx] / data['demo-jetsonlp'].sum()])
        else:
            sponza_bars.append([0, data['sponza-jetsonhp'][idx] / data['sponza-jetsonhp'].sum(), data['sponza-jetsonlp'][idx] / data['sponza-jetsonlp'].sum()]) 
            materials_bars.append([0, data['materials-jetsonhp'][idx] / data['materials-jetsonhp'].sum(), data['materials-jetsonlp'][idx] / data['materials-jetsonlp'].sum()])
            platformer_bars.append([0, data['platformer-jetsonhp'][idx] / data['platformer-jetsonhp'].sum(), data['platformer-jetsonlp'][idx] / data['platformer-jetsonlp'].sum()])
            demo_bars.append([0, data['demo-jetsonhp'][idx] / data['demo-jetsonhp'].sum(), data['demo-jetsonlp'][idx] / data['demo-jetsonlp'].sum()])
    
    sponza_bars = [[y*100 for y in x] for x in sponza_bars]
    materials_bars = [[y*100 for y in x] for x in materials_bars]
    platformer_bars = [[y*100 for y in x] for x in platformer_bars]
    demo_bars = [[y*100 for y in x] for x in demo_bars]

    sponza_sum = np.zeros(3)
    materials_sum = np.zeros(3)
    platformer_sum = np.zeros(3)
    demo_sum = np.zeros(3)
    colors = ['navy', 'steelblue', 'mediumturquoise', 'lightcoral', 'orange']

    temp_bar_list = []
    for idx in range(len(sponza_bars)):
        temp_bar = axs.bar(x - (width * 1.65), sponza_bars[idx], width, bottom=sponza_sum, color=colors[idx])
        axs.bar(x - (width * 0.55), materials_bars[idx], width, bottom=materials_sum, color=colors[idx])
        axs.bar(x + (width * 0.55), platformer_bars[idx], width, bottom=platformer_sum, color=colors[idx])
        axs.bar(x + (width * 1.65), demo_bars[idx], width, bottom=demo_sum, color=colors[idx])

        temp_bar_list.append(temp_bar)
        sponza_sum += sponza_bars[idx]
        materials_sum += materials_bars[idx]
        platformer_sum += platformer_bars[idx]
        demo_sum += demo_bars[idx]

    axs.set_xticks(x)
    axs.set_xticklabels([' S M P AR\nDesktop', ' S M P AR\nJetson HP', ' S M P AR\nJetson LP'], fontsize=16)
    axs.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=16)
    axs.set_ylim(bottom=0, top=100)
    axs.legend(temp_bar_list, ['CPU\nPower', 'GPU\nPower', 'DDR\nPower', 'SOC\nPower', 'SYS\nPower'], loc='upper left', bbox_to_anchor=(1.01, 1.00),
            fancybox=False, shadow=False, ncol=1, prop={'size': 16}, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(left=.12, right=.72, bottom=0.15)
    fig.set_size_inches(6, 4)

    plt.savefig('../Graphs/power-breakdown.pdf')


def plot_power_total():
    plt.clf()
    data = pd.read_csv('../output/power.csv').transpose()
    fig, axs = plt.subplots()

    new_header = data.iloc[0] #grab the first row for the header
    data = data[1:] #take the data less the header row
    data.columns = new_header #set the header row as the df header
    x = np.arange(3) * 2
    width = 0.40

    sponza_bars = [sum(data['sponza-desktop'][0:3]), sum(data['sponza-jetsonhp']), sum(data['sponza-jetsonlp'])]
    materials_bars = [sum(data['materials-desktop'][0:3]), sum(data['materials-jetsonhp']), sum(data['materials-jetsonlp'])]
    platformer_bars = [sum(data['platformer-desktop'][0:3]), sum(data['platformer-jetsonhp']), sum(data['platformer-jetsonlp'])]
    demo_bars = [sum(data['demo-desktop'][0:3]), sum(data['demo-jetsonhp']), sum(data['demo-jetsonlp'])]

    bars = []
    bars.append(axs.bar(x - (width * 1.65), sponza_bars, width, color='darkred'))
    bars.append(axs.bar(x - (width * 0.55), materials_bars, width, color='indigo'))
    bars.append(axs.bar(x + (width * 0.55), platformer_bars, width, color='royalblue'))
    bars.append(axs.bar(x + (width * 1.65), demo_bars, width, color='green'))

    axs.set_xticks(x)
    axs.set_xticklabels(['S  M  P AR\nDesktop', 'S  M  P AR\nJetson HP', 'S  M  P AR\nJetson LP'], fontsize=18)
    axs.set_yticklabels([10, 100, 1000], fontsize=16)
    axs.set_yscale('log')
    axs.set_ylabel('Power (Watts)', fontsize=16)
    axs.set_ylim(bottom=1, top=1000)
    axs.legend(bars, ['Sponza', 'Materials', 'Platformer', 'AR Demo'], loc='upper right',
            fancybox=False, shadow=False, ncol=1, prop={'size': 17}, fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(left=.15, right=.95, bottom=.15)
    fig.set_size_inches(6, 4)

    plt.savefig('../Graphs/power-total.pdf')


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

    axs.legend(['Jetson LP', 'Jetson HP', 'Desktop'], ncol=3, loc='upper center', prop={'size': 20})
    axs.axes.xaxis.set_visible(False)
    axs.set_ylabel('Time (ms)', fontsize=24)

    fig.set_size_inches(12, 3)
    plt.yticks(fontsize=20)
    fig.subplots_adjust(left=.08, right=.98, bottom=.05, top=.95)

    plt.xlim(left=200, right=max_val)
    plt.ylim(bottom=0, top=max_y)

    plt.savefig('../Graphs/mtp-' + title + '.pdf')


def plot_wall_time(trials):
    plt.clf()
    fig, axs = plt.subplots()

    for trial in tqdm(trials):
        if trial.conditions.machine == 'desktop' and trial.conditions.application == 'platformer':
            account_names = ['zed_camera_thread iter', 'zed_imu_thread iter', 'imu_integrator iter', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']
            
            max_val = 0
            wall_data = []
            for name in account_names:
                ts_temp = trial.ts.reset_index()
                if name == 'zed_camera_thread iter':
                    data = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'].to_numpy()
                else:
                    data = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][200:].to_numpy()
                wall_data.append(data)

                if len(data) > max_val:
                    max_val = len(data)

            colors = ['saddlebrown', 'indigo', 'lightcoral', 'navy', 'firebrick', 'yellowgreen']
            for idx, data in enumerate(wall_data):
                axs.plot(np.arange(len(data)) * (max_val / len(data)), data, color=colors[idx])


            axs.legend(['Camera', 'IMU', 'Integrator', 'Reprojection', 'Playback', 'Encoding'], loc='upper center', bbox_to_anchor=(0.5, 0.06),
                    fancybox=False, shadow=False, ncol=3, prop={'size': 20})
            axs.set_ylabel('Time (ms)', fontsize=24)

            fig.set_size_inches(12, 3)
            axs.axes.xaxis.set_visible(False)
            fig.subplots_adjust(left=.10, right=.98, bottom=.30, top=.95)

            plt.yticks(fontsize=20)
            plt.ylim(bottom=0, top=2)
            plt.savefig('../Graphs/timeseries-platformer-desktop-2.pdf')

    plt.clf()
    fig, axs = plt.subplots()

    for trial in tqdm(trials):
        if trial.conditions.machine == 'desktop' and trial.conditions.application == 'platformer':
            account_names = ['OpenVINS Camera', 'app']
            
            max_val = 0
            wall_data = []
            for name in account_names:
                ts_temp = trial.ts.reset_index()
                if name == 'OpenVINS Camera':
                    data = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'].to_numpy()
                else:
                    data = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][200:].to_numpy()
                wall_data.append(data)

                if len(data) > max_val:
                    max_val = len(data)

            colors = ['steelblue', 'orange']
            for idx, data in enumerate(wall_data):
                axs.plot(np.arange(len(data)) * (max_val / len(data)), data, color=colors[idx])


            axs.legend(['VIO', 'App'], ncol=2, loc='upper center', prop={'size': 20}, fontsize=64)
            axs.set_ylabel('Time (ms)', fontsize=24)

            fig.set_size_inches(12, 4)
            axs.axes.xaxis.set_visible(False)
            fig.subplots_adjust(left=.10, right=.98, bottom=.05, top=.95)

            plt.yticks(fontsize=20)
            plt.ylim(bottom=5, top=25)
            plt.savefig('../Graphs/timeseries-platformer-desktop-1.pdf')


def plot_frame_time(trial_num):
    plt.clf()
    mean_data = pd.read_csv('../output/frame_time_mean.csv')
    std_data = pd.read_csv('../output/frame_time_std.csv')
    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 2, 2]})

    def autolabel(rects, ax, sd, mean):
        for idx, rect in enumerate(rects):
            ax.annotate('%.2f'%(sd[idx]),
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=10, rotation=90)

    # Yes this is gross i know and im sorry DX
    def create_row(data, ax_row, trials):
        width = 0.35  # the width of the bars

        labels = ['Camera','VIO']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][0:2]
        materials_fps = data[trials[1]][0:2]
        platformer_fps = data[trials[2]][0:2]
        demo_fps = data[trials[3]][0:2]

        s_bar = ax_row[0].bar(x - (width * 2.025), sponza_fps, width, label='Sponza', color='darkred')
        autolabel(s_bar, ax_row[0], std_data[trials[0]][0:2].to_numpy(), sponza_fps.to_numpy())
        m_bar = ax_row[0].bar(x - (width * 0.675), materials_fps, width, label='Materials', color='indigo')
        autolabel(m_bar, ax_row[0], std_data[trials[1]][0:2].to_numpy(), materials_fps.to_numpy())
        p_bar = ax_row[0].bar(x + (width * 0.675), platformer_fps, width, label='Platformer', color='royalblue')
        autolabel(p_bar, ax_row[0], std_data[trials[2]][0:2].to_numpy(), platformer_fps.to_numpy())
        a_bar = ax_row[0].bar(x + (width * 2.025), demo_fps, width, label='AR Demo', color='green')
        autolabel(a_bar, ax_row[0], std_data[trials[3]][0:2].to_numpy(), demo_fps.to_numpy())

        labels = ['IMU', 'Integrator']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][2:4]
        materials_fps = data[trials[1]][2:4]
        platformer_fps = data[trials[2]][2:4]
        demo_fps = data[trials[3]][2:4]

        s_bar = ax_row[1].bar(x - (width * 2.025), sponza_fps, width, label='Sponza', color='darkred')
        autolabel(s_bar, ax_row[1], std_data[trials[0]][2:4].to_numpy(), sponza_fps.to_numpy())
        m_bar = ax_row[1].bar(x - (width * 0.675), materials_fps, width, label='Materials', color='indigo')
        autolabel(m_bar, ax_row[1], std_data[trials[1]][2:4].to_numpy(), materials_fps.to_numpy())
        p_bar = ax_row[1].bar(x + (width * 0.675), platformer_fps, width, label='Platformer', color='royalblue')
        autolabel(p_bar, ax_row[1], std_data[trials[2]][2:4].to_numpy(), platformer_fps.to_numpy())
        a_bar = ax_row[1].bar(x + (width * 2.025), demo_fps, width, label='AR Demo', color='green')
        autolabel(a_bar, ax_row[1], std_data[trials[3]][2:4].to_numpy(), demo_fps.to_numpy())

        labels = ['Application', 'Reprojection']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][4:6]
        materials_fps = data[trials[1]][4:6]
        platformer_fps = data[trials[2]][4:6]
        demo_fps =  data[trials[3]][4:6]

        s_bar = ax_row[2].bar(x - (width * 2.025), sponza_fps, width, label='Sponza', color='darkred')
        autolabel(s_bar, ax_row[2], std_data[trials[0]][4:6].to_numpy(), sponza_fps.to_numpy())
        m_bar = ax_row[2].bar(x - (width * 0.675), materials_fps, width, label='Materials', color='indigo')
        autolabel(m_bar, ax_row[2], std_data[trials[1]][4:6].to_numpy(), materials_fps.to_numpy())
        p_bar = ax_row[2].bar(x + (width * 0.675), platformer_fps, width, label='Platformer', color='royalblue')
        autolabel(p_bar, ax_row[2], std_data[trials[2]][4:6].to_numpy(), platformer_fps.to_numpy())
        a_bar = ax_row[2].bar(x + (width * 2.025), demo_fps, width, label='AR Demo', color='green')
        autolabel(a_bar, ax_row[2], std_data[trials[3]][4:6].to_numpy(), demo_fps.to_numpy())

        labels = ['Playback', 'Encoding']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][6:8]
        materials_fps = data[trials[1]][6:8]
        platformer_fps = data[trials[2]][6:8]
        demo_fps = data[trials[3]][6:8]

        s_bar = ax_row[3].bar(x - (width * 2.025), sponza_fps, width, label='Sponza', color='darkred')
        autolabel(s_bar, ax_row[3], std_data[trials[0]][6:8].to_numpy(), sponza_fps.to_numpy())
        m_bar = ax_row[3].bar(x - (width * 0.675), materials_fps, width, label='Materials', color='indigo')
        autolabel(m_bar, ax_row[3], std_data[trials[1]][6:8].to_numpy(), materials_fps.to_numpy())
        p_bar = ax_row[3].bar(x + (width * 0.675), platformer_fps, width, label='Platformer', color='royalblue')
        autolabel(p_bar, ax_row[3], std_data[trials[2]][6:8].to_numpy(), platformer_fps.to_numpy())
        a_bar = ax_row[3].bar(x + (width * 2.025), demo_fps, width, label='AR Demo', color='green')
        autolabel(a_bar, ax_row[3], std_data[trials[3]][6:8].to_numpy(), demo_fps.to_numpy())

    # Uncomment one of these at a time depending on if you want to gen Desktop/Jetson HP/Jetson LP FPS values
    if trial_num == 0:
        create_row(mean_data, axes, ['sponza-desktop', 'materials-desktop', 'platformer-desktop', 'demo-desktop'])
    elif trial_num == 1: 
        create_row(mean_data, axes, ['sponza-jetsonhp', 'materials-jetsonhp', 'platformer-jetsonhp', 'demo-jetsonhp'])
    else:
        create_row(mean_data, axes, ['sponza-jetsonlp', 'materials-jetsonlp', 'platformer-jetsonlp', 'demo-jetsonlp'])

    if trial_num == 0:
        axes[3].legend(loc='lower left', prop={'size': 10}, fontsize=14)

    labels = ['Camera','VIO']
    axes[0].set_xticks(np.arange(len(labels)) * 2)
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_xlabel('Perception (Camera)', fontsize=14)
    ylim_val = axes[0].get_ylim()
    axes[0].set_ylim(top=ylim_val[1] * 1.3)
    axes[0].axhline(y=66.666, xmin=0.02, xmax=0.98, color='r')
    axes[0].set_ylabel('Time (ms)', fontsize=14)

    labels = ['IMU', 'Integrator']
    axes[1].set_xticks(np.arange(len(labels)) * 2)
    axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_xlabel('Perception (IMU)', fontsize=14)
    ylim_val = axes[1].get_ylim()
    axes[1].set_ylim(top=ylim_val[1] * 1.3)
    axes[1].axhline(y=2, xmin=0.02, xmax=0.98, color='r')

    labels = ['App', 'Reprojection']
    axes[2].set_xticks(np.arange(len(labels)) * 2)
    axes[2].set_xticklabels(labels, fontsize=11)
    axes[2].set_xlabel('Visual', fontsize=14)
    ylim_val = axes[2].get_ylim()
    axes[2].set_ylim(top=ylim_val[1] * 1.3)
    axes[2].axhline(y=8.3333, xmin=0.02, xmax=0.98, color='r')

    labels = ['Playback', 'Encoding']
    axes[3].set_xticks(np.arange(len(labels)) * 2)
    axes[3].set_xticklabels(labels, fontsize=11)
    axes[3].set_xlabel('Audio', fontsize=14)
    ylim_val = axes[3].get_ylim()
    axes[3].set_ylim(top=ylim_val[1] * 1.3)
    axes[3].axhline(y=20.83, xmin=0.02, xmax=0.98, color='r')

    fig.set_size_inches(8, 2.5)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.30, hspace=.18)

    if trial_num == 0:
        plt.savefig('../Graphs/time-desktop.pdf')
    elif trial_num == 1:
        plt.savefig('../Graphs/time-jhp.pdf')
    else:
        plt.savefig('../Graphs/time-jlp.pdf')


# Lol
def plot_cpu_ipc():
    plt.clf()
    fig, axs = plt.subplots()

    x = np.arange(7)
    width = 0.40

    a = [60.6, 35.6, 41, 11.2, 21.8, 69.4, 85.6]
    b = [5.1, 5.3, 4.9, 9.3, 2.6, 0.3, 1.5]
    c = [12.4, 23.1, 16.3, 49.6, 12.2, 9.3, 0.9]
    d = [21.9, 36, 37.8, 29.9, 63.5, 21, 12]

    sums = []
    for idx in range(len(a)):
        sums.append(a[idx] + b[idx] + c[idx] + d[idx])

    for idx in range(len(a)):
        a[idx] /= sums[idx]
        b[idx] /= sums[idx]
        c[idx] /= sums[idx]
        d[idx] /= sums[idx]

        a[idx] *= 100
        b[idx] *= 100
        c[idx] *= 100
        d[idx] *= 100

    e = [2.22, 1.24, 1.61, 0.34, 0.88, 2.46, 3.50]
    e = [(x / 4) * 100 for x in e]

    bars = []
    bars.append(axs.bar(x, a, width, color='yellowgreen'))
    bars.append(axs.bar(x, b, width, bottom=a, color='gold'))
    bars.append(axs.bar(x, c, width, bottom=[sum(x) for x in zip(a, b)], color='orange'))
    bars.append(axs.bar(x, d, width, bottom=[sum(x) for x in zip(a, b, c)], color='firebrick'))
    bars.append(axs.bar(x, d, 0, color='black'))

    x0, y0 = [0, 1], e[0:2]
    x1, y1 = [1, 2], e[1:3]
    x2, y2 = [2, 3], e[2:4]
    x3, y3 = [3, 4], e[3:5]
    x4, y4 = [4, 5], e[4:6]
    x5, y5 = [5, 6], e[5:7]
    axs.plot(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, color='black')

    axs.set_xticks(x)
    axs.set_xticklabels(['VIO', 'Eye\nTracking', 'Scene\nReconst.', 'Reproj.', 'Hologram', 'Audio\nEncoding', 'Audio\nPlayback'], fontsize=13)
    axs.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=14)
    axs.set_ylabel('Cycle Breakdown (%)')
    axs.set_ylim(bottom=0, top=100)

    axs2 = axs.twinx()
    axs2.set_ylabel('IPC')
    axs2.set_yticklabels([0, 1, 2, 3, 4], fontsize=14)
    axs2.set_ylim(bottom=0, top=4.01)

    axs.legend(bars, ['Retiring', 'Bad Speculation', 'Frontend Bound', 'Backend Bound', 'IPC'], loc='upper center', bbox_to_anchor=(0.46, -.27),
            fancybox=False, shadow=False, ncol=5, prop={'size': 11})
    fig.tight_layout()
    fig.subplots_adjust(left=.13, right=.93, bottom=.33)
    fig.set_size_inches(8, 2.5)

    plt.savefig('../Graphs/microarchitecture.pdf')