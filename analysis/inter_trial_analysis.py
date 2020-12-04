from util import PerTrialData
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import charmonium.time_block as ch_time_block

def analysis(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    # populate_fps(trials, replaced_names)
    populate_cpu(trials, replaced_names)
    populate_gpu(trials, replaced_names)
    populate_power(trials, replaced_names)
    # populate_mtp(trials, replaced_names)

@ch_time_block.decor(print_start=False, print_args=False)   
def populate_fps(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    # account_names = trials[0].ts.index.levels[0]
    # ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'app']
    # account_list = [name for name in account_names if name not in ignore_list]
    # account_list.append('app') 
    # account_list = [replaced_names[name] if name in replaced_names else name for name in account_list]
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection', 'Hologram', 'Playback', 'Encoding']
    data = pd.DataFrame()
    data["Components"] = account_list

    for trial in tqdm(trials):
        account_names = trial.ts.index.levels[0]

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            values.append(trial.summaries["period_mean"][name])
        # values.append(trial.summaries["period_mean"]['app'])

        print(values)
        data[trial.conditions.application + '-' + trial.conditions.machine] = values

    data = data.reset_index(drop=True).transpose()

    fig, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 3, 2]})

    def create_row(data: pd.DataFrame, ax_row: List[plt.Axes], trials: List[str]) -> None:
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

        labels = ['Applicaton', 'Reprojection', 'Hologram']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][4:7]
        platformer_fps = data[trials[1]][4:7]
        materials_fps = data[trials[2]][4:7]
        demo_fps = data[trials[3]][4:7]

        ax_row[2].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[2].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[2].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[2].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

        labels = ['Playback', 'Encoding']
        x = np.arange(len(labels)) * 2  # the label locations
        sponza_fps = data[trials[0]][7:9]
        platformer_fps = data[trials[1]][7:9]
        materials_fps = data[trials[2]][7:9]
        demo_fps = data[trials[3]][7:9]

        ax_row[3].bar(x - (width * 1.5), sponza_fps, width, label='Sponza')
        ax_row[3].bar(x - (width * 0.5), platformer_fps, width, label='Platformer')
        ax_row[3].bar(x + (width * 0.5), materials_fps, width, label='Materials')
        ax_row[3].bar(x + (width * 1.5), demo_fps, width, label='AR Demo')

    # Uncomment one of these at a time depending on if you want to gen Desktop/Jetson HP/Jetson LP FPS values
    create_row(data, axes, ['sponza-desktop', 'platformer-desktop', 'materials-desktop', 'demo-desktop'])
    # create_row(data, axes[1], ['sponza-jetsonhp', 'platformer-jetsonhp', 'materials-jetsonhp', 'demo-jetsonhp'])
    # create_row(data, axes[2], ['sponza-jetsonlp', 'platformer-jetsonlp', 'materials-jetsonlp', 'demo-jetsonlp'])

    axes[0].legend()

    labels = ['Camera','OpenVINS Camera']
    axes[0].set_xticks(np.arange(len(labels)) * 2)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel('Perception Pipeline (Camera)', fontsize=18)
    labels = ['IMU', 'IMU Integrator']
    axes[1].set_xticks(np.arange(len(labels)) * 2)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('Perception Pipeline (IMU)', fontsize=18)
    labels = ['Applicaton', 'Reprojection', 'Hologram']
    axes[2].set_xticks(np.arange(len(labels)) * 2)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel('Visual Pipeline', fontsize=18)
    labels = ['Playback', 'Encoding']
    axes[3].set_xticks(np.arange(len(labels)) * 2)
    axes[3].set_xticklabels(labels)
    axes[3].set_xlabel('Audio Pipeline', fontsize=18)

    fig.set_size_inches(16, 4)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.1, hspace=.1)

    fig.savefig(Path() / ".." / "output" / "fps.png")
    plt.close()


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_cpu(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_names = trials[0].ts.index.levels[0]
    ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'app']
    account_list = [name for name in account_names if name not in ignore_list]
    account_list.append('app') 
    account_list = [replaced_names[name] if name in replaced_names else name for name in account_list]
    account_list.insert(0, "Run Name")
    data = pd.DataFrame([], columns=account_list)

    for trial in tqdm(trials):
        account_names = trial.ts.index.levels[0]

        values = {"Run Name": trial.conditions.application + '-'+ trial.conditions.machine}
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue

            formatted_name = replaced_names[name] if name in replaced_names else name
            values.update({formatted_name: trial.summaries["cpu_time_duration_sum"][name]})
        values.update({"Application": trial.summaries["cpu_time_duration_sum"]['app']})

        data = data.append(values, ignore_index=True, sort=False)
        # from IPython import embed; embed()
    data = data.reset_index(drop=True).transpose()
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
    colors = ['tab:blue', 'lightcoral', 'tab:orange', 'tab:green', 'gold', 'tab:red', 'tab:brown', 'tab:purple', 'tab:pink', 'skyblue']

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
    axs.legend(temp_bar_list, ['OpenVINS Camera', 'OpenVINS IMU', 'Playback', 'Encoding', 'Hologram', 'IMU Integrator', 'Reprojection', 'Camera', 'IMU', 'Application'], loc='upper center', bbox_to_anchor=(0.5, -0.07),
            fancybox=False, shadow=False, ncol=5)
    fig.tight_layout()
    fig.subplots_adjust(left=.05, right=.95, bottom=.2)
    fig.set_size_inches(8, 4.5)
    fig.savefig(Path() / ".." / "output" / "cpu.png")
    plt.close()

@ch_time_block.decor(print_start=False, print_args=False)   
def populate_gpu(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_names = trials[0].ts.index.levels[0]
    account_list = ['app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu']
    account_list = [replaced_names[name] if name in replaced_names else name for name in account_list]
    account_list.insert(0, "Run Name")
    data = pd.DataFrame([], columns=account_list)

    for trial in tqdm(trials):
        account_names = trial.ts.index.levels[0]

        values = {"Run Name": trial.conditions.application + '-'+ trial.conditions.machine}
        name_list = ['app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu']
        for idx, name in enumerate(name_list):

            formatted_name = replaced_names[name] if name in replaced_names else name
            values.update({formatted_name: trial.summaries["gpu_time_duration_sum"][name]})

        data = data.append(values, ignore_index=True, sort=False)
        # from IPython import embed; embed()

    data = data.reset_index(drop=True).transpose()

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

    fig.savefig(Path() / ".." / "output" / "fps.png")
    plt.close()

@ch_time_block.decor(print_start=False, print_args=False)
def populate_power(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_names = trials[0].ts.index.levels[0]
    account_list = ['CPU Power', 'GPU Power', 'DDR Power', 'SOC Power', 'SYS Power']
    account_list.insert(0, "Run Name")
    data_frame = pd.DataFrame([], columns=account_list)

    for trial in tqdm(trials):
        account_names = trial.ts.index.levels[0]

        if len(trial.power_data) == 4:
            gpu_power = trial.power_data[0]
            cpu_time = trial.power_data[1]
            cpu_energy = trial.power_data[2]
            ddr_energy = trial.power_data[3]

            cpu_power = cpu_energy / cpu_time
            ddr_power = ddr_energy / cpu_time
            values = {"Run Name": trial.conditions.application + '-'+ trial.conditions.machine, 'GPU Power': gpu_power, 'CPU Power': cpu_power, 'DDR Power': ddr_power}
            data_frame = data_frame.append(values, ignore_index=True, sort=False)
        else:
            values = {"Run Name": trial.conditions.application + '-'+ trial.conditions.machine, 'GPU Power': trial.power_data[1], 'DDR Power': trial.power_data[2], 'CPU Power': trial.power_data[3], 'SOC Power': trial.power_data[4], 'SYS Power': trial.power_data[5]}
            data_frame = data_frame.append(values, ignore_index=True, sort=False)

        # from IPython import embed; embed()

    data_frame.to_csv('../output/power.csv', index=False)

@ch_time_block.decor(print_start=False, print_args=False)    
def populate_mtp(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    for trial in tqdm(trials):
        trial.mtp.to_csv(trial.output_path / "mtp.csv", index=False)

