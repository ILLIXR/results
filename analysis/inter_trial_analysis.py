from util import PerTrialData
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import charmonium.time_block as ch_time_block

def analysis(trials: List[PerTrialData]) -> None:
    for trial in trials:
        # make sure all trials  the same ILLIXR commit
        assert trial.conditions.illixr_commit == trials[0].conditions.illixr_commit, (
            f"{trial.conditions} is from a different commit than {trials[0].conditions}"
        )
        # make sure all trials have the same components
        assert set(trial.summaries.index) == set(trials[0].summaries.index)

    trials_df = pd.concat({
        (trial.conditions.machine, trial.conditions.application): trial.summaries
        for trial in trials
    }, names=["machine", "app"]).sort_index()

    # trials_df is a DataFrame,
    # whose index is (machine, application, account_name)
    # whose columns are `["period_mean", "period_std", ..., *rows in trial.summaries]`

    # Access them using `trials_df.loc[(machine_name, application_name, account_name), column_name]`
    # `slice(None)` for the values in the index tuple refers to "all" values of that index.
    # If an index at the end is omitted, is implicitly `slice(None)`.
    # `:` for the column_name refers to all columns
    # If the column is :, that is equivalent to `slice(None)`

    # For example:
    #     trials_df.loc[(slice(None), application_name), "period_mean"]
    # returns a Series whose index is (machine_name, account_name) and whose values are from "period_mean"

    print("\U0001f600")
    plot_freq(trials_df)
    # populate_cpu(trials)
    # populate_gpu(trials)
    # populate_power(trials)
    # populate_mtp(trials)
    # populate_frame_time_mean(trials)
    # populate_frame_time_std(trials)
    # populate_frame_time_min(trials)
    # populate_frame_time_max(trials)


ms_to_s = 1e-3

@ch_time_block.decor(print_start=False, print_args=False)   
def plot_freq(trials_df: pd.DataFrame) -> None:
    # Parameters:
    account_groups = [
        (15 , 3  , "Perception (Camera)", ["Camera"  , "VIO"         ,]),
        (500, 100, "Perception (IMU)"   , ["IMU"     , "Integrator"  ,]),
        (120, 15 , "Visual"             , ["App"     , "Reprojection",]),
        (48 , 6  , "Audio"              , ["Playback", "Encoding"    ,]),
    ]
    colors = "darkred indigo royalblue green".split(" ")
    bar_width = 0.35
    intragroup_gap = 0.2 * bar_width
    intergroup_gap = 1.1 * bar_width
    size_inches = (8, 2.5)
    ylabel_fontsize = 14
    xlabel_fontsize = 11
    n_yticks = 6
    ylabel = "Rate (Hz)"
    subplots_adjust = dict(wspace=.30, hspace=.18)
    width_ratios = [2, 2, 2, 2]
    output_path = Path() / ".." / "Graphs" / "freq" 


    trials_df = trials_df.copy()
    # Omitting all indices (passing empty-tuple) means "all indices"
    trials_df.loc[(), "freq"] = 1 / (ms_to_s * trials_df.loc[(), "period_mean"])

    output_path.mkdir(exist_ok=True, parents=True)

    for machine in trials_df.index.levels[0]:
        fig, axes = plt.subplots(1, len(account_groups), gridspec_kw=dict(width_ratios=width_ratios))
        axes[0].set_ylabel(ylabel, fontsize=ylabel_fontsize)
        for ax, (ymax, ystep, title, accounts) in zip(axes, account_groups):
            apps = list(trials_df.loc[(machine,), :].index.levels[0])
            account_width = sum([
                bar_width * len(apps),
                intragroup_gap * (len(apps) - 1),
                intergroup_gap,
            ])
            account_centers = np.arange(len(accounts)) * account_width
            ax.set_xticks(account_centers)
            ax.set_xticklabels(accounts, fontsize=xlabel_fontsize)
            ax.set_xlabel(title, fontsize=ylabel_fontsize)
            ax.set_yticks(np.arange(n_yticks) * ystep)
            ax.set_ylim(0, ymax + 0.01)
            for account_no, (account_center, account) in enumerate(zip(account_centers, accounts)):
                for app_no, (color, app) in enumerate(zip(colors, apps)):
                    bar_left = account_center + (app_no - len(apps) / 2) * (bar_width + intragroup_gap) + intergroup_gap * 0.5
                    bar_height = trials_df.loc[(machine, app, account), "freq"]
                    print(bar_left, bar_height / ymax)
                    ax.bar(bar_left, bar_height, width=bar_width, label=app, color=color)

        fig.set_size_inches(*size_inches)
        fig.tight_layout()
        fig.subplots_adjust(**subplots_adjust)
        fig.savefig(output_path/ f"{machine}.pdf")

@ch_time_block.decor(print_start=False, print_args=False)   
def populate_frame_time_mean(trials: List[PerTrialData]) -> None:
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        account_names = ['zed_camera_thread iter', 'OpenVINS Camera',  'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            if name == 'audio_decoding iter' or name == 'audio_encoding iter':
                # First ~200 values seem to be garbage so omit those when calculating the mean
                ts_temp = trial.ts.reset_index()
                if trial.conditions.machine == 'jetsonlp':
                    mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][150:].mean() 
                elif trial.conditions.machine == 'jetsonhp':
                    mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][100:].mean() 
                else:
                    mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][60:].mean() 
            else:
                ts_temp = trial.ts.reset_index()
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'].mean() 

            values.append(mean)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_mean.csv', index=False)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_frame_time_std(trials: List[PerTrialData]) -> None:
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        account_names = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            if name == 'audio_decoding iter' or name == 'audio_encoding iter':
                # First ~200 values seem to be garbage so omit those when calculating the mean
                ts_temp = trial.ts.reset_index()
                if trial.conditions.machine == 'jetsonlp':
                    mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][150:].std() 
                elif trial.conditions.machine == 'jetsonhp':
                    mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][100:].std() 
                else:
                    mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][60:].std() 
            else:
                ts_temp = trial.ts.reset_index()
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'].std() 

            values.append(mean)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_std.csv', index=False)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_frame_time_min(trials: List[PerTrialData]) -> None:
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        account_names = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            ts_temp = trial.ts.reset_index()
            if trial.conditions.machine == 'jetsonlp':
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][150:].min() 
            elif trial.conditions.machine == 'jetsonhp':
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][100:].min() 
            else:
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][60:].min() 

            values.append(mean)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_min.csv', index=False)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_frame_time_max(trials: List[PerTrialData]) -> None:
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        account_names = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            ts_temp = trial.ts.reset_index()
            if trial.conditions.machine == 'jetsonlp':
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][150:].max() 
            elif trial.conditions.machine == 'jetsonhp':
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][100:].max() 
            else:
                mean = ts_temp[ts_temp["account_name"] == name]['wall_time_duration'][60:].max() 

            values.append(mean)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_max.csv', index=False)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_cpu(trials: List[PerTrialData]) -> None:
    account_names = trials[0].ts.index.levels[0]
    ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
    account_list = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

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
def populate_gpu(trials: List[PerTrialData]) -> None:
    account_names = trials[0].ts.index.levels[0]
    account_list = ['app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu']
    account_list = [replaced_names[name] if name in replaced_names else name for name in account_list]
    account_list.insert(0, "Run Name")
    data = pd.DataFrame([], columns=account_list)

    for trial in tqdm(trials):
        account_names = trial.ts.index.levels[0]

        values = {"Run Name": trial.conditions.application + '-'+ trial.conditions.machine}
        name_list = ['app_gpu1', 'app_gpu2', 'timewarp_gl gpu']
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
def populate_power(trials: List[PerTrialData]) -> None:
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
def populate_mtp(trials: List[PerTrialData]) -> None:
    for trial in tqdm(trials):
        trial.mtp.to_csv(trial.output_path / "mtp.csv", index=False)

    account_list = ['Mean', 'Std Dev']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        values = [trial.mtp['imu_to_display'][200:].mean(), trial.mtp['imu_to_display'][200:].std()]
        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values

    data_frame.to_csv('../output/MTP_Vals.csv', index=False)

