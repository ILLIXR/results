import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import NamedTuple
from util import list_concat
from typing import List
from warnings import WarningMessage
clocks = ["cpu", "wall", "gpu"]
class PerTrialData(NamedTuple):
    ts: pd.DataFrame
    summaries: pd.DataFrame
    thread_ids: pd.DataFrame
    output_path: Path
    switchboard_topic_stop: pd.DataFrame
    mtp: pd.DataFrame
    warnings_log: List[WarningMessage]
    
def analysis(data: PerTrialData) -> None:
    table_summaries(data)
    stacked_cpu_time(data)
    stacked_gpu_time(data)
    stacked_energy(data)
    time_series(data)
    account_time_series(data)
    #motion_to_photon(data)
    cpu_timeline(data)
    
def table_summaries(data: PerTrialData) -> None:
    with (data.output_path / "account_summaries.md").open("w") as f:
        f.write("# Summaries\n\n")
        columns = ["count"] + [col for col in data.summaries.columns if col != "count"]
        floatfmt = ["", ".0f", ".1f", ".1f"] + list_concat(["e", "e", "e"] for clock in clocks)
        f.write(data.summaries[columns].to_markdown(floatfmt=floatfmt))
        f.write("\n\n")
        f.write("# Totals\n\n")
        f.write(data.summaries[["cpu_time_duration_sum", "gpu_time_duration_sum"]].sum().to_markdown())
        f.write("\n\n")
        f.write("# Thread IDs (of long-running threads)\n\n")
        f.write(data.thread_ids.sort_values(["name", "sub_name"]).to_markdown())
        f.write("\n\n")
        f.write("# Switchboard topic stops\n\n")
        f.write(data.switchboard_topic_stop.to_markdown(floatfmt=["", ".0f", ".0f", ".2f"]))
        f.write("\n\n")
        f.write("# Notes\n\n")
        f.write("- All times are in milliseconds unless otherwise mentioned.\n")
        f.write("- Total wall runtime = {:.1f} sec\n".format(
            (data.ts.loc["timewarp_gl iter", "wall_time_stop"].max() - data.ts.loc["timewarp_gl iter", "wall_time_start"].min()) / 1e3
        ))
        f.write("\n\n")
        f.write("# Warnings\n\n")
        for warning in data.warnings_log:
            f.write(f"- {warning.filename}:{warning.lineno} {warning.message}\n\n")
        f.write("\n\n")
        columns = ["period"] + [col for col in data.ts.columns if col != "period"]
        account_names = data.ts.index.levels[0]
        for account_name in account_names:
            f.write(f"# {account_name}\n\n")
            df = data.ts.loc[account_name]
            f.write(pd.concat([df.head(20), df.tail(20)]).to_markdown())
            f.write("\n\n")
      
def stacked_cpu_time(data: PerTrialData) -> None:
    replaced_names = {
        'app': 'Application',
        'zed_imu_thread iter': 'IMU',
        'zed_camera_thread iter': 'Camera',
        'timewarp_gl iter': 'Reprojection',
        'hologram iter': 'Hologram',
        'audio_encoding iter': 'Encoding',
        'audio_decoding iter': 'Playback',
        # GPU Values
        'timewarp_gl gpu': 'Reprojection',
        'hologram': 'Hologram',
    }
    # Stacked graphs
    total_cpu_time = 0.0
    plt.rcParams.update({'font.size': 8})
    # App is only in this list because we want to make it appear at the top of the graph
    ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'app']
    account_names = data.ts.index.levels[0]
    for account_name in account_names:
        if account_name in ignore_list:
            continue
        total_cpu_time += data.summaries["cpu_time_duration_sum"][account_name]
    total_cpu_time += data.summaries["cpu_time_duration_sum"]['app']
    width = 0.4
    bar_plots = []
    rolling_sum = 0.0
    for idx, name in enumerate(account_names):
        if name in ignore_list:
            continue
        bar_height = data.summaries["cpu_time_duration_sum"][name]
        bar_plots.append(plt.bar(1, bar_height, width=width, bottom=rolling_sum)[0])
        rolling_sum += bar_height
    # This is only because we want the app section at the top
    bar_height = data.summaries["cpu_time_duration_sum"]['app']
    bar_plots.append(plt.bar(1, bar_height, width=width, bottom=rolling_sum)[0])
    rolling_sum += bar_height
    plt.title('CPU Time Breakdown Per Run')
    plt.xticks(np.arange(0, 1, step=1))
    plt.yticks(np.arange(0, rolling_sum+1, rolling_sum / 10))
    plt.ylabel('Total CPU Time')
    plt.subplots_adjust(right=0.7)
    account_list = [name for name in account_names if name not in ignore_list]
    account_list.append('app')
    account_list = [replaced_names[name] if name in replaced_names else name for name in account_list]
    plt.legend([x for x in bar_plots][::-1], account_list[::-1], bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.xlabel("Full System")
    plt.savefig(data.output_path / "stacked.png")
    plt.close()
  
  
def stacked_gpu_time(data: PerTrialData) -> None:
    gpu_list = ['app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu']
    total_gpu_time = 0.0
    account_names = data.ts.index.levels[0]
    for account_name in account_names:
        if account_name not in gpu_list:
            continue
        total_gpu_time += data.summaries["gpu_time_duration_sum"][account_name]
    plt.clf()
    width = 0.4
    bar_plots = []
    app_num = data.summaries["gpu_time_duration_sum"]["app_gpu1"] + data.summaries["gpu_time_duration_sum"]["app_gpu2"]
    bar_plots.append(plt.bar(1, data.summaries["gpu_time_duration_sum"]["hologram"], width=width, bottom=0, color="brown")[0])
    bar_plots.append(plt.bar(1, data.summaries["gpu_time_duration_sum"]["timewarp_gl gpu"], width=width, bottom=data.summaries["gpu_time_duration_sum"]["hologram"])[0])
    bar_plots.append(plt.bar(1, app_num, width=width, bottom=data.summaries["gpu_time_duration_sum"]["timewarp_gl gpu"] + data.summaries["gpu_time_duration_sum"]["hologram"])[0])
    plt.title('GPU Time Breakdown Per Run')
    plt.xticks(np.arange(0, 1, step=1))
    rolling_sum = app_num + data.summaries["gpu_time_duration_sum"]["timewarp_gl gpu"] + data.summaries["gpu_time_duration_sum"]["hologram"]
    plt.yticks(np.arange(0, rolling_sum+1, rolling_sum/10))
    plt.ylabel('Total GPU Time')
    plt.subplots_adjust(right=0.7)
    account_list = ['Hologram', 'Reprojection', 'Application']
    plt.legend([x for x in bar_plots][::-1], account_list[::-1], bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.xlabel("Full System")
    plt.savefig(data.output_path / "stacked_gpu.png")
    plt.close()
    
def stacked_energy(data: PerTrialData) -> None:
    #plt.clf()
    #gpu_energy = gpu_power * cpu_time
    #total_energy = cpu_energy + gpu_energy
    #width = 0.4
    #bar_plots = []
    #bar_plots.append(plt.bar(1, cpu_energy/total_energy, width=width, bottom=0)[0])
    #bar_plots.append(plt.bar(1, gpu_energy, width=width, bottom= cpu_energy/total_energy)[0])
    #plt.title('Energy Breakdown Per Run')
    #plt.xticks(np.arange(0, 1, step=1))
    #plt.yticks(np.arange(0, 1.01, .1))
    #plt.ylabel('Percent of Total Energy')
    #plt.subplots_adjust(right=0.7)
    #plt.legend([x for x in bar_plots], ['CPU Energy', 'GPU Energy'], bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    #plt.xlabel("Full System")
    #plt.savefig(data.output_path / "stacked_energy.png")
    pass
    
def time_series(data: PerTrialData) -> None:
    f = plt.figure()
    f.tight_layout(pad=2.0)
    plt.rcParams.update({'font.size': 8})
    # plot the same data on both axes
    ax = f.gca()
    ignore_list = ['app_gpu1', 'app_gpu2', 'timewarp_gl gpu', 'hologram', 'opencv', 'Runtime', 'camera_cvtfmt', 'zed_imu_thread iter', 'OpenVINS IMU', 'camera_cvtfmt']
    account_names = data.ts.index.levels[0]
    for i, account_name in enumerate(account_names):
        if account_name in ignore_list:
            continue
        x_data = data.ts.loc[account_name, "wall_time_start"].copy()
        y_data = data.ts.loc[account_name, "cpu_time_duration"].copy()
        if account_name == 'hologram iter' or account_name == 'timewarp_gl iter':
            x_data.drop(x_data.index[0], inplace=True)
            y_data.drop(y_data.index[0], inplace=True)
        ax.plot(x_data, y_data, label=account_name)
        ax.set_title(f"{account_name} CPU Time Timeseries")
        ax.set(ylabel='CPU Time (ms)')
    plt.xlabel("Timestamp (ms)")
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.subplots_adjust(right=0.6)
    plt.yscale("log")
    plt.savefig(data.output_path / "overlayed.png")
    plt.close()
    
def account_time_series(data: PerTrialData) -> None:
    ts_dir = data.output_path / "ts"
    ts_dir.mkdir(exist_ok=True)
    account_names = data.ts.index.levels[0]
    for i, account_name in enumerate(account_names):
        f = plt.figure()
        f.tight_layout(pad=2.0)
        plt.rcParams.update({'font.size': 8})
        # plot the same data on both axes
        x_data = data.ts.loc[account_name, "wall_time_start"].copy()
        y_data = data.ts.loc[account_name, "cpu_time_duration"].copy()
        if account_name == 'hologram iter' or account_name == 'timewarp_gl iter':
            x_data.drop(x_data.index[0], inplace=True)
            y_data.drop(y_data.index[0], inplace=True)
        ax = f.gca()
        ax.plot(x_data, y_data)
        ax.set_title(f"{account_name} CPU Time Timeseries")
        ax.set(ylabel='CPU Time (ms)')
        plt.xlabel("Timestamp (ms)")
        plt.yscale("log")
        plt.savefig(ts_dir / f"{account_name}.png")
        plt.close()

def motion_to_photon(data: PerTrialData) -> None:
    fig = plt.figure()
    ax = plt.gca()
    ys = (data.mtp["vsync"] - data.mtp["imu_time"]) / 1e6
    xs = (data.mtp["vsync"] - data.mtp["vsync"].iloc[0]) / 1e9
    ax.plot(xs, ys)
    ax.set_xlabel("Time since application start (sec)")
    ax.set_ylabel("Motion-to-photon (ms)")
    fig.savefig(data.output_path / "mtp.png")
    plt.close(fig)

def cpu_timeline(data: PerTrialData) -> None:
    rowNum = -1.5 
    coIndex = 0  
    edges = ['black','gray','brown']  
    edIndex = 0  
    fig, ax = plt.subplots()  
    plt.title('CPU Over Time')  
    plt.xlabel('Wall Time')  
    plt.ylabel('account_name')  
    colors = ['red','orange','yellow','green','cyan','blue','purple','pink']  
    ax.set_xticks(range(len(data.ts.index.levels[0]))) 
    ax.set_yticks(range(len(data.ts.index.levels[0])))
    ax.set_yticklabels(data.ts.index.levels[0])   
    ax.grid(True)  
    plt.ylim(-1,16) 
    for x in data.ts.index.levels[0]: 
         rowNum = rowNum + 1 
         coIndex = coIndex + 1 
         if coIndex>7: 
              coIndex = 0 
         for y in data.ts.loc[x].index[0:50]: 
             wallStart = data.ts.loc[(x,y)]['wall_time_start'] 
             wallStop = data.ts.loc[(x,y)]['wall_time_stop'] 
             if wallStart > 1e6: 
                 wStart = [(wallStart,(wallStop-wallStart))] 
                 wStop = (rowNum,1) 
                 if edIndex >2: 
                     edIndex = 0 
                 plt.broken_barh(wStart,wStop,facecolors = colors[coIndex],edgecolors = edges[edIndex])
                 fig.savefig(data.output_path / "cputl.png")
                 plt.close(fig) 
   
    # import IPython; IPython.embed()
