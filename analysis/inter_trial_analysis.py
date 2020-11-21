from util import PerTrialData
import pandas as pd
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
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

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
        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/fps.csv', index=False)

@ch_time_block.decor(print_start=False, print_args=False)   
def populate_cpu(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_names = trials[0].ts.index.levels[0]
    ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'app']
    account_list = [name for name in account_names if name not in ignore_list]
    account_list.append('app') 
    account_list = [replaced_names[name] if name in replaced_names else name for name in account_list]
    account_list.insert(0, "Run Name")
    data_frame = pd.DataFrame([], columns=account_list)

    for trial in tqdm(trials):
        account_names = trial.ts.index.levels[0]

        values = {"Run Name": trial.conditions.application + '-'+ trial.conditions.machine}
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue

            formatted_name = replaced_names[name] if name in replaced_names else name
            values.update({formatted_name: trial.summaries["cpu_time_duration_sum"][name]})
        values.update({"Application": trial.summaries["cpu_time_duration_sum"]['app']})

        data_frame = data_frame.append(values, ignore_index=True, sort=False)
        # from IPython import embed; embed()

    data_frame.to_csv('../output/cpu.csv', index=False)

@ch_time_block.decor(print_start=False, print_args=False)   
def populate_gpu(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_names = trials[0].ts.index.levels[0]
    account_list = ['app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu']
    account_list = [replaced_names[name] if name in replaced_names else name for name in account_list]
    account_list.insert(0, "Run Name")
    data_frame = pd.DataFrame([], columns=account_list)

    for trial in tqdm(trials):
        account_names = trial.ts.index.levels[0]

        values = {"Run Name": trial.conditions.application + '-'+ trial.conditions.machine}
        name_list = ['app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu']
        for idx, name in enumerate(name_list):

            formatted_name = replaced_names[name] if name in replaced_names else name
            values.update({formatted_name: trial.summaries["gpu_time_duration_sum"][name]})

        data_frame = data_frame.append(values, ignore_index=True, sort=False)
        # from IPython import embed; embed()

    data_frame.to_csv('../output/gpu.csv', index=False)

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

