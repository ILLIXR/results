from util import PerTrialData
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import charmonium.time_block as ch_time_block

def analysis(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    print("\U0001f600")
    populate_fps(trials, replaced_names)
    # populate_cpu(trials, replaced_names)
    # populate_gpu(trials, replaced_names)
    # populate_power(trials, replaced_names)
    # populate_mtp(trials, replaced_names)
    # populate_frame_time_mean(trials, replaced_names)
    # populate_frame_time_std(trials, replaced_names)
    # populate_frame_time_var_coeff(trials, replaced_names)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_fps(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        account_names = ['OpenVINS Camera', 'zed_camera_thread iter', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']
        
        import IPython; IPython.embed()
        return

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            # Convert ms to s and then convert period to hz
            values.append(1 / (trial.summaries["period_mean"][name] * .001))
        # values.append(trial.summaries["period_mean"]['app'])

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/fps.csv', index=False)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_frame_time_mean(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
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
            
            ts_temp = trial.ts.reset_index()
            mean = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].mean() 

            values.append(mean)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_mean.csv', index=False)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_frame_time_std(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
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
            mean = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].std() 

            values.append(mean)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_std.csv', index=False)


# Dont you love clean code cause same
@ch_time_block.decor(print_start=False, print_args=False)   
def populate_frame_time_var_coeff(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection', 'Playback', 'Encoding']

    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list
    for trial in tqdm(trials):
        if trial.conditions.machine != 'desktop':
            continue

        account_names = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            ts_temp = trial.ts.reset_index()
            std_dev = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].std() 
            mean = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].mean() 

            values.append(std_dev/mean * 100)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_var_coeff_desktop.csv', index=False)

    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list
    for trial in tqdm(trials):
        if trial.conditions.machine != 'jetsonhp':
            continue

        account_names = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            ts_temp = trial.ts.reset_index()
            std_dev = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].std() 
            mean = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].mean() 

            values.append(std_dev/mean * 100)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_var_coeff_jhp.csv', index=False)

    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list
    for trial in tqdm(trials):
        if trial.conditions.machine != 'jetsonlp':
            continue

        account_names = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            ts_temp = trial.ts.reset_index()
            std_dev = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].std() 
            mean = ts_temp[ts_temp["account_name"] == name]['cpu_time_duration'].mean() 

            values.append(std_dev/mean * 100)

        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values
        data_frame.to_csv('../output/frame_time_var_coeff_jlp.csv', index=False)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_cpu(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_names = trials[0].ts.index.levels[0]
    ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU']
    account_list = ['zed_camera_thread iter', 'OpenVINS Camera', 'zed_imu_thread iter', 'imu_integrator iter', 'app', 'timewarp_gl iter', 'audio_decoding iter', 'audio_encoding iter']

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
        name_list = ['app_gpu1', 'app_gpu2', 'timewarp_gl gpu']
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

    account_list = ['Mean', 'Std Dev']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        values = [trial.mtp['imu_to_display'][200:].mean(), trial.mtp['imu_to_display'][200:].std()]
        data_frame[trial.conditions.application + '-' + trial.conditions.machine] = values

    data_frame.to_csv('../output/MTP_Vals.csv', index=False)

