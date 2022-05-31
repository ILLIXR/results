from util import PerTrialData
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import charmonium.time_block as ch_time_block

def analysis(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    print("\U0001f600")
    populate_fps(trials, replaced_names)
    populate_cpu(trials, replaced_names)
    populate_gpu(trials, replaced_names)
    # populate_power(trials, replaced_names)
    populate_mtp(trials, replaced_names)
    populate_frame_time_mean(trials, replaced_names)
    populate_frame_time_std(trials, replaced_names)
    populate_frame_time_min(trials, replaced_names)
    populate_frame_time_max(trials, replaced_names)


@ch_time_block.decor(print_start=False, print_args=False)   
def populate_fps(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
    account_list = ['Camera', 'OpenVINS Camera', 'IMU', 'IMU Integrator', 'Application', 'Reprojection']
    data_frame = pd.DataFrame()
    data_frame["Components"] = account_list

    for trial in tqdm(trials):
        account_names = ['OpenVINS Camera', 'zed_camera_thread iter', 'zed_imu_thread iter', 'gtsam_integrator cb', 'app', 'timewarp_gl iter']

        # if trial.conditions.machine == 'jetsonlp' and trial.conditions.application == 'materials':
        #     # import IPython; IPython.embed()
        #     ts_temp = trial.ts.reset_index()
        #     testt = ts_temp[ts_temp["account_name"] == 'OpenVINS Camera']['wall_time_duration'].to_numpy()
        #     counter = 0
        #     for val in testt:
        #         if val > 66.666666:
        #             counter += 1
            
        #     print(counter)
        #     print(len(testt))
        #     return

        values = []
        ignore_list = ['opencv', 'Runtime', 'camera_cvtfmt', 'app_gpu1', 'app_gpu2', 'hologram', 'timewarp_gl gpu', 'OpenVINS IMU', 'audio_decoding iter', 'audio_encoding iter']
        print(trial.summaries["period_mean"])
        for idx, name in enumerate(account_names):
            if name in ignore_list:
                continue
            
            # Convert ms to s and then convert period to hz
            print(name)

            # We dont need this stuff 
            # if name == 'audio_decoding iter' or name == 'audio_encoding iter':
            #     # First ~200 values seem to be garbage so omit those when calculating the mean
            #     ts_temp = trial.ts.reset_index()
            #     if trial.conditions.machine == 'jetsonlp':
            #         mean_period = ts_temp[ts_temp["account_name"] == name]['period'][150:].mean() 
            #     elif trial.conditions.machine == 'jetsonhp':
            #         mean_period = ts_temp[ts_temp["account_name"] == name]['period'][100:].mean() 
            #     else:
            #         mean_period = ts_temp[ts_temp["account_name"] == name]['period'][60:].mean() 
            #     values.append(1 / (mean_period * .001))
            # else:
            values.append(1 / (trial.summaries["period_mean"][name] * .001))

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
def populate_frame_time_min(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
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
def populate_frame_time_max(trials: List[PerTrialData], replaced_names: Dict[str,str]) -> None:
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
            print(trial.power_data)
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

