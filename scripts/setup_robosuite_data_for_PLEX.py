import urllib.request
import argparse
import pathlib
import zipfile
import glob
import os
from attrdict import AttrDict
from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=str, default='data', help = "Directory path where the demonstration data is to be written.")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    print("Downloading Robosuite demonstrations...")
    rs_dest_file = out_path / "robosuite.zip"
    urllib.request.urlretrieve("https://download.microsoft.com/download/e/5/1/e5106eb4-0f53-4d65-afd1-58c03d60bf98/robosuite_demo_data.zip", rs_dest_file)
    with zipfile.ZipFile(rs_dest_file, 'r') as file_to_unzip:
        file_to_unzip.extractall(out_path)
    pathlib.Path.unlink(rs_dest_file)

    task_data_paths = glob.glob(str(out_path / 'robosuite/*'))

    print(task_data_paths)

    for task_data_path in task_data_paths:
        # Process each tasks's data and move the processed data from the task's "raw" to the task' "ph" directory.
        task = pathlib.PurePath(task_data_path).name
        task_files = glob.glob(str(pathlib.Path(task_data_path) / 'Panda' / 'raw' / '*'))
        print(task_files)

        for task_file, idx in zip(task_files, range(len(task_files))):
            print(f'\n=== Converting {task_file} ===')
            task_output_file = task + f"__{idx}.hdf5"

            args = AttrDict({'dataset': task_file,
                            'output_name': task_output_file,
                            'done_mode': 0,
                            'camera_names': ['agentview', 'robot0_eye_in_hand'],
                            'camera_height': 84,
                            'camera_width': 84,
                            'include_depth': False,
                            'shaped': None,
                            'n': None,
                            'copy_rewards': None,
                            'copy_dones': None})

            # Inflate the low-level-state trajectory into a trajectory with pixel observations, proprios, and actions.
            dataset_states_to_obs(args)

            task_output_file_path = pathlib.Path(task_data_path) / "Panda" / "raw" / task_output_file
            task_output_ph_dest_path = pathlib.Path(task_data_path) / "Panda" / "ph"
            task_output_ph_dest_path.mkdir(parents=True, exist_ok=True)
            os.replace(task_output_file_path,  task_output_ph_dest_path / task_output_file)
