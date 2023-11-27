import argparse
import pathlib
from metaworld.data.training_data_gen import gen_data
import metaworld

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=str, default='data', help = "Directory path where the demonstration data is to be written.")
    args = parser.parse_args()
    out_path=pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate 75 trajectories for each of the 5 target tasks of Meta-World's ML50, with noise=0.5.
    # This data will be used for pretraining PLEX's executor.
    tasks=','.join(['bin-picking-v2', 'box-close-v2', 'door-lock-v2', 'door-unlock-v2', 'hand-insert-v2'])
    gen_data(tasks, num_traj=75, noise=0.5, res=84, include_depth=False, camera='corner',
             data_dir_path=out_path, write_data=True, write_video=False, video_fps=20,
             # The following args apply only to play policy and aren't used for this data generation.
             use_play_policy=False,
             counter_max=None,
             grip_flip_p=None)

    # Generate 100 trajectories for each of the tasks of Meta-World's ML50, with noise=0.
    # For ML50's 45 pretraining tasks, the videoframe sequences from these expert trajectories
    # will be used for pretraining PLEX's planner. For the 5 target tasks, 10 of these 100 trajectories
    # will be used for finetuning the pretrained PLEX archtitecture.
    tasks = ','.join([x for x in metaworld.ML1.ENV_NAMES])
    gen_data(tasks, num_traj=100, noise=0, res=84, include_depth=False, camera='corner',
             data_dir_path=out_path, write_data=True, write_video=False, video_fps=20,
             # The following args apply only to play policy and aren't used for this data generation.
             use_play_policy=False,
             counter_max=None,
             grip_flip_p=None)
