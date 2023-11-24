from pathlib import Path
import random
import math
import h5py
import numpy as np
import torch
import os
import io
import json
from tqdm import tqdm
import cv2
from deepdiff import DeepDiff
from random import randrange, sample
from robomimic.utils.obs_utils import process_obs
import robomimic.utils.file_utils as FileUtils
from PIL import Image
from copy import deepcopy
import gym
from environments import *
from metaworld.data.dataset import *
from metaworld.data.utils import normalize_reward
import PLEX.util.globals as globals
from PLEX.util.misc import construct_rewards

MIN_REWARD = math.inf
MAX_REWARD = -math.inf

# The seed for generating the train-validation split is fixed and is separate from the torch seed.
train_val_split_rng = np.random.RandomState(0)

def train_val_split(items, val_frac):
    items = list(items)
    n_total = len(items)
    train_val_split_rng.shuffle(items)
    n_val = round(val_frac * n_total)
    return items[n_val:], items[:n_val]


def process_images(images, depth=False):
    assert images.ndim == 4
    return np.stack([process_obs(images[i], ('depth' if depth else 'rgb')) for i in range(len(images))])


def discount_cumsum(x, is_successful, gamma):
    global MIN_REWARD
    global MAX_REWARD
    discount_cumsum = np.zeros_like(x)
    # To deal with trajectories of different lenghths, we pretend that all trajectories are infinitely long
    # and define the discounted cumulative sum as
    #       discount_cumsum[-1] = max_reward / (1. - gamma)  # pretend last state is absorbing
    # for trajectories that reached the goal.
    #
    # For trajectories that timed out, we don't know the right value to infinitely "extend" them. Ideally,
    # it should be the min_reward / (1 - gamma), but we generally don't know what min_reward is for a given environment
    # and need to estimate it empirically from all of this environment's loaded data.
    discount_cumsum[-1] = x[-1] + (MAX_REWARD / (1. - gamma) if is_successful else MIN_REWARD / (1. - gamma))
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


# Shortcut to concatenate arrays along axis=1 and convert to given type (float by default)
def cat1(*arrays, dtype=np.float32):
    return np.concatenate(arrays, axis=1, dtype=dtype)


def torchify(x, device):
    if x is None:
        return None
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, list):
        return torch.cat([torchify(v, device) for v in x])
    elif isinstance(x, dict):
        return {k: torchify(v, device) for k, v in x.items()}
    else:
        raise ValueError(f'Cannot torchify {x}')


def data_reading_setup(data_dir, task):
    data_dir = Path(data_dir).expanduser()
    data_dir = data_dir/task.dataset_location
    assert not Path(data_dir).is_file()
    assert Path(data_dir).is_dir(), f'Expected a directory at {data_dir}'
    trajectories = {}
    trajectories[task.name] = []
    return data_dir, trajectories


class TrajectoryDataset:
    def __init__(self, trajectories, camera_names, contextual):
        self.trajectories = list(trajectories)
        if not globals.full_state_mode:
            self.camera_names = camera_names
            self.traj_lens = np.array([traj['len'] for traj in self.trajectories])
        else:
            self.camera_names = None
            self.traj_lens = np.array([len(traj['full_state']) for traj in self.trajectories])
        self.contextual = contextual

        if len(self.trajectories) == 0:
            return

        self.p_sample = self.traj_lens / np.sum(self.traj_lens)

        proto_traj = self.trajectories[0]
        proto_traj = proto_traj['load_images'](proto_traj)
        if not globals.full_state_mode:
            self.image_dims = None

            for cam in self.camera_names:
                image_dims = proto_traj['image'][cam].shape[1:]
                if self.image_dims is None:
                    self.image_dims = image_dims
                else:
                    assert np.all(self.image_dims == image_dims) or np.all(image_dims == ()), f'Images from a given cam should all be None or have the same size as from other cams. Other cams\' image size: {self.image_dims}, this cam\'s image size is {image_dims}.'
        else:
            self.full_state_dim = proto_traj['full_state'].shape[1]

        # Check for existence of optional keys
        self.has_proprios = 'proprio' in proto_traj
        self.has_actions = 'action' in proto_traj
        self.has_rewards = 'reward' in proto_traj

        if self.has_proprios:
            assert not globals.full_state_mode, 'We shouldn\'t be using proprios in full-state mode.'
            self.proprio_dim = proto_traj['proprio'].shape[1]
            for traj in trajectories:
                assert traj['proprio'].shape[1] == self.proprio_dim

        if self.has_actions:
            self.action_dim = proto_traj['action'].shape[1]
            for traj in trajectories:
                assert traj['action'].shape[1] == self.action_dim

    def __len__(self):
        return len(self.trajectories)

    @property
    def video_only(self):
        return not self.has_actions and not self.has_proprios and not self.has_rewards and not globals.full_state_mode

    def copy_frames(self, src, actual_trg_len, rate_ratio, raise_frame_rate, pad_frame_gaps):
        # Copies data from a source array, adjusting frame rates as necessary.

        # Allocate the destination array to be same shape as the source array,
        # except for the first dimension (time), which must be actual_trg_len.
        trg_data = np.zeros((actual_trg_len, *src.shape[1:]), dtype=src.dtype)
        actual_src_len = len(src)

        if rate_ratio == 1:
            # The frame rates match. Do a direct copy.
            trg_data[:] = src[:actual_src_len]
        elif raise_frame_rate:
            # The source frame rate is too low. Copy source items as needed.
            for i in range(rate_ratio):
                new_src_len = len(trg_data[i::rate_ratio])
                trg_data[i::rate_ratio] = src[:new_src_len]
                if pad_frame_gaps:
                    break  # Leave zeros in the intervening frames.
        else:
            # The source frame rate is too high. Skip the unneeded items.
            trg_data[:] = src[0:rate_ratio * actual_src_len:rate_ratio]
        return trg_data

    def sample_batch(self, batch_size, target_frame_rate, pad_frame_gaps, max_len, get_context, discount, device=globals.DEFAULT_DEVICE, context_from_same_traj=False):
        assert len(self.trajectories) > 0
        # We should probably factor out the code that maps trajectories to tasks so that this computation is done only once, not every time a batch is sampled.
        task_name2traj_idx_dict = {}

        for i in range(len(self.trajectories)):
            if self.trajectories[i]['task_info'].name in task_name2traj_idx_dict.keys():
                task_name2traj_idx_dict[self.trajectories[i]['task_info'].name].append(i)
            else:
                task_name2traj_idx_dict[self.trajectories[i]['task_info'].name] = [i]

        batch_inds = np.random.choice(
            np.arange(len(self.trajectories)),
            size=batch_size,
            replace=True,
            p=self.p_sample     # reweights so we sample according to timesteps
        )

        if not globals.full_state_mode:
            images = {cam: [] for cam in self.camera_names}
            contexts = {cam: [] for cam in self.camera_names} if self.contextual else None
            proprios = [] if self.has_proprios else None
        else:
            full_states = []
            contexts = []
            proprios = None

        masks =  []
        actions = [] if self.has_actions else None
        rewards, returns = ([], []) if self.has_rewards else (None, None)
        timesteps = []

        for batch_index in range(batch_size):
            traj = None
            traj_len = -1
            is_valid = False

            while not is_valid:
                traj = self.trajectories[batch_inds[batch_index]]
                traj_len = traj['len'] if not globals.full_state_mode else len(traj['full_state'])

                if self.contextual:
                    MAX_RETRIES = 3
                    retry_ctr = 0
                    while not is_valid and retry_ctr < MAX_RETRIES:
                        retry_ctr += 1
                        if self.video_only:
                            # Choose a context from the same trajectory
                            ctx, is_valid = get_context(traj, 0, traj_len)
                        else:
                            # Choose a context from another random trajectory **of the same task**.
                            if context_from_same_traj:
                                ctx_traj = traj
                            else:
                                ctx_traj_idx = task_name2traj_idx_dict[traj['task_info'].name][randrange(len(task_name2traj_idx_dict[traj['task_info'].name]))]
                                ctx_traj = self.trajectories[ctx_traj_idx]
                            ctx_traj_len = ctx_traj['len'] if not globals.full_state_mode else len(ctx_traj['full_state'])
                            ctx, is_valid = get_context(ctx_traj, 0, ctx_traj_len)

                    if is_valid and retry_ctr > 1:
                        print(f'Found a valid context only on the {retry_ctr}th attempt...')

                    if not is_valid:
                        # Sample a different trajectory
                        batch_inds[batch_index] = np.random.choice(
                            np.arange(len(self.trajectories)),
                            size=1,
                            replace=True,
                            p=self.p_sample     # reweights so we sample according to timesteps
                        )[0]
                        continue

                    if not globals.full_state_mode:
                        for cam in self.camera_names:
                            contexts[cam].append(ctx[cam][np.newaxis])
                    else:
                        contexts.append(ctx[np.newaxis])
                else:
                    # Non-contexttual trajectories don't need a context, by definition, so we'll just oveeride the context validity check.
                    is_valid = True

            src_end = random.randint(1, traj_len)
            data_frame_rate = traj['task_info'].frame_rate  # Source fps.
            max_trg_len = max_len  # trg refers to target arrays that will be returned.

            assert (data_frame_rate is None) or (target_frame_rate is None) or (
                    data_frame_rate == target_frame_rate) or self.video_only, \
                "For now, the target and data frame rates can be different only for video-only data."

            if (data_frame_rate is None) or (target_frame_rate is None) or (data_frame_rate == target_frame_rate):
                # The frame rates match. Do a direct copy.
                rate_ratio = 1
                raise_frame_rate = False
                max_src_len = max_trg_len
                src_start = max(0, src_end - max_src_len)
                actual_src_len = src_end - src_start
                trg_start = src_start
                actual_trg_len = actual_src_len
            elif data_frame_rate < target_frame_rate:
                # The source frame rate is too low. Copy each source item (or pad with zeros) as many times as needed.
                rate_ratio = target_frame_rate // data_frame_rate
                raise_frame_rate = True
                max_src_len = math.ceil(max_trg_len / rate_ratio)  # Fewer source frames will be needed.
                src_start = max(0, src_end - max_src_len)
                actual_src_len = src_end - src_start
                trg_start = src_start * rate_ratio
                actual_trg_len = min(max_trg_len, actual_src_len * rate_ratio)
            else:  # data_frame_rate > target_frame_rate
                # The source frame rate is too high. Skip the unneeded items.
                rate_ratio = data_frame_rate // target_frame_rate
                raise_frame_rate = False
                max_src_len = max_trg_len * rate_ratio  # Some source frames will be dropped.
                src_start = max(0, src_end - max_src_len)
                actual_src_len = src_end - src_start
                trg_start = src_start // rate_ratio
                actual_trg_len = min(max_trg_len, (actual_src_len + rate_ratio - 1) // rate_ratio)

            trg_end = trg_start + actual_trg_len

            if not globals.full_state_mode:
                for cam in self.camera_names:
                    traj = traj['load_images'](traj, start_idx=src_start, end_idx=src_end)
                    subseq = traj['image'][cam][src_start:src_end]
                    trg_data = self.copy_frames(subseq, actual_trg_len, rate_ratio, raise_frame_rate, pad_frame_gaps)
                    images[cam].append(cat1(
                        np.zeros((1, max_trg_len - actual_trg_len, *self.image_dims)),
                        trg_data.reshape(1, actual_trg_len, *self.image_dims)
                    ))
                if self.has_proprios:
                    proprios.append(cat1(
                        np.zeros((1, max_trg_len - actual_trg_len, self.proprio_dim)),
                        traj['proprio'][src_start:src_end].reshape(1, actual_trg_len, self.proprio_dim)
                    ))
            else:
                full_states.append(cat1(
                    np.zeros((1, max_trg_len - actual_trg_len, self.full_state_dim)),
                    traj['full_state'][src_start:src_end].reshape(1, actual_trg_len, self.full_state_dim)
                ))

            if self.has_actions:
                # Why the * -10?
                actions.append(cat1(
                    np.ones((1, max_trg_len - actual_trg_len, self.action_dim)) * -10.,
                    traj['action'][src_start:src_end].reshape(1, actual_trg_len, self.action_dim)
                ))

            if self.has_rewards:
                rewards.append(cat1(
                    np.zeros((1, max_trg_len - actual_trg_len, 1)),
                    traj['reward'][src_start:src_end].reshape(1, actual_trg_len, 1)
                ))
                if 'rtg' in traj:
                    returns.append(cat1(
                        np.zeros((1, max_trg_len - actual_trg_len, 1)),
                        traj['rtg'][src_start:src_end].reshape(1, actual_trg_len, 1)
                    ))
                else:
                    rtgs = discount_cumsum(traj['reward'][src_start:], traj['success'][-1], gamma=discount)
                    returns.append(cat1(
                        np.zeros((1, max_trg_len - actual_trg_len, 1)),
                        rtgs[:actual_trg_len].reshape(1, actual_trg_len, 1)
                    ))

            timesteps.append(cat1(
                np.zeros((1, max_trg_len - actual_trg_len), dtype=np.long),
                np.arange(trg_start, trg_end, dtype=np.long)[np.newaxis],
                dtype=np.long
            ))

            masks.append(cat1(
                np.zeros((1, max_trg_len - actual_trg_len)),
                np.ones((1, actual_trg_len))
            ))

        return [
            torchify(x, device)
            for x in (contexts, (images if not globals.full_state_mode else full_states), proprios, actions, rewards, returns, timesteps, masks)
        ]


def setup_context_sampler(style):
    def get_context(traj, start, end):
        # For simplicity, all images from a given time step will serve as context.
        traj_len = traj['len']

        if 'success' in style:
            success_indices = np.nonzero(traj['success'][:traj_len] == True)[0]
            if len(success_indices) == 0:
                if not globals.full_state_mode:
                    traj = traj['load_images'](traj, start_idx=0, end_idx=1)
                    images = traj['image']
                    return {cam: np.zeros_like(images[cam][0]) for cam in images.keys()}, False
                else:
                    states = traj['full_state']
                    return np.zeros_like(states[0]), False
            elif style == 'first-success':
                chosen_idx = success_indices[0]
            elif style == 'random-success':
                chosen_idx = random.choice(success_indices)
            else:
                raise NotImplementedError
        elif style.startswith('random-next-'):
            window_len = int(style[12:])
            high = min(end + window_len, traj_len)
            chosen_idx = np.random.randint(end - 1, high)
        elif style.startswith('random-last-'):
            window_len = int(style[12:])
            chosen_idx = np.random.randint(traj_len - window_len, traj_len)
        elif style == 'blank':
            if not globals.full_state_mode:
                images = traj['load_images'](traj, start_idx=0, end_idx=1)['image']
                return {cam: np.zeros_like(images[cam][0]) for cam in images.keys()}, True
            else:
                states = traj['full_state']
                return np.zeros_like(states[0]),True
        else:
            raise NotImplementedError

        if not globals.full_state_mode:
            images = traj['load_images'](traj, start_idx=chosen_idx, end_idx=chosen_idx+1)['image']
        else:
            states = traj['full_state']

        return {cam: images[cam][chosen_idx].astype(np.float32) for cam in images.keys()} if not globals.full_state_mode else states[chosen_idx], True
    return get_context


def setup_batch_sampler(dataset, context_style, cmdline_args, device):
    context_fn = setup_context_sampler(context_style) if dataset.contextual else lambda *args, **kwargs: None
    return lambda batch_size, target_frame_rate, pad_frame_gaps: dataset.sample_batch(batch_size,
                                                    target_frame_rate,
                                                    pad_frame_gaps,
                                                    max_len=((cmdline_args['obs_pred.K'] + cmdline_args['future_step']) if cmdline_args['model'] == 'PLEX' else cmdline_args['K']),
                                                    get_context=context_fn,
                                                    discount=cmdline_args['discount'],
                                                    device=device,
                                                    context_from_same_traj=cmdline_args['context_from_same_traj'])


def choose_n_trajectories(task, trajectories, max_trajectories, only_the_best, video_only, discount, kwargs):
    if max_trajectories is not None:
        if only_the_best:
            assert not video_only, f"The \'only_the_best\' flag doesn't make sense when loading video-only trajectories, since they don't carry reward information."
            # NOTE: we choose the best trajectories based on their discounted cumulative return, which
            # requires knowing the MAX_REWARD and MIN_REWARD across all trajectories for the given task
            # (see the discount_cumsum(.) method). At the time choose_n_trajectories(.) is called, we may not
            # have loaded all trajectories for a given task. Thus, our estimates of MAX_REWARD and MIN_REWARD may
            # be based only on the data seen so far, and hence may be approximate. As a result, so is our choice
            # of the top-N trajectories.
            rets_and_trajs = []
            for traj in trajectories[task.name]:
                rets_and_trajs.append((discount_cumsum(traj['reward'], traj['success'][-1], discount)[0], traj))

            rets_and_trajs.sort(key=lambda ret_and_traj: ret_and_traj[0], reverse=True)
            selected_trajectories = [ret_and_traj[1] for ret_and_traj in rets_and_trajs]
        elif 'data_shuffling_rng' in kwargs and kwargs['data_shuffling_rng'] is not None:
            selected_trajectories = trajectories[task.name]
            kwargs['data_shuffling_rng'].shuffle(selected_trajectories)
        else:
            selected_trajectories = trajectories[task.name]

        trajectories[task.name] = selected_trajectories[:min(len(selected_trajectories), max_trajectories)]
        print(f"Using {min(len(selected_trajectories), max_trajectories)} trajectories.")

    return trajectories


# reward_type can be 'native', 'negative', 'random', 'zero', and 'sparse'.
def load_d4rl_data(log, data_dir, task,
                        max_trajectories=None, only_the_best=False, discount=1.0,
                        camera_names=None, normalize_rewards=False, reward_type='native', video_only=False,
                        **kwargs):
    import d4rl
    global MIN_REWARD
    global MAX_REWARD

    assert not video_only, "D4RL trajectories are state-only."
    data_dir, trajectories = data_reading_setup(data_dir, task)
    print("Attempting to load from {}".format(data_dir))
    #data_paths = list(data_dir.rglob('*.hdf5'))
    num_files = 0

    env = gym.make(task.name)
    d4rl.set_dataset_path(data_dir)
    dataset = env.get_dataset()
    EPS = 1e-6
    dataset['actions'] = np.clip(dataset['actions'], -1+EPS, 1-EPS)

    dataset_size = len(dataset["actions"])
    episodic_dataset = []
    episode = []

    for i in range(dataset_size):
        observation = dataset["observations"][i]
        reward = dataset["rewards"][i]
        action = dataset["actions"][i]

        episode.append({
            "observation": observation,
            "action": action,
            "reward": reward})

        timeout = dataset["timeouts"][i]
        terminal = dataset["terminals"][i]

        if timeout or terminal:
            # Episode ends
            episodic_dataset.append(list(episode))
            episode = []

    # Add the left over episode which didn't get terminated
    if len(episode) > 0:
        episodic_dataset.append(list(episode))

    # Compute return-to-go instead of reward for decision transformer
    for i in range(0, len(episodic_dataset)):
        traj_dict = {}
        traj_dict['task_info'] = task

        episode = episodic_dataset[i]
        observations_ls = []
        actions_ls = []
        rewards = []

        for transition in episode:
            observations_ls.append(transition["observation"])
            actions_ls.append(transition["action"])
            rewards.append(transition["reward"])

        # D4RL doesn't have a good way of determining a successful end of a trajectory, since its tasks are
        # mostly process-oriented, not goal-oriented.
        successes = np.array([False] * len(rewards))
        rewards = np.array(rewards)
        rewards = construct_rewards(rewards, successes, reward_type)
        MIN_REWARD = min(MIN_REWARD, rewards.min())
        MAX_REWARD = max(MAX_REWARD, rewards.max())

        traj_dict.update({
            'full_state': np.array(observations_ls),
            'action': np.array(actions_ls),
            'reward': rewards,
            'success': successes
        })
        trajectories[task.name].append(traj_dict)

    # NOTE: This works correctly *only* if we know that we won't read any more data for this task and hence MIN_REWARD and MAX_REWARD won't change.
    for traj in trajectories[task.name]:
        rtgs = discount_cumsum(traj['reward'], traj['success'][-1], gamma=discount)
        traj['rtg'] = rtgs

    env.close()

    trajectories = choose_n_trajectories(task, trajectories, max_trajectories, only_the_best, video_only, discount, kwargs)

    return trajectories


# reward_type can be 'native', 'negative', 'random', 'zero', and 'sparse'.
def load_metaworld_data(log, data_dir, task,
                        max_trajectories=None, only_the_best=False, discount=1.0,
                        camera_names=None, normalize_rewards=False, reward_type='native', video_only=False,
                        **kwargs):
    global MIN_REWARD
    global MAX_REWARD
    data_dir, trajectories = data_reading_setup(data_dir, task)
    print("Attempting to load from {}".format(data_dir))
    data_paths = list(data_dir.rglob('*.hdf5'))
    num_files = 0
    common_env_metadata = kwargs['common_env_metadata_dict']

    # Try to load every hdf5 file found in the task subtree specified by the user.
    for path in data_paths:
        print("Attempting to load from subpath {}".format(path))
        log(f'Loading Meta-World data from {path}')

        #env_meta, all_trajs = read_trajs(path, reward_type='goal-cost') if sparse_reward else read_trajs(path, reward_type='original')

        # Read the native reward first. We will transform it if needed, per normalize_rewards and reward_type arguments.
        env_meta, all_trajs = read_trajs(path, reward_type='original')
        f = h5py.File(path, 'r')

        if not video_only:
            env_meta.pop('task_name')
            env_meta.pop('horizon')
            env_meta.pop('subgoal_breakdown')
            env_meta.pop('success_steps_for_termination')

            if common_env_metadata['metaworld'] is None:
                common_env_metadata['metaworld'] = env_meta
            else:
                env_meta_diff = DeepDiff(common_env_metadata["metaworld"], env_meta)
                assert len(env_meta_diff) == 0, f"File {path}'s metadata is critically different from the metadata in previously loaded files. Diff: {env_meta_diff}"

        num_files += 1

        for traj in tqdm(all_trajs):
            traj_dict = {}
            traj_dict['task_info'] = task
            traj_dict['len'] = len(traj['observations'])
            traj_dict['load_images'] = load_mw_images
            traj_dict['success'] = np.array(traj['successes'])

            if not globals.full_state_mode:
                traj_dict['image'] = {}
                for cam in camera_names:
                    traj_dict['image'][cam] = process_images(np.array(traj['observations']), depth=False)

            if video_only:
                assert not globals.full_state_mode, 'Loading video-only trajectories is incompatible with full-state mode.'
            else:
                if globals.full_state_mode:
                    traj_dict['full_state'] = np.array(traj['states'])
                else:
                    traj_dict['proprio'] = np.array(traj['proprio_states'])

                rewards = np.array(traj['rewards'])

                if normalize_rewards:
                    rewards = normalize_reward(rewards)

                rewards = construct_rewards(rewards, traj['successes'], reward_type)
                MIN_REWARD = min(MIN_REWARD, rewards.min())
                MAX_REWARD = max(MAX_REWARD, rewards.max())

                traj_dict.update({
                    'action': np.array(traj['actions']),
                    'reward': rewards
                })

            trajectories[task.name].append(traj_dict)
        f.close()

    trajectories = choose_n_trajectories(task, trajectories, max_trajectories, only_the_best, video_only, discount, kwargs)

    assert num_files > 0, 'No paths found to load metaworld data'

    return trajectories


# reward_type can be 'native', 'negative', 'random', 'zero', and 'sparse'.
def load_robomimic_data(log, data_dir, task,
                        max_trajectories=None, only_the_best=False, discount=1.0,
                        camera_names=None, normalize_rewards=False, reward_type='native', video_only=False,
                        **kwargs):
    if globals.full_state_mode:
        raise NotImplementedError
    global MIN_REWARD
    global MAX_REWARD
    data_dir, trajectories = data_reading_setup(data_dir, task)
    print("Attempting to load from {}".format(data_dir))
    data_paths = list(data_dir.rglob('*.hdf5'))
    num_files = 0
    common_robomimic_env_metadata = kwargs['common_env_metadata_dict']

    log(f'Found {len(data_paths)} paths in {data_dir}')

    # Try to load every hdf5 file found in the task subtree specified by the user.
    for path in data_paths:
        print("Attempting to load from subpath {}".format(path))
        log(f'Loading robomimic data from {path}')
        f = h5py.File(path, 'r')

        if not video_only:
            # This logic is for ensuring data compatibility across data files from which we are loading more than just image sequences.
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=path)
            # The data from Robomimic's dataset was generated using an older Robosuite version, which used different names
            # for some of the properties.
            if 'damping' in env_meta['env_kwargs']['controller_configs']:
                env_meta['env_kwargs']['controller_configs']['damping_ratio'] = env_meta['env_kwargs']['controller_configs']['damping']
                env_meta['env_kwargs']['controller_configs'].pop('damping')
            if 'damping_limits' in env_meta['env_kwargs']['controller_configs']:
                env_meta['env_kwargs']['controller_configs']['damping_ratio_limits'] = env_meta['env_kwargs']['controller_configs']['damping_limits']
                env_meta['env_kwargs']['controller_configs'].pop('damping_limits')

            env_meta.pop('env_name')
            env_meta['env_kwargs'].pop('camera_depths')
            if 'render_gpu_device_id' in env_meta['env_kwargs']:
                env_meta['env_kwargs'].pop('render_gpu_device_id')
            for cam in camera_names:
                assert cam in env_meta['env_kwargs']['camera_names'], f"Camera {cam} is missing from this data file. Available cameras: {env_meta['env_kwargs']['camera_names']}"
            env_meta['env_kwargs']['camera_names'] = list(camera_names)

            if common_robomimic_env_metadata['robosuite'] is None:
                common_robomimic_env_metadata['robosuite'] = env_meta
            else:
                env_meta_diff = DeepDiff(common_robomimic_env_metadata["robosuite"], env_meta)
                assert len(env_meta_diff) == 0, f"This file's metadata is critically different from the metadata in previously loaded files. Diff: {env_meta_diff}"

        num_files += 1

        for traj in tqdm(f['data'].values()):
            traj_dict = {}
            traj_dict['task_info'] = task
            traj_dict['len'] = len(traj['obs'][f'{camera_names[0]}_image'])
            assert not globals.full_state_mode, 'This codebase currently doesn\'t support working with Robosuite/Robomimic in full-state mode.'
            traj_dict['image'] = {}
            traj_dict['load_images'] = load_rs_images
            traj_dict['success'] = np.array(traj['successes'])

            if video_only:
                for cam in camera_names:
                    cam_image = f'{cam}_image'
                    found = False
                    for cam_name in ROBOMIMIC_SUBSTITUTE_IMAGE_OBS[cam_image]:
                        if cam_name in traj['obs']:
                            traj_dict['image'][cam] = process_images(np.array(traj['obs'][cam_name]), depth=False)
                            found = True
                            break
                    if not found:
                        assert found, f"Need to find cam {cam_image} or equivalent in the available observations of task {task.name}."
            else:
                for cam in camera_names:
                    traj_dict['image'][cam] = process_images(np.array(traj['obs'][f'{cam}_image']), depth=False)

                if f'{camera_names[0]}_depth' in traj['obs']:
                    traj_dict['depth'] = {
                        cam: process_images(np.array(traj['obs'][f'{cam}_depth']), depth=True)
                        for cam in camera_names
                    }

                if normalize_rewards:
                    raise NotImplementedError

                rewards = construct_rewards(traj['rewards'], traj['successes'], reward_type)
                MIN_REWARD = min(MIN_REWARD, rewards.min())
                MAX_REWARD = max(MAX_REWARD, rewards.max())

                traj_dict.update({
                    'proprio': compose_proprio(traj['obs'], keys=PROPRIO_KEYS['robosuite']),
                    'action': np.array(traj['actions']),
                    'reward': np.array(rewards),
                    'success': np.array(traj['successes'])
                    # 'done': np.array(traj['dones'])
                })

            trajectories[task.name].append(traj_dict)
        f.close()

    trajectories = choose_n_trajectories(task, trajectories, max_trajectories, only_the_best, video_only, discount, kwargs)

    assert num_files > 0, 'No paths found to load robomimic data'

    return trajectories


def load_mw_images(traj_dict, start_idx=0, end_idx=None):
    assert 'image' in traj_dict and traj_dict['image'] is not None
    return traj_dict


def load_rs_images(traj_dict, start_idx=0, end_idx=None):
    assert 'image' in traj_dict and traj_dict['image'] is not None
    return traj_dict


def load_bridge_images(traj_dict, start_idx=0, end_idx=None):
    if 'image' in traj_dict and not (traj_dict['image'] is None):
        traj_dict['image'] = None

    f = h5py.File(traj_dict['traj_info']['file_path'], 'r')
    traj = f['data'][traj_dict['traj_info']['idx']]
    traj_dict = deepcopy(traj_dict)
    traj_dict['image'] = {}

    for cam in traj_dict['camera_names']:
        if cam == 'main':
            main_cam_images = np.array([np.array(Image.open(io.BytesIO(im_bytes))) for im_bytes in traj['obs']['images0'][start_idx:end_idx]])
            images = process_images(main_cam_images, depth=False)
            traj_dict['image'][cam] = np.zeros([len(traj['obs']['images0'])] + list(images[0].shape))
            traj_dict['image'][cam][start_idx:end_idx] = images
        elif cam == 'aux':
            # Randomly sample one of the other cameras.
            aux_cams = list(traj['obs'].keys())
            aux_cams.remove('images0')
            aux_cams.remove('state')
            if len(aux_cams) > 0:
                aux_cam = sample(aux_cams, 1)[0]
            else:
                aux_cam = 'images0'
            aux_cam_images = np.array([np.array(Image.open(io.BytesIO(im_bytes))) for im_bytes in traj['obs'][aux_cam][start_idx:end_idx]])
            images = process_images(aux_cam_images, depth=False)
            traj_dict['image'][cam] = np.zeros([len(traj['obs'][aux_cam])] + list(images[0].shape))
            traj_dict['image'][cam][start_idx:end_idx] = images
        elif cam == 'left_aux':
            aux_cam = 'images2'
            aux_cam_images = np.array([np.array(Image.open(io.BytesIO(im_bytes))) for im_bytes in traj['obs'][aux_cam][start_idx:end_idx]])
            images = process_images(aux_cam_images, depth=False)
            traj_dict['image'][cam] = np.zeros([len(traj['obs'][aux_cam])] + list(images[0].shape))
            traj_dict['image'][cam][start_idx:end_idx] = images
        else:
            assert cam == 'main' or cam == 'aux' or cam == 'left_aux', f'Invalid camera name for the Bridge Dataset: {cam}'

    return traj_dict


# reward_type can be 'native', 'negative', 'random', 'zero', and 'sparse'.
def load_bridge_data(log, data_dir, task,
                        max_trajectories=None, only_the_best=False, discount=1.0,
                        camera_names=None, normalize_rewards=False, reward_type='native', video_only=False,
                        **kwargs):
    if globals.full_state_mode:
        raise NotImplementedError

    if normalize_rewards:
        raise NotImplementedError

    global MIN_REWARD
    global MAX_REWARD
    data_dir, trajectories = data_reading_setup(data_dir, task)
    print("Attempting to load bridge data from {}".format(data_dir))
    data_paths = list(data_dir.rglob('*.hdf5'))
    num_files = 0
    bridge_metadata = kwargs['common_env_metadata_dict']['bridge']
    if bridge_metadata is None:
        bridge_metadata = {'frame_rate': 5}

    target_frame_rate = kwargs['target_frame_rate']

    if target_frame_rate is None:
        print("target_frame_rate is None, so frame rates will be ignored.")
    else:
        # Compare the frame rates.
        data_frame_rate = bridge_metadata['frame_rate']
        print('target_frame_rate is {} fps'.format(target_frame_rate))
        print('data_frame_rate   is {} fps'.format(data_frame_rate))
        if data_frame_rate < target_frame_rate:
            print('In each batch, this data will be UP-SAMPLED to match the target rate.')
        elif data_frame_rate > target_frame_rate:
            print('In each batch, this data will be DOWN-SAMPLED to match the target rate.')
        assert (target_frame_rate * (data_frame_rate // target_frame_rate) == data_frame_rate) or \
               (data_frame_rate * (target_frame_rate // data_frame_rate) == target_frame_rate), \
               "The dataset and target frame rates must be integer multiples of each other."
        task.frame_rate = data_frame_rate  # This will be used to handle frame-rate mismatches.

    log(f'Found {len(data_paths)} paths in {data_dir}')

    # Try to load every hdf5 file found in the task subtree specified by the user.
    for path in data_paths:
        print("Attempting to load from subpath {}".format(path))
        log(f'Loading bridge data from {path}')
        f = h5py.File(path, 'r')

        num_files += 1

        target_camera_name = camera_names[0]

        # Loop through the dataset trajectories, each with potentially multiple camera views.
        for traj_name in tqdm(f['data'].keys()):
            traj = f['data'][traj_name]
            traj_dict = {
                'task_info': task,
                'traj_info': {'file_path': path, 'idx': traj_name},
                'camera_names': camera_names,
                'image': None,
                'load_images': load_bridge_images,
                'len': len(traj['obs']['images0']),
                'success': np.array(traj['successes'])
            }

            for cam in camera_names:
                if cam == 'main':
                    main_cam_images = np.array([np.array(Image.open(io.BytesIO(im_bytes))) for im_bytes in traj['obs']['images0']])
                    traj_dict['image'][cam] = process_images(main_cam_images, depth=False)
                elif cam == 'aux':
                    # Randomly sample one of the other cameras.
                    aux_cams = list(traj['obs'].keys())
                    aux_cams.remove('images0')
                    aux_cams.remove('state')
                    if len(aux_cams) > 0:
                        aux_cam = sample(aux_cams, 1)[0]
                    else:
                        aux_cam = 'images0'
                    aux_cam_images = np.array([np.array(Image.open(io.BytesIO(im_bytes))) for im_bytes in traj['obs'][aux_cam]])
                    traj_dict['image']['aux'] = process_images(aux_cam_images, depth=False)
                elif cam == 'images1' or cam == 'images2':
                    aux_cam = cam
                    aux_cam_images = np.array([np.array(Image.open(io.BytesIO(im_bytes))) for im_bytes in traj['obs'][aux_cam]])
                    traj_dict['image']['aux'] = process_images(aux_cam_images, depth=False)
                else:
                    assert cam == 'main' or cam == 'aux', f'Invalid camera name for the Bridge Dataset: {cam}'

            if not video_only:
                rewards = construct_rewards(traj['rewards'], traj['successes'], reward_type)
                MIN_REWARD = min(MIN_REWARD, rewards.min())
                MAX_REWARD = max(MAX_REWARD, rewards.max())

                traj_dict.update({
                    'proprio': np.array(traj['obs']['state']),
                    'action': np.array(traj['actions']),
                    'reward': np.array(rewards)
                })

            # Append this output trajectory to the growing list.
            trajectories[task.name].append(traj_dict)
        f.close()

    trajectories = choose_n_trajectories(task, trajectories, max_trajectories, only_the_best, video_only, discount, kwargs)

    assert num_files > 0, 'No paths found to load bridge data'
    log(f'Generated {len(trajectories[task.name])} trajectories for this task in the bridge dataset')

    return trajectories


def load_data(log, data_dir, tasks, max_trajectories, **kwargs):
    all_trajectories = {}
    for task, max_traj in zip(tasks, max_trajectories):
        if task.dataset_type == 'robomimic' or task.dataset_type == 'robosuite' or task.dataset_type == 'libero':
            trajectories = load_robomimic_data(log, data_dir, task, max_trajectories=max_traj, **kwargs)
        elif task.dataset_type == 'metaworld':
            trajectories = load_metaworld_data(log, data_dir, task, max_trajectories=max_traj, **kwargs)
        elif task.dataset_type == 'bridge' or task.dataset_type == 'bridge-v2':
            trajectories = load_bridge_data(log, data_dir, task, max_trajectories=max_traj, **kwargs)
        elif task.dataset_type == 'd4rl':
            trajectories = load_d4rl_data(log, data_dir, task, max_trajectories=max_traj, **kwargs)
        else:
            assert False, 'Unknown dataset type {} for task {}'.format(task.dataset_type, task)

        for task_name in trajectories:
            if task_name in all_trajectories:
                # This may happen if we are loading several sub-datasets for the same task, e.g., "ph" and "mh" subdatasets in robomimic
                # NOTE: Max trajectories limit should probably apply to *all* trajectories of this task but currently applies on a per-directory basis.
                all_trajectories[task_name].extend(trajectories[task_name])
            else:
                all_trajectories[task_name] = trajectories[task_name]
    return all_trajectories