import os
import time
import math
import random
from copy import deepcopy
from environments import DEFAULT_CAM
import numpy as np
import torch
import torch.multiprocessing as mp
from environments import RobosuiteEnv, MetaWorldEnv, d4rlEnv, init_obs_preprocessing, unprocess_image
from PLEX.models.trajectory_models.plex import PLEX
import robomimic.utils.obs_utils as ObsUtils
import cv2
import PLEX.util.globals as globals
from PLEX.util.data import setup_context_sampler, setup_batch_sampler, discount_cumsum
from PLEX.util.misc import parse_comma_sep_param_value, construct_rewards, setup_trainer


def evaluate_episode(
        task,
        model,
        ep_id,
        use_normalized_reward=False,
        reward_type='native',
        env_meta=None,
        min_time_at_goal_for_success=5,
        camera_names=None,
        image_size=84,
        device='cuda',
        max_ep_len=500,
        discount=1.,
        full_state_mode=False,
        context=None,
        target_return=None,
        record_camera=None,
        write_individual_images=False,
        record_traj_dir=None
):
    if not full_state_mode:
        # Make sure ObsUtils is set up (each process has to run this once)
        #
        # Actually, do we need this, given that it's done by the top-level module?
        # Presumably, it doesn't hurt...
        if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
            init_obs_preprocessing(camera_names, image_size)

    # Make sure that either:
    # (a) these settings are the same as at training time or
    # (b) the model was trained and is being evaluated in BC mode (i.e., rewards/returns weren't used
    # at training time and are ignored at evaluation time).
    print(f'Is the reward normalized **at evaluation time**: {use_normalized_reward}')
    print(f'Reward type **at evaluation time**: {reward_type}')
    image_obs_list = []
    if task.dataset_type in {'robosuite', 'robomimic'}:
        # Choosing a GPU for each episode in this way prevents all evluation env instances from running on the same GPU and potentially causing an OOM error.
        render_device = ep_id % torch.cuda.device_count() if device == 'cuda' else -1

        if env_meta is not None and 'robosuite' in env_meta:
            env_meta['robosuite']['env_kwargs']['render_gpu_device_id'] = render_device

        env = RobosuiteEnv(task, use_normalized_reward, full_state_mode,
                           env_meta=(env_meta['robosuite'] if env_meta is not None and 'robosuite' in env_meta else None),
                           render_gpu_device_id=render_device,
                           camera_names=camera_names,
                           image_size=image_size)
    elif task.dataset_type == 'metaworld':
        env = MetaWorldEnv(task, use_normalized_reward, full_state_mode,
                           env_meta=env_meta['metaworld'] if env_meta is not None and 'metaworld' in env_meta else None,
                           steps_at_goal=min_time_at_goal_for_success,
                           #render_gpu_device_id=render_device,
                           camera_name=camera_names[0],
                           image_size=image_size)
    elif task.dataset_type == 'd4rl':
        import d4rl
        env = d4rlEnv(task, full_state_mode)
    else:
        print("Simulator unavailable for environment type {}. Returning zeros for all metrics".format(task.dataset_type))
        episode_return = 0
        episode_success_return = 0
        episode_length = 0
        return episode_return, episode_success_return, episode_length

    obs = env.reset()

    if record_camera is not None and record_traj_dir is not None:
        record_image_key = f'{record_camera}_image'
        img_to_save = unprocess_image(obs[record_image_key])
        image_obs_list.append(img_to_save)

        if write_individual_images:
            cv2.imwrite(str(record_traj_dir/f"movie_{ep_id}_frame_{0}.png"), img_to_save)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"

    if not full_state_mode:
        context = {
            cam: torch.from_numpy(context[cam]).reshape(1, *env.obs_dims).to(device=device, dtype=torch.float32)
            for cam in camera_names
        }
        images = {
            cam: torch.from_numpy(obs[f'{cam}_image']).reshape(1, *env.obs_dims).to(device=device, dtype=torch.float32)
            for cam in camera_names
        }
        proprios = torch.from_numpy(obs['proprio']).reshape(1, env.proprio_dim).to(device=device, dtype=torch.float32)
    else:
        context = torch.from_numpy(context).reshape(1, env.obs_dims).to(device=device, dtype=torch.float32)
        full_states = torch.from_numpy(obs['state']).reshape(1, env.obs_dims).to(device=device, dtype=torch.float32)
        proprios = None

    actions = torch.zeros((0, env.action_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    print(f'Target return at evaluation time: {target_return}')
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_success, episode_length = 0, False, 0
    print("Starting an evaluation episode...")
    contig_success_steps = 0
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, env.action_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        with torch.no_grad():
            action = model.get_action_for_eval(
                context,
                (images if not full_state_mode else full_states),
                proprios,
                actions,
                rewards,
                target_return,
                timesteps
            )
        actions[-1] = action
        action = action.cpu().numpy()

        obs, reward, done, info = env.step(action)
        original_reward = reward
        reward = construct_rewards(np.array([reward]), [info['success'] if 'success' in info.keys() else False], reward_type)[0]

        if (t + 1) % 100 == 0:
            print(f"Did {t+1} steps")

        if record_camera is not None and record_traj_dir is not None:
            record_image_key = f'{record_camera}_image'
            img_to_save = unprocess_image(obs[record_image_key])
            image_obs_list.append(img_to_save)

            if write_individual_images:
                cv2.imwrite(str(record_traj_dir/f"movie_{ep_id}_frame_{t+1}.png"), img_to_save)

        if not full_state_mode:
            for cam in camera_names:
                cur_image = torch.from_numpy(obs[f'{cam}_image']).to(device=device, dtype=torch.float32).reshape(1, *env.obs_dims)
                images[cam] = torch.cat([images[cam], cur_image], dim=0)

            cur_proprio = torch.from_numpy(obs['proprio']).to(device=device).reshape(1, env.proprio_dim)
            proprios = torch.cat([proprios, cur_proprio], dim=0)
        else:
            cur_state = torch.from_numpy(obs['state']).to(device=device).reshape(1, env.obs_dims)
            full_states = torch.cat([full_states, cur_state], dim=0)

        rewards[-1] = reward
        pred_return = (target_return[0,-1] - reward) / discount
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += original_reward
        episode_length += 1

        if 'success' in info.keys() and info['success']:
            contig_success_steps += 1
        else:
            contig_success_steps = 0

        if contig_success_steps >= min_time_at_goal_for_success:
            done = True
            episode_success = True

        if done:
            break
    env.close()

    if task.dataset_type == 'd4rl':
        normalize_score = lambda returns: d4rl.get_normalized_score(task.name, returns)*100.0
        episode_return = normalize_score(episode_return)
    print(f"NATIVE undiscounted return: {episode_return}")
    print(f"Episode {('SUCCEEDED' if episode_success else 'FAILED')}")
    print(f"Episode length:{episode_length}")
    print("Finishing the evaluation episode...\n")

    if record_camera is not None and record_traj_dir is not None:
        import moviepy.editor as mpy
        clip = mpy.ImageSequenceClip(image_obs_list, fps=30)
        clip.write_videofile(str(record_traj_dir/f"movie_{ep_id}_{('SUCCESS' if episode_success else 'FAILURE')}.mp4"))

    return episode_return, episode_success, episode_length


# Helper function to do the analog of starmap where the args are a dict rather than a tuple
def _evaluate_from_dict(d):
    return evaluate_episode(**d)


def evaluate_parallel(conditions, task, model, reward_type, use_normalized_reward, env_meta, full_state_mode, device='cuda', num_workers=5, **kwargs):
    # Remember to put the model in evaluation mode!
    model.eval()
    model.to(device=device)
    arg_variants = [
        {'task': task, 'model': model, 'ep_id': id, 'use_normalized_reward': use_normalized_reward, 'reward_type': reward_type,
         'env_meta': env_meta, 'full_state_mode': full_state_mode, 'context': context, 'target_return': target_return,
         'device': device, **kwargs}
        for id, (context, target_return) in zip(range(len(conditions)), conditions)
    ]
    if num_workers > 0:
        model.share_memory()
        with mp.Pool(num_workers) as pool:
            results = pool.map(_evaluate_from_dict, arg_variants)
            pool.close()
            pool.join()
    else:
        results = [_evaluate_from_dict(d) for d in arg_variants]
    returns, succ_episodes, lengths = [], [], []
    for result in results:
        returns.append(result[0])
        succ_episodes.append(result[1])
        lengths.append(result[2])
    return returns, succ_episodes, lengths


def get_success_rate_evaluator(task, traj_data, env_metadata, cmdline_args, log_dir):
    # Taking the average return of all trajectories as the target is dangerous: we may have many trajectories with low return.
    # target_return = sum(traj['reward'].sum() for traj in val_data.trajectories) / len(val_data.trajectories)
    get_context = setup_context_sampler(cmdline_args['context_style'])

    def eval_episodes(model, step):
        conditions = []

        if cmdline_args['record_video']:
            record_traj_dir = (log_dir/f'videos_from_epoch_{step}')
            record_traj_dir.mkdir(parents=True)
        else:
            record_traj_dir = None

        returns = []

        # ASSUMPTIONS:
        # -- In goal-directed tasks, each successful (goal-reaching) trajectory has a higher score than every non-goal-reaching one.
        # -- Every goal-reaching trajectory stays at a goal state once it reaches one.
        for traj in traj_data.trajectories:
            returns.append(discount_cumsum(traj['reward'], traj['success'][-1], cmdline_args['discount'])[0])

        returns.sort(reverse=True)
        # [top_return_LO, top_return_HI] is the range of returns corresponding to the top cmdline_args['top_return_fraction'] fraction of
        # the demonstration trajectories
        top_return_LO = returns[math.ceil(cmdline_args['top_return_fraction'] * len(returns)) - 1]
        top_return_HI = returns[0]

        if not cmdline_args['bc_learning_mode']:
            print(f"Top return range: {top_return_LO} -- {top_return_HI}")

        for e in range(cmdline_args['num_eval_episodes']):
            while True:
                val_traj = random.choice(traj_data.trajectories)
                context, is_valid = get_context(val_traj, 0, len(val_traj['reward']))
                if is_valid:
                    break

            target_return = (top_return_LO + random.random() * (top_return_HI - top_return_LO))
            # If the learning mode *is* BC (as opposed to offline RL), then we will ignore
            # target return during conditioning, so its value won't matter.
            if not cmdline_args['bc_learning_mode']:
                print(f"Target return for episode {e}: {target_return}")

            conditions.append((context, target_return))

        returns, succ_episodes, lengths = evaluate_parallel(
            conditions, task, model,
            device=cmdline_args.get('device', 'cuda'),
            use_normalized_reward=cmdline_args['normalize_reward'],
            reward_type=cmdline_args['reward_type'],
            env_meta=env_metadata,
            full_state_mode = globals.full_state_mode,
            min_time_at_goal_for_success=cmdline_args['min_time_at_goal_for_success'],
            camera_names=parse_comma_sep_param_value(cmdline_args['camera_names']),
            image_size=cmdline_args['image_size'],
            num_workers=cmdline_args['num_eval_workers'],
            max_ep_len=cmdline_args['max_eval_episode_len'],
            discount=cmdline_args['discount'],
            record_camera=(DEFAULT_CAM[task.dataset_type] if cmdline_args['record_camera'] is None else cmdline_args['record_camera']),
            record_traj_dir=record_traj_dir
        )

        num_succ = len([s for s in succ_episodes if s is True])

        print(f'EPOCH {step} SUCCESS RATE: {num_succ/len(succ_episodes)*100}%')
        print(f'EPOCH {step} EXPECTED NATIVE RETURN: {np.mean(returns)}')
        print(f'EPOCH {step} MEAN EPISODE LENGTH: {np.mean(lengths)}')

        return {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'length_mean': np.mean(lengths),
            'length_std': np.std(lengths)
        }

    return eval_episodes


def get_validation_error_evaluator(dataset, cmdline_args, device):
    get_val_batch = setup_batch_sampler(dataset, cmdline_args['context_style'], cmdline_args, device)

    def validation_error(model, iter):
        errors = []
        for _ in range(cmdline_args['validation_samples']):
            contexts, images, states, actions, rewards, rtg, timesteps, attention_mask = get_val_batch(cmdline_args['batch_size'],
                                                                                                       cmdline_args['target_frame_rate'],
                                                                                                       cmdline_args['pad_frame_gaps'])
            with torch.no_grad():
                action_preds = model.forward(
                    contexts, images, states,
                    actions if isinstance(model.module, PLEX) else actions[:,:-1],
                    rewards,
                    rtg,
                    timesteps,
                    mask=attention_mask,
                )[0]

            if isinstance(model.module, PLEX):
                act_dim = action_preds.shape[2]
                attention_mask_shortened = attention_mask[:,:-cmdline_args['future_step']]
                action_preds = action_preds.reshape(-1, act_dim)[attention_mask_shortened.reshape(-1) > 0]
                action_target = torch.clone(actions[:,:-cmdline_args['future_step']]).reshape(-1, act_dim)[attention_mask_shortened.reshape(-1) > 0]
            else:
                action_target = actions[:,-1]

            # We are negating the error here for consistency with other metrics, which are maximization metrics.
            error = -torch.mean((action_preds - action_target) ** 2).item()
            errors.append(error)
        return {
            f'neg_val_error': np.mean(errors)
        }
    return validation_error


def get_finetuning_based_evaluator(ft_based_val_train_data, ft_based_val_val_data, trainable_params, cmdline_args, device):
    batch_fns = {task_name: setup_batch_sampler(task_data, cmdline_args['context_style'], cmdline_args, device) for task_name, task_data in ft_based_val_train_data.items()}
    val_fns = {task_name: get_validation_error_evaluator(task_data, cmdline_args, device) for task_name, task_data in ft_based_val_val_data.items()}

    def finetune_eval(model, step):
        print(f'==== STARTING EVALUATION FOR EPOCH {step} ====')
        model_state = deepcopy(model.state_dict())
        val_errors = {}
        for task_name, _ in ft_based_val_train_data.items():
            print(f'*** For task {task_name}... ***')
            if len(ft_based_val_train_data[task_name].trajectories) > 0:
                # Reset model for new task
                model.load_state_dict(model_state)
                batch_fn, val_fn = batch_fns[task_name], val_fns[task_name]
                trainer = setup_trainer(batch_fn, cmdline_args['finetune_learning_rate'], [], model, trainable_params, cmdline_args)
                trainer.train_iteration(
                    num_steps=cmdline_args['num_steps_per_ft_eval_iter'],
                    iter_num=f'finetune {task_name}',
                    print_fn=print
                )
            else:
                val_fn = val_fns[task_name]
            val_error_dict = val_fn(model, step)
            val_errors[task_name] = val_error_dict['neg_val_error']
        val_errors[f'neg_val_error'] = sum(val_errors.values()) / len(val_errors)
        return val_errors

    return finetune_eval
