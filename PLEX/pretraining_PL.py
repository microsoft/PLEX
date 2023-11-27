import pdb
import numpy as np
import torch
import os
import argparse
import sys
from PLEX.util.data import TrajectoryDataset, load_data, setup_batch_sampler, train_val_split
from PLEX.util.misc import parse_tasks, setup_essentials, setup_model, set_trainable_params, setup_trainer, run_training
from PLEX.util.evaluators import get_finetuning_based_evaluator
from PLEX.util.cmdline import add_common_args, add_common_pretraining_args, add_conditioning_args
from PLEX.util.log import setup_wandb_logging


def pretrain_PL(cmdline_args):
    os.environ["NCCL_DEBUG"] = "INFO"
    print("=== Pretraining the Planner ===")
    parser = argparse.ArgumentParser()

    # Add all relevant command-line arguments
    add_common_args(parser)
    add_common_pretraining_args(parser)
    add_conditioning_args(parser)
    parser.add_argument('--video_tasks', type=str, default=None)

    # Parse them and validate them
    args = parser.parse_args(cmdline_args)
    args = vars(args)
    if not args['bc_learning_mode']:
        assert 'reward' not in args['modalities_to_mask'], "If the model is expected to condition on returns, then they should not be masked out."
    assert args['best_metric'] != 'evaluation/success_rate', 'Currently, evaluation/success_rate is not a valid metric for pretraining. Use evaluation/neg_val_error instead.'

    # If we are pretraining a PLEX model, for loss computation we should use *just* the obs. embedding predictions,
    # not predictions of inverse dynamics.
    #
    # NOTE: The arguments below aren't actual command-line arguments. We are just addeing them to args[] out of convenience.
    if args['model'] == 'PLEX':
        args['grounded_inverse_dynamics_loss_weight'] = 0
        args['predicted_inverse_dynamics_loss_weight'] = 0
        args['future_prediction_loss_weight'] = 1

    log, log_to_wandb, timer, data_shuffling_rng, device, camera_names, modalities_to_mask, data_dir, common_env_metadata_dict = setup_essentials(args)
    # NOTE: common_env_metadata_dict may be modified by the calls to load_data below.

    # Load data: videos and validation trajectories (if any)
    video_tasks, video_max_trajs = parse_tasks(args['video_tasks'])
    print(f'*** The validation tasks are: {args["validation_tasks"]} ***')
    validation_tasks, validation_max_trajs = parse_tasks(args['validation_tasks'], args['robot'], args['max_validation_trajectories'])

    all_pretrain_trajectories = []
    # First, load validation data, if any
    if validation_tasks:
        print("Reading validation tasks...")
        data = load_data(log,
                         data_dir,
                         validation_tasks,
                        # NOTE: the parameter that controls this is max_validation_trajectories, *NOT* max_pretrain_trajectories.
                        max_trajectories=validation_max_trajs,
                        camera_names=camera_names,
                        image_size=args['image_size'],
                        target_frame_rate=args['target_frame_rate'],
                        normalize_rewards=args['normalize_reward'],
                        reward_type=args['reward_type'],
                        common_env_metadata_dict=common_env_metadata_dict,
                        data_shuffling_rng=data_shuffling_rng)
        val_train_data, val_val_data = {}, {}
        for k, v in data.items():
            print(f'Splitting the data of validation task {k}...')
            train_trajectories, val_trajectories = train_val_split(v, args['validation_frac'])
            val_train_data[k] = TrajectoryDataset(train_trajectories, camera_names, True)
            val_val_data[k] = TrajectoryDataset(val_trajectories, camera_names, True)
            print(f'Stored {len(val_train_data[k].trajectories)} training and {len(val_val_data[k].trajectories)} validation trajectories for task {k}...')

            """
            If we don't have a finetuning stage for evaluating the pretrained model, use the training trajectories
            of the validation tasks for pretraining the model. These tasks' validation trajectories will still be used
            for computing the pretrained model's validation loss.
            """
            if args['num_steps_per_ft_eval_iter'] <= 0 and args['validation_frac'] < 1.0:
                print(f"NOTE: since we aren't doing finetuning for evaluation at pretraining time (num_steps_per_ft_eval_iter = {args['num_steps_per_ft_eval_iter']}), we'll use some of the trajectories from validation task {k} during pretraining. These trajectries are *not* in the validation split.")
                all_pretrain_trajectories.extend(train_trajectories)
        del data

    # Then, load video-only data
    print("Reading video-only data...")
    data = load_data(log,
                    data_dir,
                    video_tasks,
                    video_only=True,
                    max_trajectories=video_max_trajs,
                    camera_names=camera_names,
                    image_size=args['image_size'],
                    target_frame_rate=args['target_frame_rate'],
                    normalize_rewards=args['normalize_reward'],
                    reward_type=args['reward_type'],
                    common_env_metadata_dict=common_env_metadata_dict,
                    data_shuffling_rng=data_shuffling_rng)
    for k, v in data.items():
        log(f'{len(v)} videos for task {k}')
        all_pretrain_trajectories.extend(v)

    video_data = TrajectoryDataset(all_pretrain_trajectories, camera_names, True)
    del data

    # Instantiate a model
    model, trainable_param_spec = setup_model(args,
                                              video_tasks[0],
                                              log,
                                              device,
                                              camera_names,
                                              modalities_to_mask,
                                              data_dir,
                                              args['bc_learning_mode'])

    # Prepare the model for training
    trainable_params = set_trainable_params(model, trainable_param_spec, log)

    # Instantiate a batch sampler over the training data we loaded above
    batch_sampler = setup_batch_sampler(video_data, args['context_style'],  args, device)

    # NOTE: We should reconsider how finetuning-based evaluator works should it be allowed to modify only exactly
    # same set of parameters that training modifies (trainable_params) or a different one (e.g., just the head)?
    #
    # Either way, in the most common evaluation case, i.e., when this evaluator just runs the model against
    # the validation tasks' data without actually doing finetuning (args['num_steps_per_ft_eval_iter'] = 0),
    # this method works correctly now.
    eval_fns = [get_finetuning_based_evaluator(val_train_data, val_val_data, trainable_params, args, device)]

    # Instantiate a trainer
    trainer = setup_trainer(batch_sampler,
                          args['pretrain_learning_rate'],
                          eval_fns,
                          model,
                          trainable_params,
                          args)

    if log_to_wandb:
        group_name = f'{args["robot"]}_pretrain'
        setup_wandb_logging(group_name, args)

    # Run training
    model_name_prefix = ('pretr_' + args['model'] + '__' if args['model'] != 'PLEX' else 'pretr_PLEX__')
    metric_values = run_training(trainer, model, args['pretrain_steps_per_iter'], model_name_prefix, args, log, log_to_wandb, timer)
    return metric_values


if __name__ == '__main__':

    # Required for multiprocessing with CUDA, and must happen in if __name__ == '__main__' block
    torch.multiprocessing.set_start_method('spawn')

    print(sys.argv[1:])
    pretrain_PL(sys.argv[1:])