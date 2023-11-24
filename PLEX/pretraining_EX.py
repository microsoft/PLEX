import os
import torch
import argparse
import sys
from PLEX.util.data import TrajectoryDataset, load_data, setup_batch_sampler, train_val_split
from PLEX.util.misc import parse_tasks, setup_essentials, setup_model, set_trainable_params, setup_trainer, run_training
from PLEX.util.evaluators import get_finetuning_based_evaluator
from PLEX.util.cmdline import add_common_args, add_common_pretraining_args
from PLEX.util.log import setup_wandb_logging


def pretrain_EX(cmdline_args):
    os.environ["NCCL_DEBUG"] = "INFO"
    print("=== Pretraining the Execuctor ===")
    parser = argparse.ArgumentParser()

    # Add all relevant command-line arguments
    add_common_args(parser)
    add_common_pretraining_args(parser)
    parser.add_argument('--noncontextual_pretrain_tasks', type=str, default=None)

    # Parse them and validate them
    args = parser.parse_args(cmdline_args)
    args = vars(args)
    assert args['best_metric'] != 'evaluation/return_mean', 'Currently, evaluation/return_mean is not a valid metric for pretraining. Use evaluation/neg_val_error instead.'

    # These parameters are needed only for evaluating the model. Since at the current stage we are pretraining just the EX
    # (inverse dynamics) part of PLEX, the values of the parameters other than bc_learning_mode don't matter, since at the
    # end of this stage the model won't yet know how to handle goal contexts.
    args['bc_learning_mode'] = True
    args['context_style'] = 'blank'
    args['context_from_same_traj'] = False
    args['reward_type'] = 'native'
    args['normalize_reward'] = False
    args['discount'] = 0

    # If we are pretraining a PLEX model, for loss computation we should use *just* the inverse dynamics predictions
    # computed based on obs. in the training trajectories (not predictions of the obs., and not predictions of inv.d. based
    # on predicted obs. -- both of these need context to be provided, and we want inv.d. to be context-independent).
    #
    # NOTE: The arguments below aren't actual command-line arguments. We are just addeing them to args[] out of convenience.
    if args['model'] == 'PLEX':
        args['grounded_inverse_dynamics_loss_weight'] = 1
        args['predicted_inverse_dynamics_loss_weight'] = 0
        args['future_prediction_loss_weight'] = 0

    log, log_to_wandb, timer, data_shuffling_rng, device, camera_names, modalities_to_mask, data_dir, common_env_metadata_dict = setup_essentials(args)
    # NOTE: common_env_metadata_dict may be modified by the calls to load_data below.

    # Load data: context-agnostic dynamics data and validation trajectories (if any)
    noncontextual_pretrain_tasks, noncontextual_pretrain_max_trajs = parse_tasks(args['noncontextual_pretrain_tasks'], args['robot'], args['max_pretrain_trajectories'])
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
                        # This doesn't matter for evaluation of pretrained executor.
                        normalize_rewards=False,
                        # This doesn't matter for evaluation of pretrained executor.
                        reward_type='sparse',
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

    # Then, load context-agnostic dynamics data
    print("Reading context-agnostic dynamics data...")
    data = load_data(log,
                    data_dir,
                    noncontextual_pretrain_tasks,
                    video_only=False,
                    max_trajectories=noncontextual_pretrain_max_trajs,
                    camera_names=camera_names,
                    image_size=args['image_size'],
                    target_frame_rate=args['target_frame_rate'],
                    # This doesn't matter for evaluation of pretrained executor.
                    normalize_rewards=False,
                    # This doesn't matter for evaluation of pretrained executor.
                    reward_type='sparse',
                    common_env_metadata_dict=common_env_metadata_dict,
                    data_shuffling_rng=data_shuffling_rng)
    for k, v in data.items():
        log(f'{len(v)} trajectories for task {k}')
        all_pretrain_trajectories.extend(v)

    noncontextual_pretrain_data = TrajectoryDataset(all_pretrain_trajectories, camera_names, False)
    del data

    # Instantiate a model
    model, trainable_param_spec = setup_model(args,
                                              noncontextual_pretrain_tasks[0],
                                              log,
                                              device,
                                              camera_names,
                                              modalities_to_mask,
                                              data_dir,
                                              bc_mode=False)

    # Prepare the model for training
    trainable_params = set_trainable_params(model, trainable_param_spec, log)

    # Instantiate a batch sampler over the training data we loaded above
    batch_sampler = setup_batch_sampler(noncontextual_pretrain_data, None, args, device)

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
    model_name_prefix = ('pretr_' + args['model'] + '__' if args['model'] != 'PLEX' else 'pretr_EX__')
    metric_values = run_training(trainer, model, args['pretrain_steps_per_iter'], model_name_prefix, args, log, log_to_wandb, timer)
    return metric_values


if __name__ == '__main__':

    # Required for multiprocessing with CUDA, and must happen in if __name__ == '__main__' block
    torch.multiprocessing.set_start_method('spawn')

    print(sys.argv[1:])
    pretrain_EX(sys.argv[1:])