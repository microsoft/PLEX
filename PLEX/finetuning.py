import os
import torch
import argparse
import sys
from PLEX.util.data import TrajectoryDataset, load_data, setup_batch_sampler, train_val_split
from PLEX.util.misc import parse_tasks, setup_essentials, setup_model, set_trainable_params, setup_trainer, run_training
from PLEX.util.evaluators import get_success_rate_evaluator, get_validation_error_evaluator
from PLEX.util.cmdline import add_common_args, add_conditioning_args
from PLEX.util.log import setup_wandb_logging


def finetune(cmdline_args):
    os.environ["NCCL_DEBUG"] = "INFO"
    print("=== Finetuning ===")
    parser = argparse.ArgumentParser()
    # Add all relevant command-line arguments
    add_common_args(parser)
    add_conditioning_args(parser)
    parser.add_argument('--finetune_learning_rate', type=float, default=1e-5)
    parser.add_argument('--finetune_steps_per_iter', type=int, default=100)
    parser.add_argument('--target_task', type=str, default=None)
    parser.add_argument('--max_target_trajectories', type=int, default=None)

    # Parse them and validate them
    args = parser.parse_args(cmdline_args)
    args = vars(args)
    if not args['bc_learning_mode']:
        assert 'reward' not in args['modalities_to_mask'], "If the model is expected to condition on returns, then they should not be masked out."

    # NOTE: The arguments below aren't actual command-line arguments. We are just addeing them to args[] out of convenience.
    # Note also that during finetuning we set predicted_inverse_dynamics_loss_weight=1, i.e., **in case the
    # finetuning trajectories contain actions**, we adapt PLEX's based on the predicted observation latents
    # from it planner PL rather than based on the actual ("grounded") observation latents contained
    # in finetuning trajectories.
    if args['model'] == 'PLEX':
        args['grounded_inverse_dynamics_loss_weight'] = 0
        args['predicted_inverse_dynamics_loss_weight'] = 1
        args['future_prediction_loss_weight'] = 1

    log, log_to_wandb, timer, data_shuffling_rng, device, camera_names, modalities_to_mask, data_dir, common_env_metadata_dict = setup_essentials(args)
    # NOTE: common_env_metadata_dict may be modified by the calls to load_data below.

    # Load data: target-task trajectories
    target_tasks, target_max_trajs = parse_tasks(args['target_task'], args['robot'], args['max_target_trajectories'])
    target_task = target_tasks[0]

    data = load_data(log,
                        data_dir,
                        target_tasks,
                        max_trajectories=target_max_trajs,
                        discount=args['discount'],
                        camera_names=camera_names,
                        image_size=args['image_size'],
                        target_frame_rate=args['target_frame_rate'],
                        normalize_rewards=args['normalize_reward'],
                        reward_type=args['reward_type'],
                        common_env_metadata_dict=common_env_metadata_dict,
                        data_shuffling_rng=data_shuffling_rng)

    assert len(data.keys()) == 1, f"There should be only one target task. Discovered {len(data.keys())}: {data.keys()}"
    #assert args['validation_tasks'] is None, f"Validation tasks other than the target tasks aren't used during finetuning and were likely specified erroneously: {args['validation_tasks']}."

    # Train/test split
    # NOTE: we don't actually need create the split if args['best_metric'] == 'evaluation/success_rate'
    if args['best_metric'] == 'evaluation/success_rate':
        print("WARNING: since the evaluation metric is success rate, the training-validation split of the target task data will be ignored, and all target-task trajectories will be used for training.")
    train_trajectories, val_trajectories = train_val_split(data[target_task.name], args['validation_frac'])
    target_all_data = TrajectoryDataset(data[target_task.name], camera_names, contextual=True)
    print(f"Total target trajectories: {len(target_all_data)}")
    target_train_data = TrajectoryDataset(train_trajectories, camera_names, contextual=True)
    target_val_data = TrajectoryDataset(val_trajectories, camera_names, contextual=True)
    del train_trajectories
    del val_trajectories
    log(f'{len(target_train_data.trajectories)} train and {len(target_val_data.trajectories)} validation trajectories')

    # Instantiate a model
    model, trainable_param_spec = setup_model(args,
                                              target_task,
                                              log,
                                              device,
                                              camera_names,
                                              modalities_to_mask,
                                              data_dir,
                                              args['bc_learning_mode'])

    # If the number of training iterations is 0, we are being asked to just evaluate the model
    if args['max_iters'] == 0:
        print("--------------- RUNNING IN EVALUATION MODE ----------------")
        # We are in the evaluation mode
        # Note that for evaluation, we are using *all* the demonstration data for the task, not just the validation data.
        # This is because get_success_rate_evaluator will use the demo trajectories only for sampling the goals/contexts.
        # We allow using the same contexts during both training and evaluation.
        evaluator = get_success_rate_evaluator(target_task, target_all_data, common_env_metadata_dict, args, log.dir)
        dummy_iter_num = 0
        outputs = evaluator(model, dummy_iter_num)

        logs = dict()
        for k, v in outputs.items():
            logs[f'evaluation/{k}'] = [v]

        for k, v in logs.items():
            print(f'{k}: {v[0]}')

        print("--------------- FINISHED EVALUATION ----------------")
        return logs

    # Otherwise, prepare the model for training
    trainable_params = set_trainable_params(model, trainable_param_spec, log)

    # Instantiate a batch sampler over the training data we loaded above
    if args['best_metric'] == 'evaluation/neg_val_error':
        batch_sampler = setup_batch_sampler(target_train_data, args['context_style'],  args, device)
    else:
        # Recall from above that if the metric is success rate, we use all target task data for training,
        # without allocating any of this data for validation.
        batch_sampler = setup_batch_sampler(target_all_data, args['context_style'],  args, device)

    # Setup a model evaluator
    eval_fn_dict = {'evaluation/neg_val_error': get_validation_error_evaluator(target_val_data, args, device),
                    'evaluation/success_rate': get_success_rate_evaluator(target_task, target_all_data, common_env_metadata_dict, args, log.dir)}
    eval_fns = [eval_fn_dict[args['best_metric']]]

    # Instantiate a trainer
    trainer = setup_trainer(batch_sampler,
                          args['finetune_learning_rate'],
                          eval_fns,
                          model,
                          trainable_params,
                          args)

    if log_to_wandb:
        group_name = f'{args["robot"]}_target-{target_task.name}'
        setup_wandb_logging(group_name, args)

    # Run training
    model_name_prefix = 'finet_' + args['model'] + target_task.name + '__'
    metric_values = run_training(trainer, model, args['finetune_steps_per_iter'], model_name_prefix, args, log, log_to_wandb, timer)
    return metric_values


if __name__ == '__main__':

    # Required for multiprocessing with CUDA, and must happen in if __name__ == '__main__' block
    torch.multiprocessing.set_start_method('spawn')

    print(sys.argv[1:])
    finetune(sys.argv[1:])