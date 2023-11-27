from PLEX.pretraining_EX import pretrain_EX
from PLEX.pretraining_PL import pretrain_PL
from PLEX.finetuning import finetune
import argparse
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--training_stage", type=str, default='ft', help = "The training stage. Can be 'ex' (pretaining the EXecutor), 'pl' (pretraining the PLanner), or 'ft' (finetuning a pretrained PLEX)")
    parser.add_argument("-d", "--data_dir", type=str, default='store/data', help = "Directory path where the training data is.")
    parser.add_argument("-l", "--log_dir", type=str, default='store/logs', help = "Directory path where to output logs and model checkpoints.")
    parser.add_argument("-m", "--model_file", type=str, default=None, help = "Model file path.")
    parser.add_argument("-t", "--target_task", type=str, default=None, help = "Directory path where the target task's data is. NOTE: applicable only if the training stage is 'ft' (finetuning).")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help = "Number of worker for running the evaluation episodes. NOTE: applicable only if the training stage is 'ft' (finetuning).")

    args = parser.parse_args()

    common_flags = ['--relative_position_encodings', '--bc_learning_mode']

    common_args = {
                    'seed': str(random.randint(0, 1000000)),
                    'data_dir': args.data_dir,
                    'log_dir': args.log_dir,
                    'robot': 'Sawyer',
                    'camera_names': 'corner',
                    'modalities_to_mask': 'proprio,action',
                    'record_camera': 'corner',
                    'image_size': '84',
                    'reward_type': 'sparse',
                    'image_encoder_arch': 'resnet18',
                    'impute_style': 'trainable',
                    'embed_dim': '256',
                    'future_step': '1',
                    'activation_function': 'relu',
                    'device': 'cuda',
                    'dropout': '0.2',
                    'weight_decay': '1e-05',
                    'warmup_steps': '200',
                    'batch_size': '256',
                    'action_output_type': 'deterministic',
                    'model': 'PLEX',
                    'obs_pred.n_layer': '3',
                    'obs_pred.n_head': '4',
                    'obs_pred.K': '30',
                    'inv_d_pred.n_layer': '3',
                    'inv_d_pred.n_head': '4',
                    'inv_d_pred.K': '30'
                  }

    common_pretraining_flags = ['--no_video']

    common_pretraining_args = {
                                'pretrain_learning_rate': '0.0005',
                                'pretrain_steps_per_iter': '250',
                                'num_steps_per_ft_eval_iter': '0',
                                'best_metric': 'evaluation/neg_val_error',
                                'validation_frac': '1.0',
                                'validation_samples': '30',
                                # Validation tasks can be any MW tasks -- we don't use validation error to stop training.
                                # We use the target tasks as validation tasks.
                                'validation_tasks': 'metaworld/hand-insert-v2/--TARGET_ROBOT--/noise0/,metaworld/door-lock-v2/--TARGET_ROBOT--/noise0/,metaworld/door-unlock-v2/--TARGET_ROBOT--/noise0/,metaworld/box-close-v2/--TARGET_ROBOT--/noise0/,metaworld/bin-picking-v2/--TARGET_ROBOT--/noise0/',
                              }

    cmdline_args = common_flags
    for k in common_args:
        cmdline_args.append('--' + k)
        cmdline_args.append(common_args[k])

    if args.training_stage == 'ex':
        cmdline_args.extend(common_pretraining_flags)
        for k in common_pretraining_args:
            cmdline_args.append('--' + k)
            cmdline_args.append(common_pretraining_args[k])

        cmdline_args.extend([
                            '--max_iters', '10',
                            # To pretrain the executor, use 75 play trajectories per task.
                            '--max_pretrain_trajectories', '75',
                            # During executor pretraining, we adapt both the executor's and the encoder's weights but keep the planner frozen.
                            '--image_encoder_tune_style', 'all',
                            '--obs_pred.transformer_tune_style', 'none',
                            '--inv_d_pred.transformer_tune_style', 'all',
                            # Use the dynamics data from Meta-World ML50's 5 downstream environments.
                            '--noncontextual_pretrain_tasks', 'metaworld/hand-insert-v2/--TARGET_ROBOT--/noise0.5/,metaworld/door-lock-v2/--TARGET_ROBOT--/noise0.5/,metaworld/door-unlock-v2/--TARGET_ROBOT--/noise0.5/,metaworld/box-close-v2/--TARGET_ROBOT--/noise0.5/,metaworld/bin-picking-v2/--TARGET_ROBOT--/noise0.5/',
                            ])
        pretrain_EX(cmdline_args)

    elif args.training_stage == 'pl':
        cmdline_args.extend(common_pretraining_flags)
        for k in common_pretraining_args:
            cmdline_args.append('--' + k)
            cmdline_args.append(common_pretraining_args[k])

        cmdline_args.extend([
                            '--max_iters', '10',
                            # To pretrain the planner, use all (100) available video demonstrations per task.
                            '--max_pretrain_trajectories', 100,
                            '--context_style', 'first-success',
                            '--context_from_diff_traj',
                            # During planner pretraining, we want to keep the encoder and the executor's weights frozen, adapting only the weights of the planner itself.
                            '--image_encoder_tune_style', 'none',
                            '--obs_pred.transformer_tune_style', 'all',
                            '--inv_d_pred.transformer_tune_style', 'none',
                            # For pretraining, use video demonstrations from Meta-World ML50's 45 pretraining tasks.
                            '--video_tasks', 'metaworld/pick-out-of-hole-v2/Sawyer/noise0/,metaworld/door-open-v2/Sawyer/noise0/,metaworld/pick-place-wall-v2/Sawyer/noise0/,metaworld/assembly-v2/Sawyer/noise0/,metaworld/faucet-close-v2/Sawyer/noise0/,metaworld/coffee-pull-v2/Sawyer/noise0/,metaworld/plate-slide-back-side-v2/Sawyer/noise0/,metaworld/dial-turn-v2/Sawyer/noise0/,metaworld/stick-push-v2/Sawyer/noise0/,metaworld/sweep-into-v2/Sawyer/noise0/,metaworld/handle-pull-side-v2/Sawyer/noise0/,metaworld/drawer-open-v2/Sawyer/noise0/,metaworld/window-open-v2/Sawyer/noise0/,metaworld/button-press-v2/Sawyer/noise0/,metaworld/assembly-v2/Sawyer/noise0/,metaworld/faucet-close-v2/Sawyer/noise0/,metaworld/coffee-pull-v2/Sawyer/noise0/,metaworld/plate-slide-back-side-v2/Sawyer/noise0/,metaworld/dial-turn-v2/Sawyer/noise0/,metaworld/stick-push-v2/Sawyer/noise0/,metaworld/sweep-into-v2/Sawyer/noise0/,metaworld/handle-pull-side-v2/Sawyer/noise0/,metaworld/shelf-place-v2/Sawyer/noise0/,metaworld/basketball-v2/Sawyer/noise0/,metaworld/button-press-topdown-v2/Sawyer/noise0/,metaworld/button-press-topdown-wall-v2/Sawyer/noise0/,metaworld/button-press-wall-v2/Sawyer/noise0/,metaworld/coffee-button-v2/Sawyer/noise0/,metaworld/coffee-push-v2/Sawyer/noise0/,metaworld/disassemble-v2/Sawyer/noise0/,metaworld/door-close-v2/Sawyer/noise0/,metaworld/drawer-close-v2/Sawyer/noise0/,metaworld/faucet-open-v2/Sawyer/noise0/,metaworld/hammer-v2/Sawyer/noise0/,metaworld/handle-press-side-v2/Sawyer/noise0/,metaworld/handle-press-v2/Sawyer/noise0/,metaworld/handle-pull-v2/Sawyer/noise0/,metaworld/lever-pull-v2/Sawyer/noise0/,metaworld/peg-insert-side-v2/Sawyer/noise0/,metaworld/reach-v2/Sawyer/noise0/,metaworld/push-back-v2/Sawyer/noise0/,metaworld/push-v2/Sawyer/noise0/,metaworld/pick-place-v2/Sawyer/noise0/,metaworld/plate-slide-v2/Sawyer/noise0/,metaworld/plate-slide-side-v2/Sawyer/noise0/,metaworld/plate-slide-back-v2/Sawyer/noise0/,metaworld/peg-unplug-side-v2/Sawyer/noise0/,metaworld/soccer-v2/Sawyer/noise0/,metaworld/stick-pull-v2/Sawyer/noise0/,metaworld/push-wall-v2/Sawyer/noise0/,metaworld/reach-wall-v2/Sawyer/noise0/,metaworld/sweep-v2/Sawyer/noise0/,metaworld/window-close-v2/Sawyer/noise0/',
                            '--load_path', args.model_file
                            ])
        pretrain_PL(cmdline_args)

    elif args.training_stage == 'ft':

        cmdline_args.extend([
                            '--max_iters', '10',
                            # Use just 10 trajectories of the target task for finetunning, randomly sampled from the set of all (100) of that task's training trajectories.
                            '--max_target_trajectories', '10',
                            '--target_task', args.target_task,
                            '--context_style', 'first-success',
                            '--context_from_diff_traj',
                            # During finetuning we adapt just the last transformer layer of the planner, keeping the planner's other layers as well as the encoder and the executor frozen.
                            # We could adapt other parts of PLEX too, but it's unnecessary to reproduce the PLEX paper's results.
                            '--image_encoder_tune_style', 'none',
                            '--obs_pred.transformer_tune_style', 'last_block',
                            '--inv_d_pred.transformer_tune_style', 'none',
                            '--finetune_learning_rate', '0.0005',
                            '--finetune_steps_per_iter', '100',
                            '--best_metric', 'evaluation/success_rate',
                            '--max_eval_episode_len', '500',
                            '--num_eval_episodes', '50',
                            '--num_eval_workers', str(args.num_workers),
                            # Remove this flag if you don't want videos of evaluation trajectories to be recorded.
                            '--record_video',
                            '--load_path', args.model_file
                            ])
        finetune(cmdline_args)
    else:
        raise ValueError(f'Invalid training stage: {args.training_stage}')
