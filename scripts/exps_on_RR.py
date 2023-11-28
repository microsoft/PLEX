from PLEX.pretraining_EX import pretrain_EX
from PLEX.pretraining_PL import pretrain_PL
from PLEX.finetuning import finetune
import argparse
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", type=str, default='plex-rel', help = "The architecture to use. Can be 'plex-rel' (PLEX w/ relative position encoding), 'plex-abs' (PLEX w/ absolute position encoding), or 'dt' (DT, which uses global position encoding)")
    parser.add_argument("-d", "--data_dir", type=str, default='store/data', help = "Directory path where the training data is.")
    parser.add_argument("-l", "--log_dir", type=str, default='store/logs', help = "Directory path where to output logs and model checkpoints.")
    parser.add_argument("-t", "--target_task", type=str, default=None, help = "Directory path where the target task's data is. NOTE: applicable only if the training stage is 'ft' (finetuning).")
    parser.add_argument("-m", "--max_tt", type=int, default=75, help = "Number of target-task demonstrations to use for training.")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help = "Number of worker for running the evaluation episodes. NOTE: applicable only if the training stage is 'ft' (finetuning).")

    args = parser.parse_args()

    common_flags = ['--record_video', '--use_raw_reward', '--context_from_diff_traj']

    common_args = {
                    'seed': str(random.randint(0, 1000000)),
                    'data_dir': args.data_dir,
                    'log_dir': args.log_dir,
                    'robot': 'Panda',
                    'target_task': args.target_task,
                    'camera_names': 'agentview,robot0_eye_in_hand',
                    'modalities_to_mask': 'action',
                    'record_camera': 'agentview',
                    'image_size': '84',
                    'reward_type': 'sparse',
                    'discount': '0.999',
                    'image_encoder_arch': 'resnet18',
                    'impute_style': 'trainable',
                    'embed_dim': '256',
                    'future_step': '1',
                    'activation_function': 'relu',
                    'device': 'cuda',
                    'dropout': '0.2',
                    # Training arguments
                    'weight_decay': '1e-05',
                    'warmup_steps': '200',
                    'batch_size': '256',
                    'finetune_steps_per_iter': '500',
                    'finetune_learning_rate': '0.0005',
                    'action_output_type': 'deterministic',
                    'context_style': 'first-success',
                    'max_iters': '10',
                    'max_target_trajectories': str(args.max_tt),
                    # Evaluation arguments
                    'best_metric': 'evaluation/success_rate',
                    'num_eval_workers': str(args.num_workers),
                    'max_eval_episode_len': '700',
                    'num_eval_episodes': '50',
                    'validation_frac': '0.2',
                    'validation_samples': '20',
                    'image_encoder_tune_style': 'all',
                  }

    common_plex_flags = ['--bc_learning_mode']

    common_plex_args = {
                        'model': 'PLEX',
                        'obs_pred.n_layer': '3',
                        'obs_pred.n_head': '4',
                        'obs_pred.K': '30',
                        'inv_d_pred.n_layer': '3',
                        'inv_d_pred.n_head': '4',
                        'inv_d_pred.K': '30',
                        'obs_pred.transformer_tune_style': 'all',
                        'inv_d_pred.transformer_tune_style': 'all',
                        }

    cmdline_args = common_flags
    for k in common_args:
        cmdline_args.append('--' + k)
        cmdline_args.append(common_args[k])

    if args.arch == 'plex-rel':
        cmdline_args.extend(common_plex_flags)
        for k in common_plex_args:
            cmdline_args.append('--' + k)
            cmdline_args.append(common_plex_args[k])

        cmdline_args.append('--relative_position_encodings')

        finetune(cmdline_args)

    elif args.arch == 'plex-abs':
        cmdline_args.extend(common_plex_flags)
        for k in common_plex_args:
            cmdline_args.append('--' + k)
            cmdline_args.append(common_plex_args[k])

        cmdline_args.append('--absolute_position_encodings')

        finetune(cmdline_args)

    elif args.arch == 'dt':
        cmdline_args.extend([
                            '--orl_learning_mode',
                            '--absolute_position_encodings',
                            '--model', 'DT',
                            '--K', '30',
                            '--n_layer', '6',
                            '--n_head', '4',
                            '--transformer_tune_style', 'all',
                            '--top_return_fraction', '1.0'
                            ])

        finetune(cmdline_args)
    else:
        raise ValueError(f'Invalid training stage: {args.training_stage}')
