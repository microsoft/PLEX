# Helper to add boolean command-line arguments because argparse doesn't handle it well.
def add_boolean_arg(parser, name, true, false, default):
    assert true.startswith('--') and false.startswith('--')
    assert type(default) is bool
    true_false = parser.add_mutually_exclusive_group()
    true_false.add_argument(true, dest=name, action='store_true')
    true_false.add_argument(false, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def add_conditioning_args(parser):
    # Chooses between behavior cloning mode (the default, involves conditioning only on a goal, if available)
    # and offline RL mode (involves conditioning on a goal, if available, and on a return).
    add_boolean_arg(parser, 'bc_learning_mode', true='--bc_learning_mode', false='--orl_learning_mode', default=True)
    parser.add_argument('--context_style', type=str, default='first-success')
    add_boolean_arg(parser, 'context_from_same_traj', true='--context_from_same_traj', false='--context_from_diff_traj', default=False)
    # reward_type can be 'native', 'negative', 'random', 'zero', or 'sparse'.
    parser.add_argument('--reward_type', type=str, default='native')
    add_boolean_arg(parser, 'normalize_reward', true='--normalize_reward', false='--use_raw_reward', default=False)
    parser.add_argument('--discount', type=float, default=0.99)


def add_common_args(parser):
    # Logging
    parser.add_argument('--log_dir', type=str, default='~/logs')
    parser.add_argument('--log_id', type=str, default=None)
    parser.add_argument('--log_to_wandb', '-w', action='store_true')

    # General setup
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    # Core model
    parser.add_argument('--model', type=str, default='DT')
    # This is the load path for the starting model. If None, the starting model is initialized randomly.
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--modalities_to_mask', type=str, default='action')
    parser.add_argument('--impute_style', type=str, default='trainable')

    # Parameters for the Gaussian action head, if used
    parser.add_argument('--std_min', type=float, default=0.001)
    parser.add_argument('--std_max', type=float, default=1.0)

    ### Decision transformer parameters
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=None) # The default is None to easily detect when this pipeline is running the DT model unintentionally.
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=128) # NOTE: embed_dim must be a multiple of n_head!
    parser.add_argument('--transformer_tune_style', type=str, default=None,
                        choices=['all', 'last_block', 'linear_probe', 'none'])
    ### PLEX parameters
    parser.add_argument('--future_step', type=int, default=1)
    parser.add_argument('--obs_pred.n_layer', type=int, default=None)
    parser.add_argument('--obs_pred.n_head', type=int, default=None)
    parser.add_argument('--obs_pred.K', type=int, default=None)
    parser.add_argument('--obs_pred.transformer_tune_style', type=str, default=None,
                        choices=['all', 'last_block', 'linear_probe', 'none'])

    parser.add_argument('--inv_d_pred.n_layer', type=int, default=None)
    parser.add_argument('--inv_d_pred.n_head', type=int, default=None)
    parser.add_argument('--inv_d_pred.K', type=int, default=None)
    parser.add_argument('--inv_d_pred.transformer_tune_style', type=str, default=None,
                        choices=['all', 'last_block', 'linear_probe', 'none'])
    ### This applies only to transformer-based models
    add_boolean_arg(parser, 'relative_position_encodings', true='--relative_position_encodings', false='--absolute_position_encodings', default=True)
    parser.add_argument('--action_output_type', type=str, default='deterministic',
                        choices=['deterministic', 'gaussian', 'gaussian_mixture'])

    # Image encoder
    parser.add_argument('--image_encoder_arch', type=str, default='resnet18')
    parser.add_argument('--image_encoder_load', type=str, default=None)
    parser.add_argument('--pool_type', type=str, default='SpatialSoftmax')
    parser.add_argument('--image_encoder_tune_style', type=str, default='all') # none, fc, lastN (N an integer), or all

    # Data
    parser.add_argument('--data_dir', type=str, default='~/data')
    # --camera_names can have a special value FULL_STATE.
    # FULL_STATE means that the agent should use the full_state field returned by the data/env, and should *not* use proprio states.
    # In this case, the encoder is automatically set to be a linear layer mapping the full state dimentsion to the model's hidden dimnesion.
    # The image_size should then have the size M,1 or 1,N, where M or N are the length of the full state vectors.
    parser.add_argument('--camera_names', type=str, default='agentview') # E.g., --camera_names=agentview,robot0_eye_in_hand
    # If --image_size is a single number N, the image is interpreted to be of dimensions N x N.
    # If it is two numbers -- M,N -- the image is interpreted to have height M and width N.
    # NOTE: If --image_size is two numbers -- M,N as above -- and either M or N is 1, the image contents are interpreted as
    # an image embedding vector. The --image_encoder_arch is then ignored, and the encoder is automatically set to be a linear layer mapping
    # the embedding dimension to the model's hidden dimnesion.
    parser.add_argument('--image_size', type=int, default=84)
    # Frames-per-second for the desired frame rate (usually, a target task's). The default is to ignore frame rates.
    parser.add_argument('--target_frame_rate', type=int, default=None)
    add_boolean_arg(parser, 'pad_frame_gaps', default=True,
                    true='--pad_frame_gaps', false='--copy_into_frame_gaps')
    # Dynamics and action spaces are generally problem-specific, so we use robot-specifc data for them, as well as for validation tasks.
    parser.add_argument('--robot', type=str, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Evaluation
    # What fraction of the top demo trajectory returns will be used during evaluation?
    # NOTE: this parameter is relevant only if we are in the offline RL mode, not BC mode.
    parser.add_argument('--top_return_fraction', type=float, default=0.2)
    parser.add_argument('--best_metric', type=str, default='evaluation/neg_val_error', choices=['evaluation/neg_val_error', 'evaluation/return_mean'])
    # NOTE: during pretraining, --validation_frac applies *only* to tasks specified by --validation_tasks,
    # and specify the fraction of these tasks' trajectrories that will be used for validation.
    #
    # NOTE: The remaining validation trajectories of these tasks will be used for pretraining.
    # I.e., if you want all data of the --validation_tasks to be used only for validation and none for pretraining,
    # set --validation_frac=1.0 (or just don't specify --validation_frac at all, since 1.0 is the default).
    #
    # During finetuning, --validation_frac applies only to --target_task, and --validation_tasks must be None.
    #
    # NOTE: If during finetuning --best_metric is evaluation/return_mean (i.e., success rate),
    # --validation_frac is ignored and all of --target_task's trajectories are used for training. In this case,
    # validation loss isn't computed.
    #
    # NOTE: the following parameters are relevant only if best_metric is negative validation error.
    parser.add_argument('--validation_frac', type=float, default=1.0)
    parser.add_argument('--validation_samples', type=int, default=100) # how many sample batches on which to measure error
    # NOTE: the following parameters are relevant only if best_metric is success rate.
    parser.add_argument('--max_eval_episode_len', type=int, default=500)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--num_eval_workers', type=int, default=5)
    parser.add_argument('--min_time_at_goal_for_success', type=int, default=5) # Minimum number of consecutive time steps an agent should spend at a goal state during an evaluation episode for the episode to terminate with a success.
    parser.add_argument('--record_camera', type=str, default=None)
    add_boolean_arg(parser, 'record_video', true='--record_video', false='--no_video', default=False)


def add_common_pretraining_args(parser):
    # Data
    # Max trajectories *per pretraining task*
    parser.add_argument('--max_pretrain_trajectories', type=int, default=None)
    parser.add_argument('--validation_tasks', type=str, default=None)
    # Max trajectories *per validation task*
    parser.add_argument('--max_validation_trajectories', type=int, default=None) # to test low-data regime

    # Training
    parser.add_argument('--pretrain_learning_rate', type=float, default=1e-4)
    parser.add_argument('--pretrain_steps_per_iter', type=int, default=100)
    # A separate parameter for the number of steps/iter to be used during finetuning-based evaluation **during pretraining**.
    # (There will be just 1 iteration, and it will have --num_steps_per_ft_eval_iter steps.)
    parser.add_argument('--num_steps_per_ft_eval_iter', type=int, default=0)
    # The following parameter is also used during the (separate) finetuning stage.
    parser.add_argument('--finetune_learning_rate', type=float, default=1e-5)
