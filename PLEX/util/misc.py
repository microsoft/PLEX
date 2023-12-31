import numpy as np
import torch
import random
import wandb
import pickle
from collections import defaultdict
import PLEX.util.globals as globals
from PLEX.util.log import setup_logging
from PLEX.util.timer import Timer
from PLEX.envs.environments import *
from PLEX.models.trajectory_models.plex import PLEX
from PLEX.models.trajectory_models.mlp_bc import MLPBCModel
from PLEX.models.trajectory_models.decision_transformer import DecisionTransformer
from PLEX.training.trainer import Trainer


class TaskDescriptor:
    def __init__(self, task_descr_string):
        self.dataset_location = task_descr_string = task_descr_string.rstrip('/').lstrip('/ ')
        parts = task_descr_string.split('/')
        self.frame_rate = None

        assert parts[0] in {'robosuite', 'robomimic', 'libero', 'metaworld', 'bridge', 'bridge-v2', 'd4rl'}
        self.dataset_type = parts[0]
        assert self.dataset_type == 'bridge-v2' or self.dataset_type in TASK_NAMES, f"ERROR! {self.dataset_type} is not in dataset type-to-task names dict! Task descr string is {task_descr_string}."

        assert self.dataset_type == 'bridge-v2' or parts[1] in TASK_NAMES[self.dataset_type]
        self.name = parts[1]

        assert parts[2] in ROBOT_NAMES
        self.robot = parts[2]


def parse_comma_sep_param_value(comma_sep_param_value_str):
    param_values = [param_value.strip() for param_value in comma_sep_param_value_str.split(',')]
    return param_values


def parse_tasks(task_spec_str, robot=None, global_max_traj=None):
    if task_spec_str is None or task_spec_str == 'None':
        return [], []

    task_specs = parse_comma_sep_param_value(task_spec_str)
    descriptors = []
    max_trajs = []
    for task_spec in task_specs:
        if task_spec.startswith('(') and task_spec.endswith(')'):
            task_spec, max_traj = [part.strip('(): ') for part in task_spec.split(':')]
            max_trajs.append(int(max_traj))
        else:
            max_trajs.append(global_max_traj)

        if robot is None:
            task = task_spec
        else:
            # --TARGET_ROBOT-- is a reserved token that can't be used to name an actual robot.
            task = task_spec.replace('--TARGET_ROBOT--', robot)
            assert task != task_spec, 'Invalid task directory string: {}. Needs to contain the \"--TARGET_ROBOT--\" token'.format(task)

        descriptors.append(TaskDescriptor(task))
    return descriptors, max_trajs


# reward_type can be 'native', 'negative', 'random', 'zero', and 'sparse'.
def construct_rewards(original_rewards, successes, reward_type):
    if reward_type == 'sparse':
        rewards = np.asarray([sparse_reward(r) for r in successes])
    elif reward_type == 'native':
        rewards = original_rewards
    elif reward_type == 'negative':
        rewards = -original_rewards
    elif reward_type == 'zero':
        rewards = np.zeros_like(original_rewards)
    elif reward_type == 'random':
        rewards = np.random.rand(*original_rewards.shape)
    else:
        raise NotImplementedError
    return rewards


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def construct_data_dir_path(cmdline_args):
    data_dir = cmdline_args['data_dir'].replace('--TARGET_ROBOT--', cmdline_args['robot'])
    data_dir = Path(data_dir).expanduser()
    return data_dir


def setup_essentials(cmdline_args):
    set_seed(cmdline_args['seed'])
    data_shuffling_rng = np.random.RandomState(cmdline_args['seed'])
    log = setup_logging(cmdline_args)
    device = cmdline_args.get('device', 'cuda')
    log_to_wandb = cmdline_args.get('log_to_wandb', False)
    timer = Timer(log)

    camera_names = parse_comma_sep_param_value(cmdline_args['camera_names'])

    # Very important! This sets up observation preprocessing (such as resizing images to a desired size and swapping their format from HWC to CWH)
    # that will be done by the robomimic library to specified observation types when these observations are loaded from robomimic's h5py files or
    # generated by robosuite.
    if 'FULL_STATE' in camera_names:
        assert len(camera_names) == 1, "If FULL_STATE is present among camera names, it must be the only camera name."
        globals.full_state_mode = True
    else:
        globals.full_state_mode = False

    if not globals.full_state_mode:
        init_obs_preprocessing(camera_names, cmdline_args['image_size'])

    modalities_to_mask = parse_comma_sep_param_value(cmdline_args['modalities_to_mask'])
    data_dir = construct_data_dir_path(cmdline_args)
    common_env_metadata_dict = {'robosuite': None, 'metaworld': None, 'bridge': None}

    for modality in modalities_to_mask:
        assert modality in globals.MODALITIES

    return log, log_to_wandb, timer, data_shuffling_rng, device, camera_names, modalities_to_mask, data_dir, common_env_metadata_dict


def get_robot_dims(example_task, camera_names, image_size):
    if example_task.dataset_type == 'robosuite' or example_task.dataset_type == 'robomimic':
        env = RobosuiteEnv(example_task, use_normalized_reward=False, full_state_mode=globals.full_state_mode, camera_names=camera_names)
        env.close()
        return env.obs_dims, env.proprio_dim, env.action_dim
    elif example_task.dataset_type == 'metaworld':
        env = MetaWorldEnv(example_task, use_normalized_reward=False, full_state_mode=globals.full_state_mode, camera_name=camera_names[0])
        env.close()
        return env.obs_dims, env.proprio_dim, env.action_dim
    elif example_task.dataset_type == 'd4rl':
        env = d4rlEnv(example_task, full_state_mode=globals.full_state_mode)
        env.close()
        return env.obs_dims, env.proprio_dim, env.action_dim
    elif example_task.dataset_type == 'bridge' or example_task.dataset_type == 'bridge-v2':
        obs_dims = (3, image_size, image_size)
        proprio_dim = 7
        action_dim = 7
        return obs_dims, proprio_dim, action_dim
    else:
        raise ValueError('Unknown dataset type: {}'.format(example_task.dataset_type))


def setup_model(cmdline_args, example_task, log, device, camera_names, modalities_to_mask, data_dir, bc_mode):
    obs_dims, proprio_dim, action_dim = get_robot_dims(example_task, camera_names, cmdline_args['image_size'])
    pretrained_state_dict = {}

    # Load pretrained weights, if applicable
    load_path = cmdline_args['load_path']
    if load_path is not None:
        load_path = load_path.replace('--TARGET_ROBOT--', cmdline_args['robot'])
        log(f'Loading pretrained weights from {load_path}')
        pretrained_state_dict = torch.load(load_path)

    std_bounds = (cmdline_args['std_min'], cmdline_args['std_max'])

    tune_style_kwargs = {}
    tune_style_kwargs['image_encoder_tune_style'] = cmdline_args['image_encoder_tune_style']

    if cmdline_args['model'] == 'PLEX':
        assert cmdline_args['obs_pred.K'] is not None
        assert cmdline_args['inv_d_pred.K'] is not None
        assert cmdline_args['obs_pred.K'] >= cmdline_args['inv_d_pred.K']
        assert cmdline_args['obs_pred.K'] % cmdline_args['inv_d_pred.K'] == 0
        obs_pred_gpt2_kwargs = dict(
            n_layer=cmdline_args['obs_pred.n_layer'],
            n_head=cmdline_args['obs_pred.n_head'],
            K=cmdline_args['obs_pred.K'],
            activation_function=cmdline_args['activation_function'],
            resid_pdrop=cmdline_args['dropout'],
            attn_pdrop=cmdline_args['dropout']
        )
        inv_d_pred_gpt2_kwargs = dict(
            n_layer=cmdline_args['inv_d_pred.n_layer'],
            n_head=cmdline_args['inv_d_pred.n_head'],
            K=cmdline_args['inv_d_pred.K'],
            activation_function=cmdline_args['activation_function'],
            resid_pdrop=cmdline_args['dropout'],
            attn_pdrop=cmdline_args['dropout']
        )

        model = PLEX(
            camera_names=camera_names,
            obs_dims=obs_dims,
            proprio_dim=proprio_dim,
            act_dim=action_dim,
            hidden_dim=cmdline_args['embed_dim'],
            # The history length for this model is always the observation prediction model's history length:
            history_len=cmdline_args['obs_pred.K'],
            image_encoder_arch=cmdline_args['image_encoder_arch'],
            image_encoder_load=cmdline_args['image_encoder_load'],
            use_random_crops=True,
            pool_type=cmdline_args['pool_type'],
            action_output_type=cmdline_args['action_output_type'],
            impute_style=cmdline_args['impute_style'],
            data_dir=data_dir,
            relative_position_encodings=cmdline_args['relative_position_encodings'],
            future_step=cmdline_args['future_step'],
            std_bounds=std_bounds,
            obs_pred_gpt2_kwargs=obs_pred_gpt2_kwargs,
            inv_d_pred_gpt2_kwargs=inv_d_pred_gpt2_kwargs,
            modalities_to_mask=modalities_to_mask,
            bc_mode=bc_mode
        ).to(device=device)

        # Record the tune style parameters
        tune_style_kwargs['obs_pred_transformer_tune_style'] = cmdline_args['obs_pred.transformer_tune_style']
        tune_style_kwargs['inv_d_pred_transformer_tune_style'] = cmdline_args['inv_d_pred.transformer_tune_style']

    elif cmdline_args['model'] == 'DT':
        # Configure the model
        gpt2_kwargs = dict(
            n_layer=cmdline_args['n_layer'],
            n_head=cmdline_args['n_head'],
            activation_function=cmdline_args['activation_function'],
            resid_pdrop=cmdline_args['dropout'],
            attn_pdrop=cmdline_args['dropout'],
            relative_position_encodings=cmdline_args['relative_position_encodings']
        )

        model = DecisionTransformer(
            camera_names=camera_names,
            obs_dims=obs_dims,
            proprio_dim=proprio_dim,
            act_dim=action_dim,
            hidden_dim=cmdline_args['embed_dim'],
            history_len=cmdline_args['K'],
            image_encoder_arch=cmdline_args['image_encoder_arch'],
            image_encoder_load=cmdline_args['image_encoder_load'],
            use_random_crops=True,
            pool_type=cmdline_args['pool_type'],
            action_output_type=cmdline_args['action_output_type'],
            impute_style=cmdline_args['impute_style'],
            data_dir=data_dir,
            gpt2_kwargs=gpt2_kwargs,
            std_bounds=std_bounds,
            modalities_to_mask=modalities_to_mask,
            bc_mode=bc_mode
        ).to(device=device)

        # Record the tune style parameters
        tune_style_kwargs['transformer_tune_style'] = cmdline_args['transformer_tune_style']

    elif cmdline_args['model'] == 'MLP':
        model = MLPBCModel(
            camera_names=camera_names,
            obs_dims=obs_dims,
            proprio_dim=proprio_dim,
            act_dim=action_dim,
            hidden_dim=cmdline_args['embed_dim'],
            history_len=cmdline_args['K'],
            image_encoder_arch=cmdline_args['image_encoder_arch'],
            image_encoder_load=cmdline_args['image_encoder_load'],
            use_random_crops=True,
            impute_style=cmdline_args['impute_style'],
            n_layer=cmdline_args['n_layer'],
            activation_function=cmdline_args['activation_function'],
            dropout=cmdline_args['dropout'],
            modalities_to_mask=modalities_to_mask,
            bc_mode=bc_mode,
            std_bounds=std_bounds,
        ).to(device=device)

        # Record the tune style parameters
        # TODO

    else:
        raise NotImplementedError(f'Unknown model type: {cmdline_args.model}')
    log('Model architecture:')
    log(str(model))

    if len(pretrained_state_dict) > 0:
        model.load_state_dict(pretrained_state_dict)
        log('Loaded successfully!')
    else:
        log('Training/finetuning the model from scratch!')

    return model, tune_style_kwargs


def set_trainable_params(model, trainable_param_spec, log):
    model.set_requires_grad(**trainable_param_spec)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable_params = sum([p.numel() for p in trainable_params])
    num_params = sum([p.numel() for p in model.parameters()])
    log(f'Training {num_trainable_params} out of {num_params} total parameters')
    return trainable_params


def setup_trainer(batch_sampler, lr, eval_fns, model, trainable_params, cmdline_args):
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=cmdline_args['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/cmdline_args['warmup_steps'], 1)
    )

    # Model-specific loss weights
    if cmdline_args['model'] == 'DT' or cmdline_args['model'] == 'MLP':
        loss_weights = {
            'action': 1.0
        }
    elif cmdline_args['model'] == 'PLEX':
        loss_weights = {
            # This is the task-conditioned latent state prediction loss weight.
            # It should be 1.0 for PL pretraining and 0.0 for EX pretraining (since EX pretraining uses
            # task-agnostic data that makes task-conditioned latent state prediction impossible).
            # It should be 1.0 for target-task finetuning as well.
            'future_prediction': cmdline_args['future_prediction_loss_weight']
        }
        # The EX part of PLEX (i.e., inversed dynamics -- action prediction based on the current and a future latent state)
        # can be trained using the future latent state of the training trajectory *or* the future latent state
        # predicted by the PL part of PLEX (the latent state predictor).
        # If we care about the former, we set grounded_inverse_dynamics_loss_weight = 1 and predicted_inverse_dynamics_loss_weight = 0.
        # If we care about the latter, then vice versa. In either case,
        # predicted_inverse_dynamics_loss_weight = 1 - grounded_inverse_dynamics_loss_weight.
        #
        # Namely, for EX pretraining we set grounded_inverse_dynamics_loss_weight = 1, because
        # the latent state predictor (PL) is unavailable at the time when EX is being pretrained.
        #
        # For PL pretraining, grounded_inverse_dynamics_loss_weight doesn't matter, because during PL pretraining
        # the inverse dynamics precictor (EX) is frozen and isn't affected by training, and the inverse dynamics
        # losses, in turn, don't affect the PL component of PLEX.
        #
        # For target-task finetuning of PLEX, we set predicted_inverse_dynamics_loss_weight = 1, because we want to adapt the
        # PL and EX components of PLEX to work together.
        for which in ['predicted', 'grounded']:
            key = f'{which}_inverse_dynamics'
            loss_weights[key] = cmdline_args[f'{key}_loss_weight']
    else:
        raise NotImplementedError

    return Trainer(
        model=model,
        optimizer=optimizer,
        get_batch=batch_sampler,
        batch_size=cmdline_args['batch_size'],
        target_frame_rate=cmdline_args['target_frame_rate'],
        pad_frame_gaps=cmdline_args['pad_frame_gaps'],
        scheduler=scheduler,
        loss_weights=loss_weights,
        eval_fns=eval_fns,
    )


def run_training(trainer, model, num_steps, model_filename_prefix, cmdline_args, log, log_to_wandb, timer):
    log(f'Commencing training...')
    metrics = defaultdict(list)
    best = float('-inf')

    if cmdline_args['model'] == 'PLEX':
        model_info = f'plK{cmdline_args["obs_pred.K"]}_plL{cmdline_args["obs_pred.n_layer"]}_plH{cmdline_args["obs_pred.n_head"]}_exK{cmdline_args["inv_d_pred.K"]}_exL{cmdline_args["inv_d_pred.n_layer"]}_exH{cmdline_args["inv_d_pred.n_head"]}_res{cmdline_args["image_size"]}_bc{cmdline_args["bc_learning_mode"]}_la{cmdline_args["future_step"]}_relpos{cmdline_args["relative_position_encodings"]}__'
    elif cmdline_args['model'] == 'DT':
        model_info = f'K{cmdline_args["K"]}_L{cmdline_args["n_layer"]}_H{cmdline_args["n_head"]}_res{cmdline_args["image_size"]}_bc{cmdline_args["bc_learning_mode"]}_relpos{cmdline_args["relative_position_encodings"]}__'
    elif cmdline_args['model'] == 'MLP':
        model_info = f'K{cmdline_args["K"]}_L{cmdline_args["n_layer"]}_res{cmdline_args["image_size"]}_bc{cmdline_args["bc_learning_mode"]}__'
    else:
        raise NotImplementedError

    for iter in range(cmdline_args['max_iters']):
        with timer.time('iteration'):
            outputs = trainer.train_iteration(
                num_steps=num_steps,
                iter_num=iter+1,
                print_fn=log
            )

        for k, v in outputs.items():
            metrics[k].append(v)

        with open(log.dir/'metrics.pkl', 'wb') as f:
            pickle.dump(dict(metrics), f)

        if log_to_wandb:
            wandb.log(outputs)

        torch.save(model.state_dict(), log.dir/(model_filename_prefix + model_info + 'latest.pt'))
        torch.save(model.state_dict(), log.dir/(model_filename_prefix + model_info + f'iter_{iter+1}.pt'))
        metric_of_interest = outputs[cmdline_args['best_metric']]
        if metric_of_interest > best:
            best = metric_of_interest
            log(f'New best: {best}')
            torch.save(model.state_dict(), log.dir/(model_filename_prefix + model_info + 'best.pt'))

    print(f"\n\nTHE BEST VALUE OF THE {cmdline_args['best_metric']} METRIC ACROSS ALL TRAINING ITERATIONS IS {best}.")
    return dict(metrics)
