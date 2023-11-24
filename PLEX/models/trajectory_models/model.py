import torch
import torch.nn as nn
import torch.distributions as D
import math
import torchvision
import os
from robomimic.models.base_nets import SpatialSoftmax, SpatialMeanPool, Module
from robomimic.models.obs_nets import obs_encoder_factory, ObservationEncoder
from torchvision.models.resnet import BasicBlock, Bottleneck

from PLEX.models.heads.distributions import GaussianHead, GaussianMixtureHead
from PLEX.models.encoders.vision import R3M_Module
from r3m.models.models_r3m import R3M
import PLEX.util.globals as globals



def _action_loss(action_preds, action_targets, mask):
    if isinstance(action_preds, D.Distribution):
        # minimize negative log-likelihood, i.e. maximize likelihood
        unmasked_losses = -action_preds.log_prob(action_targets)
    elif torch.is_tensor(action_preds):
        # minimize mean squared error
        unmasked_losses = torch.mean((action_preds - action_targets)**2, dim=-1)
    else:
        raise RuntimeError(f'Invalid action_preds: {action_preds}')

    # consider loss only in positions where mask = 1
    assert unmasked_losses.shape == mask.shape
    selected_losses = unmasked_losses[mask.bool()]
    return selected_losses.mean()


class TrajectoryModel(nn.Module):
    def __init__(self, camera_names, obs_dims,
                 proprio_dim, act_dim, hidden_dim,
                 image_encoder_arch='resnet18',
                 image_encoder_load=None,
                 use_random_crops=True,
                 pool_type='SpatialSoftmax',
                 action_output_type='gaussian',
                 action_tanh=True,
                 std_bounds=None,
                 impute_style=None,
                 data_dir=None,
                 history_len=None,
                 modalities_to_mask=['action'],
                 bc_mode=True):
        super().__init__()

        self.camera_names = camera_names
        self.obs_dims = obs_dims
        self.proprio_dim = proprio_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.image_encoder_arch = image_encoder_arch
        self.image_encoder_load = image_encoder_load
        self.use_random_crops = use_random_crops
        self.pool_type = pool_type
        self.action_output_type = action_output_type
        self.data_dir = data_dir
        self.history_len = history_len
        self.bc_mode = bc_mode
        assert type(modalities_to_mask) == list
        self.modalities_to_mask = modalities_to_mask
        # In behavior cloning mode, we don't condidtion on context return.
        # To implement this, we will map context return to a fixed embedding in this mode.
        if self.bc_mode and 'return' not in self.modalities_to_mask:
            self.modalities_to_mask.append('return')
        self.action_tanh = action_tanh
        self.std_bounds = std_bounds
        assert len(std_bounds) == 2 and std_bounds[0] < std_bounds[1]

        # For embedding inputs
        self.return_encoder = nn.Linear(1, hidden_dim)
        self.action_encoder = nn.Linear(act_dim, hidden_dim)

        # If we are in image-based mode, we will need image and proprio encoders.
        if not globals.full_state_mode:
            self.proprio_encoder = nn.Linear(proprio_dim, hidden_dim)
            # For Robomimic's resnet18 encoder, we have to tell the encoder what its output dim should be.
            if self.image_encoder_arch == 'resnet18':
                self.image_encoder_feature_dim = 64

            self.image_encoder = self._create_image_encoder()

            # For R3M, we just take the output dim from R3M itself.
            if self.image_encoder_arch.startswith('r3m'):
                self.image_encoder_feature_dim = int(self.image_encoder.output_shape()[0] / len(camera_names))

            # For combining embeddings of images into single state
            self.image_obs_combiner = nn.Linear(
                self.image_encoder_feature_dim * len(camera_names),
                hidden_dim
            )

            self.image_and_proprio_emb_combiner = nn.Linear(
                hidden_dim + hidden_dim,
                hidden_dim
            )

            # For combining embeddings of proprio data and images into single state
            self.obs_combiner = nn.Linear(
                hidden_dim + self.image_encoder_feature_dim * len(camera_names),
                hidden_dim
            )

            self.context_encoder = self.image_encoder

        else: # Otherwise we are in low-dimensional mode and we will need full state encoders.
            assert type(self.obs_dims) == int
            self.state_encoder = nn.Linear(self.obs_dims, hidden_dim)
            self.context_encoder = self.state_encoder

        # For predicting outputs
        action_input_dim = hidden_dim
        self.predict_proprio = torch.nn.Linear(hidden_dim, self.proprio_dim)
        if self.action_output_type == 'gaussian_mixture':
            num_components = 5
            self.predict_action = GaussianMixtureHead(
                num_components, action_input_dim, self.act_dim, self.std_bounds,
                squash=action_tanh
            )
        elif self.action_output_type == 'gaussian':
            self.predict_action = GaussianHead(
                action_input_dim, self.act_dim, self.std_bounds,
                squash=action_tanh
            )
        elif self.action_output_type == 'deterministic':
            layers = [
                nn.Linear(action_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.act_dim)
            ]
            if action_tanh:
                layers.append(nn.Tanh())
            self.predict_action = nn.Sequential(*layers)
        else:
            raise ValueError(f'Unknown action output type: {self.action_output_type}')
        self.predict_return = torch.nn.Linear(hidden_dim, 1)

        # For handling missing values
        assert impute_style in {'zero-embedding', 'zero-input', 'trainable'}
        self.impute_style = impute_style

        for name in ['context', 'state', 'image', 'return', 'proprio', 'action']:
            attr = f'missing_{name}_embedding'
            if impute_style == 'trainable':
                zeros = torch.zeros(hidden_dim)
                value = nn.Parameter(zeros)
                setattr(self, attr, value)
            elif impute_style == 'zero-embedding':
                pass
            else:
                print(f'Unexpected imputation style: {impute_style}')
                raise NotImplementedError

    def _create_image_encoder(self):
        if self.image_encoder_arch.startswith('r3m'):
            import robomimic.utils.obs_utils as ObsUtils
            # NOTE: for loading the model from a local directory, r3m *must* be installed from https://github.com/akolobov/r3m:
            # % git clone https://github.com/akolobov/r3m.git
            # % cd r3m
            # % pip install -e .
            from r3m import load_r3m_from_path
            assert(self.data_dir is not None)
            root_path = os.path.join(self.data_dir, 'pretrained_image_encoders/r3m')
            obs_shapes = {f'{cam}_image': self.obs_dims for cam in self.camera_names}

            # ASSUMPTION: ObsUtils.initialize_obs_modality_mapping_from_dict() has been called before this.
            model = ObservationEncoder(feature_activation=None)

            for obs_name, dims in obs_shapes.items():
                model.register_obs_key(
                    name=obs_name,
                    shape=dims,
                    net_class=None,
                    net_kwargs=None,
                    net=R3M_Module(load_r3m_from_path(self.image_encoder_arch, root_path).module),
                    randomizer=ObsUtils.OBS_RANDOMIZERS['CropRandomizer'](**{
                                                                            'input_shape': dims,
                                                                            'crop_height': math.ceil(0.9 * self.obs_dims[1]),
                                                                            'crop_width': math.ceil(0.9 * self.obs_dims[2]),
                                                                            'num_crops': 1,
                                                                            'pos_enc': False
                                                                            }),
                    share_net_from=None)

            model.make()
            return model
        elif self.image_encoder_arch == 'atari':
            image_channels = min(self.obs_dims) # note: (fair) assumption
            return nn.Sequential(nn.Conv2d(image_channels, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(3136, 512), nn.ReLU(),  # Note: 3136 assumes 84x84 image
                                 nn.Linear(512, self.hidden_dim))

        elif self.image_encoder_arch == 'resnet18':
            obs_processor_kwargs = {'feature_dimension': 'THIS_ARG_IS_IGNORED',
                                    'core_class': 'VisualCore',
                                    'core_kwargs': {'input_shape': self.obs_dims,
                                                    'backbone_class': 'ResNet18Conv',
                                                    'backbone_kwargs': {'input_channel': 3,
                                                                        'pretrained': False,
                                                                        'input_coord_conv': False},
                                                    'pool_class': self.pool_type,
                                                    'pool_kwargs': {'num_kp': 32,
                                                                    'learnable_temperature': False,
                                                                    'temperature': 1.0,
                                                                    'noise_std': 0.0},
                                                    'flatten': True,
                                                    'feature_dimension': self.image_encoder_feature_dim}}
            if self.use_random_crops:
                obs_processor_kwargs['obs_randomizer_class'] = 'CropRandomizer'
                obs_processor_kwargs['obs_randomizer_kwargs'] = {
                    'crop_height': math.ceil(0.9 * self.obs_dims[1]),
                    'crop_width': math.ceil(0.9 * self.obs_dims[2]),
                    'num_crops': 1,
                    'pos_enc': False
                }
            else:
                obs_processor_kwargs['obs_randomizer_class'] = None
                obs_processor_kwargs['obs_randomizer_kwargs'] = {}

            obs_shapes = {f'{cam}_image': self.obs_dims for cam in self.camera_names}

            # ASSUMPTION: ObsUtils.initialize_obs_modality_mapping_from_dict() has been called before this.
            model = obs_encoder_factory(obs_shapes, feature_activation=nn.ReLU, encoder_kwargs={'rgb': obs_processor_kwargs})

            if self.image_encoder_load is not None:
                if self.image_encoder_load == 'R3M':
                    from r3m import load_r3m
                    r3m_resnet = load_r3m('resnet18').module.convnet
                    r3m_nets = torch.nn.Sequential(*(list(r3m_resnet.children())[:-2]))
                    pretrained_state_dict = r3m_nets.state_dict()
                else:
                    pretrained_state_dict = torch.load(self.image_encoder_load)

                for cam in self.camera_names:
                    model.obs_nets[f'{cam}_image'].backbone.nets.load_state_dict(pretrained_state_dict)

            return model
        else:
            raise NotImplementedError

    def _get_tunables(self, image_encoder_tune_style):
        # Always train the linear encoders and prediction heads
        tunables = [
            self.predict_proprio,
            self.predict_action,
            self.predict_return
        ]

        # Include embeddings for missing modalities (if applicable)
        if self.impute_style == 'trainable':
            tunables.extend([
                getattr(self, f'missing_{x}_embedding') for x in [
                    'context', 'state', 'image', 'return', 'proprio', 'action'
                ] if hasattr(self, f'missing_{x}_embedding')
            ])

        # Image encoder
        if image_encoder_tune_style == 'none':
            pass
        else:
            if not globals.full_state_mode:
                tunables.append(self.obs_combiner)
                tunables.append(self.image_obs_combiner)
                tunables.append(self.image_and_proprio_emb_combiner)

                if image_encoder_tune_style == 'all':
                    tunables.append(self.image_encoder)
                else:
                    for obs_net in self.image_encoder.obs_nets.values():
                        if isinstance(obs_net, R3M_Module):
                            # Batch normalization layer tuning
                            tunables.append(obs_net.bn)
                            if image_encoder_tune_style == 'fc':
                                # Nothing to do -- this model doesn't have an fc layer at the end
                                # But remember that the combiners and the batch normalization layer have already been added to the tunables!
                                pass
                            elif image_encoder_tune_style.startswith('last'):
                                # Last n blocks of ResNet
                                n = int(image_encoder_tune_style[4:])
                                assert n >= 0
                                if n > 0:
                                    blocks = [m for m in obs_net.R3M_obj.convnet.modules() if (isinstance(m, torchvision.models.resnet.BasicBlock) or isinstance(m, torchvision.models.resnet.Bottleneck))]
                                    assert len(blocks) >= n
                                    tunables.extend(blocks[-n:])
                            else:
                                raise ValueError(f'Invalid image_encoder_tune_style: {image_encoder_tune_style}')
                        else: # Then it's Robomimic's encoder.
                            # Add last (fully-connected) layer
                            fc_layer = obs_net.nets[-1]
                            if fc_layer is not None and not isinstance(fc_layer, R3M):
                                assert isinstance(fc_layer, nn.Linear)
                                tunables.append(fc_layer)

                            if image_encoder_tune_style == 'fc':
                                # We already added the last (fc) layer
                                pass
                            elif image_encoder_tune_style.startswith('last'):
                                # Spatial softmax layer
                                last_layer = obs_net.nets[1]
                                if last_layer is not None and not isinstance(last_layer, R3M):
                                    assert isinstance(last_layer, SpatialSoftmax) or isinstance(last_layer, SpatialMeanPool)
                                    tunables.append(last_layer)

                                # Last n blocks of ResNet
                                convnet = obs_net.nets[0]
                                n = int(image_encoder_tune_style[4:])
                                assert n >= 0
                                if n > 0:
                                    blocks = [m for m in convnet.modules() if (isinstance(m, BasicBlock) or isinstance(m, torchvision.models.resnet.BasicBlock) or isinstance(m, torchvision.models.resnet.Bottleneck))]
                                    assert len(blocks) >= n
                                    tunables.extend(blocks[-n:])
                            else:
                                raise ValueError(f'Invalid image_encoder_tune_style: {image_encoder_tune_style}')
            else:
                tunables.append(self.state_encoder)

        return tunables

    def set_requires_grad(self, **kwargs):
        # Start by disabling gradients for all parameters
        for p in self.parameters():
            p.requires_grad = False

        # Selectively enable
        for x in self._get_tunables(**kwargs):
            if isinstance(x, nn.Parameter):
                x.requires_grad = True
            elif isinstance(x, nn.Module):
                for p in x.parameters():
                    p.requires_grad = True

    def _embed_helper(self, value, name, batch_dims):
        encoder = getattr(self, f'{name}_encoder')
        extra_conditions = (name not in self.modalities_to_mask)

        if value is not None and extra_conditions:
            return encoder(value)
        elif self.impute_style in {'trainable'}:
            return torch.tile(getattr(self, f'missing_{name}_embedding'),
                              (*batch_dims, 1))
        elif self.impute_style == 'zero-embedding':
            zeros = torch.zeros(self.hidden_dim)
            return torch.tile(zeros, (*batch_dims, 1))
        else:
            raise NotImplementedError

    def embed_return(self, rtg, batch_dims):
        return self._embed_helper(rtg, 'return', batch_dims)

    def embed_proprio(self, proprio, batch_dims):
        return self._embed_helper(proprio, 'proprio', batch_dims)

    def embed_action(self, action, batch_dims):
        return self._embed_helper(action, 'action', batch_dims)

    def embed_observations(self, obs, proprios, batch_dims):
        if not globals.full_state_mode:
            cams2images = obs
            for cam in cams2images.keys():
                c, h, w = self.obs_dims
                cams2images[cam] = cams2images[cam].reshape(-1, c, h, w)
            img_embeddings = self.image_encoder(cams2images)
            assert img_embeddings.ndim == 2
            img_embed_dim = img_embeddings.shape[1]
            img_embeddings = img_embeddings.reshape(*batch_dims, img_embed_dim)
            prop_embeddings = self.embed_proprio(proprios, batch_dims)
            return self.obs_combiner(torch.cat([img_embeddings, prop_embeddings], dim=-1))
        else:
            # Ignore proprios even if they are present.
            state = obs
            return self.state_encoder(state)

    def embed_image_observations(self, cams2images, batch_dims):
        for cam in cams2images.keys():
            c, h, w = self.obs_dims
            cams2images[cam] = cams2images[cam].reshape(-1, c, h, w)
        img_embeddings = self.image_encoder(cams2images)
        assert img_embeddings.ndim == 2
        img_embed_dim = img_embeddings.shape[1]
        img_embeddings = img_embeddings.reshape(*batch_dims, img_embed_dim)
        return self.image_obs_combiner(img_embeddings)

    def embed_context(self, context, batch_dims):
        return self.embed_observations(context, None, [batch_dims[0]])

    def forward(self, context, obs, proprios, actions, rewards, returns_to_go, timesteps, mask, **kwargs):
        raise NotImplementedError

    def compute_losses(self, forward_outputs, actions, contextual, mask=None):
        # Compute any auxiliary losses here.
        losses = {}
        return losses

    def _zero_pad(self, x):
        current_dim = x.shape[1]
        needed = self.history_len - current_dim
        if needed > 0:
            zeros_shape = list(x.shape)
            zeros_shape[1] = needed
            zeros = torch.zeros(*zeros_shape, dtype=x.dtype, device=x.device)
            return torch.cat([zeros, x], dim=1)
        else:
            return x[:,-self.history_len:]

    def get_action_for_eval(self, context, obs, proprios, actions, rewards, returns_to_go, timesteps):
        if not globals.full_state_mode:
            images = obs
            proprios = proprios.reshape(1, -1, self.proprio_dim).float()
            images = {cam: im.reshape(1, -1, *self.obs_dims).float() for cam, im in images.items()}
        else:
            states = obs
            states = states.reshape(1, -1, self.obs_dims).float()

        actions = actions.reshape(1, -1, self.act_dim).float()
        returns_to_go = returns_to_go.reshape(1, -1, 1).float()
        timesteps = timesteps.reshape(1, -1)

        if self.history_len is not None:
            reference = proprios if not globals.full_state_mode else states
            l = min(self.history_len, reference.shape[1])
            mask = self._zero_pad(
                torch.ones(1, l, dtype=torch.long, device=reference.device)
            )
            if not globals.full_state_mode:
                proprios = self._zero_pad(proprios)
                for cam, cam_images in images.items():
                    images[cam] = self._zero_pad(images[cam])
            else:
                states = self._zero_pad(states)
            actions = self._zero_pad(actions)
            returns_to_go = self._zero_pad(returns_to_go)
            timesteps = self._zero_pad(timesteps)
        else:
            mask = None

        action_preds = self.forward(
            context, (images if not globals.full_state_mode else states), proprios, actions, rewards, returns_to_go, timesteps, mask,
            compute_pred_obs=True, compute_pred_future_actions=True, compute_known_future_actions=False, eval_mode=True
        )[0]

        # The forward pass may return a distribution.
        # In this case, we return a sample from the distribution
        if isinstance(action_preds, D.Distribution):
            action_preds = action_preds.sample()

        return action_preds[-1,-1]
