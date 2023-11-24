from copy import deepcopy
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import transformers

from PLEX.models.heads.distributions import GaussianHead, GaussianMixtureHead, slice_dist
from PLEX.models.trajectory_models.model import TrajectoryModel, _action_loss
from PLEX.models.trajectory_models.trajectory_gpt2 import GPT2Model
import PLEX.util.globals as globals


# Upper bound on length of trajectories
T_MAX = 2000


# This code is minimally modified from the original DT https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py).
class DecisionTransformer(TrajectoryModel):
    def __init__(
            self,
            camera_names, obs_dims,
            proprio_dim, act_dim,
            hidden_dim,
            gpt2_kwargs={},
            **kwargs
    ):
        super().__init__(camera_names, obs_dims, proprio_dim, act_dim, hidden_dim, **kwargs)

        assert self.action_output_type == 'deterministic'
        layers = [nn.Linear(self.hidden_dim, self.act_dim)]
        if self.action_tanh:
            layers.append(nn.Tanh())
        self.predict_action = nn.Sequential(*layers)

        self.n_components = 3 # namely: return-to-go, observation, action
        n_positions = self.history_len * self.n_components
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_positions=n_positions,
            n_ctx=n_positions,
            n_embd=hidden_dim,
            **gpt2_kwargs
        )

        # NOTE: the only differences between this GPT2Model and the default Huggingface version are:
        # -- GPT2Model doesn't have absolute position embeddings (since we'll add those ourselves if needed).
        # -- Our GPT2Model has the option of using relative position embeddings.
        self.transformer = GPT2Model(config)
        if not self.transformer.config.relative_position_encodings:
            self.embed_timestep = nn.Embedding(T_MAX, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)

        assert not self.bc_mode, "The original DT should be used only in offline RL mode (with --orl_learning_mode)"

    def _get_tunables(self, image_encoder_tune_style='all', transformer_tune_style='all'):
        tunables = super()._get_tunables(image_encoder_tune_style)

        # Transformer
        if transformer_tune_style == 'all':
            tunables.extend([
                self.embed_ln,
                # The observation (state or image encoder) is already handled by super()._get_tunables(.)
                self.return_encoder,
                self.action_encoder,
                self.transformer
            ])

            if not globals.full_state_mode:
                tunables.extend([self.proprio_encoder])

            # Fine-tune everything
            if not self.transformer.config.relative_position_encodings:
                tunables.append(self.embed_timestep)  # Only for absolute position encodings.

        elif transformer_tune_style == 'last_block':
            # Fine-tune the last block of the transformer
            tunables.extend([
                self.transformer.h[-1],
                self.transformer.ln_f
            ])
        elif transformer_tune_style == 'linear_probe':
            # Only tune the predict_* networks
            pass
        else:
            raise ValueError(f'Invalid transformer_tune_style: {transformer_tune_style}')

        return tunables

    def forward(self, context, obs, proprios, actions, rewards, returns_to_go, timesteps, mask, compute_pred_obs=True, compute_pred_future_actions=True, compute_known_future_actions=False, eval_mode=False):
        # Ignore context
        context = None

        # embed each modality with a different head

        if not globals.full_state_mode:
            images = obs
            batch_dims = images[self.camera_names[0]].shape[:2]
            batch_size, seq_length = batch_dims

            # embed each modality with a different head
            obs_embeddings = self.embed_observations({f'{cam}_image': images[cam] for cam in images.keys()}, proprios, batch_dims)
        else:
            states = obs
            batch_dims = states.shape[:2]
            batch_size, seq_length = batch_dims
            obs_embeddings = self.embed_observations(states, proprios, batch_dims)

        action_embeddings = self.embed_action(actions, batch_dims)
        returns_embeddings = self.embed_return(returns_to_go, batch_dims)

        if not self.transformer.config.relative_position_encodings:
            # shift embeddings by position embedding
            position_embeddings = self.embed_timestep(timesteps)
            obs_embeddings = obs_embeddings.to(position_embeddings.device) + position_embeddings
            action_embeddings = action_embeddings.to(position_embeddings.device) + position_embeddings
            returns_embeddings = returns_embeddings.to(position_embeddings.device) + position_embeddings

        if mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # this makes the sequence look like (R_1, o_1, a_1, R_2, o_2, a_2, ...)
        # which works nice in an autoregressive sense since observations predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, obs_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.n_components*seq_length, self.hidden_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (mask, mask, mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.n_components*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, self.n_components, self.hidden_dim).permute(0, 2, 1, 3)

        # We predict only actions, not (latent) states or returns, because:
        # - Actions is ultimately what we care about predicting.
        # - Since we are working with high-d. observations whose encoders are getting trained together with DT
        # itself, optimizing w.r.t. losses that are based on predicting latent states may lead to encoders'
        # latent state collapse.
        # - We are working with robotics scenarios, where rewards are sparse and give rise to returns that are
        # easy to predict, so optimizing return prediction may lead to overfitting.
        action_preds = self.predict_action(x[:,1])  # predict next action given state and return

        return action_preds,

    def compute_losses(self, forward_outputs, actions, contextual, mask):
        losses = super().compute_losses(forward_outputs, actions, contextual, mask)
        action_preds, = forward_outputs
        losses['action'] = _action_loss(action_preds, actions, mask)
        return losses