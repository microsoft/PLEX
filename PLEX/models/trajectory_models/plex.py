from copy import deepcopy
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import transformers

from PLEX.models.heads.distributions import GaussianMixtureHead, GaussianHead, slice_dist
from PLEX.models.trajectory_models.model import TrajectoryModel, _action_loss
from PLEX.models.trajectory_models.trajectory_gpt2 import GPT2Model


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


class PLEX(TrajectoryModel):
    def __init__(
            self,
            camera_names, obs_dims,
            proprio_dim, act_dim,
            hidden_dim,
            relative_position_encodings,
            future_step=1,
            obs_pred_gpt2_kwargs={},
            inv_d_pred_gpt2_kwargs={},
            **kwargs
    ):
        super().__init__(camera_names, obs_dims, proprio_dim, act_dim, hidden_dim, **kwargs)

        # Create separately trainable positional embeddings and LayerNorms for the observational and the inverse dynamics transformer.
        self.relative_position_encodings = relative_position_encodings
        obs_pred_gpt2_kwargs['relative_position_encodings'] = relative_position_encodings
        inv_d_pred_gpt2_kwargs['relative_position_encodings'] = relative_position_encodings

        self.obs_tr_history_len = obs_pred_gpt2_kwargs['K']
        self.inv_d_tr_history_len = inv_d_pred_gpt2_kwargs['K']

        if not self.relative_position_encodings:
            self.embed_obs_tr_timestep = nn.Embedding(self.obs_tr_history_len, hidden_dim)
            self.embed_inv_d_tr_timestep = nn.Embedding(self.inv_d_tr_history_len, hidden_dim)

        self.embed_obs_tr_ln = nn.LayerNorm(hidden_dim)
        self.embed_inv_d_ln = nn.LayerNorm(hidden_dim)

        self.n_obs_tr_components = 2 # namely: target returns and image observations
        # One extra position is for the context embedding.
        n_obs_tr_positions = 1 + self.obs_tr_history_len * self.n_obs_tr_components
        obs_tr_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_positions=n_obs_tr_positions,
            n_ctx=n_obs_tr_positions,
            n_embd=hidden_dim,
            **obs_pred_gpt2_kwargs
        )
        self.obs_transformer = GPT2Model(obs_tr_config)

        self.n_inv_d_tr_components = 3 # namely: integrated observations (image obs. embeddings + proprios combined), image obs predictions, and actions
        n_inv_d_transformer_positions = self.inv_d_tr_history_len * self.n_inv_d_tr_components
        inv_d_transformer_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_positions=n_inv_d_transformer_positions,
            n_ctx=n_inv_d_transformer_positions,
            n_embd=hidden_dim,
            **inv_d_pred_gpt2_kwargs
        )
        self.inv_d_transformer = GPT2Model(inv_d_transformer_config)

        self.future_step = future_step

        # NOTE: currently, using the Gaussian head-based stochastic prediction of observation latents doesn't work very well.
        # Therefore, we'll use deterministic prediction of observation latents instead.
        self.deterministic_future_obs_emb_predictions = True
        if not self.deterministic_future_obs_emb_predictions:
            self.predict_future = GaussianHead(
                input_dim=hidden_dim, output_dim=hidden_dim,
                std_bounds=self.std_bounds,
                hidden_dim=hidden_dim
            )


    def _get_tunables(self, image_encoder_tune_style='all', obs_pred_transformer_tune_style='all', inv_d_pred_transformer_tune_style='all'):
        tunables = super()._get_tunables(image_encoder_tune_style)

        #
        # Handle the tunables of the observation prediction transformer.
        #
        if not self.deterministic_future_obs_emb_predictions and obs_pred_transformer_tune_style != 'none':
            tunables.append(self.predict_future)

        if obs_pred_transformer_tune_style == 'all':
            tunables.extend([
                self.embed_obs_tr_ln,
                self.return_encoder,
                self.obs_transformer
            ])

            if self.impute_style == 'trainable':
                tunables.extend([
                    getattr(self, f'missing_{x}_embedding') for x in [
                        'context', 'image', 'return'
                    ]
                ])

            if not self.relative_position_encodings:
                tunables.append(self.embed_obs_tr_timestep)  # Only for absolute position encodings.

        elif obs_pred_transformer_tune_style == 'last_block':
            # Fine-tune the last block of the transformer
            tunables.extend([
                self.obs_transformer.h[-1],
                self.obs_transformer.ln_f
            ])
        elif obs_pred_transformer_tune_style == 'linear_probe':
            # Only tune the predict_* networks
            pass
        elif obs_pred_transformer_tune_style == 'none':
            # Tune nothing -- no parameters got included
            pass
        else:
            raise ValueError(f'Invalid transformer_tune_style: {obs_pred_transformer_tune_style}')

        #
        # Handle the tunables of the inverse dynamics prediction transformer.
        #
        if inv_d_pred_transformer_tune_style != 'none':
            tunables.append(self.predict_action)

        if inv_d_pred_transformer_tune_style == 'all':
            tunables.extend([
                self.embed_inv_d_ln,
                self.proprio_encoder,
                self.action_encoder,
                self.image_and_proprio_emb_combiner,
                self.inv_d_transformer
            ])

            if self.impute_style == 'trainable':
                tunables.extend([
                    getattr(self, f'missing_{x}_embedding') for x in [
                        'proprio', 'action'
                    ]
                ])

            if not self.relative_position_encodings:
                tunables.append(self.embed_inv_d_tr_timestep)  # Only for absolute position encodings.

        elif inv_d_pred_transformer_tune_style == 'last_block':
            # Fine-tune the last block of the transformer
            tunables.extend([
                self.inv_d_transformer.h[-1],
                self.inv_d_transformer.ln_f
            ])
        elif inv_d_pred_transformer_tune_style == 'linear_probe':
            # Only tune the predict_* networks
            pass
        elif inv_d_pred_transformer_tune_style == 'none':
            # Tune nothing -- no parameters got included
            pass
        else:
            raise ValueError(f'Invalid transformer_tune_style: {inv_d_pred_transformer_tune_style}')

        return tunables


    def _stack_inputs_and_masks(self, n_tr_input_components, inputs, mask, seq_length, batch_size, hidden_dim):
        assert len(inputs) == n_tr_input_components
        total_seq_length = len(inputs) * seq_length
        stacked_inputs = torch.stack(inputs, dim=1)\
                              .permute(0, 2, 1, 3)\
                              .reshape(batch_size, total_seq_length, hidden_dim)  # [B, N-1, NS]

        # To make the attention mask fit the stacked inputs, have to stack it as well
        stacked_mask = torch.stack(
            [mask for _ in range(len(inputs))], dim=1
        ).permute(0, 2, 1).reshape(batch_size, total_seq_length)
        return stacked_inputs, stacked_mask


    def _predict_obs(self, context_embeddings, returns_embeddings, current_image_obs_embeddings, mask, seq_length, batch_size):
        stacked_obs_tr_inputs, stacked_obs_tr_mask = self._stack_inputs_and_masks(self.n_obs_tr_components,
                                                                                    [returns_embeddings, current_image_obs_embeddings],
                                                                                    mask,
                                                                                    seq_length,
                                                                                    batch_size,
                                                                                    self.hidden_dim)
        # Account for context conditioning for the observation prediction transformer
        stacked_obs_tr_inputs = torch.cat([
            context_embeddings.unsqueeze(1),
            stacked_obs_tr_inputs
        ], dim=1)                                                            # [B, N, NS]
        stacked_obs_tr_inputs = self.embed_obs_tr_ln(stacked_obs_tr_inputs)  # [B, N, NS]

        stacked_obs_tr_mask = torch.cat([
            torch.ones(batch_size, 1, device=stacked_obs_tr_mask.device),
            stacked_obs_tr_mask
        ], dim=1)

        # We feed the input embeddings (not word indices as in NLP) to the observation prediciton model.
        obs_tr_outputs = self.obs_transformer(
            inputs_embeds=stacked_obs_tr_inputs,
            attention_mask=stacked_obs_tr_mask
        )
        x_obs_tr = obs_tr_outputs['last_hidden_state']

        # Ignore first hidden state (corresponding to context)
        x_obs_tr = x_obs_tr[:,1:,:]

        # reshape x so that the second dimension corresponds to the original
        # returns-to-go (0), or observations (1); i.e. x[:,1,t] is the token for s_t
        x_obs_tr = x_obs_tr.reshape(batch_size, seq_length, self.n_obs_tr_components, self.hidden_dim).permute(0, 2, 1, 3)

        # Get predictions

        # For each time step, the observation prediction transformer outputs two latent states:
        # the first for return-to-go, the other for the state distribution parameters.
        predicted_obs_pos_idx = self.n_obs_tr_components - 1
        if not self.deterministic_future_obs_emb_predictions:
            future_image_obs_emb_distr = self.predict_future(x_obs_tr[:,predicted_obs_pos_idx])
            pred_future_image_obs_embeddings = future_image_obs_emb_distr.rsample()
        else:
            future_image_obs_emb_distr = None
            pred_future_image_obs_embeddings = x_obs_tr[:,predicted_obs_pos_idx]

        return pred_future_image_obs_embeddings, future_image_obs_emb_distr


    def _predict_actions(self, integrated_obs_embeddings, future_image_obs_emb, action_embeddings, mask, seq_length, batch_size):
        stacked_inv_d_inputs, stacked_inv_d_mask = self._stack_inputs_and_masks(self.n_inv_d_tr_components,
                                                                                [integrated_obs_embeddings, future_image_obs_emb, action_embeddings],
                                                                                mask,
                                                                                seq_length,
                                                                                batch_size,
                                                                                self.hidden_dim)

        inv_d_tr_outputs = self.inv_d_transformer(
            inputs_embeds=stacked_inv_d_inputs,
            attention_mask=stacked_inv_d_mask
        )
        x_inv_d_tr = inv_d_tr_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # observations (0), or actions (1); i.e. x[:,0,t] is the token for s_t
        x_inv_d_tr = x_inv_d_tr.reshape(batch_size, seq_length, self.n_inv_d_tr_components, self.hidden_dim).permute(0, 2, 1, 3)

        # For each time step, the inverse dynamics prediction transformer outputs three latent states, the last of which corresponds
        # to the action (see the call to self._stack_inputs_and_masks above). We want to predict that last component using all the data
        # that comes before it.
        predicted_action_pos_idx = self.n_inv_d_tr_components - 2
        pred_future_pred_actions = self.predict_action(x_inv_d_tr[:,predicted_action_pos_idx])
        return pred_future_pred_actions


    def forward(self, context, images, proprios, actions, rewards, returns_to_go, timesteps, mask, compute_pred_obs=True, compute_pred_future_actions=True, compute_known_future_actions=False, eval_mode=False):
        batch_dims = images[self.camera_names[0]].shape[:2]
        obs_tr_batch_size, seq_length = batch_dims
        batch_increase_ratio = self.obs_tr_history_len // self.inv_d_tr_history_len
        inv_d_batch_size = obs_tr_batch_size * batch_increase_ratio

        # NOTE: During training, the length of trajectory sequences that are fed to this method is (obs_pred.K + lookahead).
        # During evaluation, it is just obs_pred.K. So, we need to let this method's logic know about this, as below.
        if eval_mode:
            k = 0
        else:
            k = self.future_step
        seq_length -= k
        assert seq_length == self.obs_tr_history_len

        #
        # *******  STEP 1: Embed all the inputs to the model.  *******
        #

        image_obs_embeddings = self.embed_image_observations({f'{cam}_image': images[cam] for cam in images.keys()}, batch_dims)
        prop_embeddings = self.embed_proprio(proprios, batch_dims)
        integrated_obs_embeddings = self.image_and_proprio_emb_combiner(torch.cat([image_obs_embeddings, prop_embeddings], dim=-1))
        action_embeddings = self.embed_action(actions, batch_dims)
        returns_embeddings = self.embed_return(returns_to_go, batch_dims)

        # Save for later
        orig_image_obs_embeddings = image_obs_embeddings[:,k:].detach()

        passthrough_current_image_obs_embeddings = image_obs_embeddings[:,:self.obs_tr_history_len]
        stopgrad_current_image_obs_embeddings = image_obs_embeddings[:,:self.obs_tr_history_len].detach()

        known_future_image_obs_embeddings = image_obs_embeddings[:,k:].reshape(inv_d_batch_size, self.inv_d_tr_history_len, self.hidden_dim)

        image_obs_embeddings = image_obs_embeddings[:,:self.obs_tr_history_len]
        prop_embeddings = prop_embeddings[:,:self.obs_tr_history_len]
        returns_embeddings = returns_embeddings[:,:self.obs_tr_history_len]
        integrated_obs_embeddings = integrated_obs_embeddings[:,:self.obs_tr_history_len].reshape(inv_d_batch_size, self.inv_d_tr_history_len, self.hidden_dim)
        action_embeddings = action_embeddings[:,:self.obs_tr_history_len].reshape(inv_d_batch_size, self.inv_d_tr_history_len, self.hidden_dim)

        # Masks for each model
        mask_prefix = mask[:,:self.obs_tr_history_len]
        inv_d_mask = mask_prefix.reshape(inv_d_batch_size, self.inv_d_tr_history_len)

        assert np.prod(passthrough_current_image_obs_embeddings.shape) == np.prod(known_future_image_obs_embeddings.shape)
        assert np.prod(stopgrad_current_image_obs_embeddings.shape) == np.prod(known_future_image_obs_embeddings.shape)

        if not self.relative_position_encodings:
            # Shift embeddings by position embedding
            # Obs. prediction and inverse dynamics prediction transformers potentially have their own position embeddings
            position_embeddings_for_obs_tr = self.embed_obs_tr_timestep(
                torch.arange(self.obs_tr_history_len, device=self.embed_obs_tr_timestep.weight.device))
            position_embeddings_for_obs_tr = torch.tile(position_embeddings_for_obs_tr, (obs_tr_batch_size, 1, 1))

            # Image obs. embeddings and returns will be fed only into the obs. prediction transformer.
            passthrough_current_image_obs_embeddings = passthrough_current_image_obs_embeddings.to(position_embeddings_for_obs_tr.device) + position_embeddings_for_obs_tr
            stopgrad_current_image_obs_embeddings = stopgrad_current_image_obs_embeddings.to(position_embeddings_for_obs_tr.device) + position_embeddings_for_obs_tr
            returns_embeddings = returns_embeddings.to(position_embeddings_for_obs_tr.device) + position_embeddings_for_obs_tr

            position_embeddings_for_inv_d_tr = self.embed_inv_d_tr_timestep(
                torch.arange(self.inv_d_tr_history_len, device=self.embed_inv_d_tr_timestep.weight.device))
            position_embeddings_for_inv_d_tr = torch.tile(position_embeddings_for_inv_d_tr, (inv_d_batch_size, 1, 1))

            # Integrated observations and actions will be fed only into the inv.d. transformer
            integrated_obs_embeddings = integrated_obs_embeddings.to(position_embeddings_for_inv_d_tr.device) + position_embeddings_for_inv_d_tr
            # NOTE: the future image observation embeddings aren't integrated with proprios, because predicting inverse dynamics from known current
            # and future proprio would be too easy and woudn't need to rely on the future image observation embeddings.
            known_future_image_obs_embeddings = known_future_image_obs_embeddings.to(position_embeddings_for_inv_d_tr.device) + position_embeddings_for_inv_d_tr
            action_embeddings = action_embeddings.to(position_embeddings_for_inv_d_tr.device) + position_embeddings_for_inv_d_tr

        #
        # *******  STEP 2: Use the observation prediction transformer to predict the observation embeddings.  *******
        #

        # NOTE: this prediction makes sense only for trajectories with a task/context, since without one it's impossible to
        # reasonably predict the next observation. But we compute the predictions anyway and let the compute_losses(.) method ignore
        # these predictions during loss computation if needed.

        # For the obs. prediction transformer, we make the sequence look like (C, R_1, o_1, R_2, o_2, ...)
        if (compute_pred_future_actions and (actions is not None) and (context is not None)) or (compute_pred_obs and (context is not None)):
            context_embeddings = self.embed_context({f'{cam}_image': context[cam] for cam in context.keys()} if context is not None else None, batch_dims)
            passthrough_context_embeddings = context_embeddings
            stopgrad_context_embeddings = context_embeddings.detach()
            pred_future_image_obs_embeddings_from_passthrough_obs, _ = self._predict_obs(passthrough_context_embeddings, returns_embeddings, passthrough_current_image_obs_embeddings, mask_prefix, self.obs_tr_history_len, obs_tr_batch_size)
            pred_future_image_obs_embeddings_from_stopgrad_obs, future_image_obs_emb_distr_from_stopgrad_obs = self._predict_obs(stopgrad_context_embeddings, returns_embeddings, stopgrad_current_image_obs_embeddings, mask_prefix, self.obs_tr_history_len, obs_tr_batch_size)

        else:
            pred_future_image_obs_embeddings_from_passthrough_obs = None
            pred_future_image_obs_embeddings_from_stopgrad_obs = None
            future_image_obs_emb_distr_from_stopgrad_obs = None

        #
        # *******  STEP 3: Predict inverse dynamics, possibly in two ways.  *******
        #

        # For the inv. dynamics prediction transformer, we make the sequence look like (int_o_1, pred_img_o_2, a_1, int_o_2, pred_img_o_3, a_2, ...)
        # Here, int_o_X are the embeddings of combined image-proprio observations, and pred_img_o_(X+1) are the predicted embeddings
        # of the next image observation. During learning, latter can be obtained either from STEP 2 or from the image_obs_embeddings array
        # *shifted by 1 position*. In this case, Presumably, the original image observation sequence contains 1 more entry than the action array.
        #
        # NOTE that the sequence doesn't contain a task specification C, since inverse dynamics should be task-agnostic.
        #
        # NOTE: We drop the last element of each input sequence before reshaping the inputs and passing them to the
        # inverse dynamics transformer. This is because the last action in each input sequence can't be predicted,
        # reliably, since we don't have the ground truth for the following observation, we omit this action from the
        # sequence.

        # NOTE: perhaps we shouldn't include predicted observations into the history (shaping the input as (int_o_1, pred_img_o_2, a_1, int_o_2, pred_img_o_3, a_2, ... ) includes them).
        # It makes the history long for no good reason (just due to including past predictions, which don't add any information), potentially making the model the model "used to" the
        # fact that predictions carry no extra info and making it largely ignore the prediction of the latest observation latent, which is actually crucial for making the correct action prediction.
        #
        if compute_pred_future_actions and (actions is not None):
            # If compute_pred_future_actions, this means we are doing inference. At inference/execution time, we don't have future observations
            # available to us, and therefore *must* rely on those predicted in STEP 2.
            assert pred_future_image_obs_embeddings_from_passthrough_obs is not None
            pred_future_image_obs_embeddings_from_passthrough_obs = pred_future_image_obs_embeddings_from_passthrough_obs.reshape(inv_d_batch_size, self.inv_d_tr_history_len, self.hidden_dim)
            # Remember to add position encodings as appropriate
            if not self.relative_position_encodings:
                pred_future_image_obs_embeddings_from_passthrough_obs + position_embeddings_for_inv_d_tr

            pred_future_pred_actions = self._predict_actions(integrated_obs_embeddings,
                                                             ### For passing zeros instead of target vector
                                                             #torch.zeros_like(pred_future_image_obs_embeddings_from_passthrough_obs),
                                                             ### For passing goal instead of target vector
                                                             #torch.tile(passthrough_context_embeddings, (30, 1, 1)).reshape(pred_future_image_obs_embeddings_from_passthrough_obs.shape),
                                                             pred_future_image_obs_embeddings_from_passthrough_obs,
                                                             action_embeddings,
                                                             inv_d_mask,
                                                             self.inv_d_tr_history_len,
                                                             inv_d_batch_size)
        else:
            pred_future_pred_actions = None

        if compute_known_future_actions and (actions is not None):
            # If compute_loss, then we are doing learning. During learning, we know the actual future observation for each step in
            # the training trajectories, so we can use it to infer the actions.
            known_future_pred_actions = self._predict_actions(integrated_obs_embeddings,
                                                              known_future_image_obs_embeddings,
                                                              action_embeddings,
                                                              inv_d_mask,
                                                              self.inv_d_tr_history_len,
                                                              inv_d_batch_size)
        else:
            known_future_pred_actions = None

        return (
            pred_future_pred_actions,
            known_future_pred_actions,
            orig_image_obs_embeddings,
            (future_image_obs_emb_distr_from_stopgrad_obs if not self.deterministic_future_obs_emb_predictions else pred_future_image_obs_embeddings_from_stopgrad_obs)
        )


    def compute_losses(self, forward_outputs, actions, contextual, mask):
        # Include superclass's losses
        losses = super().compute_losses(forward_outputs, actions, contextual, mask)

        # Unpack model outputs into local vars
        pred_future_action_preds, grounded_action_preds, target_obs_embeddings, future_obs_distr_from_stopgrad_obs = forward_outputs

        batch_size, actual_seq_length = target_obs_embeddings.shape[:2]
        assert actual_seq_length == self.obs_tr_history_len
        obs_mask = mask[:,:self.obs_tr_history_len]

        if actions is not None:
            target_actions = actions[:,:self.obs_tr_history_len]
            if grounded_action_preds is not None:
                mask__reshaped_for_predictions = obs_mask.reshape(grounded_action_preds.shape[0], -1)
                target_actions__reshaped_for_predictions = target_actions.reshape(grounded_action_preds.shape[0], grounded_action_preds.shape[1], -1)
                losses['grounded_inverse_dynamics'] = _action_loss(grounded_action_preds,
                                                                   target_actions__reshaped_for_predictions,
                                                                   mask__reshaped_for_predictions)
            if contextual and pred_future_action_preds is not None:
                # Action prediction based on predicted observations makes sense only for contextual trajectories
                # because without a context/task, observations can't be reasonably predicted.
                mask__reshaped_for_predictions = obs_mask.reshape(pred_future_action_preds.shape[0], -1)
                target_actions__reshaped_for_predictions = target_actions.reshape(pred_future_action_preds.shape[0], pred_future_action_preds.shape[1], -1)

                if pred_future_action_preds is not None:
                    losses['predicted_inverse_dynamics'] = _action_loss(pred_future_action_preds,
                                                                        target_actions__reshaped_for_predictions,
                                                                        mask__reshaped_for_predictions)

        # Predict embedding k steps into the future.
        #
        # As with inverse dynamics computation based on predicted observations, observation prediction loss itself makes sense
        # only for contextual trajectories.
        if contextual:
            future_mask = obs_mask.bool()
            # NOTE: Here, we stop-grad the computed observation embeddings so that backpropagation affects only
            # the observation embedding prediction model, not the observation encoders. If we allow observation
            # encoders to be updated as well, the observation embeddings may eventually collapse due to
            # updates on observation-only batches. On observation-action batches, the encoders get updated anyway
            # thanks to backpropagation from the inverse dynamics.

            if not self.deterministic_future_obs_emb_predictions:
                future_embeddings = target_obs_embeddings[future_mask].detach()
                sliced_future_distr = slice_dist(future_obs_distr_from_stopgrad_obs, (slice(batch_size), slice(self.obs_tr_history_len)))
                masked_future_distr = slice_dist(sliced_future_distr, future_mask)
                future_log_probs = masked_future_distr.log_prob(future_embeddings)
                losses['future_prediction'] = -future_log_probs.mean()
            else:
                future_embeddings = target_obs_embeddings.detach()
                unmasked_losses = torch.mean((future_obs_distr_from_stopgrad_obs - future_embeddings)**2, dim=-1)
                assert unmasked_losses.shape == future_mask.shape
                selected_losses = unmasked_losses[future_mask]
                losses['future_prediction'] = selected_losses.mean()

        return losses