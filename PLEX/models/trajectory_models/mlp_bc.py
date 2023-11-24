import torch
import torch.nn as nn
import torch.nn.functional as F

from PLEX.models.trajectory_models.model import TrajectoryModel


ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU
}


class MLPBCModel(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past observations.
    """

    def __init__(
            self,
            camera_names, obs_dims,
            proprio_dim, act_dim,
            hidden_dim,
            n_layer=3,
            activation_function='relu',
            dropout=0.1,
            **kwargs
    ):
        super().__init__(camera_names, obs_dims, proprio_dim, act_dim, hidden_dim, **kwargs)

        layers = []
        prev_dim = (1 + 2 * self.history_len) * hidden_dim
        for _ in range(n_layer):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                ACTIVATION_FUNCTIONS[activation_function](),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.extend([
            nn.Linear(prev_dim, act_dim),
            nn.Tanh()
        ])
        self.mlp = nn.Sequential(*layers)

    def _get_tunables(self, image_encoder_tune_style='all'):
        tunables = super()._get_tunables(image_encoder_tune_style)
        tunables.append(self.mlp)
        return tunables

    def forward(self, context, images, proprios, actions, rewards, returns_to_go, timesteps=None, attention_mask=None):
        batch_dims = images[self.camera_names[0]].shape[:2]

        context_embeddings = self.embed_context({f'{cam}_image': context[cam] for cam in context.keys()} if context is not None else None, batch_dims)
        obs_embeddings = self.embed_observations({f'{cam}_image': images[cam] for cam in images.keys()}, proprios, batch_dims)
        action_embeddings = self.embed_action(actions, batch_dims)
        # returns_embeddings = self.embed_return(returns_to_go, batch_dims)

        context_embeddings = torch.unsqueeze(context_embeddings, dim=1)
        concatenated = torch.cat([context_embeddings, obs_embeddings, action_embeddings], dim=1)
        flattened = torch.flatten(concatenated, start_dim=1)
        action_preds = self.mlp(flattened)
        # get_action expects forward rteval to have a sequence dimension
        action_preds = torch.unsqueeze(action_preds, dim=1)
        return action_preds,

    def compute_losses(self, forward_outputs, actions, contextual, mask):
        # Include superclass's losses
        losses = super().compute_losses(forward_outputs, actions, contextual, mask)
        action_preds, = forward_outputs
        action_targets = actions[:,-1:,:]
        assert action_preds.shape == action_targets.shape
        losses['action'] = F.mse_loss(action_preds, action_targets)
        return losses