from collections import defaultdict

import numpy as np
import torch

import time
from tqdm import trange
import torch.distributions as D
def get_dist_dict(dist):
    "Save the tensors related to batch_size in a Distribution instance into a nested dict."
    batch_shape = dist.batch_shape
    tensors_dict = {}
    for k, v in dist.__dict__.items():
        if isinstance(v, D.Distribution):
            tensors_dict[k] = get_dist_dict(v)
        elif isinstance(v, torch.Tensor) and batch_shape == v.shape[:len(batch_shape)]:
            tensors_dict[k] = v
    return tensors_dict

def update_dist_by_dict(dist, tensors_dict):
    "Update the tensors related to batch_size in a Distribution instance based on a nested dict."
    old_batch_shape = dist._batch_shape
    tensor_shape = None
    for k, v in tensors_dict.items():
        if isinstance(v, torch.Tensor):
            setattr(dist, k, v)
            tensor_shape = v.shape
        else:
            update_dist_by_dict(getattr(dist, k), v)  # update is in-place

    if tensor_shape is not None:
        dist._batch_shape = tensor_shape[:len(old_batch_shape)]
    else:  # all keys are distribution objects
        dist._batch_shape = getattr(dist, k)._batch_shape[:len(old_batch_shape)]
    return dist

class DataParallel(torch.nn.DataParallel):

    def gather(self, outputs, output_device):
        """ A wrapper to handle Distribution objects. """
        # Turn Distribution objects into a nested dict of tensors
        new_outputs = []
        for outs in outputs:
            new_outs = list(map(lambda o: get_dist_dict(o) if isinstance(o, D.Distribution) else o, outs))
            new_outputs.append(new_outs)
        # Call the original gather function, which can only handle tensors
        results = super().gather(new_outputs, self.output_device)
        # Convert nested dict of tensors back to Distribution objects
        default_device_id = 0  # TODO maybe this needs to be changed.
        for i, o in enumerate(outputs[default_device_id]):
            if isinstance(o, D.Distribution):
                assert isinstance(results[i], dict)
                results[i] = update_dist_by_dict(o, results[i])
        return results

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Trainer:
    def __init__(self, model, optimizer, batch_size, get_batch,
                 target_frame_rate,
                 pad_frame_gaps,
                 loss_weights,
                 scheduler=None, eval_fns=None):
        self.model = DataParallel(model)
        self.optimizer = optimizer
        self.get_batch = get_batch
        self.batch_size = batch_size
        self.target_frame_rate = target_frame_rate
        self.pad_frame_gaps = pad_frame_gaps
        self.loss_weights = loss_weights
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = defaultdict(list)

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_fn=None):
        logs = dict()

        train_start = time.time()
        self.model.train()
        for _ in trange(num_steps):
            self.train_step()
            if self.scheduler is not None:
                self.scheduler.step()

        torch.cuda.empty_cache()

        logs['time/training'] = time.time() - train_start
        eval_start = time.time()

        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model, iter_num)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for key, values in self.diagnostics.items():
            logs[f'{key}/mean'] = np.mean(values)
            logs[f'{key}/std'] = np.std(values)
        self.diagnostics.clear() # reset for next iteration

        if print_fn is not None:
            print_fn('=' * 80)
            print_fn(f'Iteration {iter_num}')
            for k, v in sorted(logs.items()):
                print_fn(f'{k}: {v}')

        return logs

    def train_step(self):
        batch = self.get_batch(self.batch_size, self.target_frame_rate, self.pad_frame_gaps)
        context, images, proprios, actions, rewards, returns_to_go, timesteps, mask = batch

        kwargs = {}
        if 'grounded_inverse_dynamics' in self.loss_weights.keys():
            kwargs = {'compute_pred_obs' : self.loss_weights['future_prediction'] > 0,
                      'compute_pred_future_actions' : (context is not None) and (self.loss_weights['predicted_inverse_dynamics'] > 0),
                      'compute_known_future_actions' : self.loss_weights['grounded_inverse_dynamics'] > 0}

        model_outputs = self.model.forward(*batch,  **kwargs)
        losses = self.model.compute_losses(model_outputs, actions,
                                           (context is not None),
                                           mask=mask)
        total_loss = sum(self.loss_weights[k] * losses[k] for k in losses.keys())
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        for k, v in losses.items():
            v = v.item()
            self.diagnostics[f'unscaled_loss/{k}'].append(v)
            self.diagnostics[f'scaled_loss/{k}'].append(self.loss_weights[k] * v)
        self.diagnostics['total_loss'].append(total_loss.item())

        return total_loss.item()