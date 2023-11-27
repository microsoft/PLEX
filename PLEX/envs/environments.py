import pdb

import cv2
from copy import deepcopy
import gym
import numpy as np
from pathlib import Path
#import robomimic.envs.env_robosuite as env_robosuite
from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs, process_obs, ImageModality, DepthModality
import robomimic.utils.env_utils as EnvUtils
import metaworld
from metaworld.data.utils import get_mw_env


# ROBOSUITE_CAMERAS = ["agentview", "robot0_eye_in_hand"]
PROPRIO_KEYS = {
    'robosuite': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    'metaworld': ['proprio']
}
DEFAULT_CAM = {'robosuite': 'agentview',
               'metaworld': 'corner',
               'd4rl': None}
ROBOMIMIC_OTHER_IMAGE_OBS = ["shouldercamera0_image", "image", "image_side", "sideview_image"]
ROBOMIMIC_SUBSTITUTE_IMAGE_OBS = {"agentview_image": ["agentview_image", "image", "sideview_image", "shouldercamera0_image", "image_side"],
                                  "robot0_eye_in_hand_image": ["robot0_eye_in_hand_image", "image_wrist"]}
ROBOT_NAMES = {'IIWA', 'Jaco', 'Panda', 'Sawyer', 'UR5e', 'WidowX250', 'default'}

robotasks = ['Door',
            'Lift',
            'NutAssemblyRound',
            'NutAssemblySquare',
            'PickPlaceBread',
            'PickPlaceCan',
            'PickPlaceCereal',
            'PickPlaceMilk',
            'Stack',
            'Wipe',
            'ToolHang',
            'TwoArmTransport']
TASK_NAMES = {'robosuite': robotasks,
              'robomimic': robotasks,
              'd4rl': ['halfcheetah-random-v2',
                        'halfcheetah-medium-v2',
                        'halfcheetah-medium-replay-v2',
                        'halfcheetah-medium-expert-v2',
                        'halfcheetah-expert-v2',
                        'hopper-random-v2',
                        'hopper-medium-v2',
                        'hopper-medium-replay-v2',
                        'hopper-medium-expert-v2',
                        'hopper-expert-v2',
                        'walker2d-random-v2',
                        'walker2d-medium-v2',
                        'walker2d-medium-replay-v2',
                        'walker2d-medium-expert-v2',
                        'walker2d-expert-v2'],
              'metaworld': ['assembly-v2',
                            'basketball-v2',
                            'bin-picking-v2',
                            'box-close-v2',
                            'button-press-topdown-v2',
                            'button-press-topdown-wall-v2',
                            'button-press-v2',
                            'button-press-wall-v2',
                            'coffee-button-v2',
                            'coffee-pull-v2',
                            'coffee-push-v2',
                            'dial-turn-v2',
                            'disassemble-v2',
                            'door-close-v2',
                            'door-lock-v2',
                            'door-open-v2',
                            'door-unlock-v2',
                            'hand-insert-v2',
                            'drawer-close-v2',
                            'drawer-open-v2',
                            'faucet-open-v2',
                            'faucet-close-v2',
                            'hammer-v2',
                            'handle-press-side-v2',
                            'handle-press-v2',
                            'handle-pull-side-v2',
                            'handle-pull-v2',
                            'lever-pull-v2',
                            'peg-insert-side-v2',
                            'pick-place-wall-v2',
                            'pick-out-of-hole-v2',
                            'reach-v2',
                            'push-back-v2',
                            'push-v2',
                            'pick-place-v2',
                            'plate-slide-v2',
                            'plate-slide-side-v2',
                            'plate-slide-back-v2',
                            'plate-slide-back-side-v2',
                            'peg-unplug-side-v2',
                            'soccer-v2',
                            'stick-push-v2',
                            'stick-pull-v2',
                            'push-wall-v2',
                            'reach-wall-v2',
                            'shelf-place-v2',
                            'sweep-into-v2',
                            'sweep-v2',
                            'window-open-v2',
                            'window-close-v2'],
              'bridge': ['close_brown1fbox_flap__toykitchen2',
                         'close_large4fbox_flaps__toykitchen1',
                         'close_small4fbox_flaps__toykitchen1',
                         'close_small4fbox_flaps__toykitchen2',
                         'close_white1fbox_flap__toykitchen2',
                         'flip_cup_upright__toysink3_bww',
                         'flip_orange_pot_upright_in_sink__toykitchen2_room8052',
                         'flip_pot_upright_in_sink_distractors__toykitchen1',
                         'flip_pot_upright_which_is_in_sink__toysink1_room8052',
                         'flip_pot_upright_which_is_in_sink__toysink2_bww',
                         'flip_salt_upright__toykitchen2_room8052',
                         'lever_vertical_to_front__toykitchen1',
                         'lift_bowl__toykitchen2_room8052',
                         'open_brown1fbox_flap__toykitchen2',
                         'open_large4fbox_flaps__toykitchen1',
                         'open_small4fbox_flaps__toykitchen2',
                         'open_white1fbox_flap__toykitchen2',
                         'pick_up_any_cup__realkitchen1_dishwasher',
                         'pick_up_bit_holder__tool_chest',
                         'pick_up_blue_pen_and_put_into_drawer__tool_chest',
                         'pick_up_bowl_and_put_in_small4fbox__toykitchen1',
                         'pick_up_box_cutter_and_put_into_drawer__tool_chest',
                         'pick_up_closest_rainbow_Allen_key_set__tool_chest',
                         'pick_up_glass_cup__realkitchen1_dishwasher',
                         'pick_up_glue_and_put_into_drawer__tool_chest',
                         'pick_up_green_mug__realkitchen1_dishwasher',
                         'pick_up_pan_from_stove_distractors__toykitchen1',
                         'pick_up_pot_from_sink_distractors__toykitchen1',
                         'pick_up_red_srewdriver__tool_chest',
                         'pick_up_scissors_and_put_into_drawer__tool_chest',
                         'pick_up_sponge_and_wipe_plate__realkitchen1_counter',
                         'pick_up_violet_Allen_key__tool_chest',
                         'put_banana_in_pot_or_pan__toykitchen4',
                         'put_banana_on_plate__toykitchen1',
                         'put_big_spoon_from_basket_to_tray__toykitchen1',
                         'put_broccoli_in_bowl__toykitchen1',
                         'put_broccoli_in_pot_or_pan__toykitchen1',
                         'put_brush_into_pot_or_pan__toysink3_bww',
                         'put_can_in_pot__toykitchen2_room8052',
                         'put_carrot_in_bowl__toykitchen4',
                         'put_carrot_in_pot_or_pan__toykitchen2_room8052',
                         'put_carrot_on_cutting_board__toykitchen1',
                         'put_carrot_on_plate__toykitchen1',
                         'put_carrot_on_plate__toysink2_bww',
                         'put_corn_in_pan_which-is_on_stove_distractors__toykitchen1',
                         'put_corn_in_pan_which_is_on_stove_distractors__toykitchen1',
                         'put_corn_in_pot_which_is_in_sink_distractors__toykitchen1',
                         'put_corn_into_bowl__toykitchen1',
                         'put_corn_on_plate__toykitchen2_room8052',
                         'put_cup_from_anywhere_into_sink__toysink3_bww',
                         'put_cup_from_counter_or_drying_rack_into_sink__toysink2_bww',
                         'put_cup_into_pot_or_pan__toysink3_bww',
                         'put_detergent_from_sink_into_drying_rack__toysink3_bww',
                         'put_detergent_in_sink__toykitchen1',
                         'put_detergent_in_sink__toykitchen4',
                         'put_eggplant_in_pot_or_pan__toykitchen1',
                         'put_eggplant_into_pan__toysink1_room8052',
                         'put_eggplant_into_pot_or_pan__toysink2_bww',
                         'put_eggplant_on_plate__toykitchen1',
                         'put_fork_from_basket_to_tray__toykitchen1',
                         'put_green_squash_in_pot_or_pan__toykitchen1',
                         'put_green_squash_into_pot_or_pan__toysink3_bww',
                         'put_knife_in_pot_or_pan__toysink3_bww',
                         'put_knife_on_cutting_board__toykitchen1',
                         'put_knife_on_cutting_board__toykitchen2_room8052',
                         'put_knife_on_cutting_board__toysink2_bww',
                         'put_lemon_on_plate__toykitchen2_room8052',
                         'put_lid_on_pot_or_pan__toykitchen1',
                         'put_lid_on_pot_or_pan__toykitchen4',
                         'put_lid_on_pot_or_pan__toysink3_bww',
                         'put_lid_on_stove__toykitchen1',
                         'put_pan_from_drying_rack_into_sink__toysink1_room8052',
                         'put_pan_from_sink_into_drying_rack__toysink1_room8052',
                         'put_pan_from_stove_to_sink__toysink1_room8052',
                         'put_pan_in_sink__toykitchen1',
                         'put_pan_on_stove_from_sink__toysink1_room8052',
                         'put_pear_in_bowl__toykitchen1',
                         'put_pear_in_bowl__toykitchen2_room8052',
                         'put_pear_on_plate__toykitchen4',
                         'put_pepper_in_pan__toykitchen1',
                         'put_pepper_in_pot_or_pan__toykitchen1',
                         'put_pot_in_sink__toykitchen1',
                         'put_pot_on_stove_which_is_near_stove_distractors__toykitchen1',
                         'put_pot_or_pan_from_sink_into_drying_rack__toysink3_bww',
                         'put_pot_or_pan_in_sink__toykitchen2_room8052',
                         'put_pot_or_pan_on_stove__toykitchen2_room8052',
                         'put_potato_in_pot_or_pan__toykitchen2_room8052',
                         'put_potato_on_plate__toykitchen2_room8052',
                         'put_red_bottle_in_sink__toykitchen1',
                         'put_small_spoon_from_basket_to_tray__toykitchen1',
                         'put_spatula_in_pan__toykitchen2_room8052',
                         'put_spoon_in_pot__toysink2_bww',
                         'put_spoon_into_pan__toysink1_room8052',
                         'put_spoon_on_plate__realkitchen1_counter',
                         'put_strawberry_in_pot__toykitchen2_room8052',
                         'put_sushi_in_pot_or_pan__toykitchen4',
                         'put_sushi_on_plate__toykitchen1',
                         'put_sushi_on_plate__toykitchen2_room8052',
                         'put_sweet_potato_in_pan_which_is_on_stove__toykitchen1',
                         'put_sweet_potato_in_pan_which_is_on_stove_distractors__toykitchen1',
                         'put_sweet_potato_in_pot__toykitchen2_room8052',
                         'put_sweet_potato_in_pot_which_is_in_sink_distractors__toykitchen1',
                         'take_broccoli_out_of_pan__toykitchen1',
                         'take_can_out_of_pan__toykitchen1',
                         'take_carrot_off_plate__toykitchen1',
                         'take_lid_off_pot_or_pan__toykitchen1',
                         'take_lid_off_pot_or_pan__toysink3_bww',
                         'take_sushi_out_of_pan__toykitchen1',
                         'turn_faucet_front_to_left__toykitchen1',
                         'turn_lever_vertical_to-front__toysink2_bww',
                         'turn_lever_vertical_to_front__toykitchen2_room8052',
                         'turn_lever_vertical_to_front__toysink3_bww',
                         'turn_lever_vertical_to_front_distractors__toykitchen1',
                         'twist_knob_start_vertical_clockwise90__toykitchen1']
             }

def compose_proprio(obs, keys):
    return np.hstack([obs[k] for k in keys])

def get_image_keys(keys):
    return [k for k in keys if k.endswith('_image')]

def process_image(image, new_size):
    assert image.ndim == 3
    # At this point, the image array is still in the HWC format.
    if image.shape[0] == new_size and image.shape[1] == new_size:
        # Is it OK not to copy the image here?
        resized = deepcopy(image)
    else:
        resized = cv2.resize(image, dsize=(new_size, new_size))
    resized = ImageModality._default_obs_processor(resized)
    # Now the image array is in the CWH format.
    return resized

def unprocess_image(image):
    assert image.ndim == 3
    processed = deepcopy(image)
    # At this point, the image array is still in the CHW format.
    processed = ImageModality._default_obs_unprocessor(processed)
    # Now the image array is in the HWC format and scaled back to [0, 255]
    return processed

def process_depth_map(image, new_size):
    assert image.ndim == 3
    assert image.shape[2] == 1
    if image.shape[0] == new_size and image.shape[1] == new_size:
        # Is it OK not to copy the depth map here?
        resized = deepcopy(image)
    else:
        resized = cv2.resize(image, dsize=(new_size, new_size))
        # Restore the channel dimension -- cv2 discard it if num channels is 1
        resized = resized[:,:,None]
    resized = DepthModality._default_obs_processor(resized)
    return resized

def sparse_reward(goal_reached):
    return (0.0 if goal_reached else -1.0)

def init_obs_preprocessing(camera_names, target_img_size):
    def resize_img(obs):
        return process_image(obs, target_img_size)
    def resize_depth_map(obs):
        return process_depth_map(obs, target_img_size)
    ImageModality.set_obs_processor(resize_img)
    DepthModality.set_obs_processor(resize_depth_map)
    rgb_obs = ["{}_image".format(camera) for camera in camera_names]
    depth_obs = ["{}_depth".format(camera) for camera in camera_names]
    obs_modality_specs = {
        "obs": {
            "low_dim": PROPRIO_KEYS['robosuite'],
            "rgb": rgb_obs,
            "depth": depth_obs
        },
        "goal": {
            "low_dim": PROPRIO_KEYS['robosuite'],
            "rgb": rgb_obs,
            "depth": depth_obs
        }
    }

    initialize_obs_utils_with_obs_specs(obs_modality_specs)


class d4rlEnv(gym.Env):
    def __init__(self, task, full_state_mode):
        import d4rl

        self.full_state_mode = full_state_mode
        assert self.full_state_mode, "D4RL envs are to be used only in full-state mode."
        self._env = gym.make(task.name)
        obs = self.reset()
        self.obs_dims = self._env.observation_space.shape[0]
        self.proprio_dim = 1
        self.action_dim = self._env.action_space.shape[0]

    def _format_obs(self, obs):
        ret = {'state': obs}
        return ret


    def reset(self):
        obs = self._env.reset()
        return self._format_obs(obs)


    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self._format_obs(obs), reward, done, info


class MetaWorldEnv(gym.Env):
    def __init__(self, task, use_normalized_reward, full_state_mode, env_meta=None, steps_at_goal=1, render_gpu_device_id=-1,
                 camera_name=None, image_size=84, train_distrib=False):
        self.use_sparse_reward = False
        self.full_state_mode = full_state_mode
        self._env = get_mw_env(task_name=task.name,
                                cam_height=image_size,
                                cam_width=image_size,
                                cam_name=(camera_name if not self.full_state_mode else None), # This turns off rendering when we aren't using image obs.
                                goal_cost_reward=self.use_sparse_reward,
                                # We include 'proprios' in ('states', 'proprios') only for compatibility.
                                obs_types=(('images', 'proprio_states') if not self.full_state_mode else ('states', 'proprio_states')),
                                fix_task_sequence=False,
                                steps_at_goal=steps_at_goal,
                                stop_at_goal=True,
                                train_distrib=train_distrib,
                                use_normalized_reward=use_normalized_reward)
        self.camera_name = camera_name
        self.image_size = image_size

        obs = self.reset()
        if not self.full_state_mode:
            self.obs_dims = obs[f'{self.camera_name}_image'].shape
            self.proprio_dim = obs['proprio'].shape[0]
        else:
            self.obs_dims = obs['state'].shape[0]
            # There will be no proprio states. self.proprio_dim = 1 just keeps code from breaking.
            self.proprio_dim = 1
        self.action_dim = 4


    def _format_obs(self, obs):
        # process_obs ensures that image tensor entries are scaled to [0, 1], the image is in the CWH format, and has the correct size (3 x self.image_size x self.image_size).
        # An init_obs_preprocessing(.) call in experiment.py sets all the relevant parameters for process_obs.
        if not self.full_state_mode:
            ret = {'proprio': obs['proprio_state'],
                f'{self.camera_name}_image': process_obs(obs['image'], 'rgb')}
        else:
            ret = {'state': obs['state']}
        return ret


    def reset(self):
        obs = self._env.reset()
        return self._format_obs(obs)


    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self._format_obs(obs), reward, done, info


class RobosuiteEnv(gym.Env):
    # If env_meta is specified, image_size and postprocess_visual_obs will be ignored.
    def __init__(self, task, use_normalized_reward, full_state_mode, env_meta=None, render_gpu_device_id=-1,
                 camera_names=None, image_size=84, postprocess_visual_obs=True):
        self.full_state_mode = full_state_mode
        if self.full_state_mode:
            raise NotImplementedError
        if use_normalized_reward:
            raise NotImplementedError
        if isinstance(camera_names, str):
            self.camera_names = camera_names.split(',')
        elif type(camera_names) in {list, tuple}:
            self.camera_names = list(camera_names)
        else:
            raise ValueError(f"camera_names should be str or list of str, but got {camera_names}")

        if env_meta is not None:
            env_meta = deepcopy(env_meta)
            env_meta['env_kwargs']['camera_names'] = self.camera_names
            self._env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=task.name,
                use_image_obs=True
            )
        else:
            from robomimic.envs.env_robosuite import EnvRobosuite
            from robosuite.controllers import load_controller_config
            self._env = EnvRobosuite(
                env_name=task.name,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
                postprocess_visual_obs=postprocess_visual_obs,
                # ---------------------- kwargs -------------------------
                render_gpu_device_id=render_gpu_device_id,
                camera_names=self.camera_names,
                camera_widths=image_size,
                camera_heights=image_size,
                camera_depths=[False] * len(self.camera_names),
                robots=task.robot,
                # Even if use_sparse_reward it True, we will let the native env. return its native reward and convert it to
                # the particular kind of sparse reward we need on the fly.
                reward_shaping=True,
                controller_configs = load_controller_config(default_controller="OSC_POSE")
                # Explicitly add more params from here: https://robosuite.ai/docs/modules/environments.html ?
            )

        self.use_sparse_reward = False
        self.image_size = image_size

        obs = self.reset()
        self.proprio_dim = obs['proprio'].shape[0]
        self.obs_dims = obs[f'{self.camera_names[0]}_image'].shape

        self.action_dim = self._env.action_dimension

    def _format_obs(self, obs):
        # ASSUMPTION: The images tensor entries are scaled to [0, 1]. The tensor is in the CWH format and has the correct size (3 x self.image_size x self.image_size)
        # The assumption is enforced by Robomimic under the hood thanks to the init_obs_preprocessing(.) call in experiment.py.
        ret = {'proprio': compose_proprio(obs, keys=PROPRIO_KEYS['robosuite'])}
        for cam_name in self.camera_names:
            key = f'{cam_name}_image'
            ret[key] = obs[key]
        return ret

    def reset(self):
        obs = self._env.reset()
        return self._format_obs(obs)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        info['success'] = self._env.is_success()['task']
        if self.use_sparse_reward:
            reward = sparse_reward(info['success'])
        return self._format_obs(obs), reward, done, info
