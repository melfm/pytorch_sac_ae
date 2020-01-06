from env_manager import EnvManager
import argparse
import time
import glfw
import numpy as np
import argparse
import sys
import os

class DemoGenerator:
    def __init__(self, env, render):
        self.env = env
        self.render = render
        self.num_steps = 0
        self.frame_size = 84

        # store return from 1 episode
        self.episode_obs = []
        self.episode_pixel_obs = []
        self.episode_act = []
        self.episode_rwd = []
        self.episode_done = []
        self.episode_info = []

    def reset(self):
        # store return from 1 episode
        self.episode_obs = []
        self.episode_pixel_obs = []
        self.episode_act = []
        self.episode_rwd = []
        self.episode_done = []
        self.episode_info = []

        # reset env and store initial observation
        obs = self.env.reset()
        pixel_state = self.env.render(mode='rgb_array', width=self.frame_size, height=self.frame_size)
        pixel_state = pixel_state.reshape(3, pixel_state.shape[0], pixel_state.shape[1])
        self.episode_obs.append(obs)
        self.episode_pixel_obs.append(pixel_state)
        return obs

    def dump_eps(self):
        return self.episode_obs, self.episode_pixel_obs, self.episode_act, self.episode_rwd, self.episode_done, self.episode_info

    def move_to_goal(self, curr_pos, goal_pos, gripper):

        while True:
            action = [0, 0, 0, 0]
            for i in range(len(goal_pos - curr_pos)):
                action[i] = (goal_pos - curr_pos)[i] * 10
            action[-1] = gripper

            obs, reward, done, info = self.step(action)
            curr_pos = obs["observation"][:3]  # we assume that the first 3 dims are always the curr pos of the ee
            if done or np.linalg.norm(goal_pos - curr_pos) < 0.01:
                break
        return obs, reward, done, info

    def stay(self, gripper=-1.0, step=np.inf):
        done = False
        curr_step = 0
        while curr_step < step:
            action = [0, 0, 0, 0]
            action[-1] = gripper
            obs, reward, done, info = self.step(action)
            curr_step += 1
            if done:
                break
        return obs, reward, done, info

    def step(self, action):
        action = np.clip(action, -self.env.max_u, self.env.max_u)
        obs, reward, done, info = self.env.step(action)
        self.num_steps += 1
        if self.render:
            # Assuming you want human-mode
            self.env.render()

        pixel_state = self.env.render(mode='rgb_array', width=self.frame_size, height=self.frame_size)
        pixel_state = pixel_state.reshape(3, pixel_state.shape[0], pixel_state.shape[1])
        # update stats for the episode
        self.episode_obs.append(obs)
        self.episode_pixel_obs.append(pixel_state)
        self.episode_act.append(action)
        self.episode_rwd.append([reward])
        self.episode_done.append([done])
        self.episode_info.append(info)
        return obs, reward, done, info


# function that closes the render window
def close(env):
    if env.viewer is not None:
        # self.viewer.finish()
        glfw.destroy_window(env.viewer.window)
    env.viewer = None


def demo_hammer(render=True):
    env = EnvManager(env_pkg="metaworld", env_name="hammer-v1").get_env(pixel_only_based=False)
    demo_gen = DemoGenerator(env, render)

    obs = demo_gen.reset()

    # control
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][3:6]
    goal_pos[2] += 0.02
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, -1.0)
    assert not done
    obs, reward, done, info = demo_gen.stay(1.0, 15)
    assert not done
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][6:9]
    goal_pos[0] += -0.12
    goal_pos[1] += -0.2
    goal_pos[2] += 0.03
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, 1.0)
    assert not done
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][:3]
    goal_pos[1] += 0.15
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, 1.0)

    # stay until episode ends and then return
    if not done:
        obs, reward, done, info = demo_gen.stay(1.0)
    assert info["success"] == 1.0
    if render:
        close(env.env)

    return demo_gen.dump_eps()


def demo_reach(render=True):
    env = EnvManager(env_name="reach-v1").get_env()
    demo_gen = DemoGenerator(env, render)

    obs = demo_gen.reset()

    # control
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][6:9]
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, -1.0)
    assert not done
    obs, reward, done, info = demo_gen.stay(1.0, 3)

    # stay until episode ends and then return
    if not done:
        obs, reward, done, info = demo_gen.stay(1.0)
    assert info["success"] == 1.0
    if render:
        close(env.env)

    return demo_gen.dump_eps()


def demo_pick_place(render=True):
    env = EnvManager(env_name="reach-v1").get_env()
    demo_gen = DemoGenerator(env, render)

    obs = demo_gen.reset()

    # control
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][3:6]
    goal_pos[2] += 0.03
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, -1.0)
    assert not done
    obs, reward, done, info = demo_gen.stay(1.0, 4)
    assert not done
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][6:9]
    goal_pos[2] += 0.05
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, 1.0)

    # stay until episode ends and then return
    if not done:
        obs, reward, done, info = demo_gen.stay(1.0)
    assert info["success"] == 1.0
    if render:
        close(env.env)

    return demo_gen.dump_eps()

def demo_drawer_close(render=True):
    env = EnvManager(env_pkg="metaworld", env_name="drawer-close-v1").get_env()
    demo_gen = DemoGenerator(env, render)

    obs = demo_gen.reset()

    # control
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][3:6]
    goal_pos[2] += 0.1
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, -1.0)
    assert not done
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][3:6]
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, 1.0)
    assert not done
    curr_pos = obs["observation"][:3]
    goal_pos = obs["observation"][6:9]
    goal_pos[2] += 0.05
    obs, reward, done, info = demo_gen.move_to_goal(curr_pos, goal_pos, 1.0)

    # stay until episode ends and then return
    # if not done:
    obs, reward, done, info = demo_gen.stay(1.0)
    assert info["success"] == 1.0
    if render:
        close(env.env)

    return demo_gen.dump_eps()


demos = {
    "hammer-v1": demo_hammer,
    "reach-v1": demo_reach,
    "pick-place-v1": demo_pick_place,
    "drawer-close-v1": demo_drawer_close,
}


def main(env_name, exp_dir, num_itr, render, **kwargs):

    assert env_name in demos.keys(), "unregistered environment."

    demo_data_obs = []
    demo_data_obs_pix = []
    demo_data_act = []
    demo_data_done = []
    demo_data_reward = []
    demo_data_info = []

    for i in range(num_itr):
        print("Iteration: ", i)
        episode_obs, episode_obs_pix, episode_act, episode_rwd, episode_done, episode_info = demos[env_name](render=render)
        demo_data_obs.append(episode_obs)
        demo_data_obs_pix.append(episode_obs_pix)
        demo_data_act.append(episode_act)
        demo_data_reward.append(episode_rwd)
        demo_data_done.append(episode_done)
        demo_data_info.append(episode_info)

    T = len(demo_data_reward[0])
    dims = {
        "o": demo_data_obs[0][0]["observation"].shape,  # observation
        "o_pix": demo_data_obs[0][0]["pixel_obs"].shape,  # observation pixel-based
        "g": demo_data_obs[0][0]["desired_goal"].shape,  # goal (may be of shape (0,) if not exist)
        "u": demo_data_act[0][0].shape,
    }
    for key, value in demo_data_info[0][0].items():
        if type(value) == str:  # Note: for now, do not add info str to memory (replay buffer)
            continue
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims["info_{}".format(key)] = value.shape

    result = None
    for epsd in range(num_itr):  # we initialize the whole demo buffer at the start of the training
        obs, obs_pix, acts, goals, achieved_goals, rs = [], [], [], [], [], []
        info_keys = [key for key in dims.keys() if key.startswith("info")]
        info_values = [np.empty((T, *dims[key]), np.float32) for key in info_keys]
        for t in range(T):
            obs.append(demo_data_obs[epsd][t].get("observation"))
            obs_pix.append(demo_data_obs[epsd][t].get("pixel_obs"))
            goals.append(demo_data_obs[epsd][t].get("desired_goal"))
            achieved_goals.append(demo_data_obs[epsd][t].get("achieved_goal"))
            acts.append(demo_data_act[epsd][t])
            rs.append(demo_data_reward[epsd][t])
            for idx, key in enumerate(info_keys):
                info_values[idx][t] = demo_data_info[epsd][t][key.replace("info_", "")]


        obs.append(demo_data_obs[epsd][T].get("observation"))
        obs_pix.append(demo_data_obs[epsd][t].get("pixel_obs"))
        achieved_goals.append(demo_data_obs[epsd][T].get("achieved_goal"))

        episode = dict(o=obs, u=acts, g=goals, ag=achieved_goals, r=rs)
        for key, value in zip(info_keys, info_values):
            episode[key] = value

        for key, val in episode.items():
            episode[key] = np.array(val)[np.newaxis, ...]

        # UNCOMMENT to use the ring replay buffer! convert to (T x dims and add done signal)
        # episode = {k: v.reshape((-1, v.shape[-1])) for k, v in episode.items()}
        # episode["o_2"] = episode["o"][1:, ...]
        # episode["o"] = episode["o"][:-1, ...]
        # episode["ag_2"] = episode["ag"][1:, ...]
        # episode["ag"] = episode["ag"][:-1, ...]
        # episode["g_2"] = episode["g"][...]

        if result == None:
            result = episode
        else:
            for k in result.keys():
                result[k] = np.concatenate((result[k], episode[k]), axis=0)

    save_path = os.path.join(exp_dir, "demo_data.npz")
    print("Saving demonstration data of environment {} to {}".format(env_name, save_path))
    np.savez_compressed(save_path, **result)  # save the file


if __name__ == "__main__":

    exp_parser = argparse.ArgumentParser()
    exp_parser.add_argument("--env_name", help="environment", type=str, default="hammer-v1")
    exp_parser.add_argument(
        "--exp_dir", help="top level directory to store experiment results", type=str, default=os.getcwd()
    )
    exp_parser.add_argument(
        "--num_itr", help="number of iterations", type=int, default=10,
    )
    exp_parser.add_argument(
        "--render", help="render", type=bool, default=False,
    )
    exp_parser = exp_parser.parse_args()
    main(**vars(exp_parser))