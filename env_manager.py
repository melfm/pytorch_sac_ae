import numpy as np

try:
    import gym
    from gym.spaces import  Dict , Box
    from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT

    mtw_envs = {**HARD_MODE_CLS_DICT["train"], **HARD_MODE_CLS_DICT["test"]}
    mtw_args = {**HARD_MODE_ARGS_KWARGS["train"], **HARD_MODE_ARGS_KWARGS["test"]}

except:
    mtw_envs = None
    mtw_args = None

try:
    import dmc2gym
except:
    dmc2gym = None

class EnvWrapper(gym.Wrapper):
    """Wrapper of the DMSuite and Metaworld environments
    """

    def __init__(self, make_env, eps_length):
        self.env = make_env()
        if eps_length:
            self.eps_length = eps_length
        elif "_max_episode_steps" in self.env.__dict__:
            self.eps_length = self.env._max_episode_steps
        elif "max_path_length" in self.env.__dict__:
            self.eps_length = self.env.max_path_length
        else:
            assert False, "max episode length unknown."

        self.action_space = self.env.action_space
        self.frame_size = 84
        self.observation_space = Box(low=0, high=255, shape=[3, self.frame_size, self.frame_size], dtype=np.uint8)
        self.max_u = self.action_space.high[0]


    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        state = self.env.render(mode='rgb_array', width = self.frame_size, height=self.frame_size)
        state = state.reshape(3, state.shape[0], state.shape[1])
        return state

    def seed(self, seed=0):
        return self.env.seed(seed)

    def step(self, action):
        _, r, done, info = self.env.step(action)
        state = self.env.render(mode='rgb_array', width = self.frame_size, height=self.frame_size)
        state = state.reshape(3, state.shape[0], state.shape[1])
        return state, r, done, info

    def close(self):
        return self.env.close()

class EnvManager:
    def __init__(self, env_pkg, env_name, env_args={}, eps_length=50):
        self.make_env = None
        self.eps_length = eps_length

        if env_pkg == "dmsuite":
            # Mel: Not tested.
            def make_env():
                env = dmc2gym.make(
                    domain_name=args.domain_name,
                    task_name=args.task_name,
                    seed=args.seed,
                    visualize_reward=False,
                    from_pixels=(args.encoder_type == 'pixel'),
                    height=args.image_size,
                    width=args.image_size,
                    frame_skip=args.action_repeat
                )
            return env
            self.make_env = make_env

        elif env_pkg == "metaworld":
            # Search in MetaWorld Envs
            if self.make_env is None and mtw_envs is not None and env_name in mtw_envs.keys():

                def make_env():
                    args = mtw_args[env_name]["args"]
                    kwargs = mtw_args[env_name]["kwargs"]
                    kwargs["random_init"] = False  # disable random goal locations
                    kwargs["obs_type"] = "with_goal"  # disable random goal locations
                    env = mtw_envs[env_name](*args, **kwargs)

                    # The metaworld env wrapper is missing `_state_goal_idx`
                    # make sure this is set to plain to avoid concatenating the goal
                    # since we are pixel-based anyway.
                    env.obs_type = "plain"
                    return env

                self.make_env = make_env

    def get_env(self):
        return EnvWrapper(self.make_env, self.eps_length)


if __name__ == "__main__":
    from PIL import Image

    env_manager = EnvManager(env_pkg="metaworld", env_name="pick-place-v1")
    env = env_manager.get_env()

    obs = env.reset()
    print('Stepping through the env and storing frames .... \n')
    for i in range(50):
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)

        name = 'robot_obs_' + str(i) + '.png'
        im = Image.fromarray(obs)
        im.save(name)

