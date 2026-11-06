"""
A simplified version from OpenAI Baselines code to work with gym.env parallelization.
"""
import os
import contextlib
import multiprocessing
from numbers import Integral
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection

from algorithms.pbgr.step_contract import episode_done, preserve_final_observation


def configure_worker_threads(default=1):
    """Prevent each spawned environment process from oversubscribing CPUs."""
    value = int(os.environ.get("WVR_WORKER_THREADS", default))
    if value <= 0:
        raise ValueError("WVR_WORKER_THREADS must be a positive integer")
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(value)
    try:
        import torch

        torch.set_num_threads(value)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows setting inter-op threads only before parallel
            # work starts. Spawned workers normally take the first branch.
            pass
    except ImportError:
        pass
    return value


def _validate_subproc_configuration(env_fns, context, in_series):
    """Validate subprocess grouping and return the requested multiprocessing context."""
    if not isinstance(context, str):
        raise TypeError("context must be a multiprocessing start-method string")
    try:
        mp_context = multiprocessing.get_context(context)
    except ValueError as exc:
        available = ", ".join(multiprocessing.get_all_start_methods())
        raise ValueError(
            f"unsupported multiprocessing context {context!r}; available: {available}"
        ) from exc

    if isinstance(in_series, bool) or not isinstance(in_series, Integral):
        raise TypeError("in_series must be a positive integer")
    in_series = int(in_series)
    if in_series <= 0:
        raise ValueError("in_series must be a positive integer")

    env_fns = tuple(env_fns)
    nenvs = len(env_fns)
    if nenvs == 0:
        raise ValueError("env_fns must contain at least one environment factory")
    if nenvs % in_series != 0:
        raise ValueError(
            "Number of envs must be divisible by number of envs to run in series"
        )
    return env_fns, mp_context, in_series, nenvs


def _geometry_state_sequence(states, expected_length):
    if not isinstance(states, (list, tuple)):
        raise ValueError("initial geometry states must be a list or tuple")
    values = list(states)
    if len(values) != int(expected_length):
        raise ValueError("initial geometry state count must match vector environments")
    return values


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(self.envs), env.observation_space, env.action_space)

        self.actions = None
        self.num_agents = getattr(self.envs[0], "num_agents", 1)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obss, rewards, dones, infos = map(list, zip(*results))
        for i, done in enumerate(dones):
            if episode_done(done):
                infos[i] = preserve_final_observation(infos[i], obss[i])
                obss[i] = self.envs[i].reset()
        self.actions = None
        return self._flatten(obss), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        obss = [env.reset() for env in self.envs]
        return self._flatten(obss)

    def get_initial_geometry_states(self):
        return [env.get_initial_geometry_state() for env in self.envs]

    def set_initial_geometry_states(self, states):
        values = _geometry_state_sequence(states, self.num_envs)
        for env, state in zip(self.envs, values):
            env.set_initial_geometry_state(state)

    def set_initial_geometry_reset_indices(self, reset_indices):
        values = _geometry_state_sequence(reset_indices, self.num_envs)
        for env, reset_index in zip(self.envs, values):
            env.set_initial_geometry_reset_index(reset_index)

    def get_bric_weapon_action_masks(self):
        return np.stack(
            [env.get_bric_weapon_action_masks() for env in self.envs], axis=0
        )

    def get_v22_weapon_action_masks(self):
        values = [env.get_v22_weapon_action_masks() for env in self.envs]
        return {
            "target_masks": np.stack([value["target_masks"] for value in values], axis=0),
            "fire_masks": np.stack([value["fire_masks"] for value in values], axis=0),
        }

    def get_v22_platform_contract(self):
        values = [env.get_v22_platform_contract() for env in self.envs]
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("V22 vector environments disagree on platform contract")
        return dict(values[0])

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode, filepath):
        if mode == 'txt':
            self.envs[0].render(mode, filepath)

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)


def worker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """Maintain an environment instance in subprocess,
    communicate with parent-process via multiprocessing.Pipe.

    Args:
        remote (Connection): used for current subprocess to send/receive data.
        parent_remote (Connection): used for mainprocess to send/receive data. [Need to be closed in subprocess!]
        env_fn_wrappers (method): functions to create gym.Env instance.
    """
    def step_env(env, action):
        obs, reward, done, info = env.step(action)
        if episode_done(done):
            info = preserve_final_observation(info, obs)
            obs = env.reset()
        return obs, reward, done, info

    configure_worker_threads()
    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space)))
            elif cmd == 'get_num_agents':
                remote.send(CloudpickleWrapper((getattr(envs[0], "num_agents", 1))))
            elif cmd == 'get_initial_geometry_states':
                remote.send([env.get_initial_geometry_state() for env in envs])
            elif cmd == 'set_initial_geometry_states':
                values = _geometry_state_sequence(data, len(envs))
                for env, state in zip(envs, values):
                    env.set_initial_geometry_state(state)
                remote.send(True)
            elif cmd == 'set_initial_geometry_reset_indices':
                values = _geometry_state_sequence(data, len(envs))
                for env, reset_index in zip(envs, values):
                    env.set_initial_geometry_reset_index(reset_index)
                remote.send(True)
            elif cmd == 'get_bric_weapon_action_masks':
                remote.send(
                    [env.get_bric_weapon_action_masks() for env in envs]
                )
            elif cmd == 'get_v22_weapon_action_masks':
                remote.send([env.get_v22_weapon_action_masks() for env in envs])
            elif cmd == 'get_v22_platform_contract':
                remote.send([env.get_v22_platform_contract() for env in envs])
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Args:
            env_fns: iterable of callables - functions that create environments to run in subprocesses. Need to be cloud-pickleable
            context (str, optional): Defaults to 'spawn'.
            in_series (int, optional): number of environments to run in series in a single process. Defaults to 1.
                (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        env_fns, self._mp_context, self.in_series, nenvs = (
            _validate_subproc_configuration(env_fns, context, in_series)
        )
        self.start_method = self._mp_context.get_start_method()
        self.nremotes = nenvs // self.in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        # create Pipe connections to send/recv data from subprocesses,
        self.remotes, self.work_remotes = zip(
            *[self._mp_context.Pipe() for _ in range(self.nremotes)]
        )
        self.ps = [
            self._mp_context.Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv().x
        super().__init__(nenvs, observation_space, action_space)

        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv().x

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)  # [[tuple] * in_series] * nremotes => [tuple] * nenvs
        self.waiting = False
        obss, rewards, dones, infos = zip(*results)
        return self._flatten(obss), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obss = [remote.recv() for remote in self.remotes]
        obss = self._flatten_series(obss)
        return self._flatten(obss)

    def get_initial_geometry_states(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_initial_geometry_states', None))
        return self._flatten_series([remote.recv() for remote in self.remotes])

    def _send_geometry_values(self, command, values):
        self._assert_not_closed()
        values = _geometry_state_sequence(values, self.num_envs)
        chunks = [
            values[index : index + self.in_series]
            for index in range(0, self.num_envs, self.in_series)
        ]
        for remote, chunk in zip(self.remotes, chunks):
            remote.send((command, chunk))
        acknowledgements = [remote.recv() for remote in self.remotes]
        if not all(value is True for value in acknowledgements):
            raise RuntimeError("vector geometry state update was not acknowledged")

    def set_initial_geometry_states(self, states):
        self._send_geometry_values('set_initial_geometry_states', states)

    def set_initial_geometry_reset_indices(self, reset_indices):
        self._send_geometry_values(
            'set_initial_geometry_reset_indices', reset_indices
        )

    def get_bric_weapon_action_masks(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_bric_weapon_action_masks', None))
        values = self._flatten_series([remote.recv() for remote in self.remotes])
        return np.stack(values, axis=0)

    def get_v22_weapon_action_masks(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_v22_weapon_action_masks', None))
        values = self._flatten_series([remote.recv() for remote in self.remotes])
        return {
            "target_masks": np.stack([value["target_masks"] for value in values], axis=0),
            "fire_masks": np.stack([value["fire_masks"] for value in values], axis=0),
        }

    def get_v22_platform_contract(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_v22_platform_contract', None))
        values = self._flatten_series([remote.recv() for remote in self.remotes])
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("V22 vector environments disagree on platform contract")
        return dict(values[0])

    def close_extras(self):
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)

    @classmethod
    def _flatten_series(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0
        assert all([len(v_) > 0 for v_ in v])

        return [v__ for v_ in v for v__ in v_]


class ShareVecEnv(VecEnv):
    """
    Multi-agent version of VevEnv, that is, support `share_observation_space` interface.
    """
    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        super().__init__(num_envs, observation_space, action_space)
        self.share_observation_space = share_observation_space


class ShareDummyVecEnv(DummyVecEnv, ShareVecEnv):
    """
    Multi-agent version of DummyVecEnv, that is, support `share_observation_space` interface.

    DummyVecEnv is a VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case, avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(self.envs), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None
        self.num_agents = getattr(self.envs[0], "num_agents", 1)

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos = map(list, zip(*results))
        for i, done in enumerate(dones):
            if episode_done(done):
                infos[i] = preserve_final_observation(
                    infos[i], obs[i], share_obs[i]
                )
                obs[i], share_obs[i] = self.envs[i].reset()
        self.actions = None
        return self._flatten(obs), self._flatten(share_obs), self._flatten(rews), self._flatten(dones), np.array(infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs = map(np.array, zip(*results))
        return obs, share_obs


def shareworker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """Maintain an environment instance in subprocess,
    communicate with parent-process via multiprocessing.Pipe.

    Args:
        remote (Connection): used for current subprocess to send/receive data.
        parent_remote (Connection): used for mainprocess to send/receive data. [Need to be closed in subprocess!]
        env_fn_wrappers (method): functions to create gym.Env instance.
    """
    def step_env(env, action):
        obs, share_obs, reward, done, info = env.step(action)
        if episode_done(done):
            info = preserve_final_observation(info, obs, share_obs)
            obs, share_obs = env.reset()
        return obs, share_obs, reward, done, info

    configure_worker_threads()
    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].share_observation_space, envs[0].action_space)))
            elif cmd == 'get_num_agents':
                remote.send(CloudpickleWrapper((getattr(envs[0], "num_agents", 1))))
            elif cmd == 'get_initial_geometry_states':
                remote.send([env.get_initial_geometry_state() for env in envs])
            elif cmd == 'set_initial_geometry_states':
                values = _geometry_state_sequence(data, len(envs))
                for env, state in zip(envs, values):
                    env.set_initial_geometry_state(state)
                remote.send(True)
            elif cmd == 'set_initial_geometry_reset_indices':
                values = _geometry_state_sequence(data, len(envs))
                for env, reset_index in zip(envs, values):
                    env.set_initial_geometry_reset_index(reset_index)
                remote.send(True)
            elif cmd == 'get_bric_weapon_action_masks':
                remote.send(
                    [env.get_bric_weapon_action_masks() for env in envs]
                )
            elif cmd == 'get_v22_weapon_action_masks':
                remote.send([env.get_v22_weapon_action_masks() for env in envs])
            elif cmd == 'get_v22_platform_contract':
                remote.send([env.get_v22_platform_contract() for env in envs])
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class ShareSubprocVecEnv(SubprocVecEnv, ShareVecEnv):
    def __init__(self, env_fns, context='spawn', in_series=1):
        self.waiting = False
        self.closed = False
        env_fns, self._mp_context, self.in_series, nenvs = (
            _validate_subproc_configuration(env_fns, context, in_series)
        )
        self.start_method = self._mp_context.get_start_method()
        self.nremotes = nenvs // self.in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        # create Pipe connections to send/recv data from subprocesses,
        self.remotes, self.work_remotes = zip(
            *[self._mp_context.Pipe() for _ in range(self.nremotes)]
        )
        self.ps = [
            self._mp_context.Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv().x
        ShareVecEnv.__init__(self, nenvs, observation_space, share_observation_space, action_space)

        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv().x

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results) # [[tuple] * in_series] * nremotes => [tuple] * nenvs
        self.waiting = False
        obs, share_obs, rewards, dones, infos = zip(*results) 
        return self._flatten(obs), self._flatten(share_obs), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)
        obs, share_obs = zip(*results)
        return self._flatten(obs), self._flatten(share_obs)
