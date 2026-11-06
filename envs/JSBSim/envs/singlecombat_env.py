import numpy as np
from .env_base import BaseEnv
from ..utils.utils import NEU2LLA
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask
from ..human_task.HumanSingleCombatTask import HumanSingleCombatTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv 是一个一对一的竞争环境。
    已修改：增强了 reset_simulators 的随机性，以支持大规模对抗轨迹数据采集。
    """

    def __init__(self, config_name: str, reuse_fdm: bool = False,
                 config_overrides=None):
        super().__init__(
            config_name, reuse_fdm=reuse_fdm,
            config_overrides=config_overrides)
        # 环境特定初始化
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        """
        重置模拟器并注入大范围随机扰动。
        确保最终模型与历史模型对抗时，初始位置、高度、速度和姿态完全随机。
        """
        if self.init_states is None:
            # 记录配置文件中的原始初始状态
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]

        if bool(getattr(self.config, "InitialEngagement_enabled", False)):
            self._reset_engagement_geometry()
            self._tempsims.clear()
            return

        # 创建临时状态列表用于注入随机性
        temp_init_states = []
        for base_state in self.init_states:
            s = base_state.copy()
            # 1. 随机方位角：0 到 360 度完全随机
            s['ic_psi_true_deg'] = self.np_random.uniform(0, 360)

            # 2. 随机高度：在 15,000 到 25,000 英尺之间波动
            s['ic_h_sl_ft'] = self.np_random.uniform(15000, 25000)

            # 3. 随机初速：在 600 到 900 英尺/秒之间波动
            s['ic_u_fps'] = self.np_random.uniform(600, 900)

            # 4. 随机经纬度：在原始位置基础上注入 ±0.05 度的偏移量（约 ±5.5 公里）
            # 这确保了交战起始点在空间上的随机分布
            s['ic_long_gc_deg'] = s.get('ic_long_gc_deg', 120.0) + self.np_random.uniform(-0.05, 0.05)
            s['ic_lat_geod_deg'] = s.get('ic_lat_geod_deg', 60.0) + self.np_random.uniform(-0.05, 0.05)

            # 5. 随机初始角速度（微小扰动）
            s['ic_p_rad_sec'] = self.np_random.uniform(-0.01, 0.01)
            s['ic_q_rad_sec'] = self.np_random.uniform(-0.01, 0.01)
            s['ic_r_rad_sec'] = self.np_random.uniform(-0.01, 0.01)

            temp_init_states.append(s)

        # 随机交换 ego 和 opponent 的初始配置，进一步增加不确定性
        self.np_random.shuffle(temp_init_states)

        # 重新加载模拟器
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(temp_init_states[idx])

        # 清除临时生成的弹药信息等
        self._tempsims.clear()

    def _reset_engagement_geometry(self):
        """Sample a symmetric, reachable WVR engagement geometry.

        The former independent latitude/longitude and heading randomisation
        frequently created 30 km, mutually-diverging starts.  This sampler
        randomises a *relative* engagement: range, line-of-sight bearing,
        aspect, altitude and speed remain diverse, while both aircraft start
        close enough that either side can create a weapon opportunity.
        """

        if len(self.init_states) != 2:
            raise ValueError("initial engagement sampling requires exactly two aircraft")
        minimum = float(getattr(
            self.config, "InitialEngagement_min_range_m", 8000.0))
        maximum = float(getattr(
            self.config, "InitialEngagement_max_range_m", 16000.0))
        if not 0.0 < minimum <= maximum:
            raise ValueError("invalid initial engagement range")

        separation = float(self.np_random.uniform(minimum, maximum))
        bearing_deg = float(self.np_random.uniform(0.0, 360.0))
        bearing = np.deg2rad(bearing_deg)
        midpoint_jitter = float(getattr(
            self.config, "InitialEngagement_midpoint_jitter_m", 2000.0))
        midpoint = self.np_random.uniform(
            -midpoint_jitter, midpoint_jitter, size=2)
        half_offset = 0.5 * separation * np.asarray([
            np.cos(bearing), np.sin(bearing)], dtype=np.float64)
        positions = (midpoint - half_offset, midpoint + half_offset)

        min_altitude = float(getattr(
            self.config, "InitialEngagement_min_altitude_m", 5000.0))
        max_altitude = float(getattr(
            self.config, "InitialEngagement_max_altitude_m", 7500.0))
        altitude_delta = float(getattr(
            self.config, "InitialEngagement_max_altitude_delta_m", 600.0))
        midpoint_altitude = float(self.np_random.uniform(
            min_altitude + altitude_delta / 2.0,
            max_altitude - altitude_delta / 2.0))
        signed_delta = float(self.np_random.uniform(
            -altitude_delta, altitude_delta))
        altitudes = (
            midpoint_altitude - signed_delta / 2.0,
            midpoint_altitude + signed_delta / 2.0,
        )

        aspect_probabilities = np.asarray(getattr(
            self.config,
            "InitialEngagement_aspect_probabilities",
            [0.50, 0.30, 0.20],
        ), dtype=np.float64)
        if (aspect_probabilities.shape != (3,) or
                np.any(aspect_probabilities < 0.0) or
                not np.isclose(aspect_probabilities.sum(), 1.0)):
            raise ValueError(
                "InitialEngagement_aspect_probabilities must be three probabilities summing to one")
        aspect = int(self.np_random.choice(3, p=aspect_probabilities))
        if aspect == 0:  # head-on with a modest off-boresight perturbation
            offset = float(self.np_random.uniform(-20.0, 20.0))
            headings = (bearing_deg + offset, bearing_deg + 180.0 - offset)
        elif aspect == 1:  # crossing geometry
            magnitude = float(self.np_random.uniform(35.0, 70.0))
            offset = magnitude if self.np_random.random() < 0.5 else -magnitude
            headings = (bearing_deg + offset, bearing_deg + 180.0 - offset)
        else:  # pursuit; randomise which colour begins as the chaser
            chase_a = bool(self.np_random.random() < 0.5)
            common = bearing_deg if chase_a else bearing_deg + 180.0
            headings = (
                common + float(self.np_random.uniform(-12.0, 12.0)),
                common + float(self.np_random.uniform(-12.0, 12.0)),
            )

        min_speed = float(getattr(
            self.config, "InitialEngagement_min_speed_fps", 650.0))
        max_speed = float(getattr(
            self.config, "InitialEngagement_max_speed_fps", 900.0))
        states = []
        for index, base_state in enumerate(self.init_states):
            state = base_state.copy()
            lon, lat, _ = NEU2LLA(
                positions[index][0], positions[index][1], altitudes[index],
                self.center_lon, self.center_lat, self.center_alt)
            state["ic_long_gc_deg"] = float(lon)
            state["ic_lat_geod_deg"] = float(lat)
            state["ic_h_sl_ft"] = float(altitudes[index] / 0.3048)
            state["ic_psi_true_deg"] = float(headings[index] % 360.0)
            state["ic_u_fps"] = float(self.np_random.uniform(
                min_speed, max_speed))
            state["ic_p_rad_sec"] = float(self.np_random.uniform(-0.01, 0.01))
            state["ic_q_rad_sec"] = float(self.np_random.uniform(-0.01, 0.01))
            state["ic_r_rad_sec"] = float(self.np_random.uniform(-0.01, 0.01))
            states.append(state)

        for simulator, state in zip(self.agents.values(), states):
            simulator.reload(state)
