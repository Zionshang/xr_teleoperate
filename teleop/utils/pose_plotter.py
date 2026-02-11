import numpy as np


class PosePlotter:
    def __init__(
        self,
        history: int = 2000,
        title: str = "target_pose trajectory",
        axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        view_init: tuple[float, float] | None = None,
        axis_scale: float = 0.08,
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._history = history
        self._axis_limits = axis_limits
        self._default_axis_scale = axis_scale
        self._series = {}
        self._points = {}
        self._axes_lines = {}

        self._plt.ion()
        self._fig = self._plt.figure()
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_xlabel("x (m)")
        self._ax.set_ylabel("y (m)")
        self._ax.set_zlabel("z (m)")
        self._ax.set_title(title)
        self._ax.set_box_aspect([1, 1, 1])
        if axis_limits is not None:
            (xlim, ylim, zlim) = axis_limits
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.set_zlim(*zlim)
        if view_init is not None:
            self._ax.view_init(elev=view_init[0], azim=view_init[1])

        colors = plt.rcParams.get("axes.prop_cycle", None)
        if colors is None:
            self._color_cycle = iter(["b", "g", "m", "c", "y", "k"])
        else:
            self._color_cycle = iter(colors.by_key().get("color", ["b", "g", "m", "c", "y", "k"]))

    def _ensure_series(self, label: str, color: str | None = None):
        if label in self._series:
            return
        if color is None:
            color = next(self._color_cycle)
        line, = self._ax.plot([], [], [], color=color, label=label)
        point = self._ax.scatter([], [], [], c=color, s=20)
        self._series[label] = {"xs": [], "ys": [], "zs": [], "line": line}
        self._points[label] = point
        self._ax.legend(loc="upper right")

    def update(self, pose_6d: np.ndarray, label: str = "target", color: str | None = None, draw: bool = True) -> None:
        self._ensure_series(label, color=color)
        x, y, z = pose_6d[:3]
        data = self._series[label]
        data["xs"].append(float(x))
        data["ys"].append(float(y))
        data["zs"].append(float(z))

        if len(data["xs"]) > self._history:
            data["xs"] = data["xs"][-self._history :]
            data["ys"] = data["ys"][-self._history :]
            data["zs"] = data["zs"][-self._history :]

        data["line"].set_data(data["xs"], data["ys"])
        data["line"].set_3d_properties(data["zs"])
        self._points[label]._offsets3d = ([data["xs"][-1]], [data["ys"][-1]], [data["zs"][-1]])

        if draw:
            self.draw()

    def update_matrix(
        self,
        pose_mat: np.ndarray,
        label: str = "target",
        color: str | None = None,
        draw_axes: bool = False,
        axis_scale: float | None = None,
        draw: bool = True,
    ) -> None:
        pose_6d = np.zeros(6, dtype=np.float64)
        pose_6d[:3] = pose_mat[:3, 3]
        self.update(pose_6d, label=label, color=color, draw=False)

        if draw_axes:
            if label not in self._axes_lines:
                self._axes_lines[label] = [self._ax.plot([], [], [], color=c, alpha=0.7)[0] for c in ("r", "g", "b")]
            scale = self._default_axis_scale if axis_scale is None else axis_scale
            origin = pose_mat[:3, 3]
            rot = pose_mat[:3, :3]
            for i, line in enumerate(self._axes_lines[label]):
                axis_end = origin + scale * rot[:, i]
                line.set_data([origin[0], axis_end[0]], [origin[1], axis_end[1]])
                line.set_3d_properties([origin[2], axis_end[2]])

        if draw:
            self.draw()

    def draw(self, pause: float = 0.001) -> None:
        if self._axis_limits is None:
            self._ax.relim()
            self._ax.autoscale_view()
        self._plt.pause(pause)
