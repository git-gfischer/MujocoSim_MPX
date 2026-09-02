from __future__ import annotations

import matplotlib as mpl

mpl.use('TkAgg')
import contextlib
import multiprocessing as mp
import os
import random
import signal
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, RangeSlider


class MujocoPlotter:
    """TODO: DOCSTRING."""

    def __init__(self, enable=True):
        self.plots = {}

        self.legs = ['FL', 'FR', 'RL', 'RR']
        self.joint_names = ['HAA', 'HFE', 'KFE']
        self.predefined_plots = [
            'Torque', 'JointPos', 'JointVel', 'FootContacts', 'GRF', 'LinAcc', 'AngVel',
        ]
        self.grf_components = ['Fx', 'Fy', 'Fz']
        self.axis = ['X', 'Y', 'Z']

        self.all_plot_enable = enable

    def create(
        self,
        figure_name: str,
        subplot_titles: list,
        y_limits: list = None,
        rows: int = 1,
        cols: int = 1,
        window_size: int = 50,
        plots_per_ax: int = 1,
        enabled_flag=None,
        dual_secondary: bool = False,
        secondary_labels: list | None = None,
    ):
        """Create new Plot figure."""
        if y_limits is None:
            y_limits = [-1, 1]
        plotter = MultiLivePlotter(
            figure_name=figure_name,
            num_subplots=rows * cols,
            subplot_titles=subplot_titles,
            nrows=rows,
            ncols=cols,
            window_size=window_size,
            x_limits=[(0, window_size)] * (rows * cols),
            y_limits=y_limits * (rows * cols),
            plot_per_ax=plots_per_ax,
            enabled_flag=enabled_flag,
            dual_secondary=dual_secondary,
            secondary_labels=secondary_labels,
        )
        self.plots[figure_name] = plotter

    # ---------------------------------------------------
    def predefined_plot(  # noqa: D102
        self, name: str, y_limit: list, legs: list = None, joint_names: list = None, window_size: int = 50,
        enabled_flag=None,
    ):
        if name not in self.predefined_plots:
            print(f'Error: predefined plot {name} does not exist between: {self.predefined_plots}')
            return

        if legs is None:
            legs = self.legs
        if joint_names is None:
            joint_names = self.joint_names

        titles = []
        rows = 0
        cols = 0
        for leg in legs:
            rows += 1
            cols = 0
            for joint_name in joint_names:
                cols += 1
                titles += [f'{name} {leg}_{joint_name}']
        self.create(
            figure_name=name,
            rows=rows,
            cols=cols,
            window_size=window_size,
            subplot_titles=titles,
            y_limits=y_limit,
            enabled_flag=enabled_flag,
        )
        return legs, joint_names

    def torque_plot(self, legs: list = None, joint_names: list = None, window_size: int = 50, enable: bool = True, enabled_flag=None):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.torque_legs, self.torque_joint_names = self.predefined_plot(
            name='Torque',
            y_limit=[(-120, 120)],
            legs=legs,
            joint_names=joint_names,
            window_size=window_size,
            enabled_flag=enabled_flag,
        )

    def jointpos_plot(self, legs: list = None, joint_names: list = None, window_size: int = 50, enable: bool = True, enabled_flag=None):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.jp_legs, self.jp_joint_names = self.predefined_plot(
            name='JointPos',
            y_limit=[(-3.5, 3.5)],
            legs=legs,
            joint_names=joint_names,
            window_size=window_size,
            enabled_flag=enabled_flag,
        )

    def jointvel_plot(self, legs: list = None, joint_names: list = None, window_size: int = 50, enable: bool = True, enabled_flag=None):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.jv_legs, self.jv_joint_names = self.predefined_plot(
            name='JointVel',
            y_limit=[(-15, 15)],
            legs=legs,
            joint_names=joint_names,
            window_size=window_size,
            enabled_flag=enabled_flag,
        )

    def footContact_plot(
        self,
        legs: list = None,
        joint_names: list = None,
        window_size: int = 50,
        enable: bool = True,
        enabled_flag=None,
        dual_nominal: bool = True,
    ):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        if legs is None:
            legs = self.legs
        titles = [f'FootContacts {leg} (sim)' for leg in legs]
        secondary = [f'nominal {leg}' for leg in legs] if dual_nominal else None
        rows = len(legs)
        self.create(
            figure_name='FootContacts',
            rows=rows,
            cols=1,
            window_size=window_size,
            subplot_titles=titles,
            y_limits=[(-0.2, 1.2)],
            enabled_flag=enabled_flag,
            dual_secondary=dual_nominal,
            secondary_labels=secondary,
        )
        self.contact_legs = legs

    def grf_plot(
        self,
        legs: list = None,
        window_size: int = 50,
        enable: bool = True,
        enabled_flag=None,
        y_limit: tuple[float, float] = (-150.0, 150.0),
    ):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.grf_legs, self.grf_components = self.predefined_plot(
            name='GRF',
            y_limit=[y_limit],
            legs=legs,
            joint_names=self.grf_components,
            window_size=window_size,
            enabled_flag=enabled_flag,
        )

    def lin_acc_plot(self, axis: list = None, window_size: int = 50, enable: bool = True, enabled_flag=None):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return

        if axis is None:
            axis = self.axis

        _, self.lin_acc = self.predefined_plot(
            name='LinAcc',
            y_limit=[(-5, 13)],
            legs=['trunk'],
            joint_names=axis,
            window_size=window_size,
            enabled_flag=enabled_flag,
        )

    def ang_vel_plot(self, axis: list = None, window_size: int = 50, enable: bool = True, enabled_flag=None):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return

        if axis is None:
            axis = self.axis

        _, self.ang_vel = self.predefined_plot(
            name='AngVel',
            y_limit=[(-5, 5)],
            legs=['Trunk'],
            joint_names=axis,
            window_size=window_size,
            enabled_flag=enabled_flag,
        )

    def predefine_update(self, name, data, selected_legs, selected_joints, legs_attr=False):
        """TODO: Docstring required."""
        if legs_attr:
            data = [value for arr in data.to_list() for value in arr]  # convert LegArray to list

        if name not in self.predefined_plots:
            print(f'Error: predefined plot {name} does not exist between: {self.predefined_plots}')
            return

        if len(data) == 12:
            # Data corresponds to [FL,FR,RL,RR] x [HAA,HFE,KFE]
            filtered_values = [
                data[i * 3 + j]
                for i, leg in enumerate(self.legs)
                for j, joint in enumerate(self.joint_names)
                if leg in selected_legs and joint in selected_joints
            ]
        elif len(data) == 4:
            # Data corresponds to [FL,FR,RL,RR]
            filtered_values = [data[i] for i, leg in enumerate(self.legs) if leg in selected_legs]
        elif len(data) == 3:
            # Data corresponds to [X,Y,Z]
            filtered_values = [data[i] for i, axis in enumerate(self.axis) if axis in selected_legs]

        self.plots[name].send_data(filtered_values)

    # TODO: Make all this programmatically for any observable name
    def torque_update(self, torques, LegsAttr=False):  # noqa: D102
        self.predefine_update('Torque', torques, self.torque_legs, self.torque_joint_names, legs_attr=LegsAttr)

    def jointpos_update(self, jp, LegsAttr=False):  # noqa: D102
        self.predefine_update('JointPos', jp, self.jp_legs, self.jp_joint_names, legs_attr=LegsAttr)

    def jointvel_update(self, jv, LegsAttr=False):  # noqa: D102
        self.predefine_update('JointVel', jv, self.jv_legs, self.jv_joint_names, legs_attr=LegsAttr)

    def contact_update(self, contacts, nominal=None, LegsAttr=False):  # noqa: D102
        if LegsAttr:
            contacts = [value for arr in contacts.to_list() for value in arr]
        filtered = [
            contacts[i] for i, leg in enumerate(self.legs) if leg in self.contact_legs
        ]
        if nominal is not None:
            nominal = list(np.asarray(nominal).ravel())
            filtered_nom = [
                nominal[i] for i, leg in enumerate(self.legs) if leg in self.contact_legs
            ]
            self.plots['FootContacts'].send_data(filtered, filtered_nom)
        else:
            self.plots['FootContacts'].send_data(filtered)

    def grf_update(self, grf, LegsAttr=False):  # noqa: D102
        grf_arr = np.asarray(grf, dtype=np.float64).reshape(-1)
        components = self.grf_components
        filtered = [
            grf_arr[i * 3 + j]
            for i, leg in enumerate(self.legs)
            for j, comp in enumerate(components)
            if leg in self.grf_legs and comp in self.grf_components
        ]
        self.plots['GRF'].send_data(filtered)

    def lin_acc_update(self, lin_acc):  # noqa: D102
        self.predefine_update('LinAcc', lin_acc, self.lin_acc, [], legs_attr=False)

    def ang_vel_update(self, ang_vel):  # noqa: D102
        self.predefine_update('AngVel', ang_vel, self.ang_vel, [], legs_attr=False)

    def update_plot(self):
        """Update all plots."""
        for plot in self.plots.values():
            plot.update_plot()

    def start(self):
        """Start all plots and queues."""
        for plot in self.plots.values():
            plot.start()

    def stop(self):
        """Stop all plots and queues."""
        for plot in self.plots.values():
            plot.shutdown()

    def reset(self):
        """Reset all plots and queues."""
        for plot in self.plots.values():
            plot.reset_queues()


# ===========================================================================
class SignalControlPanel(mp.Process):
    """Checkbox window to toggle which signal plots are visible (live).

    Runs in its own process. It mutates the shared ``mp.Value('b')`` flags that
    each plot window reads to withdraw/deiconify itself.
    """

    def __init__(self, flags: dict):
        super().__init__()
        # list of (name, mp.Value) so it survives the fork without pickling dicts of Values
        self._items = list(flags.items())
        self.daemon = True

    def run(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons

        def _shutdown(*_):
            plt.close('all')
            os._exit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        names = [name for name, _ in self._items]
        states = [bool(flag.value) for _, flag in self._items]

        fig, ax = plt.subplots(figsize=(3.2, 0.5 * len(names) + 1.0))
        fig.canvas.manager.set_window_title('Signal Selector')
        ax.set_title('Toggle proprioception plots')
        check = CheckButtons(ax, names, states)

        def _on_clicked(label):
            for name, flag in self._items:
                if name == label:
                    flag.value = not flag.value

        check.on_clicked(_on_clicked)
        plt.show(block=True)


# ===========================================================================
class ProprioceptivePlotter:
    """Live proprioception plotter with lazy plot processes and low-latency updates.

    Only the signal windows toggled ON in the **Signal Selector** are spawned.
    The simulator writes the latest sample into shared memory each step; each plot
    process reads that snapshot at display rate (no queue backlog).

    Usage::

        plotter = ProprioceptivePlotter(window_size=200)
        plotter.start()
        plotter.update(
            torque=tau, joint_pos=qpos_joints, joint_vel=qvel_joints,
            contacts=contacts, contact_nominal=mpc_mask,
            grf=foot_grf, ang_vel=base_ang_vel, lin_acc=base_lin_acc,
        )
        plotter.stop()
    """

    SIGNALS = ['Torque', 'JointPos', 'JointVel', 'FootContacts', 'GRF', 'AngVel', 'LinAcc']
    _BUILDERS = {
        'Torque': 'torque_plot',
        'JointPos': 'jointpos_plot',
        'JointVel': 'jointvel_plot',
        'FootContacts': 'footContact_plot',
        'GRF': 'grf_plot',
        'AngVel': 'ang_vel_plot',
        'LinAcc': 'lin_acc_plot',
    }

    def __init__(
        self,
        window_size: int = 200,
        default_enabled: list | None = None,
    ):
        self.window_size = window_size
        self.flags = {name: mp.Value('b', False) for name in self.SIGNALS}
        for name in (default_enabled if default_enabled is not None else ['FootContacts', 'AngVel']):
            if name in self.flags:
                self.flags[name].value = True

        self._helper = MujocoPlotter(enable=True)
        self._active: dict[str, MultiLivePlotter] = {}
        self.control_panel = SignalControlPanel(self.flags)

    def _build_plot(self, name: str) -> MultiLivePlotter:
        """Create a fresh ``MultiLivePlotter`` for ``name`` (not started)."""
        builder = getattr(self._helper, self._BUILDERS[name])
        if name == 'FootContacts':
            builder(window_size=self.window_size, enabled_flag=self.flags[name], dual_nominal=True)
        else:
            builder(window_size=self.window_size, enabled_flag=self.flags[name])
        plot = self._helper.plots.pop(name)
        plot.interactive = True
        return plot

    def _sync_processes(self) -> None:
        """Start/stop plot processes to match the Signal Selector flags."""
        for name in self.SIGNALS:
            if self.flags[name].value:
                plot = self._active.get(name)
                if plot is None or not plot.is_alive():
                    if plot is not None:
                        plot.shutdown()
                    plot = self._build_plot(name)
                    plot.start()
                    self._active[name] = plot
            else:
                plot = self._active.pop(name, None)
                if plot is not None:
                    plot.shutdown()

    def start(self):
        """Launch the Signal Selector; plot windows start when toggled ON."""
        self.control_panel.start()
        self._sync_processes()

    def update(
        self,
        *,
        torque=None,
        joint_pos=None,
        joint_vel=None,
        contacts=None,
        contact_nominal=None,
        grf=None,
        ang_vel=None,
        lin_acc=None,
    ):
        """Publish the latest proprioception sample (sim-time, no queue backlog)."""
        self._sync_processes()

        if torque is not None and 'Torque' in self._active:
            self._active['Torque'].send_data(list(np.asarray(torque).ravel()))
        if joint_pos is not None and 'JointPos' in self._active:
            self._active['JointPos'].send_data(list(np.asarray(joint_pos).ravel()))
        if joint_vel is not None and 'JointVel' in self._active:
            self._active['JointVel'].send_data(list(np.asarray(joint_vel).ravel()))
        if contacts is not None and 'FootContacts' in self._active:
            nom = None if contact_nominal is None else list(np.asarray(contact_nominal).ravel())
            self._active['FootContacts'].send_data(
                list(np.asarray(contacts).ravel()), secondary=nom,
            )
        if grf is not None and 'GRF' in self._active:
            self._active['GRF'].send_data(list(np.asarray(grf, dtype=np.float64).reshape(-1)))
        if ang_vel is not None and 'AngVel' in self._active:
            self._active['AngVel'].send_data(list(np.asarray(ang_vel).ravel()))
        if lin_acc is not None and 'LinAcc' in self._active:
            self._active['LinAcc'].send_data(list(np.asarray(lin_acc).ravel()))

    def stop(self):
        """Terminate all plotter and control-panel processes."""
        for plot in list(self._active.values()):
            plot.shutdown()
        self._active.clear()
        with contextlib.suppress(Exception):
            self.control_panel.terminate()


# ===========================================================================
class MultiLivePlotter(mp.Process):
    """Live plotter process with shared-memory latest-value updates."""

    # Display refresh interval [ms] — decoupled from sim rate (200 Hz).
    REFRESH_MS = 16

    def __init__(
        self,
        figure_name,
        num_subplots=2,
        window_size=50,
        subplot_titles=None,
        x_limits=None,
        y_limits=None,
        nrows=1,
        ncols=None,
        y_margin=0.1,
        plot_per_ax=1,
        enabled_flag=None,
        dual_secondary: bool = False,
        secondary_labels: list | None = None,
    ):
        super(MultiLivePlotter, self).__init__()

        if plot_per_ax > 1 and nrows == 1 and ncols == 1:
            self.num_subplots = 1
            self.nBuffers = plot_per_ax
        else:
            self.num_subplots = nrows * ncols
            self.nBuffers = self.num_subplots
        self.window_size = window_size

        self.running = mp.Event()
        self._latest = mp.Array('d', self.num_subplots, lock=False)
        self._version = mp.Value('Q', 0)
        self.dual_secondary = dual_secondary
        self.secondary_labels = secondary_labels
        if dual_secondary:
            self._latest_secondary = mp.Array('d', self.num_subplots, lock=False)
            self._secondary_version = mp.Value('Q', 0)
        else:
            self._latest_secondary = None
            self._secondary_version = None

        self.data_buffers = [deque(maxlen=self.window_size) for _ in range(self.nBuffers)]
        self.secondary_buffers = (
            [deque(maxlen=self.window_size) for _ in range(self.nBuffers)]
            if dual_secondary else None
        )

        self.nrows = nrows
        self.ncols = ncols
        self.y_margin = y_margin

        self.subplot_titles = subplot_titles
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.fig_name = figure_name

        self.enabled_flag = enabled_flag
        self._withdrawn = False
        self.interactive = False
        self._visible = None

        self.daemon = True

    def signal_handler(self, signum, frame):
        """Handles external termination signals (SIGTERM, SIGINT)."""
        print(f'[{os.getpid()}] Received signal {signum}. Shutting down gracefully...')
        self.running.clear()  # Stop the update loop
        plt.close('all')  # Close figure
        with contextlib.suppress(Exception):
            self.terminate()
        plt.close('all')  # Close figure

    def run(self):
        """This method runs in a separate process and continuously updates the plots."""
        self.running.set()  # Indicate that the plotter is running

        # Register signal handlers for clean termination
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Initialize plot in a separate process
        # plt.ion()
        self.fig, axs = plt.subplots(self.nrows, self.ncols, figsize=(10, 6))
        self.axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

        # Set figure title
        # self.fig.suptitle(self.fig_name)  # Display figure title
        self.fig.canvas.manager.set_window_title(self.fig_name)  # Set window title

        # Handle subplot titles
        if self.subplot_titles is None:
            self.subplot_titles = [f'Data {i + 1}' for i in range(self.num_subplots)]
        elif len(self.subplot_titles) < self.num_subplots:
            self.subplot_titles += [f'Data {i + 1}' for i in range(len(self.subplot_titles), self.num_subplots)]

        # Create line objects for each subplot
        self.data_buffers = [deque(maxlen=self.window_size) for _ in range(self.num_subplots)]
        if self.dual_secondary:
            self.secondary_buffers = [deque(maxlen=self.window_size) for _ in range(self.num_subplots)]
        self.lines = []
        self.secondary_lines = [] if self.dual_secondary else None
        self._last_version = 0
        self._last_secondary_version = 0

        for i in range(self.num_subplots):
            (line,) = self.axs[i].plot([], [], label=self.subplot_titles[i])
            self.lines.append(line)
            if self.dual_secondary:
                sec_label = (
                    self.secondary_labels[i]
                    if self.secondary_labels and i < len(self.secondary_labels)
                    else 'nominal'
                )
                (sec_line,) = self.axs[i].plot(
                    [], [], label=sec_label, linestyle='--', alpha=0.75,
                )
                self.secondary_lines.append(sec_line)

            self.axs[i].set_title(self.subplot_titles[i])

            # Apply user-defined x and y limits
            if self.x_limits and i < len(self.x_limits):
                self.axs[i].set_xlim(*self.x_limits[i])

            if self.y_limits and i < len(self.y_limits):
                self.axs[i].set_ylim(*self.y_limits[i])
            else:
                self.axs[i].autoscale()

            self.axs[i].legend(loc='upper left')

        # Hide unused subplots
        for i in range(self.num_subplots, len(self.axs)):
            self.axs[i].axis('off')

        # Optional in-figure controls: per-subplot show/hide + live y-limits.
        if self.interactive:
            self._setup_interactive_controls()

        # Use animation for updating the plots smoothly
        self.anim = FuncAnimation(
            self.fig,
            self._update_animation,
            interval=self.REFRESH_MS,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=True)  # Block execution here to keep the figure open

    def _setup_interactive_controls(self):
        """Embed widgets to pick subplots (joint/axis) and edit y-limits live.

        Runs inside the plotter process, so the widget callbacks can mutate the
        axes directly without any cross-process communication. A checkbox column
        selects which joint/axis subplots are shown; a range slider sets the
        shared y-limits at runtime.
        """
        n = self.num_subplots
        labels = [str(self.subplot_titles[i]) for i in range(n)]
        self._visible = [True] * n

        # Reserve room on the right (subplot selector) and bottom (y-limit slider).
        self.fig.subplots_adjust(right=0.80, bottom=0.14)

        # --- Subplot selector: choose which joint/axis subplots are shown. ---
        check_ax = self.fig.add_axes([0.815, 0.10, 0.175, 0.85])
        check_ax.set_title('show', fontsize=8)
        try:
            self._check = CheckButtons(check_ax, labels, self._visible, useblit=False)
        except TypeError:  # older matplotlib without the useblit kwarg
            self._check = CheckButtons(check_ax, labels, self._visible)
        for text in self._check.labels:
            text.set_fontsize(7)

        def _on_check(label):
            if label not in labels:
                return
            i = labels.index(label)
            self._visible[i] = not self._visible[i]
            self.axs[i].set_visible(self._visible[i])
            self.fig.canvas.draw_idle()

        self._check.on_clicked(_on_check)

        # --- Live y-limit editing via a range slider (applied to every subplot). ---
        lo, hi = self._current_ylim()
        span = (hi - lo) if hi > lo else 1.0
        slider_ax = self.fig.add_axes([0.18, 0.04, 0.5, 0.03])
        self._ylim_slider = RangeSlider(
            slider_ax, 'y-lim', lo - span, hi + span, valinit=(lo, hi),
        )

        def _on_ylim(val):
            new_lo, new_hi = float(val[0]), float(val[1])
            if new_hi <= new_lo:
                return
            for ax in self.axs[:n]:
                ax.set_ylim(new_lo, new_hi)
            self.fig.canvas.draw_idle()

        self._ylim_slider.on_changed(_on_ylim)

    def _current_ylim(self):
        """Return the initial (ymin, ymax) used to seed the y-limit text boxes."""
        if self.y_limits:
            try:
                first = self.y_limits[0]
                return float(first[0]), float(first[1])
            except (TypeError, ValueError, IndexError):
                pass
        return -1.0, 1.0

    def _apply_visibility(self):
        """Show/hide the window from its own process based on ``enabled_flag``."""
        if self.enabled_flag is None:
            return
        try:
            window = self.fig.canvas.manager.window
            if self.enabled_flag.value and self._withdrawn:
                window.deiconify()
                self._withdrawn = False
            elif not self.enabled_flag.value and not self._withdrawn:
                window.withdraw()
                self._withdrawn = True
        except Exception:
            pass

    def _update_animation(self, frame):
        """Append the latest shared-memory snapshot once per display frame."""
        self._apply_visibility()
        ver = self._version.value
        if ver != self._last_version:
            self._last_version = ver
            vals = [self._latest[i] for i in range(self.num_subplots)]
            self.update_data(vals)
        if self.dual_secondary and self._secondary_version is not None:
            sec_ver = self._secondary_version.value
            if sec_ver != self._last_secondary_version:
                self._last_secondary_version = sec_ver
                sec_vals = [self._latest_secondary[i] for i in range(self.num_subplots)]
                self.update_secondary_data(sec_vals)
        return self._update_plot()

    def update_data(self, new_values):
        """Update the data buffers with new values.

        Args:
            new_values:  A list of new data points for each subplot.
        """
        if self.num_subplots > 1:
            assert len(new_values) == self.num_subplots, f'Expected {self.num_subplots} values, got {len(new_values)}.'

        for i, val in enumerate(new_values):
            self.data_buffers[i].append(val)

    def update_secondary_data(self, new_values):
        """Append secondary trace samples (e.g. nominal MPC contact mask)."""
        if not self.dual_secondary or self.secondary_buffers is None:
            return
        for i, val in enumerate(new_values):
            self.secondary_buffers[i].append(val)

    def _update_plot(self):
        """Refresh the plots with the updated sliding window data."""
        updated_lines = []

        for i in range(self.nBuffers):
            x_data = np.arange(len(self.data_buffers[i]))
            y_data = list(self.data_buffers[i])

            self.lines[i].set_data(x_data, y_data)
            updated_lines.append(self.lines[i])

            if self.dual_secondary and self.secondary_lines is not None:
                y_sec = list(self.secondary_buffers[i])
                self.secondary_lines[i].set_data(np.arange(len(y_sec)), y_sec)
                updated_lines.append(self.secondary_lines[i])

        return updated_lines

    def send_data(self, new_values, secondary=None):
        """Write the latest sample into shared memory (non-blocking, no queue)."""
        if not isinstance(new_values, list):
            new_values = [new_values]
        n = min(len(new_values), self.num_subplots)
        for i in range(n):
            self._latest[i] = float(new_values[i])
        self._version.value += 1

        if secondary is not None and self.dual_secondary and self._latest_secondary is not None:
            if not isinstance(secondary, list):
                secondary = [secondary]
            m = min(len(secondary), self.num_subplots)
            for i in range(m):
                self._latest_secondary[i] = float(secondary[i])
            self._secondary_version.value += 1

    def shutdown(self):
        """Stop the plotting process from the parent."""
        self.running.clear()
        if self.is_alive():
            with contextlib.suppress(Exception):
                self.terminate()
            self.join(timeout=1.0)

    def stop(self):
        """Alias for :meth:`shutdown` (backward compatibility)."""
        self.shutdown()

    def reset_queues(self):
        """Reset stored trace buffers."""
        try:
            for buf in self.data_buffers:
                buf.clear()
            if self.secondary_buffers is not None:
                for buf in self.secondary_buffers:
                    buf.clear()
            self._version.value = 0
            if self._secondary_version is not None:
                self._secondary_version.value = 0
        except Exception:
            pass


# ===========================================================================
if __name__ == '__main__':
    # Example usage: 2 subplots, window size of 50
    # Titles, X-limits, and Y-limits for each subplot

    titles = ['Random Stream A']  # , 'Random Stream B']
    x_lims = [(0, 50), (0, 50)]
    y_lims = [(0, 2), (0, 1)]  # Different Y ranges for demonstration

    plotter = MujocoPlotter(enable=True)
    plotter.create(
        figure_name='example',
        subplot_titles=titles,
        rows=1,
        cols=2,
        y_limits=y_lims,
        window_size=50,
    )
    plotter.start()

    # Simulate data streaming for 200 updates
    while True:
        # Generate random data for each subplot
        new_val_subplot1 = random.uniform(0, 2)
        new_val_subplot2 = random.random()

        # Update the sliding windows
        plotter.plots['example'].send_data([new_val_subplot1, new_val_subplot2])

    plotter.stop()