import sys
import numpy as np

from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QLabel, QLineEdit
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, FuncFormatter

N = 100
pts = np.linspace(0, 2 * np.pi, N, endpoint=False)
P = 1.0  # Orbital period; use days or years if desired

# ----------------------------
# Orbit engine
# ----------------------------
def orbit_points(f, a=1.0, e=0.4, inc=45.0, omega=60.0, Omega=30.0, Tp=0, n=N):
    if f is None:
        f = np.linspace(0, 2 * np.pi, n, endpoint=False)
    f = (np.asarray(f, dtype=float) + np.radians(Tp)) % (2 * np.pi)
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    i = np.radians(inc)
    w = np.radians(omega)
    O = np.radians(Omega)
    u = w + f
    x = r * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(i))
    y = r * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(i))
    z = r * np.sin(u) * np.sin(i)
    return x, y, z

# ----------------------------
# Kepler's second law
# ----------------------------
def true_to_mean_anomaly(f, e):
    E = np.arctan2(np.sqrt(1 - e**2) * np.sin(f), e + np.cos(f)) % (2 * np.pi)
    return E - e * np.sin(E)

def true_anomaly_to_time(f, e, P=1.0):
    return P * true_to_mean_anomaly(f, e) / (2 * np.pi)

def elapsed_time_from_start(f, e, Tp, P=1.0):
    M_start = true_to_mean_anomaly(np.radians(Tp) % (2 * np.pi), e)
    M_current = true_to_mean_anomaly((np.asarray(f) + np.radians(Tp)) % (2 * np.pi), e)
    delta_M = (M_current - M_start) % (2 * np.pi)
    return P * delta_M / (2 * np.pi)

def end_phase(Tp, baseline, e):
    M_start = true_to_mean_anomaly(np.radians(Tp), e)
    M_end = M_start + 2 * np.pi * baseline
    E_end = 2 * np.pi * (M_end / (2 * np.pi))  # Initial value for clarity
    E_end = np.array([M_end % (2 * np.pi)], dtype=float)[0]
    for _ in range(30):
        E_end -= (E_end - e * np.sin(E_end) - (M_end % (2 * np.pi))) / (1 - e * np.cos(E_end))
    f_end = np.arctan2(np.sqrt(1 - e**2) * np.sin(E_end), np.cos(E_end) - e)
    return np.degrees(f_end) % 360

# ----------------------------
# Matplotlib canvas
# ----------------------------
class OrbitCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)

        self.lim = 2.0

        self.ax3d = self.fig.add_axes([0.02, 0.45, 0.50, 0.50], projection="3d")
        self.axxy = self.fig.add_axes([0.52, 0.50, 0.50, 0.45])
        self.linear = self.fig.add_axes([0.06, 0.08, 0.89, 0.35])

        # 3D orbit
        self.ax3d.set_title("3D Orbit")
        self.ax3d.set_xlim(-self.lim, self.lim)
        self.ax3d.set_ylim(-self.lim, self.lim)
        self.ax3d.set_zlim(-self.lim, self.lim)
        self.ax3d.set_autoscale_on(False)
        self.ax3d.set_box_aspect((1, 1, 1), zoom=1.35)
        self.ax3d.set_anchor("C")
        self.ax3d.margins(0)
        self.ax3d.view_init(elev=20, azim=45)
        self.ax3d.xaxis.pane.set_alpha(0)
        self.ax3d.yaxis.pane.set_alpha(0)
        self.ax3d.zaxis.pane.set_alpha(0)
        self.ax3d.set_axis_off()

        self.line3d, = self.ax3d.plot([], [], [], color="k", lw=2)
        self.subset_line3d, = self.ax3d.plot([], [], [], color="goldenrod", lw=2)
        self.origin = self.ax3d.scatter([0], [0], [0], color="k", s=25)
        self.pericentre3d = self.ax3d.scatter([], [], [], color="k", s=12)
        self.apocentre3d = self.ax3d.scatter([], [], [], color="k", s=12)

        self.x_axis = self.ax3d.quiver(0, 0, 0, self.lim, 0, 0, color="r", arrow_length_ratio=0.05)
        self.y_axis = self.ax3d.quiver(0, 0, 0, 0, self.lim, 0, color="g", arrow_length_ratio=0.05)
        self.z_axis = self.ax3d.quiver(0, 0, 0, 0, 0, self.lim, color="b", arrow_length_ratio=0.05)

        self.x_label = self.ax3d.text(self.lim * 1.08, 0, 0, "X", color="r", fontsize=12)
        self.y_label = self.ax3d.text(0, self.lim * 1.08, 0, "Y", color="g", fontsize=12)
        self.z_label = self.ax3d.text(0, 0, self.lim * 1.08, "Z", color="b", fontsize=12)

        xx, yy = np.meshgrid([-self.lim, self.lim], [-self.lim, self.lim])
        zz = np.zeros_like(xx)
        self.xy_plane = self.ax3d.plot_surface(xx, yy, zz, color="lightblue", alpha=0.2, edgecolor="none", zorder=-10)

        # XY projection
        self.axxy.set_title('XY ("on-sky") Projection')
        self.axxy.set_xlim(-self.lim, self.lim)
        self.axxy.set_ylim(-self.lim, self.lim)
        self.axxy.set_aspect("equal")
        self.axxy.set_autoscale_on(False)
        self.axxy.grid(False)
        self.axxy.axhline(0, color="0.8", lw=1, zorder=0)
        self.axxy.axvline(0, color="0.8", lw=1, zorder=0)
        self.axxy.set_xlabel("X")
        self.axxy.set_ylabel("Y")

        self.origin_xy = self.axxy.scatter([0], [0], color="k", s=25, zorder=0)
        self.linexy, = self.axxy.plot([], [], color="k", lw=2, zorder=1)
        self.subset_linexy, = self.axxy.plot([], [], color="goldenrod", lw=2, zorder=1)
        self.start_xy = self.axxy.scatter([], [], color="goldenrod", s=50, marker="*", zorder=1)
        self.pericentre = self.axxy.scatter([], [], color="k", s=12, zorder=1)
        self.apocentre = self.axxy.scatter([], [], color="k", s=12, zorder=1)
        self.label = self.axxy.text(0.05, 0.95, "", transform=self.axxy.transAxes, va="top", fontsize=16, color="k")

        # Bottom plot: components versus elapsed time
        self.linear.set_autoscale_on(False)
        self.linear.set_ylim(-2, 2)
        self.linear.set_xlim(0, P)
        self.linear.set_xlabel("Fractional period")
        self.linear.set_ylabel("Amplitude")
        self.linear.xaxis.set_major_locator(MultipleLocator(P / 8))
        self.linear.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t / P:.3g}"))

        self.hline = self.linear.axhline(0, color="grey", lw=2, zorder=0)
        self.zline, = self.linear.plot([], [], color="b", lw=2, ls="--", alpha=0.7, zorder=1)
        self.xline, = self.linear.plot([], [], color="r", lw=2, zorder=1)
        self.yline, = self.linear.plot([], [], color="g", lw=2, zorder=1)
        self.sumline, = self.linear.plot([], [], color="k", lw=2, ls="--", label="Projected separation", zorder=1)

        self.linear.legend(loc="upper left", fontsize=10, frameon=False)
        self.vspan = self.linear.axvspan(0, P, color="lightgrey", alpha=0.5, zorder=2)

        self.update_plot(0.4, 45, 60, 30, 0, 0.5)

    def update_plot(self, e, inc, omega, Omega, Tp, baseline):
        # Relative true anomaly: starts at the selected Tp position
        f = pts.copy()
        x, y, z = orbit_points(f, e=e, inc=inc, omega=omega, Omega=Omega, Tp=Tp)
        r = np.sqrt(x**2 + y**2 + z**2)
        r_sky = np.sqrt(x**2 + y**2)

        # Time since the selected starting position, in chronological order
        time = elapsed_time_from_start(f, e, Tp, P=P)

        # Locate the endpoint of the selected baseline
        end_idx = int(np.searchsorted(time, baseline * P, side="right") - 1)
        end_idx = max(0, min(end_idx, N - 1))

        if baseline >= 1:
            end_idx = N - 1

        # Pericentre and apocentre locations
        x_peri, y_peri, z_peri = orbit_points(0, e=e, inc=inc, omega=omega, Omega=Omega)
        x_apo, y_apo, z_apo = orbit_points(np.pi, e=e, inc=inc, omega=omega, Omega=Omega)

        # Approximate normalised integral of projected separation over true anomaly
        a_value = np.trapz(r_sky[:end_idx + 1], f[:end_idx + 1]) / (2 * np.pi)

        # Update 3D orbit
        self.pericentre3d._offsets3d = ([x_peri], [y_peri], [z_peri])
        self.apocentre3d._offsets3d = ([x_apo], [y_apo], [z_apo])
        self.line3d.set_data(x, y)
        self.line3d.set_3d_properties(z)
        self.subset_line3d.set_data(x[:end_idx + 1], y[:end_idx + 1])
        self.subset_line3d.set_3d_properties(z[:end_idx + 1])

        # Update XY projection
        self.linexy.set_data(x, y)
        self.subset_linexy.set_data(x[:end_idx + 1], y[:end_idx + 1])
        self.pericentre.set_offsets([[x_peri, y_peri]])
        self.apocentre.set_offsets([[x_apo, y_apo]])
        self.start_xy.set_offsets([[x[end_idx], y[end_idx]]])
        self.label.set_text(f"A={a_value:.3f}")

        # Update component curves against time
        self.xline.set_data(time, x)
        self.yline.set_data(time, y)
        self.zline.set_data(time, z)
        self.sumline.set_data(time, r_sky)

        # Shade the remaining portion of the orbital period
        self.vspan.set_bounds(baseline * P, 0, (1 - baseline) * P, 1)

        self.draw_idle()

# ----------------------------
# Main window
# ----------------------------
class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASFI Orbit Simulator")

        layout = QHBoxLayout()
        controls = QVBoxLayout()
        self.setLayout(layout)

        self.e_slider = self.make_slider("e", 0, 0.99, 0.01, 0.4)
        self.i_slider = self.make_slider("i", 0, 180, 1, 45)
        self.omega_slider = self.make_slider("ω", 0, 360, 1, 60)
        self.Omega_slider = self.make_slider("Ω", 0, 360, 1, 30)
        self.Tp_slider = self.make_slider("Tp", -0.5, 0.5, 0.001, 0)
        self.baseline_slider = self.make_slider("Baseline", 0, 1, 0.01, 0.5)

        self.sliders = [self.e_slider, self.i_slider, self.omega_slider, self.Omega_slider, self.Tp_slider, self.baseline_slider]

        for slider in self.sliders:
            controls.addWidget(slider["container"])
            slider["slider"].valueChanged.connect(self.update)

        self.canvas = OrbitCanvas()
        layout.addLayout(controls, stretch=1)
        layout.addWidget(self.canvas, stretch=4)

    def update(self):
        self.canvas.update_plot(e=self.slider_value(self.e_slider), inc=self.slider_value(self.i_slider), omega=self.slider_value(self.omega_slider), Omega=self.slider_value(self.Omega_slider), Tp=360 * self.slider_value(self.Tp_slider), baseline=self.slider_value(self.baseline_slider))

    def make_slider(self, name, minv, maxv, step, initial):
        slider = QSlider(Qt.Orientation.Horizontal)
        scale = round(1 / step)
        slider.setMinimum(round(minv * scale))
        slider.setMaximum(round(maxv * scale))
        slider.setValue(round(initial * scale))

        label = QLabel(f"{name}:")
        entry = QLineEdit(f"{initial:g}")
        entry.setFixedWidth(65)
        entry.setAlignment(Qt.AlignmentFlag.AlignRight)

        def update_label(v):
            value = v / scale
            label.setText(f"{name}: {value:g}")
            if not entry.hasFocus():
                entry.setText(f"{value:g}")

        def update_slider():
            try:
                value = float(entry.text())
            except ValueError:
                entry.setText(f"{slider.value() / scale:g}")
                return
            value = max(minv, min(maxv, value))
            slider.setValue(round(value * scale))
            entry.setText(f"{slider.value() / scale:g}")

        slider.valueChanged.connect(update_label)
        entry.editingFinished.connect(update_slider)

        container = QWidget()
        layout = QVBoxLayout(container)
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(entry)
        layout.addLayout(row)
        layout.addWidget(slider)

        return {"slider": slider, "entry": entry, "container": container, "step": step, "scale": scale}

    def slider_value(self, control):
        return control["slider"].value() / control["scale"]

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())