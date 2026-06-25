import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

N=200
pts = np.linspace(0, 2*np.pi, N)

# ----------------------------
# Orbit engine (pure math)
# ----------------------------
def orbit_points(a=1.0, e=0.4, inc=45.0, omega=60.0, Omega=30.0, Tp=0, n=N):

    f = np.linspace(0, 2*np.pi, n)
    adjusted_f = f + np.radians(Tp)
    adjusted_f[adjusted_f > 2*np.pi] -= 2*np.pi

    r = a * (1 - e**2) / (1 + e*np.cos(adjusted_f))

    i = np.radians(inc)
    w = np.radians(omega)
    O = np.radians(Omega)

    cwf = np.cos(w + adjusted_f)
    swf = np.sin(w + adjusted_f)

    x = r * (np.cos(O)*cwf - np.sin(O)*swf*np.cos(i))
    y = r * (np.sin(O)*cwf + np.cos(O)*swf*np.cos(i))
    z = r * swf * np.sin(i)

    return x, y, z


# ----------------------------
# Matplotlib canvas
# ----------------------------
class OrbitCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)

        self.lim = 1.5

        self.ax3d = self.fig.add_axes([0.02, 0.45, 0.50, 0.50], projection="3d")
        self.axxy = self.fig.add_axes([0.52, 0.50, 0.50, 0.45])
        self.linear = self.fig.add_axes([0.06, 0.08, 0.89, 0.35])

        ###### 3D PLOT ######

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

        self.origin = self.ax3d.scatter([0], [0], [0], color="k", s=25)

        self.x_axis = self.ax3d.quiver(0, 0, 0, self.lim, 0, 0, color="r", arrow_length_ratio=0.05)
        self.y_axis = self.ax3d.quiver(0, 0, 0, 0, self.lim, 0, color="g", arrow_length_ratio=0.05)
        self.z_axis = self.ax3d.quiver(0, 0, 0, 0, 0, self.lim, color="b", arrow_length_ratio=0.05)

        self.x_label = self.ax3d.text(self.lim * 1.08, 0, 0, "X", color="r", fontsize=12)
        self.y_label = self.ax3d.text(0, self.lim * 1.08, 0, "Y", color="g", fontsize=12)
        self.z_label = self.ax3d.text(0, 0, self.lim * 1.08, "Z", color="b", fontsize=12)

        xx, yy = np.meshgrid([-self.lim, self.lim], [-self.lim, self.lim])
        zz = np.zeros_like(xx)

        self.xy_plane = self.ax3d.plot_surface(xx, yy, zz, color="lightblue", alpha=0.2, edgecolor="none", zorder=-10)

        ###### XY PROJECTION ######

        self.axxy.set_title("XY Projection")

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
        self.start_xy = self.axxy.scatter([], [], color="gold", s=50, marker="*", zorder=2)

        self.label = self.axxy.text(0.05, 0.95, "", transform=self.axxy.transAxes, va="top", fontsize=16, color="k")

        ###### COMPONENT PLOT ######
        #self.linear.set_title("Separated Components")

        self.linear.set_autoscale_on(False)
        self.linear.set_ylim(-2, 2)
        self.linear.set_xlim(0, 2*np.pi)
        self.linear.set_xlabel("Phase")
        self.linear.set_ylabel("Amplitude")
        
        self.hline = self.linear.axhline(0, color="grey", lw=2, zorder=0)
        self.xline, = self.linear.plot([], [], color="r", lw=2, label="X")
        self.yline, = self.linear.plot([], [], color="g", lw=2, label="Y")
        self.zline, = self.linear.plot([], [], color="b", lw=2, label="Z")
        self.sumline, = self.linear.plot([], [], color="k", lw=2, ls="--", label="Sum")

        self.update_plot(0.4, 45, 60, 30, 0)
    
    def update_plot(self, e, inc, omega, Omega, Tp):
        x, y, z = orbit_points(e=e, inc=inc, omega=omega, Omega=Omega, Tp=Tp)

        A_norm = np.sqrt(1 - e**2) * np.cos(np.radians(inc)) #* np.sqrt(np.sin(np.radians(inc))**2 * np.sin(np.radians(omega))**2)

        self.line3d.set_data(x, y)
        self.line3d.set_3d_properties(z)

        self.linexy.set_data(x, y)
        self.start_xy.set_offsets([x[0], y[0]])
        self.label.set_text(f"A={A_norm:.3f}")
        
        self.xline.set_data(pts, x)
        self.yline.set_data(pts, y)
        self.zline.set_data(pts, z)
        self.sumline.set_data(pts, x + y + z)

        self.draw_idle()

# ----------------------------
# Main window
# ----------------------------
class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Orbit Simulator (Qt)")

        layout = QHBoxLayout()
        self.setLayout(layout)

        # ----------------------------
        # Left panel: sliders
        # ----------------------------
        controls = QVBoxLayout()

        self.e_slider = self.make_float_slider("e", 0.0, 0.99, 0.01, 0.4)
        self.i_slider = self.make_slider("i", 0, 180, 45)
        self.omega_slider = self.make_slider("ω", 0, 360, 60)
        self.Omega_slider = self.make_slider("Ω", 0, 360, 30)
        self.Tp_slider = self.make_slider(r"Time of Periastron", 0, 360, 30)

        controls.addWidget(self.e_slider["container"])
        controls.addWidget(self.i_slider["container"])
        controls.addWidget(self.omega_slider["container"])
        controls.addWidget(self.Omega_slider["container"])
        controls.addWidget(self.Tp_slider["container"])

        # ----------------------------
        # Middle/right: plot canvas
        # ----------------------------
        self.canvas = OrbitCanvas()

        layout.addLayout(controls, stretch=1)
        layout.addWidget(self.canvas, stretch=4)

        # connect updates
        self.e_slider["slider"].valueChanged.connect(self.update)
        self.i_slider["slider"].valueChanged.connect(self.update)
        self.omega_slider["slider"].valueChanged.connect(self.update)
        self.Omega_slider["slider"].valueChanged.connect(self.update)
        self.Tp_slider["slider"].valueChanged.connect(self.update)
        
    def update(self):

        e = self.e_slider["slider"].value() * self.e_slider["step"]
        inc = self.i_slider["slider"].value()
        omega = self.omega_slider["slider"].value()
        Omega = self.Omega_slider["slider"].value()
        Tp = self.Tp_slider["slider"].value()

        self.canvas.update_plot(e, inc, omega, Omega, Tp)
    
    ### SLIDERS ###
    def make_slider(self, name, minv, maxv, val):

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(minv)
        slider.setMaximum(maxv)
        slider.setValue(val)

        label = QLabel(f"{name}: {val}")

        def update_label(v):
            label.setText(f"{name}: {v}")

        slider.valueChanged.connect(update_label)

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(label)
        layout.addWidget(slider)

        return {"slider": slider, "container": container}

    def make_float_slider(self, name, minv, maxv, step, initial):

        slider = QSlider(Qt.Orientation.Horizontal)

        slider.setMinimum(int(minv / step))
        slider.setMaximum(int(maxv / step))
        slider.setValue(int(initial / step))

        label = QLabel(f"{name}: {initial:.2f}")

        def update_label(v):
            val = v * step
            label.setText(f"{name}: {val:.2f}")

        slider.valueChanged.connect(update_label)

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(label)
        layout.addWidget(slider)

        return {
            "slider": slider,
            "container": container,
            "step": step
        }
    
# ----------------------------
# run app
# ----------------------------
if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())