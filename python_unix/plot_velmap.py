"""Plot EBIV velocity fields, either as a class or a CLI script."""

import glob
import os

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ebiv_params import EBIV_Config
from plot_utils import SetTimesFont
from velmap_utils import ComputeGradients2D, DetectOutliers, InterpolateOutliers

QUIVER_SCALE_REFERENCE = 25.0


class VelocityPlotter:
    """Render velocity maps using the original EBIV plotting implementation."""

    def __init__(self, config_file):
        self.config_file = os.path.abspath(str(config_file))
        self.cfg = EBIV_Config(self.config_file)
        trial_dir = os.path.dirname(self.config_file)
        self.cfg.dataDir = trial_dir + os.sep
        self.cfg.destDir = self.cfg.dataDir
        self.cfg.plotDir = os.path.join(self.cfg.destDir, 'plots') + os.sep

    def _velocity_files(self):
        velocity_dir = os.path.join(self.cfg.destDir, 'velocity_fields', self.cfg.fileStub)
        combined = os.path.join(velocity_dir, self.cfg.velocityDataFile() + '.npz')
        windows = sorted(glob.glob(os.path.join(velocity_dir, self.cfg.fileStub + '_t*us.npz')))
        if windows:
            return windows
        if os.path.isfile(combined):
            return [combined]
        raise IOError('No velocity files found in: ' + velocity_dir)

    def maximum_velocity(self, velocity_files=None):
        maximum = 0.0
        for velocity_file in velocity_files or self._velocity_files():
            with np.load(velocity_file) as np_file:
                data = np_file['velData'] if 'velData' in np_file.files else np_file[np_file.files[0]]
            maximum = max(maximum, float(np.nanmax(np.hypot(data[..., 2], data[..., 3]))))
        maximum *= 1000
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError('The velocity files contain no positive finite velocity magnitude')
        return maximum

    def run(self, vmax=None, arrow_scale=20, velocity_files=None,
            output_prefix=None, title_prefix='', color_scale='log'):
        if vmax is None:
            vmax = self.maximum_velocity()
        if vmax <= 0 or arrow_scale <= 0:
            raise ValueError('vmax and arrow_scale must be positive')
        SetTimesFont()
        cont_plot = 1
        plot_vorticity = 0
        show_title = 1
        do_validation = 1
        vec_scale = QUIVER_SCALE_REFERENCE / arrow_scale
        os.makedirs(self.cfg.plotDir, exist_ok=True)
        velocity_arrays = []
        bbox = None
        image_size = None
        for velocity_file in velocity_files or self._velocity_files():
            with np.load(velocity_file) as np_file:
                velocity_arrays.append(np_file['velData'] if 'velData' in np_file.files else np_file[np_file.files[0]])
                if 'bbox' in np_file.files and np_file['bbox'].size:
                    bbox = np.asarray(np_file['bbox'], dtype=np.int32)
                if 'image_size' in np_file.files:
                    image_size = np.asarray(np_file['image_size'], dtype=np.int32)
        vel_data = np.concatenate(velocity_arrays, axis=0)
        if bbox is not None and image_size is not None:
            vel_data = self._expand_bbox_data(vel_data, bbox, image_size)
        nt, ny, nx, _ = vel_data.shape
        ix = vel_data[:, :, :, 0].reshape(nt, ny, nx)
        iy = vel_data[:, :, :, 1].reshape(nt, ny, nx)
        vx = vel_data[:, :, :, 2].reshape(nt, ny, nx)
        vy = vel_data[:, :, :, 3].reshape(nt, ny, nx)
        plot_files = []
        for t_idx in np.arange(nt, dtype=int):
            fn_out = os.path.join(self.cfg.plotDir, output_prefix or self.cfg.velocityDataFile())
            vx_2d = vx[t_idx, :, :].reshape(ny, nx)
            vy_2d = vy[t_idx, :, :].reshape(ny, nx)
            if do_validation:
                mask = DetectOutliers(vx_2d, vy_2d, median=3.5, mag=0)
                vx_2d_int = InterpolateOutliers(vx_2d, mask)
                vy_2d_int = InterpolateOutliers(vy_2d, mask)
            else:
                vx_2d_int, vy_2d_int = vx_2d, vy_2d
            ix_2d = ix if ix.ndim == 2 else ix[t_idx, :, :].reshape(ny, nx)
            iy_2d = iy if iy.ndim == 2 else iy[t_idx, :, :].reshape(ny, nx)
            img = np.sqrt(vx_2d_int * vx_2d_int + vy_2d_int * vy_2d_int) * 1000
            cmap = 'jet'
            cbar_label = r'Velocity magnitude [pixel/s]'
            dx = np.abs(ix_2d[0, 1] - ix_2d[0, 0])
            dy = np.abs(iy_2d[1, 0] - iy_2d[0, 0])
            if plot_vorticity:
                img = ComputeGradients2D(vx_2d_int, vy_2d_int, 'vorticity', gridSpacingXY=[dx, dy], bSmooth=True) * 1000
                cmap, vmin, vmax = 'bwr', -50, 50
                color_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            elif color_scale == 'log':
                vmin = max(vmax * 1e-4, 1e-6)
                img = np.maximum(img, vmin)
                color_norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif color_scale == 'linear':
                vmin = 0
                color_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                raise ValueError("color_scale must be 'log' or 'linear'")
            fig, ax1 = plt.subplots(1, 1, figsize=(9, 5.) if cont_plot else (8, 5.))
            extent = [ix_2d.min(), ix_2d.max(), iy_2d.min(), iy_2d.max()]
            ax1.set_aspect(1)
            if cont_plot:
                img_pos = ax1.imshow(img, cmap=cmap, norm=color_norm, extent=extent, aspect=1)
            ax1.quiver(ix_2d, iy_2d, vx_2d_int, vy_2d_int, color='black', edgecolor='white', linewidth=0.5,
                       units='width', scale_units='width', scale=vec_scale, pivot='mid')
            pad = 15
            ax1.set_xlim(-pad, self.cfg.imgW + pad)
            ax1.set_ylim(-pad, self.cfg.imgH + pad)
            title = title_prefix + 'Algorithm: Motion compensation - '
            title += '%g ms, %dx%d-sample, resol:%gpx/s, single pass' % (self.cfg.t_sample / 1000, self.cfg.sampleW, self.cfg.sampleH, self.cfg.velResolX * 1000)
            if self.cfg.evalMethod in ['corr', 'sum_corr', 'corr_sum']:
                title = 'Algorithm: Sum-of-correlation - '
                title += '%g ms (%g ms offset) %dx%d-sample, single pass' % (self.cfg.t_sample / 1000, self.cfg.t_sep / 1000, self.cfg.sampleW, self.cfg.sampleH)
            title += ', validated' if do_validation else ', raw-data (no filter)'
            ax1.set_xlabel('X - [pixel]', fontsize=14)
            ax1.set_ylabel('Y - [pixel]', fontsize=14)
            if show_title:
                ax1.set_title(title, fontsize=8)
                time_label = 'Time %.2f ms' % ((self.cfg.t_start + t_idx * self.cfg.t_step) / 1000)
                xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
                ax1.text(xlim[0] - 0.1 * (xlim[1] - xlim[0]), ylim[1] + 0.02 * (ylim[1] - ylim[0]), time_label, fontsize=16)
            if cont_plot:
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='3%', pad=0.05)
                cbar = plt.colorbar(img_pos, cax=cax)
                cbar.set_label(cbar_label, fontsize=14)
                cbar.minorticks_on()
            plt.tight_layout()
            if cont_plot and plot_vorticity:
                fn_out += '_vort'
            output = fn_out + ('.png' if nt == 1 else '_%04d.png' % t_idx)
            plt.savefig(output, dpi=200)
            plot_files.append(output)
            print('plot stored in: ' + output)
            plt.close(fig)
        return plot_files

    def _expand_bbox_data(self, compact_data, bbox, image_size):
        img_h, img_w = [int(value) for value in image_size]
        full_ny = (img_h - self.cfg.sampleH) // self.cfg.sampleStepY + 1
        full_nx = (img_w - self.cfg.sampleW) // self.cfg.sampleStepX + 1
        full_data = np.zeros((compact_data.shape[0], full_ny, full_nx, 6), dtype=compact_data.dtype)
        x_positions = np.arange(full_nx) * self.cfg.sampleStepX + self.cfg.sampleW / 2
        y_positions = img_h - (np.arange(full_ny) * self.cfg.sampleStepY + self.cfg.sampleH / 2)
        full_x, full_y = np.meshgrid(x_positions, y_positions)
        full_data[:, :, :, 0] = full_x
        full_data[:, :, :, 1] = full_y
        x_indices = np.rint((compact_data[0, 0, :, 0] - self.cfg.sampleW / 2) / self.cfg.sampleStepX).astype(int)
        y_indices = np.rint((img_h - compact_data[0, :, 0, 1] - self.cfg.sampleH / 2) / self.cfg.sampleStepY).astype(int)
        x_indices = np.clip(x_indices, 0, full_nx - 1)
        y_indices = np.clip(y_indices, 0, full_ny - 1)
        for compact_y, full_y_index in enumerate(y_indices):
            for compact_x, full_x_index in enumerate(x_indices):
                full_data[:, full_y_index, full_x_index, 2:] = compact_data[:, compact_y, compact_x, 2:]
        return full_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot EBIV velocity fields')
    parser.add_argument('--config', default='../../ebiv_pipeline/config/ebiv_params.cfg')
    parser.add_argument('--vmax', type=float, required=True)
    parser.add_argument('--arrow-scale', type=float, default=20)
    args = parser.parse_args()
    VelocityPlotter(args.config).run(args.vmax, args.arrow_scale)


if __name__ == '__main__':
    main()
