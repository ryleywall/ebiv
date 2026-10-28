"""Compute EBIV velocity fields, either as a class or a CLI script."""

import os
import sys
from time import time

import numpy as np

from ebiv_params import EBIV_Config
from ebiv_utils import ComputeMotionField, ComputeMotionFieldGPU, EBIDataSet, _sample_bounds


class EBIVProcessor:
    """Run the existing EBIV velocity-field implementation for one config."""

    def __init__(self, config_file, event_file=None, output_dir=None, bbox=None):
        self.config_file = os.path.abspath(str(config_file))
        self.cfg = EBIV_Config(self.config_file)
        self.output_dir = os.path.abspath(
            str(output_dir) if output_dir is not None else os.path.dirname(self.config_file)
        )
        self.cfg.dataDir = self.output_dir + os.sep
        self.cfg.destDir = self.output_dir + os.sep
        self.cfg.plotDir = os.path.join(self.cfg.destDir, 'plots') + os.sep
        self.event_file = os.path.abspath(
            str(event_file) if event_file is not None else self.cfg.rawFile(ext='.evt')
        )
        self.duration_seconds = 0.0
        self.bbox = self._normalize_bbox(bbox)

    def _normalize_bbox(self, bbox):
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = [int(round(value)) for value in bbox]
        x_min = max(0, min(x_min, self.cfg.imgW - self.cfg.sampleW))
        y_min = max(0, min(y_min, self.cfg.imgH - self.cfg.sampleH))
        x_max = max(x_min + self.cfg.sampleW, min(x_max, self.cfg.imgW))
        y_max = max(y_min + self.cfg.sampleH, min(y_max, self.cfg.imgH))
        return np.asarray([x_min, y_min, x_max, y_max], dtype=np.int32)

    def run(self):
        tim0 = time()
        if not os.path.isfile(self.event_file):
            raise IOError('File not found: ' + self.event_file)
        vx_in = np.arange(self.cfg.VxMin, self.cfg.VxMax + self.cfg.velResolX, self.cfg.velResolX)
        vy_in = np.arange(self.cfg.VyMin, self.cfg.VyMax + self.cfg.velResolY, self.cfg.velResolY)
        use_cuda, cuda_status = self._cuda_status()
        print('CUDA acceleration: %s (%s)' % ('enabled' if use_cuda else 'disabled', cuda_status))
        print('Loading events from: ' + self.event_file)
        ev = EBIDataSet(self.event_file)
        img_h, img_w = ev.imageSize()
        sample_size = [self.cfg.sampleW, self.cfg.sampleH]
        input_duration = ev.evData['t'].max() - ev.evData['t'].min()
        print('Window length: %.3f ms' % (self.cfg.t_sample / 1000))
        print('Input file duration: %.3f ms' % (input_duration / 1000))
        velocity_dir = os.path.join(self.cfg.destDir, 'velocity_fields', self.cfg.fileStub)
        os.makedirs(velocity_dir, exist_ok=True)
        time_starts = list(range(self.cfg.t_start, self.cfg.t_end, self.cfg.t_step))
        if self.cfg.doSingleTimeStep:
            time_starts = time_starts[:1]
        x_start, x_stop, y_start, y_stop = _sample_bounds(
            img_w, img_h, self.cfg.sampleW, self.cfg.sampleH,
            self.cfg.sampleStepX, self.cfg.sampleStepY, self.bbox)
        nx_total = len(range(x_start, x_stop, self.cfg.sampleStepX))
        ny_total = len(range(y_start, y_stop, self.cfg.sampleStepY))
        total_positions = len(time_starts) * ny_total * nx_total
        total_windows = len(time_starts)

        def update_progress(window_index, window_start, started_at):
            if total_windows == 0:
                return
            elapsed = max(time() - started_at, 1e-9)
            fraction = window_index / total_windows
            rate = window_index / elapsed
            remaining = (total_windows - window_index) / rate if rate else 0
            width = 32
            filled = int(width * fraction)
            bar = '#' * filled + '-' * (width - filled)
            message = ('Windows [%s] %5.1f%% (%d/%d), t=%d us, '
                       '%.2f windows/s, ETA %s' %
                       (bar, fraction * 100, window_index, total_windows,
                        window_start, rate, _format_duration(remaining)))
            sys.stdout.write('\r' + message + ' ' * 8)
            sys.stdout.flush()

        def _format_duration(seconds):
            seconds = max(0, int(round(seconds)))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours:
                return '%dh %02dm' % (hours, minutes)
            if minutes:
                return '%dm %02ds' % (minutes, seconds)
            return '%ds' % seconds

        def update_sample_progress(completed, total):
            if total == 0:
                return
            fraction = completed / total
            if completed != total and completed % max(1, total // 10) != 0:
                return
            sys.stdout.write('\r  Current window: %5.1f%% (%d/%d samples)' %
                             (fraction * 100, completed, total))
            sys.stdout.flush()

        compute_motion_field = ComputeMotionFieldGPU if use_cuda else ComputeMotionField
        vel_data = []
        count = 0
        tim1 = time()
        ny = nx = 0
        print('Processing %d windows (%d positions/window, %s total positions)' %
              (total_windows, nx_total * ny_total, total_positions), flush=True)
        for window_index, t_0 in enumerate(time_starts, start=1):
            window_data = compute_motion_field(
                event_set=ev, frame_start=t_0, frame_duration=self.cfg.t_sample,
                sample_size=sample_size,
                sample_step=[self.cfg.sampleStepX, self.cfg.sampleStepY],
                VxIN=vx_in, VyIN=vy_in, polarity=self.cfg.polarity,
                objectiveFxn=self.cfg.objectiveFxn, interpolation=self.cfg.interpolation,
                doExponentialFit=self.cfg.doGaussPeakFit, sample_bounds=self.bbox,
                progress_callback=update_sample_progress)
            vel_data.append(window_data)
            ny, nx, _ = window_data.shape
            count += ny * nx
            window_file = os.path.join(velocity_dir, '%s_t%010dus.npz' % (self.cfg.fileStub, t_0))
            np.savez(window_file, velData=window_data.reshape(1, ny, nx, 6),
                     bbox=self.bbox if self.bbox is not None else np.asarray([], dtype=np.int32),
                     image_size=np.asarray([img_h, img_w], dtype=np.int32))
            update_progress(window_index, t_0, tim1)
            print('\nSaved velocity window: ' + window_file, flush=True)
        if total_positions > 0:
            sys.stdout.write('\n')
        if count == 0 or not vel_data:
            raise RuntimeError('No data processed')
        tim2 = time()
        self.duration_seconds = max(float(input_duration) / 1_000_000, 1e-9)
        print('Loading event data took %.1f s' % (tim1 - tim0))
        print('Processing %d positions took %.1f min' % (count, (tim2 - tim1) / 60))
        vel_data_out = np.asarray(vel_data).reshape(len(vel_data), ny, nx, 6)
        combined_file = os.path.join(velocity_dir, self.cfg.velocityDataFile())
        np.savez(combined_file, velData=vel_data_out,
             bbox=self.bbox if self.bbox is not None else np.asarray([], dtype=np.int32),
             image_size=np.asarray([img_h, img_w], dtype=np.int32))
        print('Combined velocity data stored in: ' + combined_file)
        return combined_file + '.npz'

    @staticmethod
    def _cuda_status():
        use_cuda = False
        cuda_status = 'CuPy is not installed'
        try:
            import cupy as cp
            cuda_devices = cp.cuda.runtime.getDeviceCount()
            if cuda_devices > 0:
                use_cuda = True
                cuda_status = '%d CUDA device(s) detected' % cuda_devices
            else:
                cuda_status = 'CuPy is installed, but no CUDA device was detected'
        except ImportError:
            pass
        except Exception as exc:
            cuda_status = 'CUDA check failed: %s' % exc
        return use_cuda, cuda_status


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute EBIV velocity fields')
    parser.add_argument('--config', required=True, help='EBIV configuration file')
    parser.add_argument('--events', help='EVT event file; defaults to config dataDir/fileStub.evt')
    parser.add_argument('--output-dir', help='Output directory; defaults to the config directory')
    args = parser.parse_args()
    EBIVProcessor(
        config_file=args.config,
        event_file=args.events,
        output_dir=args.output_dir,
    ).run()


if __name__ == '__main__':
    main()
