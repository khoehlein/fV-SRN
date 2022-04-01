import os
from typing import List, Optional
import argparse
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html


class MultiRunComparison(object):

    def __init__(self, base_directory: str, loss_keys: List[str]):
        self.base_directory = base_directory
        self.loss_keys = loss_keys
        self._read_hdf5_storage()

    def _read_hdf5_storage(self):
        hdf5_directory = self._get_hdf5_directory()
        run_files = {
            os.path.splitext(f)[0]: os.path.join(hdf5_directory, f)
            for f in sorted(os.listdir(hdf5_directory)) if f.endswith('.hdf5')
        }
        data = []
        for run_name, file_path in run_files.items():
            try:
                with h5py.File(file_path) as current_file:
                    loss_summary = {}
                    for loss_key in self.loss_keys:
                        min_loss, min_loss_idx, final_loss = self._summarize_loss_data(current_file[loss_key][...])
                        loss_summary.update({
                            f'{loss_key}:min_val': min_loss,
                            f'{loss_key}:min_idx': min_loss_idx,
                            f'{loss_key}:final_val': final_loss,
                        })
                    file_data = {'run_name': run_name, **current_file.attrs, **loss_summary}
            except Exception:
                pass
            else:
                data.append(file_data)
        self.data = pd.DataFrame(data)

    def _get_hdf5_directory(self):
        path = os.path.join(self.base_directory, 'results', 'hdf5')
        if not os.path.isdir(path):
            raise RuntimeError(f'[ERROR] Expected HDF5 directory at path {path}, but path is not a valid directory.')
        return path

    def get_pcp_app(self, color_key: str, drop_constant_parameters: bool = True):
        data = self.data
        if drop_constant_parameters:
            data = data.loc[:, (data != data.iloc[0]).any()]
        app = MultiRunComparison.PCPlotApp(data, color_key)
        return app

    class PCPlotApp(object):

        def __init__(self, data: pd.DataFrame, color_key: str):
            self.data = data
            fig = self._get_pc_plot(color_key)
            self.app = dash.Dash()
            self.app.layout = html.Div(
                children=[dcc.Graph(figure=fig, style = {'width': '95vw', 'height': '75vh'})],
            )

        def run(self, debug: bool = False, port: int = os.getenv('PORT', '8050')):
            self.app.run_server(port=port, debug=debug)

        def _get_pc_plot(self, color_key: Optional[str] = None,):
            # fig = px.parallel_coordinates(self.data, color=color_key)
            dimensions = []
            for i, c in enumerate(self.data.columns):
                dimension_data = self.data[c]
                if np.issubdtype(dimension_data.dtype, np.number):
                    dimensions.append(self._handle_numeric_dimension(c, dimension_data))
                else:
                    dimensions.append(self._handle_textual_dimension(c, dimension_data))
            fig = go.Figure(
                data=go.Parcoords(
                    line={'color': self.data[color_key]},
                    dimensions=dimensions
                )
            )
            return fig

        @staticmethod
        def _handle_numeric_dimension(key: str, data: pd.Series):
            return {
                'range': [data.min(), data.max()],
                'values': data,
                'label': key,
            }

        @staticmethod
        def _handle_textual_dimension(key:str, data: pd.Series):
            unique = pd.DataFrame({key: data.unique()})
            unique['dummy'] = unique.index
            merged = pd.merge(data, unique, on=key, how='left')
            return {
                'range': [unique['dummy'].min(), unique['dummy'].max()],
                'values': merged['dummy'],
                'tickvals': unique['dummy'],
                'ticktext': unique[key],
                'label': key,
            }

    @staticmethod
    def _summarize_loss_data(losses: np.ndarray):
        min_loss_idx = np.argmin(losses)
        min_loss = losses[min_loss_idx]
        final_loss = losses[-1]
        return min_loss, min_loss_idx, final_loss


def _test_comparison():
    comparison = MultiRunComparison('/home/hoehlein/PycharmProjects/results/fvsrn/multi_member_linear_ensemble_evaluation', ['total'])
    pcp_app = comparison.get_pcp_app(color_key='total:min_val', drop_constant_parameters=True)
    pcp_app.run(debug=True)
    print('[INFO] Finished')


def run_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='base directory of the multi-run experiment')
    parser.add_argument('--loss-keys', type=str, help='loss keys to show in visualization (for multiple, separate by :)', default='total')
    parser.add_argument('--color-key', type=str, help='loss key to use for color coding', default=None)
    parser.add_argument('--drop-const', action='store_true', dest='drop_const', help='drop parameters with only a single setting')
    parser.add_argument('--show-const', action='store_false', dest='drop_const', help='show parameters with only a single setting')
    parser.add_argument('--debug', action='store_true', dest='debug')
    parser.set_defaults(drop_const=True, debug=False)

    args = vars(parser.parse_args())
    loss_keys = args['loss_keys'].split(':')
    color_key = args['color_key']
    if color_key is None:
        color_key = loss_keys[0] + ':min_val'

    comparison = MultiRunComparison(args['dir'], loss_keys)
    pcp_app = comparison.get_pcp_app(color_key=color_key, drop_constant_parameters=args['drop_const'])
    pcp_app.run(debug=args['debug'])

    print('[INFO] Finished')


if __name__ == '__main__':
    run_comparison()
