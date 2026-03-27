"""fMRI-oriented transforms for feature spaces."""

from .compat import (
    event_series_to_fmrimod_event_variable,
    has_fmrimod,
    map_hrf_name,
    render_events_with_fmrimod,
    to_sampling_frame,
)
from .design import add_lags, concat_feature_series
from .hrf import hrf_convolve, hrf_kernel
from .query import (
    ExperimentGrid,
    RunGrid,
    build_experiment_grid,
    query_feature_window,
    query_feature_window_tr,
    query_feature_zoo_window_tr,
)
from .render import render_events
from .resample import build_tr_grid, resample_feature_series

__all__ = [
    "add_lags",
    "build_tr_grid",
    "concat_feature_series",
    "ExperimentGrid",
    "event_series_to_fmrimod_event_variable",
    "has_fmrimod",
    "hrf_convolve",
    "hrf_kernel",
    "map_hrf_name",
    "query_feature_window",
    "query_feature_window_tr",
    "query_feature_zoo_window_tr",
    "render_events",
    "render_events_with_fmrimod",
    "resample_feature_series",
    "RunGrid",
    "to_sampling_frame",
    "build_experiment_grid",
]
