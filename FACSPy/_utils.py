from functools import wraps
from anndata import AnnData
from typing import Optional, Union, Callable
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
import inspect

from .exceptions._exceptions import (ChannelSubsetError,
                                     GateNotFoundError,
                                     GateAmbiguityError,
                                     PopulationAsGateError,
                                     ExhaustedGatePathError,
                                     GateNameError)
from .exceptions._utils import GateNotProvidedError, ExhaustedHierarchyError

reduction_names = {
    reduction: [f"{reduction}{i}" for i in range(1, 50)]
    for reduction in ["PCA", "MDS", "UMAP", "TSNE"]
}

GATE_SEPARATOR = "/"

IMPLEMENTED_SAMPLEWISE_DIMREDS = ["MDS", "PCA", "UMAP", "TSNE"]
IMPLEMENTED_SCALERS = ["MinMaxScaler", "RobustScaler", "StandardScaler"]

cytof_technical_channels = ["event_length", "Event_length",
                            "width", "Width",
                            "height", "Height",
                            "center", "Center",
                            "residual", "Residual",
                            "offset", "Offset",
                            "amplitude", "Amplitude",
                            "dna1", "DNA1",
                            "dna2", "DNA2"]

scatter_channels = ["FSC", "SSC", "fsc", "ssc"]
time_channels = ["time", "Time"]
spectral_flow_technical_channels = ["AF"]


def _replace_in_args(args: tuple,
                     value: str,
                     replacement: str) -> tuple:
    arglist = list(args)
    arg_idx = [i for i, idx in enumerate(arglist)
               if isinstance(idx, str) and idx == value][0]
    arglist[arg_idx] = replacement
    return tuple(arglist)


def _enable_gate_aliases(func: Callable):
    """\
    Decorator function in order to enable gate aliasing.
    The passed gate is checked against a dictionary stored in
    FACSPy.settings.gate_aliases. If there is an entry present,
    the alias will be passed down.

    This function is only meant for internal use. There are no
    checks for the presence of the `gate` argument in the function
    signature.

    Note that if you want to combine it with @_default_gate or
    @_default_gate_and_default_layer, the @_enable_gate_alias has
    to be second.

    Parameters
    ----------
    func
        The function to be decorated.

    Returns
    -------
    A function where the `gate` argument has been set using aliases
    obtained from FACSPy.settings.

    Examples
    --------
    >>> @_enable_gate_aliases
    ... def my_custom_facspy_function(
    ...     adata: AnnData,
    ...     gate: Optional[str] = None,
    ...     *args,
    ...     **kwargs
    ... )
    >>> @_default_gate
    ... @_enable_gate_aliases
    ... def my_custom_facspy_function(
    ...     adata: AnnData,
    ...     gate: Optional[str] = None,
    ...     *args,
    ...     **kwargs
    ... )
    >>> @_default_gate_and_default_layer
    ... @_enable_gate_aliases
    ... def my_custom_facspy_function(
    ...     adata: AnnData,
    ...     gate: Optional[str] = None,
    ...     *args,
    ...     **kwargs
    ... )

    """
    argspec = inspect.getfullargspec(func)
    sig = inspect.signature(func)

    @wraps(func)
    def __allow_gate_aliases(*args, **kwargs):
        # we first build a dictionary with the passed positional arguments
        function_parameters = argspec[0]
        if not function_parameters:
            function_parameters = list(sig.parameters.keys())

        named_pos_args = {arg: val
                          for arg, val
                          in zip(function_parameters, args)}

        if "gate" in named_pos_args and "gate" in kwargs:
            raise ValueError(
                "This shouldnt happen. Please provide a bugreport."
            )

        from ._settings import settings

        _gate = None
        # we now check if `gate` has been passed as a positional argument
        if "gate" in named_pos_args:
            _gate = named_pos_args["gate"]
            arglist = list(args)
            # sidenote: we can't use .index since AnnData does not support
            # equality
            arg_idx = [i for i, idx in enumerate(arglist)
                       if isinstance(idx, str) and idx == _gate][0]

            # if so, we check if there is a gate alias stored in fp.settings...
            if _gate in settings.gate_aliases:
                # ... and update args
                arglist[arg_idx] = settings.gate_aliases[_gate]
                args = tuple(arglist)

        # we now check if `gate` has been passed as a keyword argument
        elif "gate" in kwargs:
            _gate = kwargs["gate"]
            # if so, we check if there is a gate alias stored in fp.settings...
            if _gate in settings.gate_aliases:
                # ... and update the kwargs
                kwargs["gate"] = settings.gate_aliases[_gate]

        return func(*args, **kwargs)
    return __allow_gate_aliases


def _default_layer(func: Callable):
    """\
    Decorator function in order to pass a default value for the
    `layer` argument. This function checks if `layer` was either
    passed as a positional or keyword argument. If it wasn't,
    the default_gate value of fp.settings is passed as a keyword
    argument.

    This function is only meant for internal use. There are no
    checks for the presence of the `layer` argument in the function
    signature.

    Parameters
    ----------
    func
        The function to be decorated.

    Returns
    -------
    A function where the `layer` argument has been set.

    Examples
    --------
    >>> @_default_layer
    ... def my_custom_facspy_function(
    ...     adata: AnnData,
    ...     layer: Optional[str] = None,
    ...     *args,
    ...     **kwargs
    ... )

    """

    sig = inspect.signature(func)

    @wraps(func)
    def __add_default_layer(*args, **kwargs):
        # we first build a dictionary with the passed positional arguments
        function_parameters = list(sig.parameters.keys())
        named_pos_args = {arg: val
                          for arg, val
                          in zip(function_parameters, args)}

        # we now check if `layer` has been passed as a positional or keyword
        # argument. if it has, we return the function as the user is allowed
        # to overwrite it
        if "layer" in named_pos_args or "layer" in kwargs:
            return func(*args, **kwargs)

        # alternatively, we set the `layer` kwarg as specified in the defaults.
        from ._settings import settings
        kwargs["layer"] = settings.default_layer
        return func(*args, **kwargs)
    return __add_default_layer


def _default_gate(func: Callable):
    """\
    Decorator function in order to pass a default value for the
    `gate` argument. This function checks if `gate` was either
    passed as a positional or keyword argument. If it wasn't,
    the default_gate value of fp.settings is passed as a keyword
    argument.

    This function is only meant for internal use. There are no
    checks for the presence of the `gate` argument in the function
    signature.

    Parameters
    ----------
    func
        The function to be decorated.

    Returns
    -------
    A function where the `gate` argument has been set.

    Examples
    --------
    >>> @_default_gate
    ... def my_custom_facspy_function(
    ...     adata: AnnData,
    ...     gate: Optional[str] = None,
    ...     *args,
    ...     **kwargs
    ... )

    """

    sig = inspect.signature(func)

    @wraps(func)
    def __add_default_gate(*args, **kwargs):
        # we first build a dictionary with the passed positional arguments
        function_parameters = list(sig.parameters.keys())
        named_pos_args = {arg: val
                          for arg, val
                          in zip(function_parameters, args)}

        # we now check if `gate` has been passed as a positional or keyword
        # argument. if it has, we return the function as the user is allowed
        # to overwrite it
        if "gate" in named_pos_args or "gate" in kwargs:
            return func(*args, **kwargs)

        # alternatively, we set the `gate` kwarg as specified in the defaults.
        from ._settings import settings
        kwargs["gate"] = settings.default_gate
        return func(*args, **kwargs)
    return __add_default_gate


def _default_gate_and_default_layer(func: Callable):
    """\
    Decorator function in order to pass a default value for the
    `gate` and `layer` argument. This function checks if `layer` and `gate`
    were either passed as a positional or keyword argument. If they weren't,
    the `default_gate` and `default_layer` value of fp.settings are passed as
    keyword arguments.

    This function is only meant for internal use. There are no
    checks for the presence of the `layer` or `gate` argument in the function
    signature.

    Devnote: The complicated syntax is due to the fact that chained
    decorators did not really work. This can be fixed in the future.
    The problem is, that a function passed through one decorator loses the
    argspec obtained by inspect.getfullargspec.

    Parameters
    ----------
    func
        The function to be decorated

    Returns
    -------
    A function where the `layer` and `gate` argument have been set.

    Examples
    --------
    >>> @_default_gate_and_default_layer
    ... def my_custom_facspy_function(
    ...     adata: AnnData,
    ...     gate: Optional[str] = None,
    ...     layer: Optional[str] = None,
    ...     *args,
    ...     **kwargs
    ... )

    """

    sig = inspect.signature(func)

    @wraps(func)
    def __add_default_gate_and_default_layer(*args, **kwargs):
        # we first build a dictionary with the passed positional arguments

        function_parameters = list(sig.parameters.keys())
        named_pos_args = {arg: val
                          for arg, val
                          in zip(function_parameters, args)}

        # we now check if `layer` has been passed as a positional or keyword
        # argument. if it has, we return the function as the user is allowed
        # to overwrite it
        user_set_layer = "layer" in named_pos_args or "layer" in kwargs
        user_set_gate = "gate" in named_pos_args or "gate" in kwargs
        if user_set_layer and user_set_gate:
            return func(*args, **kwargs)

        # alternatively, we set the `layer` and `gate` kwarg as specified in the defaults.
        from ._settings import settings
        if not user_set_layer:
            if "layer" in named_pos_args:
                args = _replace_in_args(args, "layer", settings.default_layer)
            else:
                kwargs["layer"] = settings.default_layer
        if not user_set_gate:
            if "gate" in named_pos_args:
                args = _replace_in_args(args, "gate", settings.default_gate)
            else:
                kwargs["gate"] = settings.default_gate
        return func(*args, **kwargs)

    return __add_default_gate_and_default_layer


def _check_gate_name(gate: str) -> None:
    if gate.startswith(GATE_SEPARATOR) or gate.endswith(GATE_SEPARATOR):
        raise GateNameError(GATE_SEPARATOR)
    if not gate:
        raise GateNotProvidedError(gate)


def _check_gate_path(gate_path: str) -> None:
    _check_gate_name(gate_path)
    if GATE_SEPARATOR not in gate_path:
        raise PopulationAsGateError(gate_path)


def _is_parent(adata: AnnData,
               gate: str,
               parent: str) -> bool:
    """Substring analysis to see if these are actually children"""
    parent_gate = _find_gate_path_of_gate(adata, parent)
    child_gate = _find_gate_path_of_gate(adata, gate)
    return parent_gate in child_gate and parent_gate != child_gate


def _find_current_population(gate: str) -> str:
    """Finds current population of a specified gating path

    Parameters
    ----------

    gate: str, no default
        provided gating path

    Examples
    --------
    >>> find_current_population("root/singlets")
    singlets
    >>> find_current_population("root")
    root
    >>> find_current_population("root/singlets/T_cells")
    T_cells
    """
    _check_gate_name(gate)
    return gate.split(GATE_SEPARATOR)[-1]


def _find_gate_path_of_gate(adata: AnnData,
                            gate: str) -> str:
    """\
    Finds the gate path of the specified population
    This function looks into adata.uns["gating_cols"] and selects
    the entry that endswith the provided population

    Parameters
    ----------

    adata: AnnData
        the current dataset with an uns dict
    gate: str
        the population that is looked up

    Examples
    --------

    >>> adata = ad.AnnData(uns = {"gating_cols": pd.Index(["root/singlets"])})
    >>> find_gate_path_of_gate(adata, "singlets")
    "root/singlets"

    """
    _check_gate_name(gate)
    if GATE_SEPARATOR in gate:
        n_separators = gate.count(GATE_SEPARATOR)
        n_gates = n_separators + 1
        gates = [
            gate_path
            for gate_path in adata.uns["gating_cols"]
            if _gate_path_length(gate_path) >= n_gates and
            _extract_partial_gate_path_end(gate_path, n_gates) == gate
        ]
    else:
        gates = [gate_path for gate_path in adata.uns["gating_cols"]
                 if _find_current_population(gate_path) == gate]
    if not gates:
        raise GateNotFoundError(gate)
    if len(gates) > 1:
        raise GateAmbiguityError(gates)
    return gates[0]


def _gate_path_length(gate_path: str) -> int:
    _check_gate_path(gate_path)
    return len(gate_path.split(GATE_SEPARATOR))


def _extract_partial_gate_path_end(gate_path: str,
                                   n_positions: int) -> str:
    _check_gate_path(gate_path)
    gate_components = gate_path.split(GATE_SEPARATOR)
    if len(gate_components) < n_positions:
        raise ExhaustedGatePathError(n_positions, len(gate_components))
    return GATE_SEPARATOR.join(gate_path.split(GATE_SEPARATOR)[-n_positions:])


def _extract_partial_gate_path_start(gate_path: str,
                                     n_positions: int) -> str:
    _check_gate_path(gate_path)
    gate_components = gate_path.split(GATE_SEPARATOR)
    if len(gate_components) < n_positions:
        raise ExhaustedGatePathError(n_positions, len(gate_components))
    return GATE_SEPARATOR.join(gate_path.split("/")[:n_positions])


def _find_gate_indices(adata: AnnData,
                       gate_columns: Union[list[str], str]) -> list[int]:
    """Finds the index of provided populations in adata.uns["gating_cols"]
    This function is supposed to index columns provided as a string.
    That way, the indices can be used to access the sparse matrix
    in adata.obsm["gating"] that stores the gating values.

    Parameters
    ----------
    adata
        the provided dataset
    gate_columns
        the gate columns that are supposed to be looked up

    Examples
    --------
    >>> adata = ad.Anndata(uns = {"gating_cols": pd.Index(["root/singlets",
                                                           "root/singlets/T_cells])}
    >>> find_gate_indices(adata, "root/singlets")
    [0]
    >>> find_gate_indices(adata, ["root/singlets", "root/singlets/T_cells"])
    [0,1]

    """

    if not isinstance(gate_columns, list):
        gate_columns = [gate_columns]
    return [adata.uns["gating_cols"].get_loc(gate) for gate in gate_columns]


def _find_parent_gate(gate: str) -> str:
    """Returns the parent gate path of the provided gate

    Parameters
    ----------
    gate: str
        the provided gate path

    Examples
    --------

    >>> find_parent_gate("root/singlets/T_cells")
    root/singlets
    >>> find_parent_gate("root")
    ExhaustedHierarchyError
    >>> find_parent_gate("root/singlets")
    root

    """
    _check_gate_name(gate)
    if GATE_SEPARATOR in gate:
        return GATE_SEPARATOR.join(gate.split(GATE_SEPARATOR)[:-1])
    else:
        raise ExhaustedHierarchyError(gate)


def _find_parent_population(gate: str) -> str:
    """Returns the parent population of the provided gate path

    Parameters
    ----------
    gate: str
        the provided gate path

    Examples
    --------
    >>> find_parent_population("root/singlets/T_cells")
    singlets
    >>> find_parent_population("root")
    ExhaustedHierarchyError
    >>> find_parent_population("root/singlets/")
    root
    """

    _check_gate_name(gate)
    if GATE_SEPARATOR in gate:
        return gate.split(GATE_SEPARATOR)[:-1][::-1][0]
    else:
        raise ExhaustedHierarchyError(gate)


def _find_grandparent_gate(gate: str) -> str:
    """Finds the grandparent gating path of a provided gate

    Parameters
    ----------
    gate: str
        the provided gating path

    Examples
    --------

    >>> find_grandparent_gate("root/singlets/T_cells")
    root
    >>> find_grandparent_gate("root/singlets/T_cells/cytotoxic")
    root/singlets
    >>> find_grandparent_gate("root/singlets")
    ExhaustedHieararchyError
    """
    _check_gate_name(gate)
    return _find_parent_gate(_find_parent_gate(gate))


def _find_grandparent_population(gate: str) -> str:
    """Finds the grandparent population of a provided gate

    Parameters
    ----------
    gate: str
        the provided gating path

    Examples
    --------

    >>> find_grandparent_gate("root/singlets/T_cells")
    root
    >>> find_grandparent_gate("root/singlets/T_cells/cytotoxic")
    singlets
    >>> find_grandparent_gate("root/singlets")
    ExhaustedHieararchyError
    """
    _check_gate_name(gate)
    return _find_parent_population(_find_parent_gate(gate))


def _find_parents_recursively(gate: str,
                              parent_list: Optional[list[str]] = None
                              ) -> list[str]:
    """Finds all parent gates of a specified gate

    Parameters
    ----------
    gate: str
        provided gating path
    parent_list: None
        is instantiated to None because the function is used recursively

    Examples
    --------

    >>> find_parents_recursively("root/singlets/T_cells")
    ["root_singlets", "root"]
    >>> find_parents_recursively("root")
    ExhaustedHierarchyError
    """
    if parent_list is None:
        parent_list = []
    parent = _find_parent_gate(gate)
    parent_list.append(parent)
    if parent != "root":
        return _find_parents_recursively(parent, parent_list)
    return parent_list


def _find_children_of_gate(adata: AnnData,
                           query_gate: str) -> list[str]:
    gates = adata.uns["gating_cols"]
    return [
        gate for gate in gates
        if GATE_SEPARATOR.join(gate.split(GATE_SEPARATOR)[:-1]) == query_gate
    ]


def _transform_gates_according_to_gate_transform(vertices: np.ndarray,
                                                 transforms: dict,
                                                 gate_channels: list[str]
                                                 ) -> np.ndarray:

    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [
            transform for transform in transforms
            if gate_channel in transform.id
        ]
        if len(channel_transforms) > 1:
            transform = [
                transform for transform in channel_transforms
                if "Comp-" in transform.id
            ][0]
        else:
            transform = channel_transforms[0]
        vertices[i] = transform.apply(vertices[i])
    return vertices


def _transform_vertices_according_to_gate_transform(vertices: np.ndarray,
                                                    transforms: dict,
                                                    gate_channels: list[str]
                                                    ) -> np.ndarray:
    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [
            transform for transform in transforms
            if gate_channel in transform.id
        ]
        if len(channel_transforms) > 1:
            transform = [
                transform for transform in channel_transforms
                if "Comp-" in transform.id
            ][0]
        else:
            transform = channel_transforms[0]
        vertices[:, i] = transform.apply(vertices[:, i])
    return vertices


def _inverse_transform_gates_according_to_gate_transform(
        vertices: np.ndarray,
        transforms: dict,
        gate_channels: list[str]) -> np.ndarray:

    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [
            transform for transform in transforms
            if gate_channel in transform.id
        ]
        if len(channel_transforms) > 1:
            transform = [
                transform for transform in channel_transforms
                if "Comp-" in transform.id
            ][0]
        else:
            transform = channel_transforms[0]
        vertices[i] = transform.inverse(vertices[i])
    return vertices


def _inverse_transform_vertices_according_to_gate_transform(
        vertices: np.ndarray,
        transforms: dict,
        gate_channels: list[str]) -> np.ndarray:

    for i, gate_channel in enumerate(gate_channels):
        channel_transforms = [
            transform for transform in transforms
            if gate_channel in transform.id
        ]
        if len(channel_transforms) > 1:
            transform = [
                transform for transform in channel_transforms
                if "Comp-" in transform.id
            ][0]
        else:
            transform = channel_transforms[0]
        vertices[:, i] = transform.inverse(vertices[:, i])
    return vertices


def _close_polygon_gate_coordinates(vertices: np.ndarray) -> np.ndarray:
    """Closes a polygon gate by adding the first coordinate to the bottom of
    the array.

    Parameters
    ----------

    vertices: np.ndarray
        the array that contains the gate coordinates

    Examples
    --------
    >>> coordinates = np.array([[1,2], [3,4]])
    >>> close_polygon_gate_coordinates(coordinates)
    np.array([[1,2], [3,4], [1,2]])
    """
    return np.vstack([vertices, vertices[0]])


def _create_gate_lut(wsp_dict: dict[str: dict]) -> dict:
    # TODO: needs check for group...
    _gate_lut = {}
    gated_files = []
    for file in wsp_dict:

        _gate_lut[file] = {}
        gate_list = wsp_dict[file]["gates"]

        if gate_list:
            gated_files.append(file)

        for i, _ in enumerate(gate_list):
            
            gate_name = wsp_dict[file]["gates"][i]["gate"].gate_name.replace(
                " ", "_"
            )
            _gate_lut[file][gate_name] = {}

            gate_path = GATE_SEPARATOR\
                .join(list(wsp_dict[file]["gates"][i]["gate_path"]))\
                .replace(" ", "_")
            gate_channels = [
                dim.id
                for dim in wsp_dict[file]["gates"][i]["gate"].dimensions
            ]

            gate_dimensions = np.array(
                [
                    (dim.min, dim.max)
                    for dim in wsp_dict[file]["gates"][i]["gate"].dimensions
                ],
                dtype = np.float32
            )
            gate_dimensions = _inverse_transform_gates_according_to_gate_transform(gate_dimensions,  # noqa
                                                                                   wsp_dict[file]["transforms"],  # noqa
                                                                                   gate_channels)  # noqa

            try:
                vertices = np.array(
                    wsp_dict[file]["gates"][i]["gate"].vertices
                )
                vertices = _close_polygon_gate_coordinates(vertices)
                vertices = _inverse_transform_vertices_according_to_gate_transform(vertices,  # noqa
                                                                                   wsp_dict[file]["transforms"],  # noqa
                                                                                   gate_channels)  # noqa
            except AttributeError:
                vertices = gate_dimensions

            _gate_lut[file][gate_name]["parent_path"] = gate_path
            _gate_lut[file][gate_name]["dimensions"] = gate_channels
            _gate_lut[file][gate_name]["full_gate_path"] =\
                GATE_SEPARATOR.join([gate_path, gate_name])
            _gate_lut[file][gate_name]["gate_type"] =\
                wsp_dict[file]["gates"][i]["gate"].__class__.__name__
            _gate_lut[file][gate_name]["gate_dimensions"] = gate_dimensions
            _gate_lut[file][gate_name]["vertices"] = vertices

    return _gate_lut


def _fetch_fluo_channels(adata: AnnData) -> list[str]:
    """
    compares channel names to a predefined list of common FACS and
    CyTOF channels
    """
    return adata.var.loc[adata.var["type"] == "fluo"].index.tolist()


def subset_fluo_channels(adata: AnnData,
                         as_view: bool = False,
                         copy: bool = False) -> Optional[AnnData]:
    """\
    Subsets only channels that are of type 'fluo'.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    as_view
        If True, returns an AnnDataView object.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.subset_fluo_channels(dataset)

    """
    adata = adata.copy() if copy else adata
    if as_view:
        return adata[:, adata.var["type"] == "fluo"]
    else:
        adata._inplace_subset_var(adata.var[adata.var["type"] == "fluo"].index)
    return adata if copy else None


def subset_channels(adata: AnnData,
                    channels: Optional[list[str]] = None,
                    use_panel: bool = False,
                    keep_state_channels: bool = True,
                    copy: bool = False) -> Optional[AnnData]:
    if not use_panel and channels is None:
        raise ChannelSubsetError

    if use_panel:  # overrides channels input.
        channels = adata.uns["panel"].dataframe["antigens"].to_list()

    if keep_state_channels:
        state_channels: list[str] = [
            channel for channel in adata.var_names
            if any(k in channel.lower() for k in (
                scatter_channels +
                time_channels +
                cytof_technical_channels +
                spectral_flow_technical_channels)
            )
        ]
        assert channels is not None
        channels += state_channels

    adata = adata.copy() if copy else adata
    adata._inplace_subset_var(
        adata.var.loc[adata.var["pns"].isin(channels)].index.to_list()
    )
    return adata if copy else None


def subset_gate(adata: AnnData,
                gate: str,
                as_view: bool = False,
                copy: bool = False) -> Optional[AnnData]:
    """\
    Subsets the dataset to a specific population.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    gate
        The gate to be subset to. Can be passed as a population name
        (e.g. 'CD45+'), a partial gate path (e.g. 'live/CD45+') or a
        complete gate path (e.g. 'root/cells/live/CD45+').
    as_view
        If True, returns an AnnDataView object.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.subset_gate(dataset, "CD45+")

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.subset_gate(dataset, 'live/CD45+')

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.subset_gate(dataset, 'root/cells/live/CD45+')

    """
    adata = adata.copy() if copy else adata

    gates: list[str] = adata.uns["gating_cols"].to_list()

    gate_path = _find_gate_path_of_gate(adata, gate)

    gate_idx = gates.index(gate_path)

    subset = adata[adata.obsm["gating"][:,gate_idx] == True, :]
    if as_view:
        return subset
    subset = subset.copy()
    adata._init_as_actual(subset, dtype = None)
    return adata if copy else None


def equalize_groups(adata: AnnData,
                    fraction: Optional[float] = None,
                    n_obs: Optional[int] = None,
                    on: Union[list[str], str] = "sample_ID",
                    random_state: int = 187,
                    as_view: bool = False,
                    copy: bool = False
                    ) -> Optional[AnnData]:
    """\
    Equalizes the cell count between groups. If there are discrepancies in
    cell numbers between samples or conditions, this function allows to
    equalize the cell counts in order to avoid over-/underrepresentation of
    samples. Subsampling is done by random selection of cell indices per group.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    fraction
        Fraction of cells to be kept. By default, the group with the smallest
        cell count is selected to calculate the final cell number per group by
        using n_cells * fraction.
    n_obs
        Absolute number of cells per group to be kept. If this number is
        greater than the cell count in one group, a warning will be issued and
        all cells of that group are kept.
    on
        The group variable. Select to the group to equalize. Defaults to
        `sample_ID`, but can be any column in the `.obs` slot.
    random_state
        Controls the random state for reproducible analysis.
    as_view
        If True, returns an AnnDataView object.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.equalize_groups(dataset, n_obs = 300_000, on = "disease_group")

    """
    #TODO: add "min" as a parameter
    np.random.seed(random_state)
    if n_obs and fraction:
        raise ValueError(
            "Please provide either `n_obs` or `fraction`, not both."
        )

    if n_obs is not None:
        new_n_obs = n_obs
    elif fraction is not None:
        if fraction > 1 or fraction < 0:
            raise ValueError(
                f'`fraction` needs to be within [0, 1], not {fraction}'
            )
        new_n_obs = int(adata.obs.value_counts(on).min() * fraction)
    else:
        raise ValueError("Please provide one of `n_obs` or `fraction`")

    # we check if there are enough cells per group
    n_cells_per_group = adata.obs.groupby(on, observed = True).size()
    groups_below_n_obs = n_cells_per_group[
        n_cells_per_group < new_n_obs
    ].index.tolist()
    groups_above_n_obs = n_cells_per_group[
        n_cells_per_group >= new_n_obs
    ].index.tolist()

    if on is None:
        warnings.warn(
            "Equalizing... groups to equalize are set to 'sample_ID'",
            UserWarning
        )
        on = "sample_ID"

    if groups_below_n_obs:
        warnings.warn(
            f"There are groups with n_cells smaller than the requested number "
            f"of cells. These are {groups_below_n_obs} of group {on}",
            UserWarning
        )
        obs_indices_above_n_obs = adata[
            adata.obs[on].isin(groups_above_n_obs)
        ].obs.groupby(on, observed = True).sample(new_n_obs).index.to_numpy()
        obs_indices_below_n_obs = adata[
            adata.obs[on].isin(groups_below_n_obs)
        ].obs.index.to_numpy()
        obs_indices = np.concatenate(
            [obs_indices_above_n_obs, obs_indices_below_n_obs],
            axis = 0
        )

    else:
        obs_indices = adata.obs\
            .groupby(on, observed = True)\
            .sample(new_n_obs).index\
            .to_numpy()

    if as_view:
        return adata[obs_indices]

    adata._inplace_subset_obs(obs_indices)
    return adata if copy else None


def contains_only_fluo(adata: AnnData) -> bool:
    return all(adata.var["type"] == "fluo")


def get_idx_loc(adata: AnnData,
                idx_to_loc: Union[list[str], pd.Index]) -> np.ndarray:
    return np.array([adata.obs_names.get_loc(idx) for idx in idx_to_loc])


def remove_unnamed_channels(adata: AnnData,
                            as_view: bool = False,
                            copy: bool = False) -> Optional[AnnData]:
    """\
    Removes unnamed channels. Unnamed channels are defined as channels that
    have the same value in the 'pnn' field as the 'pns' field in `adata.var`.
    This happens when channels were recorded and saved to the .fcs file that
    were not given a name via the panel. Scatter- and technical channels are
    kept, regardless of their definition in the Panel object.

    This function removes these channels in order to not include empty
    channels in further analysis.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    as_view
        If True, returns an AnnDataView object.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.remove_unnamed_channels(dataset)

    """

    adata = adata.copy() if copy else adata

    unnamed_channels = [
        channel for channel in adata.var.index if
        channel not in adata.uns["panel"].dataframe["antigens"].to_list() and
        adata.var.loc[adata.var["pns"] == channel, "type"].iloc[0] == "fluo"
    ]
    named_channels = [channel for channel in adata.var.index if
                      channel not in unnamed_channels]
    non_fluo_channels = adata.var[adata.var["type"] != "fluo"].index.to_list()

    channels_to_keep = list(set(named_channels + non_fluo_channels))

    if as_view:
        return adata[:, adata.var.index.isin(channels_to_keep)]
    adata._inplace_subset_var(channels_to_keep)
    return adata if copy else None


def _flatten_nested_list(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]


def _is_valid_sample_ID(adata: AnnData,
                        string_to_check: Optional[str]) -> bool:
    if not string_to_check:
        return False
    return string_to_check in adata.obs["sample_ID"].unique()


def _is_valid_filename(adata: AnnData,
                       string_to_check: Optional[str]) -> bool:
    if not string_to_check:
        return False
    return string_to_check in adata.obs["file_name"].unique()


def is_fluo_channel(adata: AnnData,
                    channel: str) -> bool:
    return adata.var.loc[adata.var["pns"] == channel, "type"].iloc[0] == "fluo"


def _create_comparisons(data: pd.DataFrame,
                        groupby: str,
                        splitby: Optional[str],
                        n: int = 2) -> list[tuple[str, str]]:
    groupby_values = data[groupby].unique()
    if splitby:
        splitby_values = data[splitby].unique()
        vals = [(g, s)
                for g in groupby_values
                for s in splitby_values]
    else:
        vals = groupby_values
    return list(combinations(vals, n))


def convert_cluster_to_gate(adata: AnnData,
                            cluster_key: str,
                            positive_cluster: Union[list[int], list[str], int, str],  # noqa
                            population_name: str,
                            parent_name: str,
                            copy: bool = False) -> Optional[AnnData]:
    """\
    Converts cluster information to gates. Select positive clusters and define
    a population from the positive clusters. The population will be added as a
    gate to perform analyses on.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    cluster_key
        The name of the `.obs` columns where the cluster information is stored.
    positive_cluster
        The values of the clusters to be defined as a population. In order to
        select multiple clusters, pass a list.
    population name
        The name of the newly defined population.
    parent_name
        The name of the parent population in order to define a bona-fide
        gating path.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.settings.default_gate = "CD45+"
    >>> fp.settings.default_layer = "transformed"
    >>> fp.tl.pca(dataset)
    >>> fp.tl.neighbors(dataset)
    >>> fp.tl.parc(dataset)
    >>> fp.convert_cluster_to_gate(
    ...     dataset,
    ...     cluster_key = "CD45+_transformed_parc",
    ...     positive_cluster = ["1", "4", "6"],
    ...     population_name = "Neutrophils",
    ...     parent_name = "CD45+"
    ... )

    """
    from scipy.sparse import csr_matrix, hstack
    adata = adata.copy() if copy else adata
    full_parent = _find_gate_path_of_gate(adata, parent_name)
    full_gate = GATE_SEPARATOR.join([full_parent, population_name])
    if full_gate in adata.uns["gating_cols"]:
        raise TypeError(
            "Gate already present. Please choose a different name!"
        )

    if not isinstance(positive_cluster, list):
        positive_cluster = [positive_cluster]
    gate_list = adata.obs[cluster_key]
    uniques = gate_list.unique()
    mapping = {cluster: cluster in positive_cluster for cluster in uniques}
    gate_matrix = csr_matrix(
        gate_list.map(mapping).values.reshape(len(gate_list), 1), dtype = bool
    )
    adata.obsm["gating"] = hstack([adata.obsm["gating"], gate_matrix])

    adata.uns["gating_cols"] = adata.uns["gating_cols"].append(pd.Index([full_gate]))

    return adata if copy else None


@_enable_gate_aliases
def convert_gate_to_obs(adata: AnnData,
                        gate: str,
                        key_added: Optional[str] = None,
                        copy: bool = False) -> Optional[AnnData]:
    """\
    Converts the gate information stored in `.obsm["gating"]` into an
    `.obs` column.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    gate
        The gate to transfer. Must be in `adata.uns["gating_cols"]`. Can be
        specified as a population, as a partial gate path or a full gate path.
        Also allows for gate aliasing. If a gate alias is provided, pass the
        key_added parameter in order to pass that alias forward. Otherwise,
        the full gate path is used for the `.obs` column.
    key_added
        The name of the corresponding `.obs` column. Positive events will be
        stored as specified in `key_added`, negative events will be stored as
        `other`.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.convert_gate_to_obs(dataset, "CD45+", key_added = "leukocytes")
    >>> fp.tl.pca(dataset, gate = "live", layer = "compensated")
    >>> fp.pl.pca(dataset, color = "leukocytes")

    """
    adata = adata.copy() if copy else adata

    gate_path = _find_gate_path_of_gate(adata, gate)
    gate_index = _find_gate_indices(adata, gate_path)
    gate_id = key_added or gate
    adata.obs[gate_id] = adata.obsm["gating"][:,gate_index].toarray()
    adata.obs[gate_id] = adata.obs[gate_id].map(
        {True: gate_id, False: "other"}
    )
    adata.obs[gate_id] = adata.obs[gate_id].astype("category")
    adata.obs[gate_id] = adata.obs[gate_id].cat.set_categories(
        [gate_id, "other"]
    )
    return adata if copy else None


def rename_channel(adata: AnnData,
                   old_channel_name: str,
                   new_channel_name: str,
                   copy: bool = False) -> Optional[AnnData]:
    """\
    Renames a channel name.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    current_channel_name
        Current channel name. Has to be in `adata.var_names`.
    new_channel_name
        The name that the current channel name is replaced with.
    copy
        Whether to copy the dataset.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.remove_channel(dataset, "CD16")

    
    """
    adata = adata.copy() if copy else adata
    # we need to rename it in the panel, the cofactors and var
    adata.var = adata.var.replace(old_channel_name, new_channel_name)

    # curiosity: only the uncommented line works. The commented lead to a
    # serious file save error 'TypeError: expected str, bytes or os.PathLike
    # object, not NoneType
    # Error raised while writing key 'pns' of <class 'h5py._hl.group.Group'>
    # to / in anndata\_io\h5ad.py:104
    adata.var.index = adata.var["pns"].tolist()
    # adata.var = adata.var.set_index("pns", drop = False)
    # adata.var.index.name = "" # just to keep the index clean :)

    if "panel" in adata.uns and len(adata.uns["panel"].dataframe) > 0:
        from .dataset._supplements import Panel
        panel: Panel = adata.uns["panel"]
        panel.rename_antigen(old_channel_name, new_channel_name)

    if "cofactors" in adata.uns and len(adata.uns["cofactors"].dataframe) > 0:
        from .dataset._supplements import CofactorTable
        cofactors: CofactorTable = adata.uns["cofactors"]
        cofactors.rename_channel(old_channel_name, new_channel_name)

    return adata if copy else None


def remove_channel(adata: AnnData,
                   channel: Union[list[str], str],
                   as_view: bool = False,
                   copy: bool = False) -> Optional[AnnData]:
    """\
    Removes a channel from the dataset. This function will only subset the
    AnnData object. Use in conjunction with fp.sync.synchronize_dataset()
    to update the analyzed data and the accompanying metadata with it.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    channel
        The channel to remove. Has to be in `adata.var_names`. Pass a list of
        channels to subset multiple channels at once.
    as_view
        If True, returns an AnnDataView object.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> fp.remove_channel(dataset, "CD16")


    """
    adata = adata.copy() if copy else adata

    if not isinstance(channel, list):
        channel = [channel]

    if any(ch not in adata.var_names for ch in channel):
        raise ValueError("One of the provided channels was not found.")

    if as_view:
        return adata[:, ~adata.var_names.isin(channel)]
    adata._inplace_subset_var(
        [var for var in adata.var_names if var not in channel]
    )
    return adata if copy else None


def convert_var_to_panel(adata: AnnData,
                         copy: bool = False) -> Optional[AnnData]:
    from .dataset._supplements import Panel
    adata = adata.copy() if copy else adata

    new_panel = pd.DataFrame(data = {"fcs_colname": adata.var["pnn"].to_list(),
                                     "antigens": adata.var["pns"].to_list()})
    adata.uns["panel"] = Panel(panel = new_panel)

    return adata if copy else None
