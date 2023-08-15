import os
import warnings

import numpy as np
import pandas as pd

from flowio import FlowData
from ..transforms._matrix import Matrix

from typing import Optional

class FCSFile:
    """
    intermediate representation of a sample FCS file.
    Organization into an object is meant to facilitate cleaner code
    """
    def __init__(self,
                 input_directory: str,
                 file_name: str,
                 subsample: Optional[int] = None
                 ) -> None:
        
        self.original_filename = file_name

        raw_data = self.load_fcs_file_from_disk(input_directory,
                                                file_name,
                                                ignore_offset_error = False)
        
        self.compensation_status = "uncompensated"
        self.transform_status = "untransformed"
        self.gating_status = "ungated"

        self.event_count = self.parse_event_count(raw_data)
        self.version = self.parse_fcs_version(raw_data)
        self.fcs_metadata = self.parse_fcs_metadata(raw_data)
        
        self.channels = self.parse_channel_information(raw_data)
        
        self.original_events = self.parse_and_process_original_events(raw_data,
                                                                      subsample)

        self.fcs_compensation = self.parse_compensation_matrix_from_fcs()


    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'v{self.version}, ' +
            f'{self.original_filename}, '
            f'{self.channels.shape[0]} channels, ' + 
            f'{self.event_count} events, ' +
            f'gating status: {self.gating_status}, ' + 
            f'compensation status: {self.compensation_status}, ' + 
            f'transform status: {self.transform_status})'
        )

    def get_events(self,
                   source: str) -> np.ndarray:
        if source == "raw":
            return self._get_original_events()
        elif source == "comp":
            raise NotImplementedError("not yet implemented")
            #self._get_compensated_events()
        else:
            raise NotImplementedError("Only Raw and comped events can be fetched.")

    def _get_original_events(self):
        return self.original_events
    
    def _get_compensated_events(self):
        return self.compensated_events


    def get_channel_index(self,
                          channel_label):
        return self.channels.loc[self.channels.index == channel_label, "channel_numbers"] - 1

    def parse_compensation_matrix_from_fcs(self) -> Matrix:
        
        if "spill" not in self.fcs_metadata:
            fluoro_channels_no = len(
                [
                    channel
                    for channel in self.channels.index
                    if any(k not in channel.lower() for k in ["fsc", "ssc", "time"])
                ]
            )
            return Matrix(matrix_id = "FACSPy_empty",
                        detectors = self.channels.index,
                        fluorochromes = self.channels["pns"],
                        spill_data_or_file = np.eye(N = fluoro_channels_no, M = fluoro_channels_no)
                        )

        return Matrix(matrix_id = "acquisition_defined",
                      detectors = self.channels.index,
                      fluorochromes = self.channels["pns"],
                      spill_data_or_file = self.fcs_metadata["spill"]
                    )

    def parse_event_count(self,
                          fcs_data: FlowData):
        return fcs_data.event_count

    def subsample_events(self,
                         events: np.ndarray,
                         size: int) -> np.ndarray:
        if size >= events.shape[0]:
            return events
        
        return events[np.random.randint(events.shape[0],
                                        size = size), :]

    def parse_and_process_original_events(self,
                                          fcs_data: FlowData,
                                          subsample: Optional[int]) -> np.ndarray:
        tmp_orig_events = self.parse_original_events(fcs_data)
        if subsample is not None:
            tmp_orig_events = self.subsample_events(tmp_orig_events,
                                                    subsample)
        tmp_orig_events = self.process_original_events(tmp_orig_events)
        return tmp_orig_events

    def process_original_events(self,
                                tmp_orig_events: np.ndarray) -> np.ndarray:
        tmp_orig_events = self.adjust_time_channel(tmp_orig_events)
        tmp_orig_events = self.adjust_decades(tmp_orig_events)
        tmp_orig_events = self.adjust_channel_gain(tmp_orig_events)
        return tmp_orig_events

    def adjust_channel_gain(self,
                            events: np.ndarray) -> np.ndarray:
        channel_gains = self.channels.sort_values("channel_numbers")["png"].to_numpy()
        return np.divide(events, channel_gains)
    
    def adjust_decades(self,
                       events: np.ndarray) -> np.ndarray:
        for (decades, log0), channel_number, channel_range in zip(self.channels["pne"],
                                                                  self.channels["channel_numbers"],
                                                                  self.channels["pnr"]):
            if decades > 0:
                events[:, channel_number - 1] = (
                    10 ** (decades * events[:, channel_number - 1] / channel_range)
                    ) * log0
        
        return events
    
    def adjust_time_channel(self,
                            events: np.ndarray) -> np.ndarray:
        if self.time_channel_exists:
            time_index, time_step = self.find_time_channel()
            events[:, time_index] = events[:, time_index] * time_step
        return events

    def find_time_channel(self) -> tuple[int, float]:
        time_step = float(self.fcs_metadata["timestep"]) if "timestep" in self.fcs_metadata else 1.0
        time_index = int(self.channels.loc[self.channels.index.isin(["Time", "time"]), "channel_numbers"].iloc[0]) -1
        return (time_index, time_step)

    def time_channel_exists(self) -> bool:
        return any(
            time_symbol in self.channels.index for time_symbol in ["Time", "time"]
        )

    def parse_original_events(self,
                              fcs_data: FlowData) -> np.ndarray:
        return np.array(
            fcs_data.events,
            dtype=np.float64
        ).reshape(-1, fcs_data.channel_count)

    def remove_disallowed_characters_from_string(self,
                                  input_string: str) -> str:
        for char in [" ", "/", "-"]:
            if char in input_string:
                input_string = input_string.replace(char, "_")
        return input_string
    
    def parse_channel_information(self,
                                  fcs_data: FlowData) -> pd.DataFrame:
        
        channels: dict = fcs_data.channels
        
        pnn_labels = [self.parse_pnn_label(channels, channel_number) for
                      channel_number in channels]
        
        pns_labels = [self.parse_pns_label(channels, channel_number) for
                      channel_number in channels]

        channel_gains = [self.parse_channel_gain(channel_number) for
                         channel_number in channels]
        
        channel_lin_log = [self.parse_channel_lin_log(channel_number) for
                           channel_number in channels]
        
        channel_ranges = [self.parse_channel_range(channel_number) for
                          channel_number in channels]
        
        channel_numbers = [int(k) for k in channels]

        channel_frame = pd.DataFrame(
            data = {"pns": pns_labels,
                    "png": channel_gains,
                    "pne": channel_lin_log,
                    "pnr": channel_ranges,
                    "channel_numbers": channel_numbers
                    },
            index = pnn_labels
        )

        return channel_frame.sort_values("channel_numbers")

    def parse_pnn_label(self,
                        channels: dict,
                        channel_number: str) -> str:
        return channels[channel_number]["PnN"]

    def parse_pns_label(self,
                        channels: dict,
                        channel_number: str) -> str:
        try:
            return self.remove_disallowed_characters_from_string(channels[channel_number]["PnS"]) 
        except KeyError:
            return "" 

    def parse_channel_range(self,
                            channel_number: str) -> int:
        return int(self.fcs_metadata[f"p{channel_number}r"])

    def parse_channel_lin_log(self,
                              channel_number: str) -> tuple[float, float]:
        try:
            (decades, log0) = [float(x) for x in self.fcs_metadata[f"p{channel_number}e"].split(",")] 
            if log0 == 0.0 and decades !=0:
                log0 = 1.0 # FCS std states to use 1.0 for invalid 0 value
            return (decades, log0)
        except KeyError:
            return (0.0, 0.0)
    
    def parse_channel_gain(self,
                           channel_number: str) -> float:
        
        if self.fcs_metadata[f"p{channel_number}n"] in ["Time", "time"]:
            return 1.0
        
        try:
            return float(self.fcs_metadata[f"p{channel_number}g"])
        except KeyError:
            return 1.0

    def parse_fcs_metadata(self,
                             fcs_data: FlowData) -> dict[str: str]:
        """Returns fcs metadata as a dictionary"""
        return fcs_data.text

    def parse_fcs_version(self,
                            fcs_data: FlowData) -> Optional[str]:
        """returns the fcs version"""
        try:
            return str(fcs_data.header["version"])
        except KeyError:
            return None
        
    def load_fcs_file_from_disk(self,
                                input_directory: str,
                                file_name: str,
                                ignore_offset_error: bool) -> FlowData:
        try:
            return FlowData(os.path.join(input_directory, file_name), ignore_offset_error)
        except ValueError:
            warnings.warn("FACSPy IO: FCS file could not be read with " + 
                        f"ignore_offset_error set to {ignore_offset_error}. " +
                        "Parameter is set to True.")
            return FlowData(input_directory, ignore_offset_error = True)
        
    def compensate(self):
        pass