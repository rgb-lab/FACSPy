import anndata as ad
import numpy as np
import pandas as pd
import flowio
import copy
import logging
from ..transforms._matrix import Matrix
from ..gates.gating_strategy import GatingStrategy
from ..io.load import load_fcs_file_from_disk

from .workspaces import FlowJoWorkspace, DivaWorkspace
from flowio import FlowData
from typing import Union, Optional

class Sample:
    """
    intermediate representation of a sample FCS file,
    is meant to return an anndata representation.
    Organization into an object is meant to facilitate cleaner code
    """
    def __init__(self,
                 input_directory,
                 workspace: Union[FlowJoWorkspace, DivaWorkspace],
                 subsample = False
                 ):
        
        raw_data = load_fcs_file_from_disk(input_directory,
                                           ignore_offset_error = False)
        
        self.compensation_status = "uncompensated"
        self.transform_status = "untransformed"
        self.gating_status = "ungated"

        self.event_count = self.parse_event_count(raw_data)
        self.version = self.parse_fcs_version(raw_data)
        self.fcs_metadata = self.parse_fcs_metadata(raw_data)
        self.channels = self.parse_channel_information(raw_data)
        #self.original_events = self.parse_and_process_original_events(raw_data)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'v{self.version}, ' +
            #f'{self.original_filename}, '
            #f'{len(self.pnn_labels)} channels, ' + 
            f'{self.event_count} events, ' +
            f'gating status: {self.gating_status}, ' + 
            f'compensation status: {self.compensation_status}, ' + 
            f'transform status: {self.transform_status})'
        )

    def parse_event_count(self,
                          fcs_data: FlowData):
        return fcs_data.event_count

    def parse_and_process_original_events(self,
                                          fcs_data: FlowData) -> np.ndarray:
        tmp_orig_events = self.parse_original_events(fcs_data)
        tmp_orig_events = self.adjust_time_channel(tmp_orig_events)
        tmp_orig_events = self.adjust_decades(tmp_orig_events)
        tmp_orig_events = self.adjust_channel_gain(tmp_orig_events)

    def adjust_channel_gain(self,
                            events: np.ndarray) -> np.ndarray:
        channel_gains = self.channels.sort_values("channel_numbers")["png"].to_numpy()
        return channel_gains + 1
    
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
        time_index = int(self.channels.loc[self.channels.index.isin(["Time", "time"]), "channel_number"]) -1
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

        return pd.DataFrame(
            data = {"pns": pns_labels,
                    "png": channel_gains,
                    "pne": channel_lin_log,
                    "pnr": channel_ranges,
                    "channel_numbers": channel_numbers
                    },
            index = pnn_labels
        )

    def parse_pnn_label(self,
                        channels: dict,
                        channel_number: str) -> str:
        return channels[channel_number]["PnN"]

    def parse_pns_label(self,
                        channels: dict,
                        channel_number: str) -> str:
        try:
            return channels[channel_number]["PnS"] 
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