import time
import re
from datetime import datetime, timezone
from attrs import define, field, Factory
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from mne.channels.montage import DigMontage
from pathlib import Path
import numpy as np
import pandas as pd
import mne

# ElectrodeHelper module
from pathlib import Path
# import trimesh
import importlib.resources as resources


@define(slots=False)
class ElectrodeHelper:
    """
    A helper class for creating MNE montages from Emotiv electrode data
    and projecting them onto scalp surfaces from MRI data.
    
    
    Basic Electrode Positions Loading:
        from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper
        from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper

        # Just create montage from your electrode positions
        # electrode_positions_path = Path(r"E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts/emotiv_wellAlignedPho.ced").resolve()
        electrode_positions_path = Path(r"E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts/emotiv.ced").resolve()
        assert electrode_positions_path.exists()
        montage = ElectrodeHelper.create_complete_montage_workflow(electrode_positions_path)
        montage
        
    Stateful Electrode Positions Loading:
    
        from mne.channels.montage import DigMontage
        from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper

        active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage()
        emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage
        emotiv_epocX_montage

        # Just create montage from your electrode positions
        print("Montage created successfully!")
        print(f"Channel names: {emotiv_epocX_montage.ch_names}")
        

        
    """
    active_montage: DigMontage = field()

    
    @classmethod
    def init_EpocX_montage(cls, electrode_positions_path: Optional[Path] = None) -> "ElectrodeHelper":
        if electrode_positions_path is None:
            # electrode_pos_parent_folder: Path = Path("E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts").resolve()
            # electrode_positions_path = electrode_pos_parent_folder.joinpath('ElectrodePositions_2025-08-14', 'brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv')

            # electrode_pos_parent_folder: Path = Path("E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts").resolve()
            # electrode_positions_path = electrode_pos_parent_folder.joinpath('phopymnehelper/resources/ElectrodeLayouts/brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv')

            electrode_positions_path = Path(resources.files("phopymnehelper").joinpath("resources/ElectrodeLayouts/brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv")).resolve()


        assert electrode_positions_path.exists(), f"electrode_positions_path: '{electrode_positions_path}' does not exist!"
        
        emotiv_epocX_montage: DigMontage = ElectrodeHelper.montage_from_subjece_space_mm_tsv(electrode_positions_path=electrode_positions_path)
        return cls(active_montage=emotiv_epocX_montage)
        
    
    @classmethod
    def montage_from_subjece_space_mm_tsv(cls, electrode_positions_path: Path) -> DigMontage:
        """ 
        Loads a Brainstorm Exported Electrode configuration 
        To export from Brainstorm, right click an EEGLAB channels (14) object and go to:
            File > Export to file...
            From the "Files of type:" dropdown, select "EEG: BIDS electrodes.tsv, subject space mm (*.tsv)"
            For "File name:" I used 'brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv'
            
            
        Usage:
            from mne.channels.montage import DigMontage
            from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper

            electrode_pos_parent_folder = Path("E:/Dropbox (Personal)/Hardware/Consumer EEG Headsets/Emotiv Epoc EEG/ElectrodeLayouts").resolve()
            electrode_positions_path = electrode_pos_parent_folder.joinpath('ElectrodePositions_2025-08-14', 'brainstorm_electrode_positions_PhoHAle_eeg_subjectspacemm.tsv')

            mont: DigMontage = ElectrodeHelper.montage_from_subjece_space_mm_tsv(electrode_positions_path=electrode_positions_path)
            mont
        """
        # subjectspacemm
        assert electrode_positions_path.exists()
        df = pd.read_csv(electrode_positions_path, sep="\t")  # or '\s+' if whitespace separated
        # Map column names from Brainstorm's format to MNE
        # Assuming columns: Name   X   Y   Z   Type
        df.columns = [c.strip().lower() for c in df.columns]  # normalize
        ch_pos = {}
        fid = {}
        for _, row in df.iterrows():
            name = row["name"]
            pos_m = (row["x"]/1000.0, row["y"]/1000.0, row["z"]/1000.0)  # mm → m
            if name.upper() in {"NAS","LPA","RPA"}:
                fid[name.upper()] = pos_m
            else:
                ch_pos[name] = pos_m

        mont: DigMontage = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=fid.get("NAS"), lpa=fid.get("LPA"), rpa=fid.get("RPA"), coord_frame="head")
        return mont


    @staticmethod
    def _parse_ced(ced_file_path: Path) -> Dict[str, np.ndarray]:
        """Very tolerant .ced parser: returns dict[label] = np.array([x,y,z]) (units: unknown)."""
        coords = {}
        float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        with open(ced_file_path, "r", encoding="utf8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("%") or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # label is first token that is not purely numeric
                label = parts[0]
                # find numeric tokens in the rest of the line
                nums = float_re.findall(line)
                # if the first numeric occurs after the label, extract up to 3 numeric values
                if len(nums) >= 3:
                    # choose first three numeric tokens as x,y,z
                    x, y, z = map(float, nums[:3])
                    coords[label] = np.array([x, y, z], dtype=float)
                else:
                    # fallback: if no numeric triple, skip
                    continue
        return coords


    @staticmethod
    def visualize_montage(montage: mne.channels.DigMontage):
        """Simple visualization using MNE's built-in plot; opens interactive 3D viewer if available."""
        montage.plot(kind="3d")




