from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

try:
    from pyairtable import Table
    AIRTABLE_AVAILABLE = True
except ImportError:
    AIRTABLE_AVAILABLE = False

import mne


def export_eeg_dataset_to_airtable(raw: mne.io.BaseRaw, 
                                    airtable_base_id: str,
                                    airtable_table_name: str,
                                    airtable_api_key: str,
                                    xdf_file_path: Optional[Path] = None,
                                    additional_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Export a loaded EEG dataset from an XDF file to Airtable.
    
    This function extracts metadata and summary information from an MNE Raw object
    (loaded from an XDF file) and creates a record in Airtable.
    
    Args:
        raw: MNE Raw object containing the EEG data
        airtable_base_id: Airtable base ID (e.g., "appXXXXXXXXXXXXXX")
        airtable_table_name: Name of the Airtable table to write to
        airtable_api_key: Airtable API key
        xdf_file_path: Optional path to the source XDF file
        additional_fields: Optional dictionary of additional fields to include in the record
        
    Returns:
        Dictionary with 'success' (bool), 'record_id' (str if successful), and 'error' (str if failed)
        
    Usage:
        from phopylslhelper.core.xdf_files import LabRecorderXDF
        from phoofflineeeganalysis.analysis.airtable_export import export_eeg_dataset_to_airtable
        
        # Load XDF file
        xdf_file = Path("path/to/file.xdf")
        lab_recorder_xdf = LabRecorderXDF.init_from_lab_recorder_xdf_file(
            a_xdf_file=xdf_file,
            should_load_full_file_data=True
        )
        
        # Get the first EEG Raw object
        eeg_raws = lab_recorder_xdf.datasets_dict.get('EEG', [])
        if eeg_raws:
            raw = eeg_raws[0]
            
            # Export to Airtable
            result = export_eeg_dataset_to_airtable(
                raw=raw,
                airtable_base_id="appXXXXXXXXXXXXXX",
                airtable_table_name="EEG Recordings",
                airtable_api_key="your_api_key_here",
                xdf_file_path=xdf_file
            )
            print(result)
    """
    if not AIRTABLE_AVAILABLE:
        return {
            'success': False,
            'error': 'pyairtable is not installed. Install it with: pip install pyairtable'
        }
    
    try:
        # Initialize Airtable table
        table = Table(airtable_api_key, airtable_base_id, airtable_table_name)
        
        # Extract metadata from Raw object
        info = raw.info
        
        # Get measurement date
        meas_date = info.get('meas_date', None)
        if meas_date is not None:
            if isinstance(meas_date, tuple):
                meas_date = datetime.fromtimestamp(meas_date[0], tz=timezone.utc)
            elif hasattr(meas_date, 'timestamp'):
                if meas_date.tzinfo is None:
                    meas_date = meas_date.replace(tzinfo=timezone.utc)
                else:
                    meas_date = meas_date.astimezone(timezone.utc)
        
        # Get file path from description or provided path
        file_path_str = None
        if xdf_file_path is not None:
            file_path_str = str(xdf_file_path.resolve())
        elif 'description' in info:
            file_path_str = info['description']
        
        # Extract channel information
        ch_names = info.get('ch_names', [])
        ch_types = raw.get_channel_types()
        n_channels = len(ch_names)
        sfreq = info.get('sfreq', None)
        
        # Get duration
        duration_sec = None
        if len(raw.times) > 0:
            duration_sec = float(raw.times[-1])
        
        # Get device information if available
        device_info = info.get('device_info', {})
        device_type = device_info.get('type', None)
        device_model = device_info.get('model', None)
        device_serial = device_info.get('serial', None)
        
        # Extract stream info if available
        stream_info = device_info.get('stream_info', {})
        stream_name = stream_info.get('name', None)
        source_id = stream_info.get('source_id', None)
        hostname = stream_info.get('hostname', None)
        
        # Get annotations summary
        annotations = raw.annotations
        n_annotations = len(annotations) if annotations is not None else 0
        annotation_descriptions = []
        if annotations is not None and len(annotations) > 0:
            annotation_descriptions = list(set(annotations.description))
        
        # Build Airtable record fields
        fields = {
            'Recording Date': meas_date.isoformat() if meas_date else None,
            'File Path': file_path_str,
            'File Name': Path(file_path_str).name if file_path_str else None,
            'Number of Channels': n_channels,
            'Channel Names': ', '.join(ch_names) if ch_names else None,
            'Channel Types': ', '.join(set(ch_types)) if ch_types else None,
            'Sampling Rate (Hz)': sfreq,
            'Duration (seconds)': duration_sec,
            'Device Type': device_type,
            'Device Model': device_model,
            'Device Serial': device_serial,
            'Stream Name': stream_name,
            'Source ID': source_id,
            'Hostname': hostname,
            'Number of Annotations': n_annotations,
            'Annotation Types': ', '.join(annotation_descriptions) if annotation_descriptions else None,
        }
        
        # Add any additional fields
        if additional_fields:
            fields.update(additional_fields)
        
        # Remove None values (Airtable doesn't like None)
        fields = {k: v for k, v in fields.items() if v is not None}
        
        # Create record in Airtable
        record = table.create(fields)
        
        return {
            'success': True,
            'record_id': record['id'],
            'fields': record['fields']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def export_multiple_eeg_datasets_to_airtable(raws: List[mne.io.BaseRaw],
                                             airtable_base_id: str,
                                             airtable_table_name: str,
                                             airtable_api_key: str,
                                             xdf_file_paths: Optional[List[Path]] = None,
                                             additional_fields_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Export multiple EEG datasets to Airtable.
    
    Args:
        raws: List of MNE Raw objects
        airtable_base_id: Airtable base ID
        airtable_table_name: Name of the Airtable table
        airtable_api_key: Airtable API key
        xdf_file_paths: Optional list of XDF file paths (must match length of raws)
        additional_fields_list: Optional list of additional field dictionaries
        
    Returns:
        List of result dictionaries from export_eeg_dataset_to_airtable
    """
    results = []
    
    for idx, raw in enumerate(raws):
        xdf_path = xdf_file_paths[idx] if xdf_file_paths and idx < len(xdf_file_paths) else None
        additional = additional_fields_list[idx] if additional_fields_list and idx < len(additional_fields_list) else None
        
        result = export_eeg_dataset_to_airtable(
            raw=raw,
            airtable_base_id=airtable_base_id,
            airtable_table_name=airtable_table_name,
            airtable_api_key=airtable_api_key,
            xdf_file_path=xdf_path,
            additional_fields=additional
        )
        results.append(result)
    
    return results


__all__ = ['export_eeg_dataset_to_airtable', 'export_multiple_eeg_datasets_to_airtable']
