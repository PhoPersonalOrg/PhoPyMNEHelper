from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json
import pandas as pd
import numpy as np

import mne


class JSONExporter:

    @classmethod
    def _convert_to_json_serializable(cls, obj: Any) -> Any:
        """
        Convert various Python types to JSON-serializable formats.
        
        Handles:
        - numpy arrays -> lists
        - numpy scalars -> Python native types
        - datetime objects -> ISO format strings
        - pandas Timestamp -> ISO format strings
        - NaN/Inf -> None
        - pandas Series/DataFrame -> dict/list
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if obj is None:
            return None
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            # Check for NaN or Inf values
            if np.any(np.isnan(obj)) or np.any(np.isinf(obj)):
                # Convert NaN/Inf to None
                obj = np.where(np.isnan(obj) | np.isinf(obj), None, obj)
            return obj.tolist()
        
        # Handle numpy scalars
        if isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj.item()
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            else:
                obj = obj.astimezone(timezone.utc)
            return obj.isoformat()
        
        # Handle pandas Timestamp
        if isinstance(obj, pd.Timestamp):
            if obj.tzinfo is None:
                obj = obj.tz_localize(timezone.utc)
            else:
                obj = obj.tz_convert(timezone.utc)
            if pd.isna(obj):
                return None
            return obj.isoformat()
        
        # Handle pandas Timedelta
        if isinstance(obj, pd.Timedelta):
            if pd.isna(obj):
                return None
            return obj.total_seconds()
        
        # Handle pandas Series
        if isinstance(obj, pd.Series):
            return cls._convert_to_json_serializable(obj.tolist())
        
        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return cls._convert_to_json_serializable(obj.to_dict('records'))
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [cls._convert_to_json_serializable(item) for item in obj]
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: cls._convert_to_json_serializable(v) for k, v in obj.items()}
        
        # Handle float NaN/Inf
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        
        # For other types, try to return as-is (will fail during JSON serialization if not serializable)
        return obj


    @classmethod
    def _extract_metadata_from_raw(cls, raw: mne.io.BaseRaw, stream_info_row: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Extract metadata from an MNE Raw object and optional stream info DataFrame row.
        
        Args:
            raw: MNE Raw object
            stream_info_row: Optional pandas Series with stream metadata from stream_infos_df
            
        Returns:
            Dictionary with metadata
        """
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
        
        # Get file path from description
        file_path_str = info.get('description', None)
        
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
        
        # Build metadata dictionary
        metadata = {
            'recording_datetime': meas_date,
            'file_path': file_path_str,
            'file_name': Path(file_path_str).name if file_path_str else None,
            'n_channels': n_channels,
            'channel_names': ch_names,
            'channel_types': ch_types,
            'sampling_rate_hz': sfreq,
            'duration_seconds': duration_sec,
            'device_type': device_type,
            'device_model': device_model,
            'device_serial': device_serial,
            'stream_name': stream_name,
            'source_id': source_id,
            'hostname': hostname,
            'n_annotations': n_annotations,
            'annotation_types': annotation_descriptions,
        }
        
        # Add stream info from DataFrame if provided
        if stream_info_row is not None:
            # Add relevant columns from stream_infos_df
            for col in ['xdf_filename', 'xdf_dataset_idx', 'name', 'fs', 'n_samples', 'n_channels',
                        'created_at_dt', 'first_timestamp_dt', 'last_timestamp_dt', 'duration_sec']:
                if col in stream_info_row.index:
                    value = stream_info_row[col]
                    # Convert pandas Timestamp/Timedelta to serializable format
                    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        if pd.isna(value):
                            continue
                    metadata[col] = value
            ## END for col in ['xdf_filename', ...]:...
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return metadata


    @classmethod
    def _extract_raw_data_from_raw(cls, raw: mne.io.BaseRaw, include_raw_data: bool = True, max_samples_per_stream: Optional[int] = None, sample_interval: int = 1) -> Optional[Dict[str, Any]]:
        """
        Extract raw time-series data from an MNE Raw object.
        
        Args:
            raw: MNE Raw object
            include_raw_data: Whether to include raw data (if False, returns None)
            max_samples_per_stream: Optional maximum number of samples to export per stream
            sample_interval: Take every Nth sample (default: 1, meaning all samples)
            
        Returns:
            Dictionary with timestamps and channel data, or None if include_raw_data is False
        """
        if not include_raw_data:
            return None
        
        try:
            # Get data and times
            data, times = raw.get_data(return_times=True)
            
            # Get measurement date for absolute timestamps
            meas_date = raw.info.get('meas_date', None)
            if meas_date is not None:
                if isinstance(meas_date, tuple):
                    meas_date = datetime.fromtimestamp(meas_date[0], tz=timezone.utc)
                elif hasattr(meas_date, 'timestamp'):
                    if meas_date.tzinfo is None:
                        meas_date = meas_date.replace(tzinfo=timezone.utc)
                    else:
                        meas_date = meas_date.astimezone(timezone.utc)
            else:
                meas_date = None
            
            # Apply sampling interval
            if sample_interval > 1:
                indices = np.arange(0, len(times), sample_interval)
                times = times[indices]
                data = data[:, indices]
            
            # Apply max_samples limit if specified
            if max_samples_per_stream is not None and len(times) > max_samples_per_stream:
                # Take evenly spaced samples
                indices = np.linspace(0, len(times) - 1, max_samples_per_stream, dtype=int)
                times = times[indices]
                data = data[:, indices]
            
            # Convert relative times to absolute timestamps if meas_date is available
            timestamps_iso = None
            if meas_date is not None:
                timestamps_iso = [meas_date + pd.Timedelta(seconds=float(t)) for t in times]
                timestamps_iso = [ts.isoformat() if not pd.isna(ts) else None for ts in timestamps_iso]
            
            # Get channel names
            ch_names = raw.info.get('ch_names', [])
            
            # Build channel data dictionary
            channels_data = {}
            for i, ch_name in enumerate(ch_names):
                channels_data[ch_name] = data[i, :].tolist()
            ## END for i, ch_name in enumerate(ch_names):...
            
            # Build result dictionary
            result = {
                'timestamps_relative_seconds': times.tolist(),
                'channels': channels_data,
            }
            
            if timestamps_iso is not None:
                result['timestamps'] = timestamps_iso
            
            return result
            
        except Exception as e:
            # If data extraction fails, return None (will be handled by caller)
            print(f"Warning: Failed to extract raw data: {e}")
            return None


    @classmethod
    def export_xdf_data_to_json(cls, eeg_raws: List[mne.io.BaseRaw], stream_infos_df: pd.DataFrame, output_path: Path, include_raw_data: bool = True, max_samples_per_stream: Optional[int] = None, sample_interval: int = 1) -> Path:
        """
        Export loaded XDF data (MNE Raw objects and stream metadata) to a JSON file
        compatible with augmented-analytics.kanaries.net.
        
        Args:
            eeg_raws: List of MNE Raw objects from loaded XDF files
            stream_infos_df: DataFrame with stream metadata (from XDFDataStreamAccessor)
            output_path: Path where JSON file will be saved
            include_raw_data: Whether to include raw time-series data (default: True)
            max_samples_per_stream: Optional maximum number of samples per stream (for large files)
            sample_interval: Take every Nth sample (default: 1, meaning all samples)
            
        Returns:
            Path to the created JSON file
            
        Usage:
            from phopymnehelper.exporters.JSON_Exporter import JSONExporter
            from pathlib import Path
            
            output_path = Path("xdf_export.json")
            JSONExporter.export_xdf_data_to_json(
                eeg_raws=_out_eeg_raw,
                stream_infos_df=_out_xdf_stream_infos_df,
                output_path=output_path,
                include_raw_data=True,
                max_samples_per_stream=10000  # Optional: limit for large files
            )
        """
        if not eeg_raws:
            raise ValueError("eeg_raws list is empty")
        
        if stream_infos_df.empty:
            raise ValueError("stream_infos_df is empty")
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build top-level metadata
        export_date = datetime.now(timezone.utc)
        total_duration = sum([float(raw.times[-1]) if len(raw.times) > 0 else 0.0 for raw in eeg_raws])
        
        top_level_metadata = {
            'export_date': export_date.isoformat(),
            'num_streams': len(eeg_raws),
            'total_duration_seconds': total_duration,
        }
        
        # Process each stream
        streams_data = []
        
        for stream_idx, raw in enumerate(eeg_raws):
            try:
                # Get corresponding stream info row if available
                stream_info_row = None
                if 'xdf_dataset_idx' in stream_infos_df.columns:
                    # Try to match by xdf_dataset_idx
                    matching_rows = stream_infos_df[stream_infos_df['xdf_dataset_idx'] == stream_idx]
                    if not matching_rows.empty:
                        stream_info_row = matching_rows.iloc[0]
                elif stream_idx < len(stream_infos_df):
                    # Fallback to index-based matching
                    stream_info_row = stream_infos_df.iloc[stream_idx]
                
                # Extract metadata
                file_info = cls._extract_metadata_from_raw(raw, stream_info_row)
                
                # Extract channel info
                ch_names = raw.info.get('ch_names', [])
                ch_types = raw.get_channel_types()
                sfreq = raw.info.get('sfreq', None)
                
                channel_info = {
                    'channel_names': ch_names,
                    'channel_types': ch_types,
                    'sampling_rate': sfreq,
                }
                
                # Extract raw data
                data_dict = cls._extract_raw_data_from_raw(
                    raw,
                    include_raw_data=include_raw_data,
                    max_samples_per_stream=max_samples_per_stream,
                    sample_interval=sample_interval
                )
                
                # Build stream entry
                stream_entry = {
                    'stream_index': stream_idx,
                    'file_info': file_info,
                    'channel_info': channel_info,
                }
                
                if data_dict is not None:
                    stream_entry['data'] = data_dict
                
                streams_data.append(stream_entry)
                
                # Progress indicator
                if (stream_idx + 1) % 10 == 0:
                    print(f"Processed {stream_idx + 1}/{len(eeg_raws)} streams...")
                    
            except Exception as e:
                print(f"Warning: Failed to process stream {stream_idx}: {e}")
                # Continue with next stream
                continue
        ## END for stream_idx, raw in enumerate(eeg_raws)...
        
        # Build final JSON structure
        json_data = {
            'metadata': top_level_metadata,
            'streams': streams_data,
        }
        
        # Convert to JSON-serializable format
        json_data = cls._convert_to_json_serializable(json_data)
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Successfully exported {len(streams_data)} streams to {output_path}")
            return output_path
            
        except Exception as e:
            raise IOError(f"Failed to write JSON file: {e}")


    @classmethod
    def export_timeline_to_json(cls, timeline, output_path: Path, include_raw_data: bool = True, max_samples_per_stream: Optional[int] = None) -> Path:
        """
        Export timeline datasources to JSON format compatible with JSON_Exporter.
        
        This extracts data from the timeline's datasources (which contain processed
        DataFrames) rather than requiring the original MNE Raw objects.
        
        Args:
            timeline: SimpleTimelineWidget instance with loaded datasources
            output_path: Path where JSON file will be saved
            include_raw_data: Whether to include raw time-series data (default: True)
            max_samples_per_stream: Optional maximum number of samples per stream
            
        Returns:
            Path to the created JSON file
            
        Usage:
            from pathlib import Path
            from phopymnehelper.exporters.JSON_Exporter import JSONExporter
            
            # After timeline is built and all details are loaded
            output_path = Path("timeline_export.json")
            JSONExporter.export_timeline_to_json(timeline, output_path)

            # Export to JSON:
            JSONExporter.export_timeline_to_json(timeline, output_path, include_raw_data=True, max_samples_per_stream=10000)
        """
        if not hasattr(timeline, 'track_datasources'):
            raise ValueError("Timeline does not have track_datasources attribute")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect all EEG datasources
        eeg_datasources = []
        for track_name, datasource in timeline.track_datasources.items():
            # Check if this is an EEG datasource (has EEG channel data)
            if hasattr(datasource, 'detailed_df') and datasource.detailed_df is not None:
                detailed_df = datasource.detailed_df
                # Check if it has EEG-like channel columns
                eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 
                              'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
                has_eeg_channels = any(ch in detailed_df.columns for ch in eeg_channels)
                if has_eeg_channels and 't' in detailed_df.columns:
                    eeg_datasources.append((track_name, datasource))
        ## END for track_name, datasource in timeline.track_datasources.items()...
        
        if not eeg_datasources:
            raise ValueError("No EEG datasources found in timeline")
        
        # Build top-level metadata
        export_date = datetime.now(timezone.utc)
        total_duration = 0.0
        
        # Process each EEG datasource
        streams_data = []
        
        for stream_idx, (track_name, datasource) in enumerate(eeg_datasources):
            try:
                intervals_df = datasource.intervals_df
                detailed_df = datasource.detailed_df
                
                # Calculate duration from intervals
                if not intervals_df.empty and 't_duration' in intervals_df.columns:
                    stream_duration = intervals_df['t_duration'].sum()
                elif 't' in detailed_df.columns and len(detailed_df) > 1:
                    t_min = detailed_df['t'].min()
                    t_max = detailed_df['t'].max()
                    if isinstance(t_min, (datetime, pd.Timestamp)) and isinstance(t_max, (datetime, pd.Timestamp)):
                        stream_duration = (t_max - t_min).total_seconds()
                    else:
                        stream_duration = float(t_max - t_min)
                else:
                    stream_duration = 0.0
                
                total_duration += stream_duration
                
                # Extract metadata from intervals and detailed data
                file_info = {
                    'track_name': track_name,
                    'datasource_name': datasource.custom_datasource_name,
                }
                
                # Get recording datetime from intervals
                if not intervals_df.empty and 't_start' in intervals_df.columns:
                    t_start = intervals_df['t_start'].iloc[0]
                    if isinstance(t_start, (datetime, pd.Timestamp)):
                        file_info['recording_datetime'] = t_start
                    elif isinstance(t_start, (float, int)):
                        # Assume Unix timestamp
                        file_info['recording_datetime'] = datetime.fromtimestamp(t_start, tz=timezone.utc)
                
                # Extract channel information from detailed_df
                eeg_channels = [col for col in detailed_df.columns if col != 't']
                ch_types = ['eeg'] * len(eeg_channels)
                
                # Estimate sampling rate from time differences
                sfreq = None
                if 't' in detailed_df.columns and len(detailed_df) > 1:
                    time_diffs = detailed_df['t'].diff().dropna()
                    if len(time_diffs) > 0:
                        # Check if time_diffs contains Timedelta objects (from datetime diff)
                        first_diff = time_diffs.iloc[0]
                        if isinstance(first_diff, pd.Timedelta):
                            # Convert all Timedelta objects to seconds
                            avg_diff = time_diffs.apply(lambda x: x.total_seconds() if isinstance(x, pd.Timedelta) else float(x)).median()
                        elif isinstance(first_diff, (datetime, pd.Timestamp)):
                            # This shouldn't happen (diff of datetime gives Timedelta), but handle it
                            avg_diff = time_diffs.apply(lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)).median()
                        else:
                            # Numeric differences (already in seconds)
                            avg_diff = float(time_diffs.median())
                        if avg_diff > 0:
                            sfreq = 1.0 / avg_diff
                
                channel_info = {
                    'channel_names': eeg_channels,
                    'channel_types': ch_types,
                    'sampling_rate': sfreq,
                }
                
                # Extract raw data if requested
                data_dict = None
                if include_raw_data and detailed_df is not None and len(detailed_df) > 0:
                    # Prepare data for export
                    data_df = detailed_df.copy()
                    
                    # Convert 't' column to timestamps
                    if 't' in data_df.columns:
                        timestamps_iso = []
                        timestamps_relative = []
                        
                        t_col = data_df['t']
                        t_first = t_col.iloc[0]
                        
                        for t_val in t_col:
                            if isinstance(t_val, (datetime, pd.Timestamp)):
                                timestamps_iso.append(t_val.isoformat())
                                if isinstance(t_first, (datetime, pd.Timestamp)):
                                    rel_sec = (t_val - t_first).total_seconds()
                                else:
                                    rel_sec = 0.0
                                timestamps_relative.append(rel_sec)
                            elif isinstance(t_val, (float, int)):
                                # Assume relative seconds
                                timestamps_relative.append(float(t_val))
                                if isinstance(t_first, (datetime, pd.Timestamp)):
                                    abs_time = t_first + pd.Timedelta(seconds=float(t_val))
                                    timestamps_iso.append(abs_time.isoformat())
                                else:
                                    timestamps_iso.append(None)
                            else:
                                timestamps_iso.append(None)
                                timestamps_relative.append(0.0)
                        ## END for t_val in t_col:...
                        
                        # Apply max_samples limit if specified
                        if max_samples_per_stream is not None and len(data_df) > max_samples_per_stream:
                            indices = np.linspace(0, len(data_df) - 1, max_samples_per_stream, dtype=int)
                            data_df = data_df.iloc[indices].reset_index(drop=True)
                            timestamps_iso = [timestamps_iso[i] for i in indices]
                            timestamps_relative = [timestamps_relative[i] for i in indices]
                        
                        # Build channel data dictionary
                        channels_data = {}
                        for ch_name in eeg_channels:
                            if ch_name in data_df.columns:
                                channels_data[ch_name] = data_df[ch_name].tolist()
                        ## END for ch_name in eeg_channels:...
                        
                        data_dict = {
                            'timestamps_relative_seconds': timestamps_relative,
                            'channels': channels_data,
                        }
                        
                        if any(ts is not None for ts in timestamps_iso):
                            data_dict['timestamps'] = timestamps_iso
                
                # Build stream entry
                stream_entry = {
                    'stream_index': stream_idx,
                    'file_info': file_info,
                    'channel_info': channel_info,
                }
                
                if data_dict is not None:
                    stream_entry['data'] = data_dict
                
                streams_data.append(stream_entry)
                
            except Exception as e:
                print(f"Warning: Failed to process datasource '{track_name}': {e}")
                import traceback
                traceback.print_exc()
                continue
        ## END for stream_idx, (track_name, datasource) in enumerate(eeg_datasources):...
        
        # Build final JSON structure
        top_level_metadata = {
            'export_date': export_date.isoformat(),
            'num_streams': len(streams_data),
            'total_duration_seconds': total_duration,
        }
        
        json_data = {
            'metadata': top_level_metadata,
            'streams': streams_data,
        }
        
        # Convert to JSON-serializable format
        json_data = cls._convert_to_json_serializable(json_data)
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Successfully exported {len(streams_data)} streams from timeline to {output_path}")
            return output_path
            
        except Exception as e:
            raise IOError(f"Failed to write JSON file: {e}")


__all__ = ['JSONExporter']

