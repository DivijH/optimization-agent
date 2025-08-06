#!/usr/bin/env python3
"""
Real-time GCS logging handler for optimization agent.

This module provides a custom logging handler that uploads log entries to GCS in real-time
with intelligent buffering to balance responsiveness with efficiency.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from google.cloud import storage
from dataclasses import dataclass
import json


@dataclass
class GCSLoggingConfig:
    """Configuration for GCS logging handler."""
    bucket_name: str
    gcs_prefix: str
    project: str = 'etsy-search-ml-dev'
    buffer_size: int = 10  # Number of log entries to buffer before upload
    flush_interval: float = 30.0  # Seconds to wait before force-flushing buffer
    max_retries: int = 3
    retry_delay: float = 1.0


class RealtimeGCSLogHandler(logging.Handler):
    """
    Custom logging handler that uploads log entries to GCS in real-time.
    
    Features:
    - Buffers log entries to reduce API calls
    - Automatic periodic flushing
    - Async upload without blocking logging
    - Error recovery and retries
    - Thread-safe operation
    """
    
    def __init__(self, config: GCSLoggingConfig, local_log_file: Path):
        super().__init__()
        self.config = config
        self.local_log_file = local_log_file
        self.gcs_log_path = self._normalize_gcs_path(str(local_log_file))
        
        # Buffer for log entries
        self._buffer = []
        self._buffer_lock = threading.Lock()
        
        # GCS client
        self._gcs_client: Optional[storage.Client] = None
        self._bucket = None
        self._blob = None
        
        # Background upload task
        self._upload_task = None
        self._shutdown_event = threading.Event()
        self._last_flush_time = time.time()
        
        # Start background upload thread
        self._start_background_uploader()
    
    def _normalize_gcs_path(self, path: str) -> str:
        """Convert absolute local path to relative GCS path."""
        if not path:
            return path
        
        path_obj = Path(path)
        
        if path_obj.is_absolute():
            path_parts = path_obj.parts
            try:
                opt_agent_idx = path_parts.index('optimization-agent')
                relative_parts = path_parts[opt_agent_idx + 1:]
                
                if relative_parts and relative_parts[0] == 'src':
                    relative_parts = relative_parts[1:]
                
                relative_path = str(Path(*relative_parts)) if relative_parts else ""
                return relative_path
            except ValueError:
                return path_obj.name
        
        return str(path_obj)
    
    def _get_gcs_client(self) -> Optional[storage.Client]:
        """Initialize and return GCS client."""
        if self._gcs_client is None:
            try:
                self._gcs_client = storage.Client(project=self.config.project)
                self._bucket = self._gcs_client.bucket(self.config.bucket_name)
                self._blob = self._bucket.blob(f"{self.config.gcs_prefix}/{self.gcs_log_path}")
            except Exception as e:
                print(f"Failed to initialize GCS client: {e}")
                return None
        
        return self._gcs_client
    
    def emit(self, record: logging.LogRecord):
        """Handle a log record by adding it to buffer."""
        try:
            # Format the log record
            formatted_message = self.format(record)
            
            # Add to buffer
            with self._buffer_lock:
                self._buffer.append({
                    'timestamp': record.created,
                    'level': record.levelname,
                    'message': formatted_message,
                    'formatted_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
                })
                
                # Check if we should flush immediately
                should_flush = (
                    len(self._buffer) >= self.config.buffer_size or
                    time.time() - self._last_flush_time >= self.config.flush_interval
                )
            
            if should_flush:
                self._trigger_flush()
                
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Error in GCS log handler: {e}")
    
    def _trigger_flush(self):
        """Trigger an immediate flush of the buffer."""
        # This will be picked up by the background thread
        pass
    
    def _start_background_uploader(self):
        """Start the background thread that handles uploads."""
        def upload_worker():
            while not self._shutdown_event.is_set():
                try:
                    # Check if we need to flush
                    should_flush = False
                    buffer_size = 0
                    
                    with self._buffer_lock:
                        buffer_size = len(self._buffer)
                        should_flush = (
                            buffer_size >= self.config.buffer_size or
                            (buffer_size > 0 and time.time() - self._last_flush_time >= self.config.flush_interval)
                        )
                    
                    if should_flush:
                        self._flush_buffer()
                    
                    # Wait a bit before checking again
                    self._shutdown_event.wait(1.0)
                    
                except Exception as e:
                    print(f"Error in background uploader: {e}")
                    time.sleep(5.0)  # Wait longer on error
        
        self._upload_thread = threading.Thread(target=upload_worker, daemon=True)
        self._upload_thread.start()
    
    def _flush_buffer(self):
        """Flush the current buffer to GCS."""
        if not self._get_gcs_client():
            return
        
        # Get current buffer contents
        with self._buffer_lock:
            if not self._buffer:
                return
            
            buffer_copy = self._buffer.copy()
            self._buffer.clear()
            self._last_flush_time = time.time()
        
        # Upload to GCS
        try:
            # Create log content
            log_content = ""
            for entry in buffer_copy:
                log_content += f"{entry['formatted_time']} - {entry['level']} - {entry['message']}\n"
            
            # Read existing content from GCS if it exists
            existing_content = ""
            try:
                existing_content = self._blob.download_as_text()
            except Exception:
                # File doesn't exist yet or other error, start fresh
                pass
            
            # Append new content
            full_content = existing_content + log_content
            
            # Upload back to GCS
            self._blob.upload_from_string(full_content, content_type='text/plain')
            
            print(f"📤 Uploaded {len(buffer_copy)} log entries to GCS: gs://{self.config.bucket_name}/{self.config.gcs_prefix}/{self.gcs_log_path}")
            
        except Exception as e:
            print(f"❌ Failed to upload logs to GCS: {e}")
            # Put the entries back in the buffer for retry
            with self._buffer_lock:
                self._buffer = buffer_copy + self._buffer
    
    def flush(self):
        """Flush any buffered log entries immediately."""
        self._flush_buffer()
    
    def close(self):
        """Close the handler and upload any remaining log entries."""
        # Signal shutdown
        self._shutdown_event.set()
        
        # Final flush
        self._flush_buffer()
        
        # Wait for background thread to finish
        if hasattr(self, '_upload_thread'):
            self._upload_thread.join(timeout=10.0)
        
        super().close()


def create_realtime_gcs_logger(
    logger_name: str,
    local_log_file: Path,
    gcs_config: GCSLoggingConfig,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Create a logger with both local file and real-time GCS handlers.
    
    Args:
        logger_name: Name for the logger
        local_log_file: Path to local log file
        gcs_config: GCS configuration
        log_level: Logging level
    
    Returns:
        Configured logger with both local and GCS handlers
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create local file handler
    local_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(local_log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create real-time GCS handler
    gcs_handler = RealtimeGCSLogHandler(gcs_config, local_log_file)
    gcs_handler.setLevel(log_level)
    gcs_handler.setFormatter(formatter)
    logger.addHandler(gcs_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Context manager for proper cleanup
class RealtimeGCSLogger:
    """Context manager for real-time GCS logging."""
    
    def __init__(self, logger_name: str, local_log_file: Path, gcs_config: GCSLoggingConfig):
        self.logger_name = logger_name
        self.local_log_file = local_log_file
        self.gcs_config = gcs_config
        self.logger = None
    
    def __enter__(self):
        self.logger = create_realtime_gcs_logger(
            self.logger_name,
            self.local_log_file,
            self.gcs_config
        )
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            # Ensure all handlers are properly closed
            for handler in self.logger.handlers:
                handler.close()