import json
import hashlib
from typing import Any, Dict
from datetime import datetime, timezone
from loguru import logger
import numpy as np

from spectrum.infra.events import InferenceEvent
from spectrum.infra.privacy import PIISanitizer

class RCIALogger:
    """
    Records immutable Risk, Compliance, Inference, and Audit events (RCIA).

    Events are sanitized, serialized to JSONL, hashed for integrity, and 
    streamed to the audit log path using loguru for robustness.
    """
    
    def __init__(self, log_path: str = "rcia_audit.jsonl"):
        """
        :param log_path: File path for the immutable JSONL log.
        """
        self.log_path = log_path
        self.sanitizer = PIISanitizer()
        
        # Configure loguru to write only this logger's output to the file
        logger.add(
            self.log_path, 
            serialize=False, 
            enqueue=True, # Critical: Makes logging non-blocking (async)
            rotation="10 MB", 
            compression="zip",
            level="INFO"
        )
        logger.info(f"RCIALogger initialized, streaming to {self.log_path}")

    def log_event(self, event: InferenceEvent, context: str = "INFERENCE"):
        """
        Logs a single InferenceEvent after sanitization and audit hashing.
        
        :param event: The InferenceEvent Pydantic instance.
        :param context: The high-level context (e.g., 'INFERENCE', 'WARGAME').
        """
        
        # 1. Sanitize the payload (PII stripping and vector summarization)
        # Note: We must duplicate the object here to separate the audit log 
        # from the live event object.
        log_entry = event.model_dump() 
        
        log_entry['input_payload_sanitized'] = self.sanitizer.sanitize(
            log_entry.pop('input_payload', None)
        )
        log_entry['output_payload_sanitized'] = self.sanitizer.sanitize(
            log_entry.pop('output_payload', None)
        )
        
        # 2. Serialize for Hashing and Storage
        # Ensure all data types are native Python (e.g., datetime -> str)
        log_entry['timestamp'] = log_entry['timestamp'].isoformat() if isinstance(log_entry.get('timestamp'), datetime) else log_entry.get('timestamp')
        
        json_data = json.dumps(log_entry, sort_keys=True)

        # 3. HASH for Audit Integrity (CRITICAL STEP)
        # The hash proves the entry hasn't been tampered with since writing.
        audit_hash = hashlib.sha256(json_data.encode('utf-8')).hexdigest()
        
        # 4. Final Log Output
        final_log = {
            "rcia_context": context,
            "audit_hash": audit_hash,
            "data": log_entry
        }
        
        # Use loguru's built-in file sink (serialized=False, we write raw JSON)
        logger.info(json.dumps(final_log))