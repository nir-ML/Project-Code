"""
Base Module - Abstract base class for all modules
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class PipelineStatus(Enum):
    """Module execution status"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class Message:
    """Message passed between modules"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str = "request"  # request, response, error
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Result:
    """Result returned by an module"""
    module_name: str
    status: PipelineStatus
    data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModule(ABC):
    """
    Abstract base class for all modules in the system.
    
    Each module must implement:
    - process(): Main processing logic
    - validate_input(): Input validation
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = PipelineStatus.IDLE
        self.message_history: List[Message] = []
        self._initialized = False
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Result:
        """Main processing method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data - must be implemented by subclasses"""
        pass
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the module with necessary resources"""
        self._initialized = True
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Result:
        """
        Execute the module's task with timing and error handling.
        This is the main entry point called by the orchestrator.
        """
        start_time = time.time()
        self.status = PipelineStatus.RUNNING
        
        try:
            # Validate input
            is_valid, error_msg = self.validate_input(input_data)
            if not is_valid:
                self.status = PipelineStatus.FAILED
                return Result(
                    module_name=self.name,
                    status=PipelineStatus.FAILED,
                    data={},
                    errors=[f"Input validation failed: {error_msg}"],
                    execution_time=time.time() - start_time
                )
            
            # Process
            result = self.process(input_data)
            result.execution_time = time.time() - start_time
            
            self.status = result.status
            return result
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            return Result(
                module_name=self.name,
                status=PipelineStatus.FAILED,
                data={},
                errors=[f"Execution error: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    def send_message(self, receiver: str, content: Dict[str, Any], 
                     message_type: str = "request") -> Message:
        """Create and log a message to another module"""
        message = Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type
        )
        self.message_history.append(message)
        return message
    
    def receive_message(self, message: Message):
        """Receive and log a message from another module"""
        self.message_history.append(message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status"""
        return {
            "name": self.name,
            "status": self.status.value,
            "initialized": self._initialized,
            "message_count": len(self.message_history)
        }
    
    def reset(self):
        """Reset module state"""
        self.status = PipelineStatus.IDLE
        self.message_history = []
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', status={self.status.value})"
