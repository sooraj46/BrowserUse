import asyncio

class AgentState:
    _instance = None

    def __init__(self):
        if not hasattr(self, '_stop_requested'):
            self._stop_requested = asyncio.Event()
            self.last_valid_state = None  # store the last valid browser state
        
        if not hasattr(self, '_pause_requested'):
            self._pause_requested = asyncio.Event()
        if not hasattr(self, 'pending_chat_messages'):
           self.pending_chat_messages = []  # list to store mid–task chat messages

    def add_chat_message(self, message: str):
        """Add a chat message to be included mid–task."""
        self.pending_chat_messages.append(message)

    def get_pending_chat_messages(self) -> list:
        """Retrieve and clear all pending chat messages."""
        messages = self.pending_chat_messages.copy()
        self.pending_chat_messages.clear()
        return messages


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentState, cls).__new__(cls)
        return cls._instance

    def request_stop(self):
        self._stop_requested.set()

    def clear_stop(self):
        self._stop_requested.clear()
        self.last_valid_state = None

    def is_stop_requested(self):
        return self._stop_requested.is_set()
   
    def request_pause(self):
        """Set the 'pause_requested' event to True."""
        self._pause_requested.set()

    def clear_pause(self):
        """Clear the 'pause_requested' event (resume)."""
        self._pause_requested.clear()

    def is_pause_requested(self):
        """Check if the agent is paused."""
        return self._pause_requested.is_set()

    def set_last_valid_state(self, state):
        self.last_valid_state = state

    def get_last_valid_state(self):
        return self.last_valid_state