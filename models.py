"""
Data models for the Multi-Agent RAG System
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


@dataclass
class Citation:
    """Represents a citation with exact source information"""
    claim: str  # The fact/claim being cited
    source_url: str  # Original source URL
    source_title: str  # Title of the source
    exact_text: str  # Exact quote from source
    char_start: int  # Character position in source
    char_end: int  # End position in source
    context: str  # Surrounding context
    chunk_id: int  # ID of the chunk in vector DB
    confidence: float  # Confidence/relevance score
    highlight_score: Optional[float] = None  # Exa highlight score
    
    def format_inline(self) -> str:
        """Format as inline citation"""
        return f"[{self.source_title}]({self.source_url})"
    
    def format_detailed(self) -> str:
        """Format with full details"""
        return f"""
Source: {self.source_title}
URL: {self.source_url}
Quote: "{self.exact_text}"
Location: Characters {self.char_start}-{self.char_end}
Confidence: {self.confidence:.2f}
        """.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "claim": self.claim,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "exact_text": self.exact_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "context": self.context,
            "chunk_id": self.chunk_id,
            "confidence": self.confidence,
            "highlight_score": self.highlight_score,
        }


@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    chunk_id: int
    text: str
    source_url: str
    source_title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Represents a search result from Exa or other sources"""
    url: str
    title: str
    text: str
    score: float
    highlights: List[str] = field(default_factory=list)
    author: Optional[str] = None
    published_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "highlights": self.highlights,
            "author": self.author,
            "published_date": self.published_date,
        }


@dataclass
class Message:
    """Represents a message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[Citation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "citations": [c.to_dict() for c in self.citations],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary"""
        citations = [Citation(**c) for c in data.get("citations", [])]
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            citations=citations,
        )


@dataclass
class MemoryBlock:
    """Represents a memory block in the system"""
    label: str  # e.g., "user_profile", "task_context"
    content: str
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "label": self.label,
            "content": self.content,
            "last_updated": self.last_updated.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """Create from dictionary"""
        return cls(
            label=data["label"],
            content=data["content"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )


@dataclass
class Session:
    """Represents a conversation session"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)
    memory_blocks: Dict[str, MemoryBlock] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message):
        """Add a message to the session"""
        self.messages.append(message)
        self.last_updated = datetime.now()
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get the n most recent messages"""
        return self.messages[-n:]
    
    def update_memory_block(self, label: str, content: str):
        """Update or create a memory block"""
        self.memory_blocks[label] = MemoryBlock(label=label, content=content)
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "memory_blocks": {k: v.to_dict() for k, v in self.memory_blocks.items()},
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create from dictionary"""
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        memory_blocks = {
            k: MemoryBlock.from_dict(v) 
            for k, v in data.get("memory_blocks", {}).items()
        }
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            messages=messages,
            memory_blocks=memory_blocks,
            metadata=data.get("metadata", {}),
        )
    
    def save(self, filepath: str):
        """Save session to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Session':
        """Load session from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_name: str
    content: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_name": self.agent_name,
            "content": self.content,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class QueryAnalysis:
    """Analysis of a user query"""
    original_query: str
    intent: str  # "question", "search", "summarize", etc.
    requires_search: bool
    requires_retrieval: bool
    requires_summarization: bool
    extracted_keywords: List[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "requires_search": self.requires_search,
            "requires_retrieval": self.requires_retrieval,
            "requires_summarization": self.requires_summarization,
            "extracted_keywords": self.extracted_keywords,
            "confidence": self.confidence,
        }
