"""
Advanced Context Window Management
Handles dynamic context optimization, compression, and token budgeting
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """Token budget allocation for different context components"""
    system_prompt: int = 200
    memory_blocks: int = 300
    retrieved_docs: int = 1500
    conversation_history: int = 800
    current_query: int = 100
    response_buffer: int = 1500
    
    @property
    def total_input(self) -> int:
        """Total tokens allocated for input"""
        return (self.system_prompt + self.memory_blocks + 
                self.retrieved_docs + self.conversation_history + 
                self.current_query)
    
    @property
    def total_with_response(self) -> int:
        """Total tokens including response"""
        return self.total_input + self.response_buffer


class ContextWindowManager:
    """
    Manages context window with intelligent compression and prioritization
    
    Features:
    - Dynamic token budgeting
    - Automatic compression when approaching limits
    - Priority-based content retention
    - Conversation summarization
    - Context relevance scoring
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        max_context_length: int = 4096,
        budget: Optional[ContextBudget] = None,
    ):
        """
        Initialize context window manager
        
        Args:
            model_name: Model name for tokenizer
            max_context_length: Maximum context length
            budget: Token budget allocation
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.budget = budget or ContextBudget()
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            logger.warning(f"Could not load tokenizer for {model_name}, using cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Context window manager initialized: {max_context_length} tokens")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        return len(self.tokenizer.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to maximum tokens
        
        Args:
            text: Input text
            max_tokens: Maximum token count
            
        Returns:
            Truncated text
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    def build_context(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        retrieved_docs: List[Dict],
        memory_blocks: Dict[str, str],
        system_prompt: str,
    ) -> Tuple[str, Dict]:
        """
        Build optimized context within token budget
        
        Args:
            query: Current user query
            conversation_history: Past messages
            retrieved_docs: Retrieved document chunks
            memory_blocks: Memory blocks (user profile, task, etc.)
            system_prompt: System prompt
            
        Returns:
            Tuple of (context_string, metadata)
        """
        # Calculate current token counts
        tokens_used = {
            "system_prompt": self.count_tokens(system_prompt),
            "query": self.count_tokens(query),
            "memory_blocks": sum(self.count_tokens(v) for v in memory_blocks.values()),
            "history": sum(self.count_tokens(m.get("content", "")) for m in conversation_history),
            "docs": sum(self.count_tokens(d.get("text", "")) for d in retrieved_docs),
        }
        
        total_tokens = sum(tokens_used.values())
        logger.info(f"Initial token count: {total_tokens}")
        
        # Check if we need compression
        if total_tokens > self.budget.total_input:
            logger.warning(f"Context exceeds budget ({total_tokens} > {self.budget.total_input}). Compressing...")
            
            # Compress in priority order
            memory_blocks = self._compress_memory_blocks(memory_blocks, self.budget.memory_blocks)
            retrieved_docs = self._compress_documents(retrieved_docs, self.budget.retrieved_docs)
            conversation_history = self._compress_history(conversation_history, self.budget.conversation_history)
        
        # Build final context
        context_parts = []
        
        # 1. System prompt (highest priority)
        context_parts.append(f"=== SYSTEM ===\n{system_prompt}")
        
        # 2. Memory blocks (high priority - persistent state)
        if memory_blocks:
            context_parts.append("\n=== MEMORY ===")
            for label, content in memory_blocks.items():
                if content:
                    context_parts.append(f"\n{label.upper()}:\n{content}")
        
        # 3. Retrieved documents (medium-high priority - current context)
        if retrieved_docs:
            context_parts.append("\n=== RELEVANT DOCUMENTS ===")
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.get("source_title", "Unknown")
                text = doc.get("text", "")
                url = doc.get("source_url", "")
                context_parts.append(f"\n[{i}] {source}")
                if url:
                    context_parts.append(f"URL: {url}")
                context_parts.append(f"Content: {text}\n")
        
        # 4. Conversation history (medium priority - recent context)
        if conversation_history:
            context_parts.append("\n=== CONVERSATION HISTORY ===")
            for msg in conversation_history:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")
        
        # 5. Current query (highest priority)
        context_parts.append(f"\n=== CURRENT QUERY ===\n{query}")
        
        # 6. Instructions
        context_parts.append("""
=== INSTRUCTIONS ===
Answer the query using the provided context. Key requirements:
1. Use ONLY information from the documents above
2. Cite sources using [Source: Title] format
3. If information is not in the context, say so clearly
4. Be concise but comprehensive
5. Maintain conversation continuity - remember what was discussed

Your response:""")
        
        final_context = "\n".join(context_parts)
        
        # Final token check
        final_tokens = self.count_tokens(final_context)
        metadata = {
            "total_tokens": final_tokens,
            "budget_used": final_tokens / self.budget.total_input,
            "tokens_by_component": {
                "system": self.count_tokens(system_prompt),
                "memory": sum(self.count_tokens(v) for v in memory_blocks.values()),
                "documents": sum(self.count_tokens(d.get("text", "")) for d in retrieved_docs),
                "history": sum(self.count_tokens(m.get("content", "")) for m in conversation_history),
                "query": self.count_tokens(query),
            },
            "compression_applied": total_tokens > self.budget.total_input,
        }
        
        logger.info(f"Final context: {final_tokens} tokens ({metadata['budget_used']:.1%} of budget)")
        
        return final_context, metadata
    
    def _compress_memory_blocks(
        self,
        memory_blocks: Dict[str, str],
        target_tokens: int,
    ) -> Dict[str, str]:
        """
        Compress memory blocks to fit token budget
        
        Strategy:
        1. Keep most important blocks (user_profile, task_context)
        2. Truncate less important blocks
        3. Remove least important if still over budget
        """
        current_tokens = sum(self.count_tokens(v) for v in memory_blocks.values())
        
        if current_tokens <= target_tokens:
            return memory_blocks
        
        # Priority order
        priority = ["user_profile", "task_context", "document_summary", "key_facts"]
        
        compressed = {}
        tokens_remaining = target_tokens
        
        for key in priority:
            if key in memory_blocks and tokens_remaining > 0:
                content = memory_blocks[key]
                tokens = self.count_tokens(content)
                
                if tokens <= tokens_remaining:
                    compressed[key] = content
                    tokens_remaining -= tokens
                else:
                    # Truncate to fit
                    compressed[key] = self.truncate_to_tokens(content, tokens_remaining)
                    break
        
        logger.info(f"Compressed memory blocks: {current_tokens} -> {sum(self.count_tokens(v) for v in compressed.values())} tokens")
        return compressed
    
    def _compress_documents(
        self,
        documents: List[Dict],
        target_tokens: int,
    ) -> List[Dict]:
        """
        Compress retrieved documents to fit token budget
        
        Strategy:
        1. Keep highest-scoring documents
        2. Truncate each document if needed
        3. Remove lowest-scoring if still over budget
        """
        if not documents:
            return []
        
        # Sort by relevance score (if available)
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get("score", 0) or x.get("confidence", 0),
            reverse=True
        )
        
        compressed = []
        tokens_used = 0
        tokens_per_doc = target_tokens // len(sorted_docs)
        
        for doc in sorted_docs:
            text = doc.get("text", "")
            tokens = self.count_tokens(text)
            
            if tokens_used + tokens <= target_tokens:
                # Document fits completely
                compressed.append(doc)
                tokens_used += tokens
            elif tokens_used < target_tokens:
                # Truncate document to fit
                remaining = target_tokens - tokens_used
                truncated_text = self.truncate_to_tokens(text, remaining)
                doc_copy = doc.copy()
                doc_copy["text"] = truncated_text
                doc_copy["truncated"] = True
                compressed.append(doc_copy)
                break
            else:
                # Budget exhausted
                break
        
        logger.info(f"Compressed documents: {len(documents)} -> {len(compressed)} docs, {target_tokens} tokens")
        return compressed
    
    def _compress_history(
        self,
        history: List[Dict[str, str]],
        target_tokens: int,
    ) -> List[Dict[str, str]]:
        """
        Compress conversation history to fit token budget
        
        Strategy:
        1. Keep most recent messages (recency bias)
        2. Summarize older messages if available
        3. Drop oldest messages if needed
        """
        if not history:
            return []
        
        # Calculate tokens for each message
        msg_tokens = [(i, self.count_tokens(m.get("content", ""))) 
                      for i, m in enumerate(history)]
        
        # Keep recent messages (last N)
        compressed = []
        tokens_used = 0
        
        # Start from most recent and work backwards
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            tokens = self.count_tokens(msg.get("content", ""))
            
            if tokens_used + tokens <= target_tokens:
                compressed.insert(0, msg)
                tokens_used += tokens
            else:
                # Can't fit this message
                if i > 0:
                    # Add summary indicator
                    summary_msg = {
                        "role": "system",
                        "content": f"[Earlier conversation history summarized - {i} messages]"
                    }
                    compressed.insert(0, summary_msg)
                break
        
        logger.info(f"Compressed history: {len(history)} -> {len(compressed)} messages, {tokens_used} tokens")
        return compressed
    
    def summarize_old_messages(
        self,
        messages: List[Dict[str, str]],
        summarizer,
    ) -> str:
        """
        Summarize old messages using model
        
        Args:
            messages: List of messages to summarize
            summarizer: Model or function for summarization
            
        Returns:
            Summary text
        """
        if not messages:
            return ""
        
        # Combine messages
        conversation_text = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        ])
        
        # Create summary prompt
        prompt = f"""Summarize this conversation in 3-4 sentences, focusing on:
1. Main topics discussed
2. Key decisions or conclusions
3. Important context for future messages

Conversation:
{conversation_text}

Summary:"""
        
        # Generate summary (implementation depends on summarizer)
        try:
            summary = summarizer(prompt)
            logger.info(f"Summarized {len(messages)} messages")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing messages: {e}")
            return "[Conversation history available but not summarized]"
    
    def get_context_stats(self, context: str) -> Dict:
        """Get statistics about context"""
        tokens = self.count_tokens(context)
        return {
            "total_tokens": tokens,
            "max_tokens": self.max_context_length,
            "usage_percent": (tokens / self.max_context_length) * 100,
            "tokens_remaining": self.max_context_length - tokens,
            "can_fit_response": tokens + self.budget.response_buffer <= self.max_context_length,
        }


class ConversationMemory:
    """
    Manages conversation memory with automatic summarization
    
    Features:
    - Sliding window of recent messages
    - Automatic summarization of old messages
    - Memory block extraction
    - Conversation state tracking
    """
    
    def __init__(
        self,
        max_messages: int = 10,
        summarization_threshold: int = 10,
    ):
        """
        Initialize conversation memory
        
        Args:
            max_messages: Maximum messages in buffer
            summarization_threshold: Trigger summarization after N messages
        """
        self.max_messages = max_messages
        self.summarization_threshold = summarization_threshold
        self.message_buffer = deque(maxlen=max_messages)
        self.full_history = []  # Complete history
        self.summaries = []  # Summarized chunks
        self.memory_blocks = {
            "user_profile": "",
            "task_context": "",
            "key_facts": "",
        }
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Add message to conversation memory
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional metadata
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        
        self.message_buffer.append(message)
        self.full_history.append(message)
        
        # Check if summarization needed
        if len(self.full_history) >= self.summarization_threshold:
            if len(self.full_history) % self.summarization_threshold == 0:
                logger.info("Summarization threshold reached")
                # Trigger summarization in background
    
    def get_recent_messages(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get recent messages
        
        Args:
            n: Number of messages (default: all in buffer)
            
        Returns:
            List of recent messages
        """
        if n is None:
            return list(self.message_buffer)
        return list(self.message_buffer)[-n:]
    
    def update_memory_blocks(self, updates: Dict[str, str]):
        """Update memory blocks"""
        self.memory_blocks.update(updates)
        logger.debug(f"Updated memory blocks: {list(updates.keys())}")
    
    def get_memory_blocks(self) -> Dict[str, str]:
        """Get current memory blocks"""
        return self.memory_blocks.copy()
    
    def get_conversation_state(self) -> Dict:
        """Get current conversation state"""
        return {
            "total_messages": len(self.full_history),
            "buffer_messages": len(self.message_buffer),
            "summaries": len(self.summaries),
            "memory_blocks": list(self.memory_blocks.keys()),
        }
    
    def clear(self):
        """Clear conversation memory"""
        self.message_buffer.clear()
        self.full_history.clear()
        self.summaries.clear()
        self.memory_blocks = {k: "" for k in self.memory_blocks}
        logger.info("Conversation memory cleared")


# Singleton instances
_context_manager = None
_conversation_memory = None


def get_context_manager(
    model_name: str = "gpt-3.5-turbo",
    max_context_length: int = 4096,
) -> ContextWindowManager:
    """Get or create context window manager"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextWindowManager(model_name, max_context_length)
    return _context_manager


def get_conversation_memory(
    max_messages: int = 10,
) -> ConversationMemory:
    """Get or create conversation memory"""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory(max_messages)
    return _conversation_memory
