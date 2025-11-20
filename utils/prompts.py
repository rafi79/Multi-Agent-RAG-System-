"""
Prompt templates for different agents and tasks
"""
from typing import List, Dict
from config import SYSTEM_PROMPTS, TASK_PROMPTS


class PromptBuilder:
    """Build prompts for different agents and tasks"""
    
    @staticmethod
    def build_master_agent_prompt(
        query: str,
        context: str = "",
        memory_blocks: Dict[str, str] = None,
        conversation_history: List[Dict] = None,
    ) -> str:
        """
        Build prompt for master/orchestrator agent
        
        Args:
            query: User query
            context: Retrieved document context
            memory_blocks: Dictionary of memory blocks
            conversation_history: Recent conversation messages
            
        Returns:
            Formatted prompt
        """
        prompt_parts = [SYSTEM_PROMPTS["master_agent"]]
        
        # Add memory blocks if available
        if memory_blocks:
            prompt_parts.append("\n=== MEMORY BLOCKS ===")
            for label, content in memory_blocks.items():
                if content:
                    prompt_parts.append(f"\n{label.upper()}:")
                    prompt_parts.append(content)
        
        # Add context if available
        if context:
            prompt_parts.append("\n=== RELEVANT DOCUMENTS ===")
            prompt_parts.append(context)
        
        # Add conversation history
        if conversation_history:
            prompt_parts.append("\n=== CONVERSATION HISTORY ===")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"\n{role.upper()}: {content}")
        
        # Add current query
        prompt_parts.append(f"\n=== CURRENT QUERY ===\n{query}")
        
        # Add instructions
        prompt_parts.append("""
=== INSTRUCTIONS ===
1. Answer the query using ONLY the provided context
2. If the context doesn't contain the answer, say so clearly
3. Cite sources using [Source: Title] format
4. Be concise but comprehensive
5. If you make any claims, back them up with citations

Your response:""")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_answer_with_context_prompt(
        question: str,
        context: str,
    ) -> str:
        """
        Build prompt for answering questions with context
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        return TASK_PROMPTS["answer_with_context"].format(
            context=context,
            question=question,
        )
    
    @staticmethod
    def build_summarization_prompt(
        document: str,
        max_length: int = 200,
    ) -> str:
        """
        Build prompt for document summarization
        
        Args:
            document: Document to summarize
            max_length: Maximum summary length in words
            
        Returns:
            Formatted prompt
        """
        base_prompt = TASK_PROMPTS["summarize_document"].format(
            document=document
        )
        return f"{base_prompt}\n\nProvide a summary in approximately {max_length} words."
    
    @staticmethod
    def build_claim_extraction_prompt(text: str) -> str:
        """
        Build prompt for extracting claims from text
        
        Args:
            text: Text to extract claims from
            
        Returns:
            Formatted prompt
        """
        return TASK_PROMPTS["extract_claims"].format(text=text)
    
    @staticmethod
    def build_query_analysis_prompt(query: str) -> str:
        """
        Build prompt for analyzing user query
        
        Args:
            query: User query
            
        Returns:
            Formatted prompt
        """
        return f"""Analyze the following user query and determine:
1. Intent (question, search, summarize, chat, etc.)
2. Whether it requires web search
3. Whether it requires document retrieval
4. Whether it requires summarization
5. Extract key keywords

Query: {query}

Provide your analysis in JSON format:
{{
    "intent": "...",
    "requires_search": true/false,
    "requires_retrieval": true/false,
    "requires_summarization": true/false,
    "keywords": ["...", "..."]
}}"""
    
    @staticmethod
    def build_context_string(chunks: List[Dict], max_chunks: int = 5) -> str:
        """
        Build context string from document chunks
        
        Args:
            chunks: List of document chunks with metadata
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = chunk.get("text", "")
            source = chunk.get("source_title", "Unknown")
            url = chunk.get("source_url", "")
            
            context_parts.append(f"""
--- Source {i}: {source} ---
URL: {url}
Content: {text}
""")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def build_conversation_history_string(messages: List[Dict]) -> str:
        """
        Build conversation history string
        
        Args:
            messages: List of messages
            
        Returns:
            Formatted history string
        """
        history_parts = []
        
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    @staticmethod
    def build_multimodal_message(
        text: str,
        images: List[str] = None,
    ) -> List[Dict]:
        """
        Build multimodal message for Qwen2.5-VL
        
        Args:
            text: Text prompt
            images: List of image URLs/paths
            
        Returns:
            Message list in Qwen format
        """
        content = []
        
        # Add images first if any
        if images:
            for img in images:
                content.append({"type": "image", "url": img})
        
        # Add text
        content.append({"type": "text", "text": text})
        
        return [{
            "role": "user",
            "content": content
        }]
    
    @staticmethod
    def format_citations(citations: List[Dict]) -> str:
        """
        Format citations for display
        
        Args:
            citations: List of citation dictionaries
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        citation_parts = ["\n\n=== SOURCES ==="]
        
        for i, citation in enumerate(citations, 1):
            title = citation.get("source_title", "Unknown")
            url = citation.get("source_url", "")
            exact_text = citation.get("exact_text", "")
            
            citation_parts.append(f"""
[{i}] {title}
    URL: {url}
    Quote: "{exact_text[:100]}..."
""")
        
        return "\n".join(citation_parts)
