from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

def create_summary_prompt() -> PromptTemplate:
    """Create prompt template for video summarization."""
    template = """
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "Timestamp" indicates the timestamp when that part begins in the video.

    VIDEO CONTEXT:
    {transcript}

    Summary:
    """

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )

    return prompt


def create_answer_prompt() -> ChatPromptTemplate:
    """Create chat prompt template for answering a question from video context."""
    return ChatPromptTemplate.from_messages([
        ("system", 
        """You are an AI assistant tasked with providing answers based on video content.

        ANSWER PRIORITY:
        1. First, check the provided video context.
        2. If the answer is in the context:
            - Extract the actual definition or explanation using wording as close as possible to the transcript
            - Include any examples given by the speaker
            - DEDUPLICATE: Mention each concept only once
            - ALWAYS cite the single most relevant timestamp from the context in your answer, in the format (Timestamp: HH:MM:SS), so it stays available if the user later asks where this came from
        3. If NOT in the context:
            - Say: "The video does not mention this."
        4. THEN, only if you are certain the general knowledge answer is accurate, add:
            - "However, [general knowledge answer]"
            - If you are not certain it is accurate, do not guess — omit this step entirely.
        5. NEVER fabricate timestamps or claim something is in the video when it is not.


        Your responses should be:
            - Specific and detailed (extract actual definitions and examples from context)
            - Clear and concise
            - Focused solely on answering the user's question

        Note: In the transcript, "Text" refers to the spoken words in the video, "Timestamp" indicates when that part begins in the video."""),

        MessagesPlaceholder(variable_name="conversation_history"),
        
        ("user", """VIDEO CONTEXT:
        ============================================
        {context}
        ============================================

        QUESTION: {question}

        ANSWER:"""),
        ])

def create_chat_prompt() -> ChatPromptTemplate:
    """Create chat prompt template for questions that don't need the video transcript."""
    return ChatPromptTemplate.from_messages([
        ("system",
        """You are a friendly AI assistant for a YouTube video Q&A tool.

        This question does not require a new video transcript lookup (e.g. a greeting, small
        talk, or a question about the conversation itself, such as asking for a timestamp or
        detail a previous answer already cited). Respond naturally and warmly using the
        conversation history for context. If the user is asking about something already stated
        in a previous answer (e.g. "what timestamp was that?"), answer directly from that prior
        answer's text instead of saying you don't have it. Only if the conversation history
        truly has nothing relevant, gently encourage the user to ask a question about the
        video."""),

        MessagesPlaceholder(variable_name="conversation_history"),

        ("user", "{question}"),
        ])


def create_router_prompt() -> PromptTemplate:
    router_prompt = """
    You are a routing AI assistant. Decide if this question needs the video transcript.

    Question: {question}
    Conversation history: {conversation_history}

    RESPOND WITH ONLY THIS JSON, NO EXPLANATION:
    {{
    "needs_transcript": true/false,
    "confidence": 0.0-1.0
    }}

    Rules:
    - needs_transcript=true if the question references the video/speaker/content, or asks
      the assistant to summarize, explain, or recall something that was said.
    - needs_transcript=false if the question is a greeting, small talk, a general-knowledge
      question unrelated to the video, or a question about something already stated earlier
      in the conversation history (e.g. asking for a timestamp, source, or detail that a
      previous answer already gave) — that can be answered from history alone, without a new
      transcript lookup.
    - confidence: your confidence (0.0-1.0) in the needs_transcript decision.

    Examples:
    - "Hi" → {{"needs_transcript": false, "confidence": 0.95}}
    - "What's up?" → {{"needs_transcript": false, "confidence": 0.9}}
    - "What's the capital of France?" → {{"needs_transcript": false, "confidence": 0.9}}
    - "What does the speaker say about AI?" → {{"needs_transcript": true, "confidence": 0.95}}
    - "Based on the video, what are hidden layers?" → {{"needs_transcript": true, "confidence": 0.95}}
    - "What is the first message i sent you?" → {{"needs_transcript": false, "confidence": 0.9}}
    - "What timestamp was that from?" (referring to your previous answer) → {{"needs_transcript": false, "confidence": 0.9}}
    - "Where did you get that from?" (referring to your previous answer) → {{"needs_transcript": false, "confidence": 0.9}}

    """

    prompt_template = PromptTemplate(
        input_variables = ["question", "conversation_history"],
        template = router_prompt
    )

    return prompt_template

def create_revision_prompt() -> PromptTemplate:
    """Create prompt template for revising an answer based on evaluator feedback."""
    template = """
    You are revising an answer to a question about a YouTube video.

    This question was routed as: {needs_transcript}

    If routed as "retrieve" (the question needed the video transcript):
        - Use only the provided video context.
        - Do not fabricate timestamps.
    If routed as "generate" (the question did NOT need the video transcript - e.g. a greeting
    or general-knowledge question):
        - Ignore the video context entirely and do not reference it.
        - Revise for tone, clarity, and engagement instead.

    Do not mention that you are revising.
    Do not mention evaluator feedback.

    VIDEO CONTEXT:
    {context}

    QUESTION:
    {question}

    PREVIOUS ANSWER:
    {previous_answer}

    EVALUATOR FEEDBACK:
    {feedback}

    REVISED ANSWER:
    """

    return PromptTemplate(
        input_variables=["context", "question", "previous_answer", "feedback", "needs_transcript"],
        template=template,
    )


def create_queries_prompt() -> PromptTemplate:
    multiqueries_template = """
    Generate 3 alternative phrasings of the query.

    QUERY: "{query}"

    RESPOND WITH ONLY JSON FORMAT, NO EXPLANATION:
    {{"queries": ["alternative 1", "alternative 2", "alternative 3"]}}
    """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template = multiqueries_template
    )

    return prompt_template

def create_eval_prompt()-> PromptTemplate:
    eval_prompt="""
        You are an evaluator judging the quality of an AI response about a YouTube video.

        This question was routed as: {needs_transcript}

        If routed as "retrieve" (the question needed the video transcript), evaluate based on:
        1. Correctness (is it factually correct based on context?)
        2. Relevance (does it answer the question?)
        3. Completeness (is it sufficient?)
        4. Grounding (is the answer supported by the provided context?)

        If routed as "generate" (the question did NOT need the video transcript - e.g. a greeting
        or general-knowledge question), do NOT evaluate or penalize grounding. Instead evaluate:
        1. Relevance (does it answer the question?)
        2. Tone (is it friendly, engaging, and encouraging?)
        3. Whether it invites the user to ask a video-related question next.

        Context: {context}
        Question: {question}
        Answer: {answer}

        RESPOND WITH ONLY VALID JSON, NO EXTRA TEXT:
        {{"score": <1-10>, "decision": "PASS or needs_improvement or FAIL", "feedback": "<brief feedback>"}}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "answer", "needs_transcript"],
        template = eval_prompt
    )
    return prompt_template
