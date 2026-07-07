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
    Please summarize the following YouTube video transcript:

    Video content:
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
        """You are an expert assistant providing answers based on video content.

        ANSWER PRIORITY:
        1. First, check the provided video context.
        2. If the answer is in the context:
            - Extract the ACTUAL DEFINITION or explanation using wording as close as possible to the transcript
            - Include any examples given by the speaker
            - DEDUPLICATE: Mention each concept only once
            - Use ONLY ONE timestamp (the earliest or most relevant)
        3. If NOT in the context:
            - Say: "The video does not mention this."
        4. THEN, only if highly confident, add:
            - "However, [general knowledge answer]"
        5. NEVER fabricate timestamps or claim something is in the video when it is not.

        EDGE CASES:
            - If multiple conflicting explanations exist, choose the clearest one and use the earliest timestamp.

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


def create_revision_prompt() -> PromptTemplate:
    """Create prompt template for revising an answer based on evaluator feedback."""
    template = """
    You are revising an answer to a question about a YouTube video.

    Use only the provided video context.
    Do not mention that you are revising.
    Do not mention evaluator feedback.
    Do not fabricate timestamps.

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
        input_variables=["context", "question", "previous_answer", "feedback"],
        template=template,
    )


def create_queries_prompt() -> PromptTemplate:
    multiqueries_template = """
    Generate 3 alternative phrasings of the question.

    Question: "{question}"

    RESPOND WITH ONLY JSON FORMAT, NO EXPLANATION:
    {{"queries": ["alternative 1", "alternative 2", "alternative 3"]}}
    """

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template = multiqueries_template
    )

    return prompt_template

def create_eval_prompt()-> PromptTemplate:
    eval_prompt="""
        You are an evaluator judging the quality of an AI response grounded in video content.

        Evaluate based on:
        1. Correctness (is it factually correct based on context?)
        2. Relevance (does it answer the question?)
        3. Completeness (is it sufficient?)
        4. Grounding (is the answer supported by the provided context?)

        Context: {context}
        Question: {question}
        Answer: {answer}

        RESPOND WITH ONLY VALID JSON, NO EXTRA TEXT:
        {{"score": <1-10>, "decision": "PASS or needs_improvement or FAIL", "feedback": "<brief feedback>"}}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "answer"],
        template = eval_prompt
    )
    return prompt_template
