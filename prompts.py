from langchain_core.prompts import PromptTemplate


def create_summary_prompt() -> PromptTemplate:
    """Create prompt template for video summarization."""
    template = """
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "Timestamp" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
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


def create_qa_prompt() -> PromptTemplate:
    """Create prompt template for context-grounded question answering."""
    qa_template = """
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:

    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    5. Mention at most ONE timestamp from the context.
    6. Always place the timestamp at the end of the answer.

    Only answer using the provided context.
    If the answer is not in the context, say "The video does not mention this."

    Note: In the transcript, "Text" refers to the spoken words in the video, timestamp indicated when that part begins in the video.
 
    Relevant Video Context: {context}
    Based on the above context, please answer the following question: {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template

def create_general_prompt() -> PromptTemplate:
    '''Create prompt template for general knowledge'''

    general_template = """
    You are a helpful assistant. Answer the user's question concisely and naturally.
    User's question: {question}

    """

    prompt_template = PromptTemplate(
        input_variables =["question"],
        template = general_template
    )

    return prompt_template


def create_router_prompt() -> PromptTemplate:
    router_prompt = """
    You are a routing agent. Decide if this question needs the video transcript.

    Question: {question}

    RESPOND WITH ONLY THIS JSON, NO EXPLANATION:
    {{"needs_transcript": true/false, "confidence": 0.0-1.0}}

    Rules:
    - needs_transcript=true if question references VIDEO/SPEAKER/CONTENT
    - needs_transcript=false if question asks general knowledge
    - confidence: your confidence (0.0-1.0) in this decision

    Examples:
    - "What does the speaker say about AI?" → {{"needs_transcript": true, "confidence": 0.95}}
    - "What's the capital of France?" → {{"needs_transcript": false, "confidence": 0.9}}
    - "What are hidden layers?" → {{"needs_transcript": false, "confidence": 0.85}} (ambiguous, no video reference)
    - "How does he explain neural networks?" → {{"needs_transcript": true, "confidence": 0.9}} (explicit video reference)
    """

    prompt_template = PromptTemplate(
        input_variables = ["question"],
        template = router_prompt
    )

    return prompt_template


def create_queries_prompt() -> PromptTemplate:
    multiqueries_template = """
    TASK: Generate 3 alternative phrasings of the question.

    Question: "{question}"

    RESPOND WITH ONLY THIS JSON, NO EXPLANATION, NO CODE:
    {{"queries": ["alternative 1", "alternative 2", "alternative 3"]}}
    """

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template = multiqueries_template
    )

    return prompt_template