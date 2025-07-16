def _query_rephrase_prompt(
    query: str,
    previous_conversation: str,
):
    """
    Generates a prompt for rephrasing a query.
    Args:
        query (str): The query to be rephrased.
        previous_conversation (str): The previous conversation.

    Returns:
        List[Dict[str, str]]: A list of messages containing the prompt and the user query.
    """
    prompt = """You are a query rephrasing tool that rephrases follow-up questions into standalone questions which can be understood independently without relying on previous question and answer.
    
    Your task:
    - If the current question is related to the chat history (i.e., it's a follow-up using ambiguous references like "this", "it", "the above", etc.), rephrase it into a **fully self-contained question** using the relevant details from the past conversation.
    - If the current question is not related to chat history, return it **as is**.

    ### Rules:
    - ALWAYS     produce a **self-contained question**.
    - Remove vague references like "it", "this", "that", "above question", etc.


Objective: Analyze the chat history enclosed within triple backticks, carefully to create standalone question independent of terms like 'it', 'that', etc.
For queries that are not a follow-up ones or not related to the conversation, you will respond with a predetermined message: 'Not a follow-up question'
'''
{previous_conversation}
'''
Here is the current user query which you can you use to under previous question and create a new standlone question.
'''
{query}
'''
## Output Format:
    A JSON dict with 1 key:
        - 'rephrased_query'(str): It Contains the rephrased query formed by following the above instructions."""
    prompt = prompt.format(previous_conversation=previous_conversation, query=query)

    messages = []
    messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": f"""Query: {query}"""})
    return messages
