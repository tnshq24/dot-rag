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

Objective: Analyze the chat history enclosed within triple backticks, carefully to create standalone question independent of terms like 'it', 'that', etc.
For queries that are not a follow-up ones or not related to the conversation, you will respond with a predetermined message: 'Not a follow-up question'
'''
{previous_conversation}
'''

## Output Format:
    A JSON dict with 1 key:
        - 'rephrased_query'(str): It Contains the rephrased query formed by following the above instructions."""
    prompt = prompt.format(previous_conversation=previous_conversation)

    messages = []
    messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": f"""Query: {query}"""})
    return messages
