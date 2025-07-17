import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(
    filename="logs.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Logger Initialised..")


azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_client = AzureOpenAI(
    api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_api_version,
)


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


async def is_language_query(query: str) -> bool:
    """
    Detect if a query is about English/language/literature/word meanings
    """
    check_prompt = f"""Question: "{query}"\nIs this question about English language, grammar, literature, or word meanings? Answer only "Yes" or "No"."""

    messages = [
        {"role": "system", "content": "You are an intent classifier."},
        {"role": "user", "content": check_prompt},
    ]

    try:
        response = openai_client.chat.completions.create(
            model=(azure_openai_chat_deployment),
            messages=messages,
            max_tokens=3,
            temperature=0.0,
        )
        intent = response.choices[0].message.content.strip().lower()
        return intent == "yes"
    except Exception as e:
        logger.error(f"Intent classification failed: {str(e)}")
        return False  # fallback to allow the query if uncertain
