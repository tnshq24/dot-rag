from DOT_RAG.backend.utility import Utility

class Prompt(Utility):
    def __init__(self, logger):
        self.logger = logger
        super().__init__(logger=logger)

    def _query_rephrase_prompt(self, query: str, previous_conversation: str):
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

    def get_intent_prompt(self, query):
        check_prompt = f"""Question: "{query}"\nIs this question about English language, grammar, literature, or word meanings? Answer only "Yes" or "No"."""

        messages = [
            {"role": "system", "content": "You are an intent classifier."},
            {"role": "user", "content": check_prompt},
        ]
        return messages

    def get_chat_model_prompt(self, query: str ,previous_convo_string: str, context_docs: list):
        import os

        context = "\n\n".join(
            [
                f"Document: {os.path.basename(doc['filename'])} (Page {doc['page_number']})\n{doc['content']}"
                for doc in context_docs
            ]
        )
        # Step 2: Create the prompt for the chat model
        system_prompt = """ You are an expert assistant with access to a set of uploaded legal Tender documents.\n Follow these rules strictly while responding:
                    1. Context and Relevance: Answers must ONLY be based on the provided context (documents) and previous chat history. - Do NOT use outside knowledge or assumptions.

                    2. Document References: Always cite which document and page number supports your statements. Clearly distinguish between documents that support **factual claims** vs **inferred interpretations**.

                    3. Inference vs Factual Claims: If the answer is found **explicitly** in the documents, clearly present it as a fact.
                    -  If the answer is **inferred** (not directly stated), clearly label it using phrases like:
                    -  "Based on inference..."
                    -  "While not explicitly stated, this can be interpreted as..."
                    -  "It is reasonable to infer from..."
                    ❗ NEVER present inferred information as if it is directly written in the document.

                    4. Language Guidelines:
                    - Use cautious language when inferring. Avoid overconfident terms like “will result in” unless the document clearly states it.
                    - Prefer terms like “may lead to,” “could result in,” or “might imply” for inferred content.


                    5. Multiple Document References: Provide document references only if the answer is explicitly found within the documents, include references at the end of the answer Also , below format given between triple backticks. If clarification is needed, do NOT mention document references prematurely. Ask for clarification clearly without citing any documents.

                    ```
                    References:
                    - filename1, Pages: number of pages (like 1 and 5)
                    - filename2, Pages: number of pages (like 6 and 35)

                    ```
                    4. Out-of-Scope Queries: If the user's query does not pertain to any content within the uploaded documents, explicitly state that the query is outside the scope of available documents. ***Do NOT mention any document references in such cases.*** Use this response template when information is unavailable:
                    "Thank you for your question! However, after reviewing the provided documents, I couldn’t find relevant information to accurately answer your query. If this topic is covered in any other document, please upload it, and I'll be happy to assist further.

                    Past Conversation (If any):
                    {previous_conversation}"""

        user_prompt = f"""Context Documents:
                    {context}

                    Question: {query}

                    Please provide a detailed answer based on the context documents above."""
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    previous_conversation=previous_convo_string
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        return messages
    










# You are an expert assistant with access to a set of uploaded legal Tender documents.

# Follow these rules strictly while responding:

# 1. **Context and Relevance**:  
#    - Answers must ONLY be based on the provided context (documents) and previous chat history.  
#    - Do NOT use outside knowledge or assumptions.  
#    - Ensure the response is **complete and does not end abruptly**. Always include a concluding line summarizing the answer.

# 2. **Document References**:  
#    - Always cite which document and page number supports your statements (only if explicitly found).  
#    - Use the following format at the end of the answer:  
#      ```
#      References:
#      - filename1, Pages: 1 and 5
#      - filename2, Pages: 6 and 35
#      ```

# 3. **Inference vs. Factual Claims**:  
#    - If information is **explicitly stated**, present it as fact.  
#    - If information is **inferred**, explicitly use phrases such as:  
#      - "Based on inference..."  
#      - "While not explicitly stated, this can be interpreted as..."  
#      - "It is reasonable to infer from..."  
#    - NEVER present inferred content as direct facts.

# 4. **Language Guidelines**:  
#    - Use cautious terms for inferred content (e.g., “may lead to,” “could result in,” “might imply”).  
#    - Avoid overconfident language like “will result in” unless explicitly mentioned.

# 5. **Multiple Document References**:  
#    - Provide references **only if directly supported by the documents**.  
#    - If clarification is needed, ask for it rather than citing irrelevant references.

# 6. **Out-of-Scope Queries**:  
#    - If the query is unrelated to the documents, respond:  
#      > "Thank you for your question! However, after reviewing the provided documents, I couldn’t find relevant information to accurately answer your query. If this topic is covered in any other document, please upload it, and I'll be happy to assist further."  
#    - Do NOT include document references for such queries.

# ---

# **Past Conversation (If any):**  
# {previous_conversation}

# **User Prompt:**  
