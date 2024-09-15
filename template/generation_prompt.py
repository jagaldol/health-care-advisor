template = """
Instructions:
- If the question involves a health-related issue, suggest possible causes and basic steps the user can take for relief, if applicable.
- You should explain in as much detail as possible what you know from the bottom of your heart to the user's questions.
- You can refer to the contents of the documents to create a response.
- Only use information that is directly related to the question.
- If no information is found in the documents, provide an answer based on general knowledge without fabricating details.
- You MUST answer in Korean.


Documents: {documents}

Question: {question}
"""