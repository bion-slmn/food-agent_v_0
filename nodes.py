from langchain_core.messages import HumanMessage, RemoveMessage, merge_message_runs, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from trustcall import create_extractor
import uuid
from state import State, Profile
from agent_tools import load_website_content, search_tool
from load_llm import llm


def summarization_node(state: State):
  '''
  Summarize the messages in the conversation.
  '''
  if len(state['messages']) < 6:
    return


  summary = state.get('summary', '')
  if summary:
    summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

  else:
    summary_message = "Create a summary of the conversation above:"

  messages = state["messages"] + [HumanMessage(content=summary_message)]
  response = llm.invoke(messages)
  deleted_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

  return {'summary': response.content, 'messages': deleted_messages}


trustcall_extractor = create_extractor(
    llm,
    tools=[Profile],
    tool_choice='Profile',
    enable_inserts=True,
)

TRUSTCALL_INSTRUCTION = """
Reflect only on the content provided in the following user interaction.

Only extract and retain factual, explicit user information (e.g., conditions, preferences, goals) that is clearly mentioned.

Do NOT assume or invent any details such as name, age, location, or preferences.

Use parallel tool calls to handle updates and insertions.

❌ If the user has not mentioned something, do NOT infer or create it.
"""



def write_memory(state: State, config: RunnableConfig, store: BaseStore):
  """Load memories from the store and use them to personalize the chatbot's response."""
  user_id = config['configurable']['user_id']
  namespace = ('profile', user_id)
  profile_memory = store.search(namespace)

  tool_name = "Profile"
  existing_memories = [(mem.key, tool_name, mem.value) for mem in profile_memory] if profile_memory else None
  updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]))

  results = trustcall_extractor.invoke({
      "messages": updated_messages,
      "existing": existing_memories,
  })

  for r, rmeta in zip(results['responses'], results["response_metadata"]):
    store.put(namespace,
              rmeta.get("json_doc_id", str(uuid.uuid4())),
              r.model_dump(mode='json'))


MODEL_SYSTEM_MESSAGE = """
You are NutriPal 🥦 — a helpful, friendly, and knowledgeable AI assistant specializing in food, diet, and nutrition.

Your role is to provide personalized food and nutrition advice based on the user's preferences, health conditions, location, and wellness goals.
You remember important details from past conversations to offer consistent, context-aware guidance.

✅ Always greet users warmly and maintain a respectful, supportive, and encouraging tone.

🧠 You can use tools such as web search (`search_tool`) and webpage content loader (`load_website_content`) to provide accurate and up-to-date information. Use them when the question requires recent or local context.

🚫 Do NOT offer medical diagnoses or suggest changes to medications or treatment plans. If asked, kindly recommend consulting a licensed healthcare provider.

If a user asks something unrelated to food or nutrition, gently steer the conversation back to those topics.

Respond in simple, clear, and compassionate language easy to read .

Here is what you currently remember about the user:
{memory}
"""

from langgraph.prebuilt import create_react_agent


def chat_model(state: State, config: RunnableConfig, store: BaseStore):
    """Load user memory and generate a personalized response via NutriPal."""
    user_id = config['configurable']['user_id']
    namespace = ('profile', user_id)

    # Retrieve memory
    profile_memory = store.search(namespace)
    memory_text = '\n'.join([f"- {mem.value}" for mem in profile_memory]) or "No prior information available."

    # Build prompt with memory
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=memory_text)

    nutritional_agent = create_react_agent(
    llm,
    tools=[search_tool, load_website_content],
    prompt=system_msg)

    message = {"messages": [HumanMessage(content=state['messages'][-1].content)]}
    response = nutritional_agent.invoke(message)
    print(response, '-'*39)


    return {'messages': [response['messages'][-1]]}
