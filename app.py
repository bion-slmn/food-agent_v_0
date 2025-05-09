import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from food_agent import graph  # Your precompiled LangGraph graph

st.title("🍽️ Food Agent Chatbot")
st.write("Ask any food-related question and let the agent help!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Your question:")
if user_input:
    # Append user message
    message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(message)

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show spinner while processing
    with st.spinner("Thinking..."):
        try:
            # Run the graph
            config = {
                "configurable": {
                    "user_id": str(st.session_state.get("user_id", "default")),
                    "thread_id": str(st.session_state.get("thread_id", "default"))
                }
            }
            result = graph.invoke({"messages": st.session_state.chat_history}, config=config)

            # Extract and display new messages
            new_messages = result.get("messages", [])
            st.session_state.chat_history.extend(new_messages)

            for msg in new_messages:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

        except Exception as e:
            # Display error message
            error_message = f"❌ An error occurred: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_message)
