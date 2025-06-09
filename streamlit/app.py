import base64
import streamlit as st
from PIL import Image
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
from datetime import datetime
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Get the absolute path to the directory containing this file
BASE_DIR = Path(__file__).resolve().parent

# Construct the path to memory.db
DB_PATH = BASE_DIR.parent / 'memory' / 'memory.db'

# Custom Memory Implementation using SQLite
class SQLiteMemory:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def store(self, data):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (user_id, content, metadata) VALUES (?, ?, ?)",
            (data["thread_id"], data["content"], json.dumps(data.get("metadata", {})))
        )
        self.conn.commit()

    def get_by_thread_id(self, thread_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT content, timestamp, metadata FROM conversations WHERE user_id = ? ORDER BY timestamp DESC",
            (thread_id,)
        )
        results = cursor.fetchall()
        return [
            {
                "content": content,
                "timestamp": timestamp,
                "metadata": json.loads(metadata)
            }
            for content, timestamp, metadata in results
        ]

    def close(self):
        self.conn.close()

# Initialize Memory
memory = SQLiteMemory()

# Function to save conversation
def save_conversation(user_id, conversation):
    memory.store({
        "thread_id": user_id,
        "content": conversation,
        "metadata": {
            "timestamp": datetime.now().isoformat()
        }
    })

# Function to retrieve conversation history
def retrieve_conversation(user_id):
    return memory.get_by_thread_id(user_id)

# Simulated External Medical Search Function
#def external_medical_search(query: str) -> str:
#    # In a production scenario, integrate with an external API such as NIH or PubMed.
#    return f"Simulated external search result for query: '{query}'. (This would normally return up-to-date medical guidelines.)"

# Memory Retrieval Tool: extracts past diagnoses from stored conversation history
def memory_retrieval_tool_prev(query: str, user_id: str) -> str:
    conversation_history = retrieve_conversation(user_id)
    if not conversation_history:
        return "No past medical history available."
    # Combine past conversation texts
    combined_text = "\n".join([item["content"] for item in conversation_history])
    # Use the LLM to summarize and extract relevant diagnoses
    prompt = (
        f"Below are past medical records. Extract and summarize any diagnoses or relevant medical conditions mentioned "
        f"that relate to the query: '{query}'.\n\nRecords:\n{combined_text}\n\nSummary:"
    )
    summary = llm.invoke([HumanMessage(content=prompt)])
    return summary

def memory_retrieval_tool(query: str, user_id: str) -> str:
    conversation_history = retrieve_conversation(user_id)
    if not conversation_history:
        return "No past conversations available."
    
    # Combine all previous conversations
    combined_text = "\n".join([f"- {item['timestamp']}: {item['content']}" for item in conversation_history])

    # Modify the prompt to summarize the entire conversation history
    summary_prompt = f"""
		You are an AI assistant summarizing previous user conversations. Focus only on the discussions between the user and the AI. Do not mention system
		functions, tools, or implementation details. 
		
		Here is the conversation history:
		
		{combined_text}
		
		Summarize the key points discussed so far in simple, clear language for the user.
		"""
    
    summary = llm.invoke([HumanMessage(content=summary_prompt)])
    return summary

# Agent Query Function using LangChain's Agent Executor
def agent_query(user_query: str, user_id: str) -> str:
    # Define the tools for the agent
    tools = [
        Tool(
            name="MemoryRetrieval",
            func=lambda q: memory_retrieval_tool_prev(q, user_id),
            description="Retrieves and summarizes past diagnoses from stored medical records."
        ),
        #Tool(
            #name="ExternalMedicalSearch",
            #func=external_medical_search,
            #description="Searches external medical databases for the latest medical guidelines and treatment updates."
        #),
        Tool(
            name="LLMResponse",
            func=lambda q: llm.invoke([HumanMessage(content=q)]),
            description="Generates a human-like response using the LLM."
        )
    ]
    # Initialize the agent with a ReAct strategy. The agent will decide which tool(s) to use.
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    result = agent.run(user_query)
    return result

# Streamlit App
def main():
    st.title("🏥 AI-Powered Medical Image Analysis Agent")
    st.write("Upload a medical image and provide patient details for analysis and report generation. "
             "The agent can now also intelligently reason over past interactions and external data.")

    # User Identification
    user_id = st.text_input("Enter User ID", placeholder="e.g., user001")
    if not user_id:
        st.warning("Please enter a User ID to proceed.")
        return

    # Retrieve Conversation History
    if st.button("Load Previous Conversations"):
        conversation_history = retrieve_conversation(user_id)

        if conversation_history:
            st.subheader("Conversation History")
            for item in conversation_history:
                st.markdown(f"**Timestamp:** {item['timestamp']}")
                st.markdown(item["content"])
                st.markdown("---")
            
            # Download Data Here
            download_data = "\n\n---\n\n".join([
                f"Timestamp: {item['timestamp']}\n\n{item['content']}"
                for item in conversation_history
                ])

            st.download_button(
                label="Download Conversation History",
                data=download_data,
                file_name=f"{user_id}_medical_history.txt",
                mime="text/plain"
                )
        else:
            st.info("No previous conversations found.")

    # Upload Image
    uploaded_image = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

    def process_uploaded_image(uploaded_file):
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image')
            file_content = uploaded_file.getvalue()
            base64_image = base64.b64encode(file_content).decode('utf-8')
            return base64_image
        return None

    base64_image = process_uploaded_image(uploaded_image)

    # Input Patient Metadata
    with st.form("patient_form"):
        patient_name = st.text_input("Patient Name", "John Doe")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        patient_symptoms = st.text_input("Symptoms", "Shortness of breath, cough")
        submit_button = st.form_submit_button("Generate Medical Report")

    if submit_button and base64_image:
        patient_details = f"Name: {patient_name}, Age: {patient_age}, Gender: {patient_gender}, Symptoms: {patient_symptoms}"

        # Generate Image Insights using the LLM with the uploaded image and symptoms
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"""Analyze the following medical image and provide **only** the key observations:
								        - Correlate findings with patient symptoms: {patient_symptoms}
								        - Identify observed abnormalities
								        - Suggest possible conditions
								        - Indicate affected organs
								        - Propose prevention strategies (if applicable)

								        🚫 **DO NOT** include:
								        - Formal document elements (e.g., "Physician Name", "Signature")
								        - Assumptions beyond what can be inferred from the image
								        - General disclaimers or irrelevant text

								        Keep the response **detailed, structured, and strictly medical**.
								        """ },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        )
        image_insights = llm.invoke([message])

        # Generate Full Medical Report using an LLM Chain
        report_prompt = PromptTemplate(
		    template="""
					Based on the following details, generate a **detailed, structured medical report**:

					1. **Patient Details:** {patient_details}  
					2. **Image Insights:** {image_insights}  

					**Your report should include:**
					- **Patient summary** (brief overview)
					- **Observed abnormalities** (with medical relevance)
					- **Possible conditions/diagnosis**
					- **Recommendations or next steps** (if applicable)

					🚫 **DO NOT** include:
					- Physician name, signature, or formal document headers
					- Extra disclaimers or non-medical text
					- Assumptions not supported by findings

					**Report:**
					""",
			input_variables=["patient_details", "image_insights"],
		)

        report_chain = LLMChain(llm=llm, prompt=report_prompt)
        full_report = report_chain.run(patient_details=patient_details, image_insights=str(image_insights))

        # Save to Memory
        conversation_entry = f"""
		### Patient Information
		{patient_details}

		### Image Analysis
		{str(image_insights)}

		### Medical Report
		{full_report}
		"""
        save_conversation(user_id, conversation_entry)

        # Display the Report
        st.subheader("Generated Medical Report")
        st.markdown(full_report)


    # Agent Query Section: Ask questions like "What diseases have I had in the past?" or "What are the latest treatments for Brain Tumor?"
    st.markdown("## Agent Assistant")
    st.write("Ask the agent a question about your medical history or for updated medical information. Like, try: "
             "'What diseases have I had in the past?' or 'What are the latest treatment guidelines for Brain Tumor?'")
    agent_user_query = st.text_input("Enter your query for the agent")
    if st.button("Ask Agent") and agent_user_query:
        with st.spinner("Processing your query..."):
            agent_response = agent_query(agent_user_query, user_id)
        st.markdown("### Agent Response")
        st.markdown(agent_response)
        # Optionally, save the agent interaction in the conversation history
        save_conversation(user_id, f"**Agent Query:** {agent_user_query}\n\n**Agent Response:** {agent_response}")

if __name__ == "__main__":
    main()
