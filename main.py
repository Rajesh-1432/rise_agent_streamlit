import streamlit as st
import os
from phi.agent import Agent
from phi.model.openai import OpenAIChat
import joblib
from typing import List
from phi.tools import Toolkit
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from phi.tools.duckduckgo import DuckDuckGo
import webbrowser
import sqlite3
import re
from dotenv import load_dotenv

load_dotenv()



#------------Live DB-------------------------------
from datetime import datetime

# Database file
DB_FILE = "my_database.db"

# Function to create tables
def create_tables():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Create user_queries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create agent_responses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            query TEXT NOT NULL,
            ml_triage_agent_response TEXT,
            nlp_triage_agent_response TEXT,
            ticket_analysis_agent_response TEXT,
            csv_agent_response TEXT,
            web_search_agent_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES user_queries(query_id) ON DELETE CASCADE
        );
    """)

    # Commit changes and close connection
    conn.commit()
    conn.close()

# Function to store a user query
def store_user_query(query):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_queries (query) VALUES (?)", (query,))
    conn.commit()
    conn.close()

# Function to store an agent response
def store_agent_response(query_id, query, ml_response, nlp_response, ticket_analysis_response, csv_response, web_search_response):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO agent_responses (query_id, query, ml_triage_agent_response, nlp_triage_agent_response, ticket_analysis_agent_response, csv_agent_response, web_search_agent_response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (query_id, query, ml_response, nlp_response, ticket_analysis_response, csv_response, web_search_response))
    conn.commit()
    conn.close()

# Function to fetch all stored responses (for debugging or viewing data)
def fetch_responses():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM agent_responses")
    results = cursor.fetchall()
    conn.close()
    return results

# Load ML Triage Model
class AssignmentGroupPredictor(Toolkit):
    def __init__(self, model_path: str, vectorizer_path: str):
        super().__init__(name="ml_triage_agent_predict_tool")
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.register(self.ml_triage_agent_predict_tool)

    def ml_triage_agent_predict_tool(self, incident_description: List[str]) -> str:
        try:
            vectorized_incident = self.vectorizer.transform(incident_description)
            predicted_group = self.model.predict(vectorized_incident)
            return predicted_group[0]
        except Exception as e:
            return f"Error: {e}"


#------------- ML Triage Agent --------------------------

# Paths for ML Model
model_path = "Reference/ML model/support_vector_machine.pkl"
vectorizer_path = "Reference/ML model/tfidf_vectorizer.pkl"
triage_toolkit = AssignmentGroupPredictor(model_path, vectorizer_path)

ml_traige_agent = Agent(
    name="Triage Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[triage_toolkit],
    instructions=[
        "When given an incident description, call the `ml_triage_agent_predict_tool` function to determine the assignment group.",
        "Do not generate hypothetical code. Use the provided function to make predictions.",
        "Suggest a possible next step or action based on the predicted group.",
        "Format:\n",
        "**Predicted Group:** <with no description, just the group name>\n",
        "**Next Step:**\n\n",
    ],
    show_tool_calls=True,
    markdown=True
)

# Load NLP Triage Model
def create_or_load_faiss(file_path: str, faiss_index_path: str):
    loader = CSVLoader(file_path=file_path, encoding='latin-1')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
        vectorstore.save_local(faiss_index_path)
    return vectorstore.as_retriever()

# Paths for NLP Model
file_path = "Reference/NLP_incident  Vernova Analysis.csv"
faiss_index_path = "Reference/NLP_incident_faiss_index"
retriever = create_or_load_faiss(file_path, faiss_index_path)

def nlp_triage_agent_predict_tool(issue_description: str) -> str:
    retrieved_docs = retriever.invoke(issue_description)
    return json.dumps([doc.page_content for doc in retrieved_docs], indent=2)

nlp_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[nlp_triage_agent_predict_tool],
    description="AI assistant for SAP GRC issue categorization.",
    instructions=[
        "Retrieve relevant context for the given issue description using `nlp_triage_agent_predict_tool`.",
        "Use the retrieved context to classify the issue into one of these predefined categories:",
        "1. New Account Tickets",
        "2. Mirror Role Access",
        "3. Authorization Issues",
        "4. Access Issues",
        "5. T-Code Access Issues",
        "6. GRC Related Issues",
        "7. Lock/Unlock Issues",
        "8. Production Access Requests",
        "9. Fiori Related Issues",
        "10. Existing User",
        "11. FFID Issue",
        "12. Password Related Issues",
        "Format: \n",
        "**Predictied Group:** <with no description, just the group name> \n",
        "**Reasoning:** \n",
    ],
    markdown=True,
    show_tool_calls=True,
)


# -------------------- CSV SEARCH AGENT --------------------
def load_csv_data(directory_path):
    csv_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".csv")]
    data = []
    for file_path in csv_files:
        loader = CSVLoader(file_path=file_path, encoding='latin-1')
        data.extend(loader.load())
    return data

def preprocess_data(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    return text_splitter.split_documents(data)

def initialize_vectordb(all_splits, vectordb_path, embedding_model="text-embedding-3-large"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    if os.path.exists(vectordb_path):
        vectordb = FAISS.load_local(vectordb_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(documents=all_splits, embedding=embeddings)
        vectordb.save_local(folder_path=vectordb_path)
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})


def query_csv_rag(query: str) -> str:
    """Retrieve the most relevant CSV-based response using FAISS."""
    top_k_docs = retriever.invoke(query)
    response_text = "\n".join([doc.page_content for doc in top_k_docs])
    return response_text if response_text else "No relevant data found."

# Main execution
def csv_search():
    directory_path = "Reference/CSV search"
    vectordb_path = "Reference/CSV search/StoreFAISSCSV"
    
    data = load_csv_data(directory_path)
    all_splits = preprocess_data(data)
    global retriever
    retriever = initialize_vectordb(all_splits, vectordb_path)
    
    csv_search_agent = Agent(
    name="CSV Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[query_csv_rag],
    description="You are an AI assistant that retrieves information from CSV files.",
    instructions=[
        "You are an assistant for generating responses from CSV prompts. ",
        "Use the retrieved context to answer the question. ",
        "If the answer is not present, say: 'I don't know'. ",
        "Structure the response with Markdown formatting.\n\n",
        "Format:\n",
        "**Step-1:** \n",
        "**Step-2:** \n",
        "**Step-3:** \n",
        "**Related Information:** \n",
    ],
    show_tool_calls=True,
    markdown=True
)
    response = csv_search_agent.run(user_input).content
    csv_response = response
    return csv_response

# -------------------- WEB SEARCH AGENT --------------------
web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=[
                  "User will write about thier issues, they are facing and you understand in simpler terms like:",
                  "For Example:",
                  "User Query: Tcode 500 and 600 is filling to be mapped to user OHRID: 703394911, while searching Input: How to resolve when Tcode is failling to be mapped",
                  "Another Example:",
                  "User Query: I am unable to login to SAP GUI because of account locked, while searching Input: What to do for SAP login Failure."
                  "Always include sources",
                  ],
    show_tool_calls=True,
    markdown=True,
)

def query_web_agent(query: str) -> str:
    """Query the Web Agent for real-time search results."""
    response = web_agent.run(query)
    return response.content if response else "No relevant web data found."

# -------------------- Ticket Analysis AGENT --------------------
# Load CSV data
def load_incident_data(directory_path):
    incident_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".csv")]
    data = []
    for file_path in incident_files:
        loader = CSVLoader(file_path=file_path, encoding='latin-1')
        data.extend(loader.load())
    return data

# Preprocess data
def preprocess_incident_data(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    return text_splitter.split_documents(data)

# Initialize FAISS vector database
def initialize_incident_vectordb(all_splits, vectordb_path, embedding_model="text-embedding-3-large"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    if os.path.exists(vectordb_path):
        vectordb = FAISS.load_local(vectordb_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(documents=all_splits, embedding=embeddings)
        vectordb.save_local(folder_path=vectordb_path)
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

def ticket_analysis_rag_agent(query: str) -> str:
    """Retrieve the most relevant CSV-based response using FAISS."""
    top_k_docs = incident_retriever.invoke(query)
    response_text = "\n".join([doc.page_content for doc in top_k_docs])
    return response_text if response_text else "No relevant previous Ticket found."

def ticket_analysis():
    directory_path = "Reference/Ticket Analysis"
    vectordb_path = "Reference/Ticket Analysis/StoreFAISS_Ticket_Analysis"
    
    data = load_incident_data(directory_path)
    all_splits = preprocess_incident_data(data)
    global incident_retriever
    incident_retriever = initialize_incident_vectordb(all_splits, vectordb_path)
    
    ticket_analysis_agent = Agent(
    name="Ticket Analysis Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[ticket_analysis_rag_agent],
    description="You are an AI assistant that retrieves information from CSV files.",
    instructions=[
        "You need to find related incidents. ",
        "Analyze those related incidents found",
        "Looking at these related incidents, you need to provide solutions to the user about the current incident.",
        "If the similar tickets is not found, say: 'No similar tickets found.'",
        "Structure the response with Markdown formatting.\n\n",
        "Always have Incident Number",
        "Format:\n",
        "**Similar Incident Found:** \n",
        "**Similar Incident Found:** \n",
        "**Similar Incident Found:** \n",
        "**Previous Ticket Analysis:** \n",
    ],
    show_tool_calls=True,
    markdown=True
)
    response = ticket_analysis_agent.run(user_input).content
    return response
# -------------------- Live agents --------------------


# -------------------- STREAMLIT UI --------------------
st.title("GRC Agentic AI Support App")

user_input = st.text_area("Write down the issues that you are facing: ", height=150)
query_id = store_user_query(user_input)

if st.button("Submit"):
    if user_input:
        with st.spinner("Performing ML Triage Agent..."):
            ml_response = ml_traige_agent.run(user_input).content
        st.subheader("ML Triage Result")
        st.write(ml_response)

        with st.spinner("Performing NLP Triage Agent..."):
            nlp_response = nlp_agent.run(f"Categorize the following issue:\n{user_input}").content
        st.subheader("NLP Triage Result")
        st.write(nlp_response)

        st.subheader("Let me try to process it like any human would do: Running Multiple Searches")

        with st.spinner("Previous Ticket Analysis..."):
            ticket_agent_response =ticket_analysis()
        st.subheader("Previous Ticket Analysis")
        st.write(ticket_agent_response)

        with st.spinner("Searching CSV for related information..."):
            csv_agent_response =csv_search()
        st.subheader("CSV Search Result")
        st.write(csv_agent_response)

        with st.spinner("Searching the web for additional insights..."):
            web_agent_response = query_web_agent(user_input)
        st.subheader("Web Search Result")
        st.write(web_agent_response)

        store_agent_response(query_id, user_input, ml_response, nlp_response, ticket_agent_response, csv_agent_response, web_agent_response)

        st.subheader("Let me try to trigger one of my agents: Displaying all the agents")

        match = re.search(r"Predicted Group:\s*(.*)", ml_response)
        if match:
            predicted_group = match.group(1).strip().lower()

            # List of keywords to trigger Pwd Agent
            pwd_related_groups = ["pwd reset", "password related issues", "password reset", "password"]

            # Check if the predicted group matches any item in the list
            if any(keyword in predicted_group for keyword in pwd_related_groups):
                st.write("Triggering Pwd Agent")
                print("password rest trigered")
            
            # List of keywords to trigger Pwd Agent
            lock_unlock_related_groups = ["lock/unlock issues", "lock/unlock related issues", "lock reset", "lock", "unlock", "login", "login issues"]

            # Check if the predicted group matches any item in the list
            if any(keyword in predicted_group for keyword in lock_unlock_related_groups):
                st.write("Lock/Unlock Agent")
                print("Lock Unlock agent trigered")

                # Define the Toolkit class
                class UserAccountToolkit(Toolkit):
                    def __init__(self):
                        super().__init__(name="user_account_tools")
                        self.register(self.manage_user_account)
                        self._conn = sqlite3.connect('user_accounts.db', timeout=10)
                    
                    def manage_user_account(self, action: str, email: str, ohr_id: str) -> str:
                        """Locks or unlocks a user account based on the given action."""
                        with self._conn:
                            cursor = self._conn.cursor()
                            if action == "lock":
                                status = 0
                            elif action == "unlock":
                                status = 1
                            else:
                                return "Invalid action. Use 'lock' or 'unlock'."

                            cursor.execute("UPDATE users SET status = ? WHERE email = ? AND ohr_id = ?", (status, email, ohr_id))

                            if cursor.rowcount == 0:
                                return "No matching user found."
                            return f"Account {action}ed successfully for email: {email}"

                # Initialize the custom toolkit
                user_account_toolkit = UserAccountToolkit()

                # Define the Agent
                lock_unlock_agent = Agent(
                    model=OpenAIChat(id="gpt-4o"),
                    tools=[user_account_toolkit],
                    description="Manages user account lock/unlock operations in the database.",
                    instructions=[
                        "Accepts user email and OHR ID to either lock or unlock an account.",
                        "If action is 'lock', set status to 0. If 'unlock', set status to 1.",
                        "If the user has asked for lock account, and value is already set 0, agent is expected to say: The account is locked only. Maybe the issue is something else.",
                        "If the user has asked for unlock account, and value is already set 1, agent is expected to say: The account is unlocked only. Maybe the issue is something else.",
                        "Returns confirmation of account status change."
                    ],
                )

                def process_account_action(action: str, email: str, ohr_id: str):
                    """Processes a user account action request using the agent."""
                    response = lock_unlock_agent.run(f"{action} {email} {ohr_id}")
                    return response

                # Streamlit UI
                st.title("User Account Management")

                # User Input
                action = st.radio("Select Action", ("lock", "unlock"))
                email = st.text_input("Enter Email")
                ohr_id = st.text_input("Enter OHR ID")

                # Button to Process Request
                if st.button("Submit", key="submit_button"):
                    if email and ohr_id:
                        response = process_account_action(action, email, ohr_id)
                        st.info(response.content)
                    else:
                        st.error("Please enter both Email and OHR ID.")
                print("Lock Unlock agent action completed")

        #db Info
        #st.write(fetch_responses())
        #print(fetch_responses())
        

    else:
        st.error("Please enter an issue description.") 