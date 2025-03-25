import streamlit as st
from typing import Optional
import sqlite3
from phi.agent import Agent
from phi.tools import Toolkit
from phi.model.openai import OpenAIChat
import os
from dotenv import load_dotenv

load_dotenv()


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
if st.button("Submit"):
    if email and ohr_id:
        response = process_account_action(action, email, ohr_id)
        st.info(response.content)
    else:
        st.error("Please enter both Email and OHR ID.")
