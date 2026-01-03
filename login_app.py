import streamlit as st
from auth_db import AuthDatabase

import base64
import os

# Initialize database
db = AuthDatabase()

# Page configuration
st.set_page_config(
    page_title="Login & Signup",
    page_icon="🔐",
    layout="centered"
)

def get_base64_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    
    /* Make the main containers semi-transparent black */
    [data-testid="stForm"], .stTabs, [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {{
        background-color: rgba(0, 0, 0, 0.75) !important;
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.6);
    }}
    
    /* Input fields styling */
    .stTextInput > div > div > input {{
        background-color: rgba(20, 20, 20, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }}
    
    /* Placeholder styling */
    .stTextInput > div > div > input::placeholder {{
        color: rgba(255, 255, 255, 0.4) !important;
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 5px;
        border-radius: 10px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        color: white !important;
    }}

    /* Titles and text color */
    h1, h2, h3, p, span, label {{
        color: white !important;
    }}
    
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background if it exists
if os.path.exists("background.jpg"):
    set_background("background.jpg")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

def logout():
    """Logout function"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_info = None
    st.rerun()

def login_page():
    """Display login page"""
    st.title("🔐 Welcome Back!")
    st.markdown("---")
    
    with st.form("login_form"):
        username_or_email = st.text_input("Username or Email", placeholder="Enter your username or email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submit = st.form_submit_button("🚀 Login")
        
        if submit:
            if not username_or_email or not password:
                st.error("⚠️ Please fill in all fields!")
            else:
                success, user_info, message = db.verify_user(username_or_email, password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = user_info['username']
                    st.session_state.user_info = user_info
                    st.success(f"✅ {message}")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

def signup_page():
    """Display signup page"""
    st.title("📝 Create Account")
    st.markdown("---")
    
    with st.form("signup_form"):
        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password", placeholder="Choose a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        
        submit = st.form_submit_button("✨ Create Account")
        
        if submit:
            # Validation
            if not username or not email or not password or not confirm_password:
                st.error("⚠️ Please fill in all fields!")
            elif len(password) < 6:
                st.error("⚠️ Password must be at least 6 characters long!")
            elif password != confirm_password:
                st.error("⚠️ Passwords do not match!")
            elif "@" not in email:
                st.error("⚠️ Please enter a valid email address!")
            else:
                success, message = db.create_user(username, email, password)
                
                if success:
                    st.success(f"✅ {message}")
                    st.info("You can now login with your credentials!")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")

def dashboard():
    """Display user dashboard after login"""
    st.title(f"👋 Welcome, {st.session_state.username}!")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 Your Dashboard")
        st.write("You are successfully logged in!")
        
        if st.session_state.user_info:
            st.markdown(f"""
            **Account Information:**
            - **Username:** {st.session_state.user_info['username']}
            - **Email:** {st.session_state.user_info['email']}
            """)
    
    with col2:
        if st.button("🚪 Logout"):
            logout()
    
    st.markdown("---")
    st.info("💡 **Next Steps:** Integrate this authentication system with your Meeting Summarizer app!")
    
    st.markdown("""
    ### 🔗 Integration Guide
    
    To protect your Meeting Summarizer app with authentication:
    
    1. Import the authentication check at the top of your app:
    ```python
    import streamlit as st
    from auth_db import AuthDatabase
    
    # Check if user is logged in
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.error("Please login first!")
        st.stop()
    ```
    
    2. Add a logout button in your app sidebar:
    ```python
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    ```
    """)

# Main app logic
def main():
    if st.session_state.logged_in:
        # Show dashboard if logged in
        dashboard()
    else:
        # Show login/signup tabs if not logged in
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            login_page()
        
        with tab2:
            signup_page()

if __name__ == "__main__":
    main()
