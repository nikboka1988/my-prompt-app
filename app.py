# app.py (IMPROVED AND FIXED VERSION)

import sys
import os
import streamlit as st
import logging
from datetime import datetime
import uuid
import random
from typing import List, Dict, Any, Optional
import re  # For sanitization

# Project root setup
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Graceful config import
try:
    from config.settings import config
except ImportError:
    class MockConfig:
        APP_ICON = "🎨"
        APP_TITLE = "AI Prompt Studio"
    config = MockConfig()

# Service imports
from services.ai_service import AIService
from services.auth_service import AuthService

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INSPIRE_ME_LIMIT = 5
RETRY_ATTEMPTS = 3

def sanitize_input(text: str) -> str:
    """Sanitize input to prevent XSS or injection by stripping HTML tags."""
    if not text:
        return ""
    return re.sub(r'<[^>]*>', '', str(text)).strip()

def build_enhanced_director_brief(
    subject: str,
    environment: str,
    shot_type: str,
    lighting: str,
    mood: str,
    color: str,
    style: List[str],
    negative: str,
    prompt_type: str,
    target_model: str
) -> str:
    """Builds a structured director brief from sanitized form inputs."""
    brief_parts: List[str] = []
    if subject: brief_parts.append(f"SUBJECT & ACTION: {sanitize_input(subject)}")
    if environment: brief_parts.append(f"ENVIRONMENT & SETTING: {sanitize_input(environment)}")
    
    technical_parts: List[str] = []
    if shot_type and shot_type != "Default": 
        technical_parts.append(f"Shot Type: {sanitize_input(shot_type)}")
    if lighting: 
        technical_parts.append(f"Lighting: {sanitize_input(lighting)}")
    if mood and mood != "Default": 
        technical_parts.append(f"Mood: {sanitize_input(mood)}")
    if color: 
        technical_parts.append(f"Color Palette: {sanitize_input(color)}")
    if technical_parts: 
        brief_parts.append(f"TECHNICAL DETAILS: {' | '.join(technical_parts)}")
    
    if style: 
        brief_parts.append(f"ARTISTIC STYLE: {', '.join([sanitize_input(s) for s in style])}")
    if negative: 
        brief_parts.append(f"AVOID (Negative Prompts): {sanitize_input(negative)}")
    brief_parts.append(f"TARGET: A {prompt_type} prompt for {target_model}.")
    
    return "\n\n".join(brief_parts)

def generate_inspiration(ai_service: AIService, auth_service: Optional[AuthService] = None) -> None:
    """Generates random creative idea using AI and populates subject field."""
    # FIX 1: Ensure the random generator is re-seeded on every single click.
    random.seed()
    
    if 'brief_result' in st.session_state:
        del st.session_state['brief_result']
    
    if 'inspire_me_count' not in st.session_state:
        st.session_state['inspire_me_count'] = 0
    
    if st.session_state['inspire_me_count'] >= INSPIRE_ME_LIMIT:
        st.warning("Inspire Me limit reached for this session. Please generate manually.")
        return
    
    st.session_state['inspire_me_count'] += 1

    # FIX 2: Expanded the list of categories for more variety.
    categories = [
        "Cyberpunk", "Fantasy Noir", "Cosmic Horror", "Solarpunk", "Steampunk",
        "Biopunk", "Dieselpunk", "Mythpunk", "Gothic Fiction", "Afrofuturism",
        "Surrealism", "Magical Realism", "Hard Science Fiction", "Space Opera",
        "Post-Apocalyptic", "Dystopian", "Urban Fantasy"
    ]
    
    # FIX 3: Added a creativity request to the prompt itself.
    creativity_injection = "Incorporate a surprising, unexpected element or twist."
    
    if random.choice(['mashup', 'masterpiece']) == 'mashup':
        cats = random.sample(categories, 2)
        prompt = f"Generate a 'world-seed' concept combining '{cats[0]}' and '{cats[1]}'. Focus on a character and their situation. Be concise and evocative. {creativity_injection}"
        message = f"Helios is mixing {cats[0]} with {cats[1]}..."
    else:
        cat = random.choice(categories)
        prompt = f"Generate a 'masterpiece' world-seed concept in the '{cat}' genre. Focus on a character and their situation. Be concise and evocative. {creativity_injection}"
        message = f"Helios is crafting a masterpiece in the {cat} genre..."
    
    with st.spinner(message):
        last_error = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response_dict = ai_service.get_ai_response(prompt)
                
                creative_text = ""
                if isinstance(response_dict, dict):
                    creative_text = response_dict.get('creative_brief', '') or response_dict.get('technical_prompt', '')
                    if not creative_text:
                        content = response_dict.get('content', {})
                        if isinstance(content, dict):
                            creative_text = content.get("creative_brief", "") or content.get("technical_prompt", "")
                        elif isinstance(content, str):
                            creative_text = content
                        else:
                            creative_text = str(response_dict)
                elif isinstance(response_dict, str):
                    creative_text = response_dict
                else:
                    creative_text = str(response_dict)
                
                if creative_text and creative_text.strip():
                    st.session_state['director_subject'] = creative_text.strip()
                    
                    if auth_service and 'username' in st.session_state:
                        try:
                            inspire_data = {
                                'id': str(uuid.uuid4()), 'input': prompt, 'content': {'creative_brief': creative_text},
                                'model': 'gemini-1.5-pro-002', 'timestamp': datetime.now(), 'type': 'inspire_me',
                                'title': f"Inspiration: {creative_text[:30]}..."
                            }
                            auth_service.save_generation_to_history(st.session_state['username'], inspire_data)
                        except Exception as log_error:
                            logger.warning(f"Failed to log usage: {log_error}")
                    
                    return
                else:
                    logger.warning(f"Empty response on attempt {attempt+1}")
                    last_error = "Received an empty response from the AI service."
                    
            except Exception as e:
                last_error = e
                logger.error(f"Inspire Me attempt {attempt+1} failed: {e}")
        
        st.session_state['director_subject'] = ""
        st.error(f"Failed to generate inspiration. Please check your AI service configuration or network.\n\n**Details:** {str(last_error)}")

def initialize_auth_service() -> AuthService:
    if 'auth_service_instance' not in st.session_state:
        st.session_state['auth_service_instance'] = AuthService()
    return st.session_state['auth_service_instance']

def initialize_ai_service() -> AIService:
    if 'ai_service_instance' not in st.session_state:
        st.session_state['ai_service_instance'] = AIService()
    return st.session_state['ai_service_instance']

def get_user_history_cached(username: str, auth_service: AuthService) -> List[Dict]:
    return auth_service.get_user_history(username) if username else []

def show_unified_app_page(ai_service: AIService, auth_service: AuthService) -> None:
    """Displays the main app page with sidebar history and form."""
    if st.session_state.pop('just_logged_in', False):
        st.toast(f"Welcome back, {st.session_state.get('name', '')}!", icon="🎉")

    with st.sidebar:
        st.title(f"👋 Welcome, {st.session_state.get('name', '')}!")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.divider()
        st.markdown("### 📚 Recent Generations")
        
        username = st.session_state.get('username')
        if not username:
            st.caption("Log in to see your history.")
        else:
            history = get_user_history_cached(username, auth_service)
            if not history:
                st.info("📭 No history yet.")
            else:
                for item in history:
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([4, 1, 1])
                        with col1:
                            st.markdown(f"**{item.get('title', 'Untitled')}**")
                            st.caption(f"{item.get('type', 'N/A').title()} | {item.get('model', 'N/A')}")
                        with col2:
                            if st.button("🔄", key=f"load_{item['id']}", help="Reload this prompt"):
                                full_brief = item.get('full_brief', {})
                                st.session_state['director_subject'] = full_brief.get('subject', '')
                                st.session_state['director_environment'] = full_brief.get('environment', '')
                                st.session_state['director_shot_type'] = full_brief.get('shot_type', 'Default')
                                st.session_state['director_lighting'] = full_brief.get('lighting', '')
                                st.session_state['director_mood'] = full_brief.get('mood', 'Default')
                                st.session_state['director_style'] = full_brief.get('style', [])
                                st.session_state['director_negative'] = full_brief.get('negative', '')
                                st.session_state['director_color'] = full_brief.get('color', '')
                                st.toast("Full prompt reloaded!", icon="🔄")
                                # This is the final fix for the RerunData error
                        with col3:
                            if st.button("🗑️", key=f"del_{item['id']}", help="Delete history item"):
                                if auth_service.delete_history_item(username, item['id']):
                                    st.toast("Deleted!", icon="🗑️")
                                    st.rerun()

    st.markdown(f"<h1>{config.APP_ICON} AI Prompt Studio</h1>", unsafe_allow_html=True)
    st.markdown("Use the 'Scene Director' panel below to craft your vision. Or, for a spark of creativity, click the 'Inspire Me' button.")

    if st.button("🎲 Inspire Me!", use_container_width=True):
        generate_inspiration(ai_service, auth_service)

    st.markdown("---")
    st.markdown("## 🎬 Scene Director")

    default_values = {
        'director_subject': '', 'director_environment': '', 'director_shot_type': 'Default',
        'director_lighting': '', 'director_mood': 'Default', 'director_style': [],
        'director_negative': '', 'director_color': ''
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    with st.form(key='prompt_generation_form'):
        subject = st.text_input(
            "1. **Subject & Action**", value=st.session_state.get('director_subject', ''),
            placeholder="e.g., A lone android contemplating a holographic flower...", key='form_subject'
        )
        environment = st.text_area(
            "2. **Environment & Setting**", value=st.session_state.get('director_environment', ''),
            placeholder="e.g., ...on a rain-slicked rooftop overlooking a sprawling neon city in 2242.",
            height=100, key='form_environment'
        )
        st.markdown("3. **Artistic & Technical Details**")
        col1, col2 = st.columns(2)
        with col1:
            shot_type = st.selectbox("Shot Type", ["Default", "Cinematic Wide Shot", "Epic Low-Angle Shot", "Intimate Close-Up", "Dynamic Action Shot", "Drone Shot"], key='form_shot_type')
            lighting = st.text_input("Lighting", value=st.session_state.get('director_lighting', ''), placeholder="e.g., Chiaroscuro with volumetric fog", key='form_lighting')
            mood = st.selectbox("Mood", ["Default", "Ominous & Foreboding", "Serene & Peaceful", "Epic & Grandiose", "Melancholic", "Energetic & Chaotic"], key='form_mood')
        with col2:
            style = st.multiselect("Style Keywords", ["Photorealistic", "Cinematic", "Steampunk", "Cyberpunk", "Fantasy", "Film Noir", "Watercolor", "Oil Painting", "Gothic"], default=st.session_state.get('director_style', []), key='form_style')
            color = st.text_input("Color Palette", value=st.session_state.get('director_color', ''), placeholder="e.g., Dominated by deep blues and electric pinks", key='form_color')
            negative = st.text_input("Negative Prompts", value=st.session_state.get('director_negative', ''), placeholder="e.g., blurry, cartoon, disfigured", key='form_negative')
        
        col3, col4 = st.columns(2)
        with col3:
            prompt_type = st.radio("Target Format:", options=["Photo", "Video"], horizontal=True, key='form_prompt_type')
        with col4:
            model_options = ["Imagen 4", "Midjourney"] if prompt_type == "Photo" else ["Veo", "Sora"]
            target_model = st.selectbox("Target Model", options=model_options, key='form_target_model')
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Generate Brief & Prompt", use_container_width=True)
        
        if submitted:
            if not subject.strip():
                st.warning("Please describe the 'Subject & Action' for your scene.")
            else:
                st.session_state.update({
                    'director_subject': subject, 'director_environment': environment, 'director_shot_type': shot_type,
                    'director_lighting': lighting, 'director_mood': mood, 'director_style': style,
                    'director_color': color, 'director_negative': negative
                })
                
                director_brief_text = build_enhanced_director_brief(
                    subject=subject, environment=environment, shot_type=shot_type, lighting=lighting, mood=mood,
                    color=color, style=style, negative=negative, prompt_type=prompt_type, target_model=target_model
                )
                
                with st.spinner("Helios is crafting your masterpiece..."):
                    try:
                        response_dict = ai_service.get_ai_response(director_brief_text)
                        
                        content = {}
                        if isinstance(response_dict, dict):
                            content = response_dict
                        elif isinstance(response_dict, str):
                            content = {'creative_brief': response_dict, 'technical_prompt': response_dict}
                        else:
                            content = {'technical_prompt': str(response_dict)}

                        st.session_state['brief_result'] = {
                            'id': str(uuid.uuid4()), 'content': content, 'model': target_model,
                            'timestamp': datetime.now(), 'type': prompt_type, 'input': subject,
                            'full_brief': {
                                'subject': subject, 'environment': environment, 'shot_type': shot_type,
                                'lighting': lighting, 'mood': mood, 'color': color, 'style': style, 'negative': negative
                            },
                            'title': subject[:50] + "..." if len(subject) > 50 else subject
                        }
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        logger.error(f"AI response generation failed: {e}")
                        with st.expander("Debug Information", expanded=False):
                            st.code(f"Error: {str(e)}\nBrief: {director_brief_text}", language="text")

    if st.session_state.get('brief_result'):
        result = st.session_state['brief_result']
        content = result.get('content', {})
        st.markdown("---")
        st.success("✅ Your creative package is ready!")
        
        if content.get('creative_brief'): 
            st.subheader("📖 Creative Brief")
            st.markdown(content['creative_brief'])
        
        if content.get('technical_prompt'): 
            st.subheader("🤖 Technical Prompt")
            st.code(content['technical_prompt'], language="text")
        
        st.caption(f"Optimized for: {result.get('model', 'N/A')}")
        
        if content.get('quality_report'):
            with st.expander("📊 Quality Report", expanded=False):
                qr = content['quality_report']
                score = qr.get('score', 0)
                color = "green" if score >= 85 else "orange" if score >= 60 else "red"
                st.progress(score / 100)
                st.markdown(f"**Overall Quality Score:** <span style='color:{color}; font-size: 1.2em;'>{score}/100</span>", unsafe_allow_html=True)
                if qr.get('issues'):
                    st.markdown("**Areas for Improvement:**")
                    for issue in qr['issues']: st.warning(f"🔸 {issue}")
                else:
                    st.success("✅ No major issues detected. Great prompt!")
        
        if st.button("💾 Save to History"):
            if auth_service.save_generation_to_history(st.session_state.get('username'), result):
                st.toast("Saved!", icon="💾")
            else:
                st.error("Failed to save to history.")

def display_auth_page(auth_service: AuthService) -> None:
    """Displays authentication tabs for login and registration."""
    st.markdown(f"<h1 style='text-align: center;'>{config.APP_ICON} Welcome to AI Prompt Studio</h1>", unsafe_allow_html=True)
    
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                if username and password:
                    try:
                        success, message, user_name = auth_service.login_user(username, password)
                        if success:
                            st.session_state['authentication_status'] = True
                            st.session_state['name'] = user_name
                            st.session_state['username'] = username.lower().strip()
                            st.session_state['just_logged_in'] = True
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Login error: {str(e)}")
                        logger.error(f"Login failed: {e}")
                else:
                    st.error("Please enter username and password.")
    
    with register_tab:
        with st.form("register_form"):
            username_reg = st.text_input("Username")
            name = st.text_input("Name")
            email_reg = st.text_input("Email")
            password_reg = st.text_input("Password", type="password")
            password_confirm = st.text_input("Repeat Password", type="password")
            
            if st.form_submit_button("Register"):
                if password_reg != password_confirm:
                    st.error("Passwords do not match.")
                elif not all([username_reg, name, email_reg, password_reg]):
                    st.error("Please fill all fields.")
                else:
                    try:
                        success, message = auth_service.register_user(name, email_reg, username_reg, password_reg)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Registration error: {str(e)}")
                        logger.error(f"Registration failed: {e}")

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title=config.APP_TITLE, 
        page_icon=config.APP_ICON, 
        layout="wide"
    )
    
    try:
        auth_service = initialize_auth_service()
        
        if not st.session_state.get("authentication_status", False):
            display_auth_page(auth_service)
        else:
            ai_service = initialize_ai_service()
            show_unified_app_page(ai_service, auth_service)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {e}")

if __name__ == "__main__":
    main()