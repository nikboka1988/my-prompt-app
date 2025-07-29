# services/auth_service.py (PERFECTED VERSION: Firestore with Improved Handling)
import os
import logging
import bcrypt
import re
from google.cloud import firestore
from datetime import datetime
import uuid
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self):
        """Initialize Firestore connection."""
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.db: Optional[firestore.Client] = None
        self.users_collection = None
        self.is_initialized = False
        self._initialize_firestore()

    def _initialize_firestore(self) -> bool:
        """Initialize Firestore client and test connection."""
        try:
            if not self.project_id:
                logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
                return False
            self.db = firestore.Client(project=self.project_id)
            self.users_collection = self.db.collection('users')
            self.users_collection.limit(1).get()  # Test connection
            self.is_initialized = True
            logger.info("AuthService initialized successfully with Firestore")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}", exc_info=True)
            return False

    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """Validate password strength."""
        if len(password) < 6: return False, "Password must be at least 6 characters long"
        if not re.search(r'[A-Za-z]', password) or not re.search(r'\d', password):
            return False, "Password must contain at least one letter and one number"
        return True, ""

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def register_user(self, name: str, email: str, username: str, password: str) -> Tuple[bool, str]:
        """Register a new user with validation and hashing."""
        if not self.is_initialized: return False, "Database connection error."
        name, email, username = name.strip(), email.lower().strip(), username.lower().strip()
        
        password_valid, password_error = self._validate_password(password)
        if not password_valid: return False, password_error
        
        try:
            if self.users_collection.document(username).get().exists: return False, "This username is already taken."
            email_query = self.users_collection.where('email', '==', email).limit(1).stream()
            if len(list(email_query)) > 0: return False, "An account with this email already exists."

            user_data = {
                'name': name, 'email': email, 'password_hash': self._hash_password(password),
                'created_at': firestore.SERVER_TIMESTAMP, 'generation_count': 0, 'subscription_tier': 'free',
                'history': []  # Initialize empty history
            }
            self.users_collection.document(username).set(user_data)
            return True, "Registration successful! You can now log in."
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}", exc_info=True)
            return False, "An unexpected error occurred."

    def login_user(self, username: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """Authenticate user and update last login."""
        if not self.is_initialized: return False, "Database connection error.", None
        username = username.lower().strip()
        try:
            user_doc = self.users_collection.document(username).get()
            if not user_doc.exists: return False, "Invalid username or password.", None
            user_data = user_doc.to_dict()
            if not self._verify_password(password, user_data.get('password_hash', '')):
                return False, "Invalid username or password.", None
            self.users_collection.document(username).update({'last_login': firestore.SERVER_TIMESTAMP})
            return True, "Login successful!", user_data.get('name', username)
        except Exception as e:
            logger.error(f"Unexpected error during login: {e}", exc_info=True)
            return False, "An unexpected error occurred.", None

    def save_generation_to_history(self, username: str, result_data: Dict) -> bool:
        """Save generation to user history and increment count."""
        if not self.is_initialized: return False
        try:
            generation_id = result_data.get('id', str(uuid.uuid4()))
            input_text = result_data.get('input', '')
            history_data = {
                'id': generation_id, 'input': input_text,
                'content': result_data.get('content', {}), 'model': result_data.get('model', 'unknown'),
                'timestamp': result_data.get('timestamp', firestore.SERVER_TIMESTAMP),
                'type': result_data.get('type', 'unknown'),
                'title': input_text[:50] + ('...' if len(input_text) > 50 else '')
            }
            # Append to array
            self.users_collection.document(username.lower().strip()).update({
                'history': firestore.ArrayUnion([history_data]),
                'generation_count': firestore.Increment(1)
            })
            return True
        except Exception as e:
            logger.error(f"Error saving history for {username}: {e}", exc_info=True)
            return False

    def get_user_history(self, username: str, limit: int = 50) -> List[Dict]:
        """Retrieve user history, sorted by timestamp descending."""
        if not self.is_initialized: return []
        try:
            user_doc = self.users_collection.document(username.lower().strip()).get()
            if not user_doc.exists: return []
            history = user_doc.to_dict().get('history', [])
            # Sort by timestamp descending (newest first)
            history.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            return history[:limit]
        except Exception as e:
            logger.error(f"Error retrieving history for {username}: {e}")
            return []

    def delete_history_item(self, username: str, generation_id: str) -> bool:
        """Delete a specific history item."""
        if not self.is_initialized: return False
        try:
            user_doc = self.users_collection.document(username.lower().strip()).get()
            if not user_doc.exists: return False
            history = user_doc.to_dict().get('history', [])
            new_history = [item for item in history if item.get('id') != generation_id]
            self.users_collection.document(username.lower().strip()).update({'history': new_history})
            return True
        except Exception as e:
            logger.error(f"Error deleting history item for {username}: {e}")
            return False