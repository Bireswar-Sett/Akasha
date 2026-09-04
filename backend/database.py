import os
from typing import Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, firestore

from config import get_settings


def _initialize_firebase():
    """
    Initialize Firebase Admin SDK if it hasn't already been initialized.
    This is independent of firebase_service.py to avoid circular imports.
    """

    if firebase_admin._apps:
        return firebase_admin.get_app()

    settings = get_settings()

    key_path = settings.firebase_service_account_key_path
    bucket_name = settings.firebase_storage_bucket
    project_id = settings.firebase_project_id

    options = {}

    if bucket_name:
        options["storageBucket"] = bucket_name

    if project_id:
        options["projectId"] = project_id

    if key_path and os.path.isfile(key_path):
        cred = credentials.Certificate(key_path)
        return firebase_admin.initialize_app(
            cred,
            options,
        )

    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        cred = credentials.ApplicationDefault()
        return firebase_admin.initialize_app(
            cred,
            options,
        )

    return firebase_admin.initialize_app(
        options=options
    )


def get_firestore():
    """
    Get the Firestore database client.
    """

    _initialize_firebase()

    return firestore.client()


def create_user(
    user_id: str,
    email: str,
    password_hash: str,
) -> Dict[str, Any]:

    db = get_firestore()

    user_data = {
        "id": user_id,
        "email": email.lower(),
        "password_hash": password_hash,
    }

    db.collection("users").document(user_id).set(user_data)

    return user_data


def get_user_by_email(
    email: str,
) -> Optional[Dict[str, Any]]:

    db = get_firestore()

    query = (
        db.collection("users")
        .where(
            "email",
            "==",
            email.lower(),
        )
        .limit(1)
        .stream()
    )

    for document in query:
        return document.to_dict()

    return None


def get_user_by_id(
    user_id: str,
) -> Optional[Dict[str, Any]]:

    db = get_firestore()

    document = (
        db.collection("users")
        .document(user_id)
        .get()
    )

    if not document.exists:
        return None

    return document.to_dict()