import { initializeApp, getApps, getApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup as fbSignInWithPopup,
  signInWithEmailAndPassword as fbSignInWithEmailAndPassword,
  createUserWithEmailAndPassword as fbCreateUserWithEmailAndPassword,
  signOut as fbSignOut
} from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';

const rawApiKey = import.meta.env.VITE_FIREBASE_API_KEY || '';
const firebaseConfig = {
  apiKey: rawApiKey,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || '',
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || 'akasha-v1',
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || '',
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || '',
  appId: import.meta.env.VITE_FIREBASE_APP_ID || ''
};

const hasRequiredFirebaseConfig = Boolean(
  firebaseConfig.apiKey &&
  firebaseConfig.authDomain &&
  firebaseConfig.projectId &&
  firebaseConfig.storageBucket &&
  firebaseConfig.appId
);
const isDemoMode = !hasRequiredFirebaseConfig ||
  rawApiKey === 'your_firebase_api_key_here' ||
  rawApiKey.includes('your_firebase');

let app = null;
let auth = null;
let db = null;
let storage = null;
let googleProvider = null;

if (!isDemoMode) {
  try {
    app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
    auth = getAuth(app);
    db = getFirestore(app);
    storage = getStorage(app);
    googleProvider = new GoogleAuthProvider();
  } catch (err) {
    console.error('[AKASHA] Firebase initialization failed. Check the frontend Firebase config.', err);
    throw err;
  }
} else {
  console.info(
    '[AKASHA] Firebase credentials not set. ' +
    'Running in local Demo / Sandbox mode. ' +
    'To connect real Firebase, add credentials to frontend/.env'
  );
}

const signInWithPopup = async (authInstance, provider) => {
  if (isDemoMode) {
    return {
      user: {
        uid: 'demo-google-user',
        email: 'operator@akasha.ai',
        displayName: 'Akasha Operator',
        photoURL: null,
      }
    };
  }
  if (!auth || !authInstance) {
    throw new Error('Firebase auth is not initialized.');
  }
  return fbSignInWithPopup(authInstance, provider);
};

const signInWithEmailAndPassword = async (authInstance, email, password) => {
  if (isDemoMode) {
    return {
      user: {
        uid: `demo-${Date.now()}`,
        email,
        displayName: email.split('@')[0],
        photoURL: null,
      }
    };
  }
  if (!auth || !authInstance) {
    throw new Error('Firebase auth is not initialized.');
  }
  return fbSignInWithEmailAndPassword(authInstance, email, password);
};

const createUserWithEmailAndPassword = async (authInstance, email, password) => {
  if (isDemoMode) {
    return {
      user: {
        uid: `demo-${Date.now()}`,
        email,
        displayName: email.split('@')[0],
        photoURL: null,
      }
    };
  }
  if (!auth || !authInstance) {
    throw new Error('Firebase auth is not initialized.');
  }
  return fbCreateUserWithEmailAndPassword(authInstance, email, password);
};

const signOut = async (authInstance) => {
  if (isDemoMode) {
    return;
  }
  if (!auth || !authInstance) {
    throw new Error('Firebase auth is not initialized.');
  }
  return fbSignOut(authInstance);
};

export {
  auth,
  db,
  storage,
  googleProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  isDemoMode
};

