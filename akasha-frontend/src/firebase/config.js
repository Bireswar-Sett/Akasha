import { initializeApp } from "firebase/app";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyARo8FLMjUQ10l4JN9b3pK3vP0UCme_K1U",
  authDomain: "akasha-satquery.firebaseapp.com",
  projectId: "akasha-satquery",
  storageBucket: "akasha-satquery.firebasestorage.app",
  messagingSenderId: "964695172038",
  appId: "1:964695172038:web:c730bb66385441c89548c7",
};

const app = initializeApp(firebaseConfig);

export const storage = getStorage(app);