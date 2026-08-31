import {
  ref,
  uploadBytes,
  getDownloadURL,
} from "firebase/storage";

import { storage } from "../firebase/config";

export async function uploadSatelliteImage(file, taskId, imageType) {
  if (!file) {
    throw new Error("No image selected.");
  }

  const filePath = `satellite-images/${taskId}/${imageType}-${file.name}`;

  const storageRef = ref(storage, filePath);

  await uploadBytes(storageRef, file);

  const downloadURL = await getDownloadURL(storageRef);

  return {
    path: filePath,
    url: downloadURL,
    name: file.name,
    size: file.size,
    type: file.type,
  };
}