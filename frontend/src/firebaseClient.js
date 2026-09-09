/**
 * firebaseClient.js
 *
 * Firebase has been fully removed from AKASHA.
 * Auth → AWS Cognito (aws-amplify)
 * Storage → AWS S3 (presigned URLs via backend)
 * Sessions → localStorage
 *
 * This file is retained only to export `isDemoMode` so that any
 * component that still imports it doesn't break during cleanup.
 * It will be deleted once all import sites are updated.
 */

export const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true';

// Stubs — nothing below is functional.
export const auth = null;
export const db = null;
export const storage = null;
export const googleProvider = null;
