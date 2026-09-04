import React, { useState } from 'react';
import { Mail, Lock, Eye, EyeOff, AlertCircle, ArrowRight, Satellite, Globe, Layers } from 'lucide-react';
import {
  auth,
  googleProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  isDemoMode
} from '../firebaseClient';
import logoSrc from '../assets/logo.png';

/* Clean Google SVG icon */
const GoogleIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
    <path
      fill="#4285F4"
      d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.66-5.17 3.66-9.17z"
    />
    <path
      fill="#34A853"
      d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.34 24 12 24z"
    />
    <path
      fill="#FBBC05"
      d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.18 0 9.99 0 12s.45 3.82 1.25 5.42l4.03-3.15z"
    />
    <path
      fill="#EA4335"
      d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.34 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.98 6.72-4.98z"
    />
  </svg>
);

export default function Login({ onLogin }) {
  const [tab, setTab] = useState('signin'); // 'signin' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [videoError, setVideoError] = useState(false);

  const clearError = () => setError('');

  /* ── Firebase email/password submit ── */
  const handleSubmit = async (e) => {
    e.preventDefault();
    clearError();

    if (!email.trim() || !password.trim()) {
      setError('Please enter your email and password.');
      return;
    }
    if (tab === 'signup' && password !== confirmPwd) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }

    setLoading(true);

    // Demo mode – fallback if credentials not configured in .env
    if (isDemoMode) {
      setTimeout(() => {
        setLoading(false);
        onLogin({
          id: 'demo-user',
          email,
          name: email.split('@')[0],
          avatar: null,
          provider: 'email',
        });
      }, 500);
      return;
    }

    try {
      let userCredential;
      if (tab === 'signin') {
        userCredential = await signInWithEmailAndPassword(auth, email, password);
      } else {
        userCredential = await createUserWithEmailAndPassword(auth, email, password);
      }

      const user = userCredential.user;
      setLoading(false);
      onLogin({
        id: user.uid,
        email: user.email,
        name: user.displayName || user.email.split('@')[0],
        avatar: user.photoURL || null,
        provider: 'email',
      });
    } catch (err) {
      let friendlyMessage = 'Authentication failed. Please check your credentials.';
      if (err.code === 'auth/user-not-found' || err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
        friendlyMessage = 'Invalid email or password.';
      } else if (err.code === 'auth/email-already-in-use') {
        friendlyMessage = 'An account already exists with this email.';
      } else if (err.message) {
        friendlyMessage = err.message;
      }
      setError(friendlyMessage);
      setLoading(false);
    }
  };

  /* ── Firebase Google OAuth ── */
  const handleGoogle = async () => {
    clearError();
    if (isDemoMode) {
      setLoading(true);
      setTimeout(() => {
        setLoading(false);
        onLogin({
          id: 'demo-google-user',
          email: 'user@akasha.ai',
          name: 'Akasha Operator',
          avatar: null,
          provider: 'google',
        });
      }, 500);
      return;
    }

    setLoading(true);
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      setLoading(false);
      onLogin({
        id: user.uid,
        email: user.email,
        name: user.displayName || 'Operator',
        avatar: user.photoURL || null,
        provider: 'google',
      });
    } catch (err) {
      setError(err.message || 'Google authentication failed.');
      setLoading(false);
    }
  };

  return (
    <div className="login-page" role="main">
      <div className="login-bg-grid" aria-hidden="true" />

      <div className="login-split-container">
        {/* Left Side: Video Animation Showcase */}
        <div className="login-video-panel">
          {!videoError ? (
            <video
              src="/login-video.mp4"
              autoPlay
              loop
              muted
              playsInline
              className="login-video-element"
              onError={() => setVideoError(true)}
            />
          ) : (
            <div className="login-video-fallback-bg">
              <div className="login-video-fallback-grid" />
              <div style={{ textAlign: 'center', zIndex: 2, padding: '20px', color: 'rgba(255,255,255,0.7)' }}>
                <Satellite size={48} color="#ffffff" style={{ margin: '0 auto 12px', opacity: 0.85 }} />
                <p style={{ fontSize: '0.85rem', fontWeight: 500, color: '#ffffff' }}>Video Preview Ready</p>
                <p style={{ fontSize: '0.74rem', color: '#a1a1aa', marginTop: '4px' }}>
                  Place your animation in <code>frontend/public/login-video.mp4</code>
                </p>
              </div>
            </div>
          )}

          <div className="login-video-overlay" />

          {/* Captions / Showcase Text */}
          <div className="login-video-content">
            <h2 className="login-video-title">
              Autonomous Satellite Analysis & Temporal Inference
            </h2>
            <p className="login-video-subtitle">
              Multi-model remote sensing orchestration across high-resolution optical and SAR imagery.
            </p>
          </div>
        </div>

        {/* Right Side: Auth Card */}
        <div className="login-card-panel">
          <div className="login-card" role="dialog" aria-label="Authentication">
            {/* Header Branding */}
            <div className="login-header">
              <img
                src={logoSrc}
                alt="AKASHA"
                className="login-logo-img"
                draggable="false"
              />
              <h1 className="login-title">AKASHA</h1>
              <p className="login-tagline">
                Earth Observation & Geospatial AI
              </p>
            </div>

            {/* Demo mode notice */}
            {isDemoMode && (
              <div className="login-demo-notice">
                <AlertCircle size={14} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--text-muted)' }} />
                <span>
                  Local sandbox mode. Enter any credentials to continue.
                </span>
              </div>
            )}

            {/* Segmented Tab Switcher */}
            <div className="login-tabs" role="tablist">
              <button
                role="tab"
                aria-selected={tab === 'signin'}
                className={`login-tab ${tab === 'signin' ? 'login-tab-active' : ''}`}
                onClick={() => { setTab('signin'); clearError(); }}
                id="tab-signin"
                type="button"
              >
                Sign In
              </button>
              <button
                role="tab"
                aria-selected={tab === 'signup'}
                className={`login-tab ${tab === 'signup' ? 'login-tab-active' : ''}`}
                onClick={() => { setTab('signup'); clearError(); }}
                id="tab-signup"
                type="button"
              >
                Create Account
              </button>
            </div>

            {/* Auth Form */}
            <form
              onSubmit={handleSubmit}
              className="login-form"
              aria-labelledby={`tab-${tab}`}
              noValidate
            >
              {/* Email input */}
              <div className="login-field">
                <Mail size={15} className="login-field-icon" />
                <input
                  id="login-email"
                  type="email"
                  autoComplete="email"
                  placeholder="name@organization.com"
                  className="glass-input login-input"
                  value={email}
                  onChange={e => { setEmail(e.target.value); clearError(); }}
                  disabled={loading}
                  required
                  aria-label="Email address"
                />
              </div>

              {/* Password input */}
              <div className="login-field">
                <Lock size={15} className="login-field-icon" />
                <input
                  id="login-password"
                  type={showPwd ? 'text' : 'password'}
                  autoComplete={tab === 'signup' ? 'new-password' : 'current-password'}
                  placeholder="Password"
                  className="glass-input login-input"
                  value={password}
                  onChange={e => { setPassword(e.target.value); clearError(); }}
                  disabled={loading}
                  required
                  aria-label="Password"
                />
                <button
                  type="button"
                  className="login-pwd-toggle"
                  onClick={() => setShowPwd(v => !v)}
                  aria-label={showPwd ? 'Hide password' : 'Show password'}
                  tabIndex={0}
                >
                  {showPwd ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>

              {/* Confirm Password (sign-up only) */}
              {tab === 'signup' && (
                <div className="login-field animate-fade-in">
                  <Lock size={15} className="login-field-icon" />
                  <input
                    id="login-confirm-password"
                    type={showPwd ? 'text' : 'password'}
                    autoComplete="new-password"
                    placeholder="Confirm password"
                    className="glass-input login-input"
                    value={confirmPwd}
                    onChange={e => { setConfirmPwd(e.target.value); clearError(); }}
                    disabled={loading}
                    aria-label="Confirm password"
                  />
                </div>
              )}

              {/* Error notice */}
              {error && (
                <div className="login-error" role="alert">
                  <AlertCircle size={14} style={{ flexShrink: 0 }} />
                  <span>{error}</span>
                </div>
              )}

              {/* Primary Submit Button */}
              <button
                type="submit"
                id="login-submit-btn"
                className="btn-primary"
                disabled={loading}
                aria-busy={loading}
                style={{ width: '100%', marginTop: '4px' }}
              >
                {loading ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    <span>Authenticating…</span>
                  </>
                ) : (
                  <>
                    <span>{tab === 'signin' ? 'Sign In' : 'Get Started'}</span>
                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </form>

            {/* Divider */}
            <div className="login-divider" aria-hidden="true">
              <span className="login-divider-line" />
              <span className="login-divider-text">or continue with</span>
              <span className="login-divider-line" />
            </div>

            {/* Google OAuth Button */}
            <button
              type="button"
              id="login-google-btn"
              className="login-google-btn"
              onClick={handleGoogle}
              disabled={loading}
              aria-label="Sign in with Google"
            >
              <GoogleIcon />
              <span>Google Account</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
