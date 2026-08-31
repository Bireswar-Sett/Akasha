import React, { useState, useEffect, useRef } from 'react';
import { Mail, Lock, Eye, EyeOff, Satellite, Globe, Wifi, AlertCircle } from 'lucide-react';
import {
  auth,
  googleProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  isDemoMode
} from '../firebaseClient';
import logoSrc from '../assets/logo.png';

/* ─────────────────────────────────────────────
   Starfield canvas drawn purely in JS/Canvas
   ───────────────────────────────────────────── */
function StarField() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const STAR_COUNT = 220;
    const stars = Array.from({ length: STAR_COUNT }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.5 + 0.3,
      alpha: Math.random(),
      speed: Math.random() * 0.008 + 0.002,
      dir: Math.random() > 0.5 ? 1 : -1,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      stars.forEach(s => {
        s.alpha += s.speed * s.dir;
        if (s.alpha >= 1 || s.alpha <= 0) s.dir *= -1;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200, 220, 255, ${s.alpha})`;
        ctx.fill();
      });
      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}
    />
  );
}

/* ─────────────────────────────────────────────
   Orbit rings decorations
   ───────────────────────────────────────────── */
function OrbitalRings() {
  return (
    <div className="login-orbital-rings" aria-hidden="true">
      <div className="orbit-ring orbit-ring-1" />
      <div className="orbit-ring orbit-ring-2" />
      <div className="orbit-ring orbit-ring-3" />
      <div className="orbit-dot orbit-dot-1">
        <Satellite size={14} color="#00D4FF" />
      </div>
      <div className="orbit-dot orbit-dot-2">
        <Globe size={11} color="#6C63FF" />
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Google Icon SVG
   ───────────────────────────────────────────── */
const GoogleIcon = () => (
  <svg width="20" height="20" viewBox="0 0 48 48" aria-hidden="true">
    <path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303C33.654 32.657 29.332 36 24 36c-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/>
    <path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 19.001 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/>
    <path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.314 0-9.828-3.553-11.298-8.413H6.428C9.922 36.626 16.4 44 24 44z"/>
    <path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303a11.966 11.966 0 01-4.087 5.571l6.19 5.238C42.021 36.021 44 30.5 44 24c0-1.341-.138-2.65-.389-3.917z"/>
  </svg>
);

/* ─────────────────────────────────────────────
   Connection log messages
   ───────────────────────────────────────────── */
const LOG_SEQUENCE = [
  { msg: 'Initialising uplink protocol…', delay: 0 },
  { msg: 'Establishing secure satellite channel…', delay: 700 },
  { msg: 'Verifying identity token…', delay: 1500 },
  { msg: 'Authentication successful. Welcome aboard.', delay: 2400 },
];

function ConnectionLog({ visible }) {
  const [lines, setLines] = useState([]);

  useEffect(() => {
    if (!visible) { setLines([]); return; }
    const timers = LOG_SEQUENCE.map(({ msg, delay }) =>
      setTimeout(() => setLines(prev => [...prev, msg]), delay)
    );
    return () => timers.forEach(clearTimeout);
  }, [visible]);

  if (!visible && lines.length === 0) return null;

  return (
    <div className="login-conn-log" aria-live="polite">
      {lines.map((l, i) => (
        <div key={i} className="conn-log-line">
          <span className="conn-log-prompt">&gt;&nbsp;</span>
          <span>{l}</span>
        </div>
      ))}
      {visible && lines.length < LOG_SEQUENCE.length && (
        <div className="conn-log-cursor" />
      )}
    </div>
  );
}

export default function Login({ onLogin }) {
  const [tab, setTab] = useState('signin'); // 'signin' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [error, setError] = useState('');

  const clearError = () => setError('');

  /* ── Firebase email/password submit ── */
  const handleSubmit = async (e) => {
    e.preventDefault();
    clearError();

    if (!email.trim() || !password.trim()) {
      setError('Please fill in all fields.');
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
    setShowLog(true);

    // Demo mode – no real credentials set in .env
    if (isDemoMode) {
      await new Promise(r => setTimeout(r, 2800));
      setLoading(false);
      setShowLog(false);
      onLogin({
        id: 'demo-user',
        email,
        name: email.split('@')[0],
        avatar: null,
        provider: 'email',
      });
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
      await new Promise(r => setTimeout(r, 2800));
      setLoading(false);
      setShowLog(false);
      onLogin({
        id: user.uid,
        email: user.email,
        name: user.displayName || user.email.split('@')[0],
        avatar: user.photoURL || null,
        provider: 'email',
      });
    } catch (err) {
      let friendlyMessage = 'An error occurred during authentication.';
      if (err.code === 'auth/user-not-found' || err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
        friendlyMessage = 'Invalid email or password.';
      } else if (err.code === 'auth/email-already-in-use') {
        friendlyMessage = 'An account already exists with this email.';
      } else if (err.message) {
        friendlyMessage = err.message;
      }
      setError(friendlyMessage);
      setLoading(false);
      setShowLog(false);
    }
  };

  /* ── Firebase Google OAuth ── */
  const handleGoogle = async () => {
    clearError();
    if (isDemoMode) {
      setLoading(true);
      setShowLog(true);
      await new Promise(r => setTimeout(r, 2800));
      setLoading(false);
      setShowLog(false);
      onLogin({
        id: 'demo-google-user',
        email: 'demo@gmail.com',
        name: 'Demo Astronaut',
        avatar: null,
        provider: 'google',
      });
      return;
    }

    setLoading(true);
    setShowLog(true);

    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      await new Promise(r => setTimeout(r, 2800));
      setLoading(false);
      setShowLog(false);
      onLogin({
        id: user.uid,
        email: user.email,
        name: user.displayName || 'Astronaut',
        avatar: user.photoURL || null,
        provider: 'google',
      });
    } catch (err) {
      setError(err.message || 'Google authentication failed.');
      setLoading(false);
      setShowLog(false);
    }
  };

  return (
    <div className="login-page" role="main">
      {/* Starfield & orbital decorations */}
      <StarField />
      <OrbitalRings />

      {/* Login Card */}
      <div className="login-card" role="dialog" aria-label="Authentication">
        {/* Logo */}
        <div className="login-logo-wrap">
          <div className="login-logo-glow" aria-hidden="true" />
          <img
            src={logoSrc}
            alt="AKASHA – AI That Sees From the Sky"
            className="login-logo-img"
            draggable="false"
          />
        </div>

        {/* Tagline */}
        <p className="login-tagline">
          <Wifi size={13} style={{ marginRight: 5, verticalAlign: 'middle', color: '#00D4FF' }} />
          Satellite Intelligence Platform
        </p>

        {/* Demo mode notice */}
        {isDemoMode && (
          <div className="login-demo-notice">
            <AlertCircle size={13} />
            <span>Demo mode — configure <code>VITE_FIREBASE_API_KEY</code> in <code>.env</code> to connect GCP Identity Platform.</span>
          </div>
        )}

        {/* Tab switcher */}
        <div className="login-tabs" role="tablist">
          {['signin', 'signup'].map(t => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              className={`login-tab ${tab === t ? 'login-tab-active' : ''}`}
              onClick={() => { setTab(t); clearError(); }}
              id={`tab-${t}`}
            >
              {t === 'signin' ? 'Sign In' : 'Sign Up'}
            </button>
          ))}
        </div>

        {/* Form */}
        <form
          onSubmit={handleSubmit}
          className="login-form"
          aria-labelledby={`tab-${tab}`}
          noValidate
        >
          {/* Email */}
          <div className="login-field">
            <Mail size={16} className="login-field-icon" />
            <input
              id="login-email"
              type="email"
              autoComplete="email"
              placeholder="Email address"
              className="glass-input login-input"
              value={email}
              onChange={e => { setEmail(e.target.value); clearError(); }}
              disabled={loading}
              required
              aria-label="Email address"
            />
          </div>

          {/* Password */}
          <div className="login-field">
            <Lock size={16} className="login-field-icon" />
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
            <div className="login-field login-field-animate">
              <Lock size={16} className="login-field-icon" />
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

          {/* Error message */}
          {error && (
            <div className="login-error" role="alert">
              <AlertCircle size={14} />
              <span>{error}</span>
            </div>
          )}

          {/* Connection log */}
          <ConnectionLog visible={showLog} />

          {/* Submit */}
          <button
            type="submit"
            id="login-submit-btn"
            className="login-submit-btn glass-button"
            disabled={loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Connecting…
              </>
            ) : (
              <>
                <Satellite size={16} />
                {tab === 'signin' ? 'Sign In to AKASHA' : 'Create Account'}
              </>
            )}
          </button>
        </form>

        {/* Divider */}
        <div className="login-divider" aria-hidden="true">
          <span className="login-divider-line" />
          <span className="login-divider-text">or</span>
          <span className="login-divider-line" />
        </div>

        {/* Google Sign-In */}
        <button
          type="button"
          id="login-google-btn"
          className="login-google-btn"
          onClick={handleGoogle}
          disabled={loading}
          aria-label="Sign in with Google"
        >
          <GoogleIcon />
          <span>Continue with Google</span>
        </button>

        {/* Footer */}
        <p className="login-footer">
          {tab === 'signin'
            ? <>No account? <button className="login-footer-link" onClick={() => setTab('signup')}>Sign up free</button></>
            : <>Already have an account? <button className="login-footer-link" onClick={() => setTab('signin')}>Sign in</button></>
          }
        </p>
      </div>
    </div>
  );
}
