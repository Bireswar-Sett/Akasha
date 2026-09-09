import React, { useState } from 'react';
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  AlertCircle,
  ArrowRight,
  Satellite
} from 'lucide-react';
import {
  signIn,
  signUp,
  signInWithRedirect,
  getCurrentUser
} from 'aws-amplify/auth';
import logoSrc from '../assets/logo.png';

const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true';

/* Google icon */
const GoogleIcon = () => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <path
      fill="#4285F4"
      d="M21.35 12.27c0-.79-.07-1.55-.2-2.27H12v4.3h5.25a4.49 4.49 0 0 1-1.95 2.95v2.45h3.15c1.84-1.69 2.9-4.18 2.9-7.43z"
    />
    <path
      fill="#34A853"
      d="M12 21.9c2.63 0 4.84-.87 6.45-2.36l-3.15-2.45c-.87.58-1.98.92-3.3.92-2.54 0-4.69-1.72-5.46-4.03H3.29v2.53A9.75 9.75 0 0 0 12 21.9z"
    />
    <path
      fill="#FBBC05"
      d="M6.54 13.98A5.87 5.87 0 0 1 6.23 12c0-.69.12-1.36.31-1.98V7.49H3.29A9.74 9.74 0 0 0 2.25 12c0 1.57.38 3.06 1.04 4.51l3.25-2.53z"
    />
    <path
      fill="#EA4335"
      d="M12 5.99c1.43 0 2.71.49 3.72 1.46l2.79-2.79C16.83 3.09 14.62 2.1 12 2.1a9.75 9.75 0 0 0-8.71 5.39l3.25 2.53C7.31 7.71 9.46 5.99 12 5.99z"
    />
  </svg>
);

export default function Login({ onLogin }) {
  const [tab, setTab] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [videoError, setVideoError] = useState(false);

  const clearError = () => setError('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    clearError();

    const cleanEmail = email.trim();

    if (!cleanEmail || !password.trim()) {
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

    /*
     * ------------------------------------------------------------
     * DEMO / JUDGING MODE
     * ------------------------------------------------------------
     * Judges can enter arbitrary email/password credentials.
     * No Cognito account or verification is required.
     */
    if (isDemoMode) {
      const demoEmail = cleanEmail || 'judge@akasha.demo';
      const demoName = demoEmail.includes('@')
        ? demoEmail.split('@')[0]
        : demoEmail;

      onLogin({
        id: `demo-${demoEmail.toLowerCase()}`,
        email: demoEmail,
        name: demoName || 'Demo User',
        avatar: null,
        provider: 'demo'
      });

      setLoading(false);
      return;
    }

    /*
     * ------------------------------------------------------------
     * REAL COGNITO AUTHENTICATION
     * ------------------------------------------------------------
     */
    try {
      if (tab === 'signin') {
        const result = await signIn({
          username: cleanEmail,
          password
        });

        if (result.isSignedIn) {
          const user = await getCurrentUser();

          onLogin({
            id: user.userId,
            email: cleanEmail,
            name:
              user.username ||
              cleanEmail.split('@')[0],
            avatar: null,
            provider: 'email'
          });

          setLoading(false);
          return;
        }

        setLoading(false);
        setError(
          `Additional authentication step required: ${
            result.nextStep?.signInStep ||
            'please continue'
          }`
        );

        return;
      }

      const result = await signUp({
        username: cleanEmail,
        password,
        options: {
          userAttributes: {
            email: cleanEmail
          }
        }
      });

      if (result.isSignUpComplete) {
        /*
         * The Cognito Pre Sign-up trigger automatically confirms the
         * account, so sign the user in immediately instead of sending
         * them back to the sign-in form.
         */
        const signInResult = await signIn({
          username: cleanEmail,
          password
        });

        if (signInResult.isSignedIn) {
          const user = await getCurrentUser();

          onLogin({
            id: user.userId,
            email: cleanEmail,
            name:
              user.username ||
              cleanEmail.split('@')[0],
            avatar: null,
            provider: 'email'
          });

          setLoading(false);
          return;
        }

        /*
         * Defensive fallback in case Cognito requires an additional
         * authentication step despite the account being created.
         */
        setTab('signin');
        setPassword('');
        setConfirmPwd('');
        setLoading(false);
        setError(
          `Account created, but an additional authentication step is required: ${
            signInResult.nextStep?.signInStep ||
            'please continue'
          }`
        );

        return;
      }

      setLoading(false);
      setError(
        'Account created, but additional authentication is required. Please continue.'
      );
    } catch (err) {
      console.error(
        '[AKASHA] Cognito authentication error:',
        err
      );

      let friendlyMessage =
        'Authentication failed. Please check your credentials.';

      switch (err.name) {
        case 'UserNotFoundException':
        case 'NotAuthorizedException':
          friendlyMessage = 'Invalid email or password.';
          break;

        case 'UsernameExistsException':
          friendlyMessage =
            'An account already exists with this email.';
          break;

        case 'InvalidPasswordException':
          friendlyMessage =
            'Password does not meet the required security policy.';
          break;

        case 'UserNotConfirmedException':
          friendlyMessage =
            'Your account has not been verified yet.';
          break;

        case 'LimitExceededException':
        case 'TooManyRequestsException':
          friendlyMessage =
            'Too many attempts. Please wait a moment and try again.';
          break;

        default:
          if (err?.message) {
            friendlyMessage = err.message;
          }
      }

      setError(friendlyMessage);
      setLoading(false);
    }
  };

  /*
   * ------------------------------------------------------------
   * GOOGLE
   * ------------------------------------------------------------
   * IMPORTANT:
   *
   * Do NOT manually construct the /oauth2/authorize URL here.
   * Amplify must own the OAuth transaction so that its PKCE/state
   * information survives the redirect and the returned code is
   * exchanged for Cognito tokens correctly.
   *
   * SELECT_ACCOUNT asks Google to show the account chooser even
   * when a Google session already exists.
   */
  const handleGoogle = async () => {
    clearError();
    setLoading(true);

    try {
      await signInWithRedirect({
        provider: 'Google'
      });
    } catch (err) {
      console.error(
        '[AKASHA] Google authentication error:',
        err
      );

      setError(
        err?.message || 'Google authentication failed.'
      );

      setLoading(false);
    }
  };

  return (
    <div className="login-page" role="main">
      <div
        className="login-bg-grid"
        aria-hidden="true"
      />

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

              <div
                style={{
                  textAlign: 'center',
                  zIndex: 2,
                  padding: '20px',
                  color: 'rgba(255,255,255,0.7)'
                }}
              >
                <Satellite
                  size={48}
                  color="#ffffff"
                  style={{
                    margin: '0 auto 12px',
                    opacity: 0.85
                  }}
                />

                <p
                  style={{
                    fontSize: '0.85rem',
                    fontWeight: 500,
                    color: '#ffffff'
                  }}
                >
                  Video Preview Ready
                </p>

                <p
                  style={{
                    fontSize: '0.74rem',
                    color: '#a1a1aa',
                    marginTop: '4px'
                  }}
                >
                  Place your animation in{' '}
                  <code>
                    frontend/public/login-video.mp4
                  </code>
                </p>
              </div>
            </div>
          )}

          <div className="login-video-overlay" />

          <div className="login-video-content">
            <h2 className="login-video-title">
              Autonomous Satellite Analysis & Temporal
              Inference
            </h2>

            <p className="login-video-subtitle">
              Multi-model remote sensing orchestration
              across high-resolution optical and SAR
              imagery.
            </p>
          </div>
        </div>

        {/* Right Side: Auth Card */}
        <div className="login-card-panel">
          <div
            className="login-card"
            role="dialog"
            aria-label="Authentication"
          >
            {/* Header Branding */}
            <div className="login-header">
              <img
                src={logoSrc}
                alt="AKASHA"
                className="login-logo-img"
                draggable="false"
              />

              <h1 className="login-title">
                AKASHA
              </h1>

              <p className="login-tagline">
                Earth Observation & Geospatial AI
              </p>
            </div>

            {/* Demo mode notice */}
            {isDemoMode && (
              <div className="login-demo-notice">
                <AlertCircle
                  size={14}
                  style={{
                    flexShrink: 0,
                    marginTop: '2px',
                    color: 'var(--text-muted)'
                  }}
                />

                <span>
                  Local demo mode. Enter any
                  credentials to continue.
                </span>
              </div>
            )}

            {/* Segmented Tab Switcher */}
            <div
              className="login-tabs"
              role="tablist"
            >
              <button
                role="tab"
                aria-selected={tab === 'signin'}
                className={`login-tab ${
                  tab === 'signin'
                    ? 'login-tab-active'
                    : ''
                }`}
                onClick={() => {
                  setTab('signin');
                  clearError();
                }}
                id="tab-signin"
                type="button"
              >
                Sign In
              </button>

              <button
                role="tab"
                aria-selected={tab === 'signup'}
                className={`login-tab ${
                  tab === 'signup'
                    ? 'login-tab-active'
                    : ''
                }`}
                onClick={() => {
                  setTab('signup');
                  clearError();
                }}
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
              {/* Email */}
              <div className="login-field">
                <Mail
                  size={15}
                  className="login-field-icon"
                />

                <input
                  id="login-email"
                  type="email"
                  autoComplete="email"
                  placeholder="name@organization.com"
                  className="glass-input login-input"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    clearError();
                  }}
                  disabled={loading}
                  required
                  aria-label="Email address"
                />
              </div>

              {/* Password */}
              <div className="login-field">
                <Lock
                  size={15}
                  className="login-field-icon"
                />

                <input
                  id="login-password"
                  type={
                    showPwd ? 'text' : 'password'
                  }
                  autoComplete={
                    tab === 'signup'
                      ? 'new-password'
                      : 'current-password'
                  }
                  placeholder="Password"
                  className="glass-input login-input"
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    clearError();
                  }}
                  disabled={loading}
                  required
                  aria-label="Password"
                />

                <button
                  type="button"
                  className="login-pwd-toggle"
                  onClick={() =>
                    setShowPwd((v) => !v)
                  }
                  aria-label={
                    showPwd
                      ? 'Hide password'
                      : 'Show password'
                  }
                  tabIndex={0}
                >
                  {showPwd ? (
                    <EyeOff size={15} />
                  ) : (
                    <Eye size={15} />
                  )}
                </button>
              </div>

              {/* Confirm Password */}
              {tab === 'signup' && (
                <div className="login-field animate-fade-in">
                  <Lock
                    size={15}
                    className="login-field-icon"
                  />

                  <input
                    id="login-confirm-password"
                    type={
                      showPwd ? 'text' : 'password'
                    }
                    autoComplete="new-password"
                    placeholder="Confirm password"
                    className="glass-input login-input"
                    value={confirmPwd}
                    onChange={(e) => {
                      setConfirmPwd(e.target.value);
                      clearError();
                    }}
                    disabled={loading}
                    aria-label="Confirm password"
                  />
                </div>
              )}

              {/* Error */}
              {error && (
                <div
                  className="login-error"
                  role="alert"
                >
                  <AlertCircle
                    size={14}
                    style={{
                      flexShrink: 0
                    }}
                  />

                  <span>{error}</span>
                </div>
              )}

              {/* Submit */}
              <button
                type="submit"
                id="login-submit-btn"
                className="btn-primary"
                disabled={loading}
                aria-busy={loading}
                style={{
                  width: '100%',
                  marginTop: '4px'
                }}
              >
                {loading ? (
                  <>
                    <span
                      className="spinner"
                      aria-hidden="true"
                    />

                    <span>
                      Authenticating…
                    </span>
                  </>
                ) : (
                  <>
                    <span>
                      {tab === 'signin'
                        ? 'Sign In'
                        : 'Get Started'}
                    </span>

                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </form>

            {/* Divider */}
            <div
              className="login-divider"
              aria-hidden="true"
            >
              <span className="login-divider-line" />

              <span className="login-divider-text">
                or continue with
              </span>

              <span className="login-divider-line" />
            </div>

            {/* Google OAuth */}
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