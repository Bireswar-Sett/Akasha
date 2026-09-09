import { Amplify } from 'aws-amplify';

/*
 * Cognito configuration is read from Vite environment variables.
 *
 * Required vars in .env / .env.production:
 *   VITE_COGNITO_USER_POOL_ID
 *   VITE_COGNITO_CLIENT_ID
 *   VITE_COGNITO_DOMAIN       (e.g. eu-north-1r71nmtdez.auth.eu-north-1.amazoncognito.com)
 *   VITE_COGNITO_REDIRECT_URL (comma-separated list, or defaults to current origin)
 */

const userPoolId   = import.meta.env.VITE_COGNITO_USER_POOL_ID   || 'eu-north-1_R71nMtdeZ';
const clientId     = import.meta.env.VITE_COGNITO_CLIENT_ID      || '2dj154bemrmeifpl2n6aob10ck';
const cognitoDomain = import.meta.env.VITE_COGNITO_DOMAIN        || 'eu-north-1r71nmtdez.auth.eu-north-1.amazoncognito.com';

// Support a comma-separated list of allowed redirect URLs or fall back to
// the current origin so the app works both locally and on Amplify.
const rawRedirects = import.meta.env.VITE_COGNITO_REDIRECT_URLS;
const baseRedirects = rawRedirects
  ? rawRedirects.split(',').map(u => u.trim()).filter(Boolean)
  : [
      'http://localhost:5173/',
      'https://main.d5t6w8xnwognw.amplifyapp.com/',
      'https://akasha.web.app/',
    ];

const redirectSet = new Set();
baseRedirects.forEach(u => {
  redirectSet.add(u);
  if (u.endsWith('/')) {
    redirectSet.add(u.slice(0, -1));
  } else {
    redirectSet.add(u + '/');
  }
});
if (typeof window !== 'undefined' && window.location?.origin) {
  redirectSet.add(window.location.origin);
  redirectSet.add(window.location.origin + '/');
}
const redirectUrls = Array.from(redirectSet);

const cognitoConfig = {
  Auth: {
    Cognito: {
      userPoolId,
      userPoolClientId: clientId,

      loginWith: {
        email: true,

        oauth: {
          domain: cognitoDomain,
          scopes: ['openid', 'email', 'profile', 'aws.cognito.signin.user.admin'],
          redirectSignIn:  redirectUrls,
          redirectSignOut: redirectUrls,
          responseType: 'code',
        },
      },

      signUpVerificationMethod: 'code',

      userAttributes: {
        email: {
          required: true,
        },
      },
    },
  },
};

Amplify.configure(cognitoConfig);

export default cognitoConfig;
