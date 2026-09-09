from __future__ import annotations

import logging
import time
from typing import Any, Dict

import jwt
import requests
from fastapi import HTTPException, status
from jwt.algorithms import RSAAlgorithm

from config import get_settings


logger = logging.getLogger("akasha.cognito")


class CognitoTokenVerifier:
    """
    Verify Amazon Cognito User Pool JWTs using the pool's JWKS.

    Supports:
      - Cognito access tokens
      - Cognito ID tokens

    The caller's bearer token must:
      - have a valid RSA signature
      - be unexpired
      - come from this Cognito user pool
      - be intended for the configured application
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.region = settings.cognito_region
        self.user_pool_id = settings.cognito_user_pool_id
        self.client_id = settings.cognito_client_id

        self.issuer = (
            f"https://cognito-idp.{self.region}.amazonaws.com/"
            f"{self.user_pool_id}"
        )

        self.jwks_url = (
            f"{self.issuer}/.well-known/jwks.json"
        )

        self._jwks: dict[str, Any] | None = None
        self._jwks_loaded_at = 0.0

        # Refresh keys periodically rather than fetching JWKS
        # for every API request.
        self._jwks_ttl_seconds = 3600

    def _load_jwks(self, force: bool = False) -> dict[str, Any]:
        now = time.time()

        if (
            not force
            and self._jwks is not None
            and (now - self._jwks_loaded_at)
            < self._jwks_ttl_seconds
        ):
            return self._jwks

        try:
            response = requests.get(
                self.jwks_url,
                timeout=10,
            )

            response.raise_for_status()

            jwks = response.json()

            if not isinstance(jwks, dict) or "keys" not in jwks:
                raise ValueError(
                    "Invalid Cognito JWKS response"
                )

            self._jwks = jwks
            self._jwks_loaded_at = now

            return jwks

        except Exception as exc:
            logger.exception(
                "Failed to load Cognito JWKS"
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service temporarily unavailable",
            ) from exc

    def _get_signing_key(
        self,
        kid: str,
    ) -> Any:
        jwks = self._load_jwks()

        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return RSAAlgorithm.from_jwk(key)

        # Cognito can rotate signing keys. Refresh immediately
        # once if the key wasn't found.
        jwks = self._load_jwks(force=True)

        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return RSAAlgorithm.from_jwk(key)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    def verify(
        self,
        token: str,
    ) -> Dict[str, Any]:
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials were not provided",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        try:
            # Read header only to determine which JWKS key to use.
            header = jwt.get_unverified_header(token)

            kid = header.get("kid")

            if not kid:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                    headers={
                        "WWW-Authenticate": "Bearer"
                    },
                )

            algorithm = header.get("alg")

            if algorithm != "RS256":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unsupported authentication token",
                    headers={
                        "WWW-Authenticate": "Bearer"
                    },
                )

            signing_key = self._get_signing_key(kid)

            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=self.issuer,
                options={
                    "require": [
                        "exp",
                        "iat",
                        "iss",
                    ]
                },
            )

            token_use = claims.get("token_use")

            if token_use == "access":
                # Cognito access tokens identify the app through
                # client_id rather than the ID-token `aud` claim.
                if claims.get("client_id") != self.client_id:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token was not issued to this application",
                        headers={
                            "WWW-Authenticate": "Bearer"
                        },
                    )

            elif token_use == "id":
                if claims.get("aud") != self.client_id:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token was not issued to this application",
                        headers={
                            "WWW-Authenticate": "Bearer"
                        },
                    )

            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                    headers={
                        "WWW-Authenticate": "Bearer"
                    },
                )

            return claims

        except HTTPException:
            raise

        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token has expired",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            ) from exc

        except jwt.InvalidIssuerError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication issuer",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            ) from exc

        except jwt.InvalidTokenError as exc:
            logger.warning(
                "Cognito JWT validation failed: %s",
                type(exc).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected Cognito token verification error"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            ) from exc


_verifier: CognitoTokenVerifier | None = None


def get_cognito_verifier() -> CognitoTokenVerifier:
    global _verifier

    if _verifier is None:
        _verifier = CognitoTokenVerifier()

    return _verifier


def verify_cognito_token(
    token: str,
) -> Dict[str, Any]:
    return get_cognito_verifier().verify(token)