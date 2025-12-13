import os
from nacl.bindings import (
    crypto_aead_xchacha20poly1305_ietf_encrypt,
    crypto_aead_xchacha20poly1305_ietf_decrypt,
)

KEY = os.environ["APP_CRYPTO_KEY"].encode()
NONCE_SIZE = 24

def encrypt_text(text: str):
    nonce = os.urandom(NONCE_SIZE)
    ciphertext = crypto_aead_xchacha20poly1305_ietf_encrypt(
        text.encode(),
        b"",
        nonce,
        KEY
    )
    return ciphertext, nonce

def decrypt_text(ciphertext: bytes, nonce: bytes):
    return crypto_aead_xchacha20poly1305_ietf_decrypt(
        ciphertext, b"", nonce, KEY
    ).decode()