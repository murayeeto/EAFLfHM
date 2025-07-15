"""
Cryptographic utilities for secure federated learning
"""

import base64
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os
from typing import Tuple, Union


def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generate RSA key pair for secure communication
    
    Returns:
        Tuple of (private_key, public_key) in PEM format
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem


def encrypt_data(data: Union[str, bytes], public_key_pem: bytes) -> bytes:
    """
    Encrypt data using RSA public key with hybrid encryption
    (RSA for key exchange, AES for actual data)
    
    Args:
        data: Data to encrypt
        public_key_pem: Public key in PEM format
        
    Returns:
        Encrypted data
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    # Load public key
    public_key = serialization.load_pem_public_key(
        public_key_pem, 
        backend=default_backend()
    )
    
    # Generate AES key
    aes_key = os.urandom(32)  # 256-bit AES key
    iv = os.urandom(16)  # 128-bit IV
    
    # Encrypt data with AES
    cipher = Cipher(
        algorithms.AES(aes_key), 
        modes.CBC(iv), 
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    
    # Pad data to block size
    padding_length = 16 - (len(data) % 16)
    padded_data = data + bytes([padding_length] * padding_length)
    
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    # Encrypt AES key with RSA
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Combine encrypted key, IV, and encrypted data
    result = {
        'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
        'iv': base64.b64encode(iv).decode('utf-8'),
        'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8')
    }
    
    return json.dumps(result).encode('utf-8')


def decrypt_data(encrypted_data: bytes, private_key_pem: bytes) -> str:
    """
    Decrypt data using RSA private key with hybrid encryption
    
    Args:
        encrypted_data: Encrypted data package
        private_key_pem: Private key in PEM format
        
    Returns:
        Decrypted data as string
    """
    # Parse encrypted package
    package = json.loads(encrypted_data.decode('utf-8'))
    encrypted_key = base64.b64decode(package['encrypted_key'])
    iv = base64.b64decode(package['iv'])
    encrypted_content = base64.b64decode(package['encrypted_data'])
    
    # Load private key
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None,
        backend=default_backend()
    )
    
    # Decrypt AES key
    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Decrypt data with AES
    cipher = Cipher(
        algorithms.AES(aes_key), 
        modes.CBC(iv), 
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    
    padded_data = decryptor.update(encrypted_content) + decryptor.finalize()
    
    # Remove padding
    padding_length = padded_data[-1]
    data = padded_data[:-padding_length]
    
    return data.decode('utf-8')


def sign_data(data: bytes, private_key_pem: bytes) -> bytes:
    """
    Sign data using RSA private key
    
    Args:
        data: Data to sign
        private_key_pem: Private key in PEM format
        
    Returns:
        Digital signature
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None,
        backend=default_backend()
    )
    
    signature = private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    return signature


def verify_signature(data: bytes, signature: bytes, public_key_pem: bytes) -> bool:
    """
    Verify digital signature
    
    Args:
        data: Original data
        signature: Digital signature
        public_key_pem: Public key in PEM format
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        public_key = serialization.load_pem_public_key(
            public_key_pem, 
            backend=default_backend()
        )
        
        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False


class SecureAggregator:
    """
    Secure aggregation for federated learning
    Implements simplified secure multi-party computation
    """
    
    def __init__(self):
        self.participant_keys = {}
    
    def add_participant(self, participant_id: str, public_key: bytes):
        """Add a participant's public key"""
        self.participant_keys[participant_id] = public_key
    
    def aggregate_securely(self, encrypted_updates: dict) -> dict:
        """
        Perform secure aggregation of model updates
        
        Note: This is a simplified implementation.
        In practice, you would use more sophisticated secure aggregation protocols.
        """
        # Placeholder for secure aggregation logic
        # In a real implementation, this would involve:
        # 1. Secret sharing of model parameters
        # 2. Homomorphic encryption operations
        # 3. Secure multi-party computation protocols
        
        aggregated_params = {}
        participant_count = len(encrypted_updates)
        
        # Simple averaging (not actually secure in this implementation)
        for participant_id, encrypted_update in encrypted_updates.items():
            # In practice, you would perform operations on encrypted data
            # For now, we'll simulate the aggregation
            pass
        
        return aggregated_params


def add_differential_privacy_noise(
    gradients: dict, 
    noise_multiplier: float = 1.1, 
    max_grad_norm: float = 1.0
) -> dict:
    """
    Add differential privacy noise to gradients
    
    Args:
        gradients: Model gradients
        noise_multiplier: Noise scale
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Noisy gradients
    """
    import torch
    
    noisy_gradients = {}
    
    for name, grad in gradients.items():
        if isinstance(grad, torch.Tensor):
            # Clip gradients
            grad_norm = torch.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
            
            # Add Gaussian noise
            noise = torch.normal(
                mean=0.0, 
                std=noise_multiplier * max_grad_norm, 
                size=grad.shape
            )
            noisy_gradients[name] = grad + noise
        else:
            noisy_gradients[name] = grad
    
    return noisy_gradients
