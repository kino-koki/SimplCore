package com.qingxiaotuan.simpcore.security;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;

public class ModelEncryption {

    private static final String ALGORITHM = "AES";
    private static final String TRANSFORMATION = "AES/GCM/NoPadding";
    private static final int GCM_IV_LENGTH = 12;
    private static final int GCM_TAG_LENGTH = 128;

    private final byte[] key;

    public ModelEncryption(byte[] key) {
        this.key = deriveKey(key);
    }

    public ModelEncryption(String passphrase) {
        this(passphrase.getBytes(StandardCharsets.UTF_8));
    }

    private byte[] deriveKey(byte[] input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return digest.digest(input);
        } catch (GeneralSecurityException e) {
            throw new SecurityException("Key derivation failed", e);
        }
    }

    public byte[] encrypt(float[] data) {
        try {
            byte[] plaintext = floatArrayToBytes(data);
            byte[] iv = generateIV();

            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            GCMParameterSpec spec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, ALGORITHM), spec);

            byte[] ciphertext = cipher.doFinal(plaintext);

            ByteBuffer buffer = ByteBuffer.allocate(iv.length + ciphertext.length);
            buffer.put(iv);
            buffer.put(ciphertext);
            return buffer.array();
        } catch (GeneralSecurityException e) {
            throw new SecurityException("Encryption failed", e);
        }
    }

    public float[] decrypt(byte[] encryptedData) {
        try {
            ByteBuffer buffer = ByteBuffer.wrap(encryptedData);
            byte[] iv = new byte[GCM_IV_LENGTH];
            buffer.get(iv);

            byte[] ciphertext = new byte[buffer.remaining()];
            buffer.get(ciphertext);

            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            GCMParameterSpec spec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, ALGORITHM), spec);

            byte[] plaintext = cipher.doFinal(ciphertext);
            return bytesToFloatArray(plaintext);
        } catch (GeneralSecurityException e) {
            throw new SecurityException("Decryption failed", e);
        }
    }

    public String encryptToString(float[] data) {
        byte[] encrypted = encrypt(data);
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public float[] decryptFromString(String encryptedString) {
        byte[] encrypted = Base64.getDecoder().decode(encryptedString);
        return decrypt(encrypted);
    }

    private byte[] generateIV() {
        byte[] iv = new byte[GCM_IV_LENGTH];
        new SecureRandom().nextBytes(iv);
        return iv;
    }

    private byte[] floatArrayToBytes(float[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * Float.BYTES);
        for (float f : data) {
            buffer.putFloat(f);
        }
        return buffer.array();
    }

    private float[] bytesToFloatArray(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        float[] data = new float[bytes.length / Float.BYTES];
        for (int i = 0; i < data.length; i++) {
            data[i] = buffer.getFloat();
        }
        return data;
    }

    public static byte[] generateRandomKey() {
        byte[] key = new byte[32];
        new SecureRandom().nextBytes(key);
        return key;
    }

    public static String generateKeyString() {
        return Base64.getEncoder().encodeToString(generateRandomKey());
    }
    
    public byte[] encryptWithHMAC(float[] data) {
        try {
            byte[] plaintext = floatArrayToBytes(data);
            byte[] iv = generateIV();
            
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            GCMParameterSpec spec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, ALGORITHM), spec);
            
            byte[] ciphertext = cipher.doFinal(plaintext);
            
            byte[] hmac = computeHMAC(iv, ciphertext);
            
            ByteBuffer buffer = ByteBuffer.allocate(iv.length + hmac.length + ciphertext.length);
            buffer.put(iv);
            buffer.put(hmac);
            buffer.put(ciphertext);
            return buffer.array();
        } catch (GeneralSecurityException e) {
            throw new SecurityException("Encryption with HMAC failed", e);
        }
    }
    
    public float[] decryptWithHMAC(byte[] encryptedData) {
        try {
            ByteBuffer buffer = ByteBuffer.wrap(encryptedData);
            byte[] iv = new byte[GCM_IV_LENGTH];
            buffer.get(iv);
            
            byte[] expectedHmac = new byte[32];
            buffer.get(expectedHmac);
            
            byte[] ciphertext = new byte[buffer.remaining()];
            buffer.get(ciphertext);
            
            byte[] actualHmac = computeHMAC(iv, ciphertext);
            if (!MessageDigest.isEqual(expectedHmac, actualHmac)) {
                throw new SecurityException("HMAC verification failed");
            }
            
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            GCMParameterSpec spec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, ALGORITHM), spec);
            
            byte[] plaintext = cipher.doFinal(ciphertext);
            return bytesToFloatArray(plaintext);
        } catch (GeneralSecurityException e) {
            throw new SecurityException("Decryption with HMAC failed", e);
        }
    }
    
    private byte[] computeHMAC(byte[] iv, byte[] data) throws GeneralSecurityException {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(iv);
        digest.update(data);
        digest.update(key);
        return digest.digest();
    }
    
    public byte[] encryptData(byte[] data) {
        try {
            byte[] iv = generateIV();
            
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            GCMParameterSpec spec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, ALGORITHM), spec);
            
            byte[] ciphertext = cipher.doFinal(data);
            
            ByteBuffer buffer = ByteBuffer.allocate(iv.length + ciphertext.length);
            buffer.put(iv);
            buffer.put(ciphertext);
            return buffer.array();
        } catch (GeneralSecurityException e) {
            throw new SecurityException("Data encryption failed", e);
        }
    }
    
    public byte[] decryptData(byte[] encryptedData) {
        try {
            ByteBuffer buffer = ByteBuffer.wrap(encryptedData);
            byte[] iv = new byte[GCM_IV_LENGTH];
            buffer.get(iv);
            
            byte[] ciphertext = new byte[buffer.remaining()];
            buffer.get(ciphertext);
            
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            GCMParameterSpec spec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, ALGORITHM), spec);
            
            return cipher.doFinal(ciphertext);
        } catch (GeneralSecurityException e) {
            throw new SecurityException("Data decryption failed", e);
        }
    }
    
    public byte[] encryptString(String data) {
        return encryptData(data.getBytes(StandardCharsets.UTF_8));
    }
    
    public String decryptString(byte[] encryptedData) {
        return new String(decryptData(encryptedData), StandardCharsets.UTF_8);
    }
}