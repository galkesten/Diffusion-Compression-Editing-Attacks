import numpy as np
import galois

# --- 1. Define the Channel and Code Parameters ---
k_payload = 20000        # Your actual data bits (x)
p = 10**-3               # Noisier channel bit flip probability
t = 45                   # Number of errors we want to correct

# We operate in GF(2^15) because the block is ~20k bits
n_max = 32767
parity_bits = t * 15     # 45 errors * 15 bits per error = 675 parity bits
k_max = n_max - parity_bits # 32092 max data bits

print(f"Initializing BCH code to correct up to {t} errors...")
print(f"This requires {parity_bits} parity bits (approx {(parity_bits/k_payload)*100:.2f}% overhead).")

# Initialize the BCH object
bch = galois.BCH(n_max, k_max)

# --- 2. Generate and Encode the Payload ---
# Generate 20,000 random bits (e.g., representing a compressed image payload)
payload = np.random.randint(0, 2, k_payload)

# Pad the front with zeros to reach the mathematical k_max
padding_length = k_max - k_payload
padded_payload = np.concatenate((np.zeros(padding_length, dtype=int), payload))

# Encode to get the parity bits (returns a strict galois.FieldArray)
encoded_padded = bch.encode(padded_payload)
calculated_parity = encoded_padded[k_max:]

# The transmitted block: 20,000 data bits + 675 parity bits = 20,675 bits total
transmitted_codeword_gf = np.concatenate((payload, calculated_parity))

# Convert the Galois Field array back to a standard NumPy integer array
# so we can easily do standard math on it in the simulated channel.
transmitted_codeword = np.asarray(transmitted_codeword_gf).astype(int)

print(f"\nTransmitted Codeword Length: {len(transmitted_codeword)} bits")

# --- 3. Simulate the Noisy Channel (p = 10^-3) ---
# Generate random flips across the ENTIRE 20,675 bit transmitted codeword
flips = np.random.rand(len(transmitted_codeword)) < p
num_errors = np.sum(flips)

# Apply the flips using standard numpy modulo 2 addition
received_codeword = (transmitted_codeword + flips.astype(int)) % 2
print(f"Errors introduced by the channel: {num_errors} (Expected average: ~20.7)")

# --- 4. Decode and Correct ---
# Pad the front with the same imaginary zeros before decoding
print('A')
padded_received = np.concatenate((np.zeros(padding_length, dtype=int), received_codeword))

print('B')
# Decode the array. The decode function handles standard numpy arrays smoothly.
decoded_padded_payload, corrected_errors = bch.decode(padded_received)

# --- 5. Verify the Results ---
print('C')
if corrected_errors != -1:
    # Strip the imaginary zeros off the front
    print('D')
    final_payload = decoded_padded_payload[padding_length:]

    print(f"\nDecoding successful! The algorithm found and fixed {corrected_errors} errors.")
    if np.array_equal(payload, final_payload):
        print("Success: The decoded payload matches your original data perfectly.")
else:
    print(f"\nDecoding FAILED. The number of channel errors ({num_errors}) exceeded our budget of {t}.")
