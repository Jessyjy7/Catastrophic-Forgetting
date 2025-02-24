import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_random_hvs(num, length, max_val):
    return np.random.randint(low=0, high=max_val, size=(num, length))

def boolean_invert(x):
    where_false = x[x == False]
    where_true = x[x == True]
    x[where_false] = True
    x[where_true] = False
    return x

def kronecker_hadamard(n, row_index):
    row = np.array([1])
    for i in range(int(np.log2(n))):
        if (row_index >> i) & 1:
            row = np.hstack((row, -row))
        else:
            row = np.hstack((row, row))
    return row

def binding(hv_length, i, random_hv):
    key_vector = kronecker_hadamard(hv_length, i)
    binded_HV = key_vector * random_hv
    print(f"Key K_{i}: {key_vector}")
    print(f"Binded HV for {i}: {binded_HV}\n")
    return binded_HV

def bundling(hv_length, n_hvs, random_hvs):
    bundled_HV = np.zeros(hv_length)
    for i in range(n_hvs):
        bundled_HV += binding(hv_length, i, random_hvs)
    print(f"\nBundled HV: {bundled_HV}\n")
    return bundled_HV

def unbinding(hv_length, i, bundled_HV, epsilon=1e-8):
    """
    New unbinding using dot-product normalization:
    1. Compute the inverse key (elementwise reciprocal of the Hadamard row).
    2. Compute the alignment as the dot product between the inverse key and the bundled HV.
    3. Normalize the bundled HV by dividing by this scalar.
    4. Multiply elementwise by the inverse key.
    """
    key_vector = kronecker_hadamard(hv_length, i)
    key_inverse = np.reciprocal(key_vector)  # For keys ±1, reciprocal is the same as the key.
    
    # Compute alignment (a scalar) via dot product:
    alignment = np.dot(key_inverse, bundled_HV)
    
    # Guard against division by zero:
    if np.abs(alignment) < epsilon:
        alignment = epsilon if alignment >= 0 else -epsilon
    
    # Normalize and then unbind:
    unbound_HV = key_inverse * (bundled_HV / alignment)
    print(f"Unbound HV for {i} (dot-product normalized): {unbound_HV}")
    return unbound_HV

def unbundling(hv_length, n_hvs, bundled_HV, epsilon=1e-8):
    unbundled_hvs = []
    for i in range(n_hvs):
        # Use the new dot-product normalization unbinding for each key
        projection = unbinding(hv_length, i, bundled_HV, epsilon)
        unbundled_hvs.append(projection)
        print(f"Unbundled HV for index {i}: {projection}")
    print("\n")
    return unbundled_hvs

def calculate_similarity(n_hvs, random_hvs, unbundled_hvs):
    for i in range(n_hvs):
        original_hv = random_hvs[i]
        decoded_hv = unbundled_hvs[i]
        sim = cosine_similarity(original_hv, decoded_hv)
        print(f"Cosine similarity for index {i}: {sim}")
