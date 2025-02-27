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
    """
    Generates one row of a Kronecker-based Hadamard matrix of size n.
    n must be a power of 2. row_index < n.
    """
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
    print(f"\nBundled HV for {i}: {bundled_HV}\n")
    return bundled_HV

def unbinding(hv_length, i, binded_HV, epsilon=1e-8):
    """
    Unbind using elementwise multiplication,
    then apply a dot-product-based normalization step.
    """
    key_vector = kronecker_hadamard(hv_length, i)
    # For ±1 keys, reciprocal is the same as the key, but we'll do it explicitly:
    key_inverse = np.reciprocal(key_vector)
    
    # 1) Naive elementwise unbind:
    naive_unbound = binded_HV * key_inverse
    
    # 2) Compute a global alignment scalar (dot product with the key):
    #    This is a matched-filter style approach to scale the unbound HV.
    dot_val = np.dot(binded_HV, key_vector)
    norm_key = np.dot(key_vector, key_vector)  # Should be hv_length if ±1
    alpha = dot_val / (norm_key + epsilon)     # Avoid division by zero
    
    # 3) Rescale the naive unbound by alpha:
    unbound_HV = naive_unbound * alpha
    
    print(f"Unbound HV (dot-product normalized) for {i}: {unbound_HV}")
    return unbound_HV

def unbundling(hv_length, n_hvs, bundled_HV):
    """
    Same as before, but calls the new unbinding with dot-product normalization.
    """
    unbundled_hvs = []
    for i in range(n_hvs):
        unbound_HV = unbinding(hv_length, i, bundled_HV)
        unbundled_hvs.append(unbound_HV)
    print("\n")
    return unbundled_hvs

def calculate_similarity(n_hvs, random_hvs, unbundled_hvs):
    for i in range(n_hvs):
        original_hv = random_hvs[i]
        decoded_hv = unbundled_hvs[i]
        sim = cosine_similarity(original_hv, decoded_hv)
        print(f"Cosine similarity for index {i}: {sim}")
