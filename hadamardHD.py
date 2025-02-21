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
    Generate a Hadamard row of length n based on row_index.
    Note: n must be a power of 2.
    """
    row = np.array([1])
    for i in range(int(np.log2(n))):
        if (row_index >> i) & 1:
            row = np.hstack((row, -row))
        else:
            row = np.hstack((row, row))
    return row

# --- Continuous Group Binding using Scaled Hadamard Keys ---
# Instead of keys being ±1, we scale them by 2 so that:
#   key_vector = 2 * kronecker_hadamard(hv_length, i)
# Then binding is done by element-wise multiplication,
# and unbinding is simply element-wise division by the key.
    
def binding(hv_length, i, random_hv):
    # Compute the Hadamard key and scale it by 2 (continuous group)
    key_vector = 2 * kronecker_hadamard(hv_length, i)
    binded_HV = key_vector * random_hv
    print(f"Key K_{i}: {key_vector}")
    print(f"Bound HV for index {i}: {binded_HV}\n")
    return binded_HV

def bundling(hv_length, n_hvs, random_hvs):
    bundled_HV = np.zeros(hv_length)
    for i in range(n_hvs):
        bundled_HV += binding(hv_length, i, random_hvs[i])
    print(f"\nBundled HV: {bundled_HV}\n")
    return bundled_HV

def unbinding(hv_length, i, binded_HV):
    # Compute the same scaled key
    key_vector = 2 * kronecker_hadamard(hv_length, i)
    # Compute its element-wise inverse. For key_vector elements ±2, reciprocal is ±0.5.
    key_inverse = np.reciprocal(key_vector)
    unbound_HV = binded_HV * key_inverse
    print(f"Unbound HV for index {i}: {unbound_HV}")
    return unbound_HV

def unbundling(hv_length, n_hvs, bundled_HV):
    unbundled_hvs = []
    for i in range(n_hvs):
        key_vector = 2 * kronecker_hadamard(hv_length, i)
        key_inverse = np.reciprocal(key_vector)
        projection = bundled_HV * key_inverse
        unbundled_hvs.append(projection)
        print(f"Unbundled HV for index {i}: {projection}")
    print("\n")
    return unbundled_hvs

def calculate_similarity(n_hvs, random_hvs, unbundled_hvs):
    for i in range(n_hvs):
        original_hv = random_hvs[i]
        decoded_hv = unbundled_hvs[i]
        sim = cosine_similarity(original_hv, decoded_hv)
        print(sim)
