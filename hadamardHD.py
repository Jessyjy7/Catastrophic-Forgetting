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
      if(row_index >> i) & 1:
        row = np.hstack((row, -row))
      else:
        row = np.hstack((row, row))
    return row


# n_hvs = 2
# hv_length = 10
# n_hvs = 3
# hv_length = 16

# random_hvs = generate_random_hvs(n_hvs, hv_length, 10)
# keys = generate_random_hvs(n_hvs, hv_length, 2)
# keys[keys == 0] = -1 

# for i, hv in enumerate(random_hvs):
#     print(f"Original A_{i}: {hv}")

# def binding (hv_length, i, random_hvs):
#     binded_HV = np.zeros(hv_length)
#     key_vector = kronecker_hadamard(hv_length, i)
#     print(f"Key K_{i}: {key_vector}")
#     binded_HV = key_vector * random_hvs[i]
#     print(f"\nBinded HV: {binded_HV}\n")
#     return binded_HV

def binding(hv_length, i, random_hv):
    key_vector = kronecker_hadamard(hv_length, i)
    binded_HV = key_vector * random_hv
    print(f"Key K_{i}: {key_vector}")
    print(f"Binded HV for {i}: {binded_HV}\n")
    return binded_HV

def bundling (hv_length, n_hvs, random_hvs):
    bundled_HV = np.zeros(hv_length)
    for i in range(n_hvs):
        bundled_HV += binding (hv_length, i, random_hvs)
    print(f"\nBundled HV for {i}: {bundled_HV}\n")
    return bundled_HV

def unbinding (hv_length, i, binded_HV):
    unbound_HV = np.zeros(hv_length)
    key_vector = kronecker_hadamard(hv_length, i)
    key_inverse = np.reciprocal(key_vector)
    unbound_HV = binded_HV * key_inverse
    # unbinded_HV.append(projection)
    print(f"Unbound HV for {i}: {unbound_HV}")
    return unbound_HV
        
def unbundling (hv_length, n_hvs, bundled_HV):
    unbundled_hvs = []
    for i in range(n_hvs):
        key_vector = kronecker_hadamard(hv_length, i)
        key_inverse = np.reciprocal(key_vector)
        projection = bundled_HV * key_inverse
        unbundled_hvs.append(projection)
        print(f"Unbundled HV for {i} index: {projection}")
    print(f"\n")
    return unbundled_hvs
	
def calculate_similarity(n_hvs, random_hvs, unbundled_hvs):
    for i in range(n_hvs):
        original_hv = random_hvs[i]
        decoded_hv = unbundled_hvs[i]
        sim = cosine_similarity(original_hv, decoded_hv)
        print(sim)
