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
n_hvs = 3
hv_length = 16

random_hvs = generate_random_hvs(n_hvs, hv_length, 10)
# keys = generate_random_hvs(n_hvs, hv_length, 2)
# keys[keys == 0] = -1 

for i, hv in enumerate(random_hvs):
    print(f"Original A_{i}: {hv}")

HV_0 = np.zeros(hv_length)
for i in range(n_hvs):
    key_vector = kronecker_hadamard(hv_length, i)
    print(f"Key K_{i}: {key_vector}")
    # key_vector = keys[i]
    HV_0 += key_vector * random_hvs[i]

print(f"\nEncoded HV_0: {HV_0}\n")

decoded_hvs = []
for i in range(n_hvs):
    # key = keys[i]
    # key_inverse = np.reciprocal(key)
    key_vector = kronecker_hadamard(hv_length, i)
    key_inverse = np.reciprocal(key_vector)
    projection = HV_0 * key_inverse
    decoded_hvs.append(projection)
    print(f"Decoded vector for A_{i}: {projection}")
	
for i in range(n_hvs):
    original_hv = random_hvs[i]
    decoded_hv = decoded_hvs[i]
    sim = cosine_similarity(original_hv, decoded_hv)
    print(sim)