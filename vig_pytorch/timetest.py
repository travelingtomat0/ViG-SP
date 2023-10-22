from timeit import default_timer as timer
import numpy as np

A = np.random.randint(low=0, high= 20, size=(224,224), dtype=np.int32)
print(A)

start = timer()

x = np.diff(A)

end = timer()
print(f"np diff took {end - start}")


start = timer()

y = np.equal(A[0:-1, :], A[1:, :])

end = timer()
print(f"np equal took {end - start}")

print(y)
print(y.sum())