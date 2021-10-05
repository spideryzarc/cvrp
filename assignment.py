import numpy as np
def assignment(C):
    assert isinstance(C, np.matrix), "use np.matrix"
    n = len(C)
    # Zij = Cij - ui - vj
    z = np.matrix(C)
    u = np.zeros(n)
    v = np.zeros(n)
    # dual cost
    dcost = 0
    # assignemt
    x = -np.ones(n, dtype=int)
    # first phase
    for i in range(n):
        minv = z[i, :].min()
        if minv > 0:
            u[i] += minv
            z[i, :] -= minv
            dcost += minv
    for j in range(n):
        minv = z[:, j].min()
        if minv > 0:
            v[j] += minv
            z[:, j] -= minv
            dcost += minv
    # print("dual cost", dcost)

    flagR = np.zeros(n, dtype=bool)
    flagC = np.zeros(n, dtype=bool)
    while True:
        # assigment attempt
        x.fill(-1)
        flagR.fill(False)
        flagC.fill(False)
        zAsArray = np.array(z)
        zeroC = n - np.count_nonzero(zAsArray, axis=0)
        zeroR = n - np.count_nonzero(zAsArray, axis=1)
        xcount = 0
        zeros = np.column_stack(np.where(z == 0))
        while len(zeros) > 0:

            i, j = min(zeros, key=lambda a: min(zeroR[a[0]], zeroC[a[1]]))


            if zeroR[i] >= zeroC[j]:
                flagR[i] = True
            else:
                flagC[j] = True
            # flagR[i] = True
            # flagC[j] = True
            for k, l in zeros:
                if k == i or l == j:
                    zeroR[k] -= 1
                    zeroC[l] -= 1
            x[i] = j

            xcount += 1
            zeros = [z for z in zeros if not flagR[z[0]] and not flagC[z[1]]]
            # zeros = [z for z in zeros if z[0] != i and z[1] != j]

        # print("assigned ", xcount)
        if xcount < n:
            minv = np.inf
            for i in range(n):
                if not flagR[i]:
                    for j in range(n):
                        if not flagC[j] and minv > z[i, j]:
                            minv = z[i, j]
                            if minv == 0:
                                print("ops")
                # print(C)

            assert minv > 0, 'minv deve ser > zero'

            for k in range(n):
                if not flagR[k]:
                    v[k] += minv
                    z[k, :] -= minv
                    dcost += minv
                if flagC[k]:
                    u[k] -= minv
                    z[:, k] += minv
                    dcost -= minv
            # print('dcost', dcost)

        else:
            break

    return dcost, x, v, u