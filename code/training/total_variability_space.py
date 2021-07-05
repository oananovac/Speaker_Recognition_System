import numpy as np
from scipy.stats import norm


class TotalVariabilitySpace(object):
    def __init__(self, space_dimension, iterations_number):
        self.space_dimension = space_dimension
        self.iterations_number = iterations_number

    def create_t_matrix(self, N, F, ubm, numFeatures, numComp):
        sigma = np.zeros(shape=(len(ubm.covariances)*len(ubm.covariances[0])))
        k = 0
        for i in range(len(ubm.covariances[0])):
            for j in range(len(ubm.covariances)):
                sigma[k] = ubm.covariances[j][i]
                k += 1
        count_covs = np.shape(sigma)[0]

        np.random.seed(1)
        T = norm.ppf(np.random.rand(count_covs, self.space_dimension))
        normT = np.linalg.norm(T)

        newT = np.divide(T, np.full(np.shape(T), normT))

        I = np.eye( self.space_dimension, dtype=float)

        for i in range(self.iterations_number):

            Ey = []
            Eyy = []
            Linv = []
            final_matrix = []

            print("Iteratia :" + str(i))
            rep_sigma = np.repeat(np.transpose(sigma)[:, np.newaxis], np.shape(newT)[1], axis=1)
            TtimesInverseSSdiag = np.transpose(np.divide(newT, rep_sigma))

            for s in range(len(N)):
                mul = np.matmul(np.multiply(TtimesInverseSSdiag, np.repeat(N[s], np.shape(newT)[1], axis=0)), newT)
                L = np.add(I, mul)
                Linv.append(np.linalg.pinv(L))
                Ey.append(np.matmul((np.matmul(Linv[s], TtimesInverseSSdiag)), F[s]))
                Eyy.append(np.add(Linv[s], np.matmul(Ey[s], np.transpose(Ey[s]))))

            Eymat = []
            for it in range(len(Ey)):
                if it == 0:
                    Eymat = Ey[it]
                else:
                    Eymat = np.concatenate((Eymat, Ey[it]), axis=1)
            FFmat = np.zeros(shape=(len(F[0]), len(F)))

            for k in range(len(F)):
                for l in range(len(F[0])):
                    FFmat[l][k] = F[k][l]
            Kt = np.matmul(FFmat, np.transpose(Eymat))
            K = np.zeros(shape=(numComp, numFeatures, self.space_dimension))
            i = 0

            for k in range(numComp):
                for k1 in range(numFeatures):
                    K[k][k1] = Kt[i]
                    i += 1

            K = np.transpose(K, (0, 2, 1))
            newnT = []
            for c in range(numComp):
                AcLocal = np.zeros(shape=(self.space_dimension, self.space_dimension))

                for s in range(np.shape(N)[0]):
                    nc = np.full(np.shape(Eyy[s]), N[s][0][c])
                    AcLocal = np.add(AcLocal, np.multiply(nc, Eyy[s]))

                newnT.append(np.transpose(np.matmul(np.linalg.pinv(AcLocal), K[c])))

            for it in range(len(newnT)):
                if it == 0:
                    final_matrix = newnT[it]
                else:
                    final_matrix = np.concatenate((final_matrix, newnT[it]), axis=0)

            lastT = newT
            newT = final_matrix
            check = np.allclose(lastT, newT, rtol=0, atol=1e-2)
            if check == True:
                break

        return final_matrix

    def save_t_matrix(self, t_matrix, path, components_number):
        f = open(path + "/models/T_matrix_" + str(components_number) + ".txt",
                 "wb")
        np.save(f, t_matrix)
        f.close


def load_t_matrix(path, components_number):
    f = open(path + "/models/T_matrix_" + str(components_number) + ".txt",
             "rb")
    t_matrix = np.load(f)
    f.close
    return t_matrix