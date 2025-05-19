import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time

# crude label assigner
def assign_label(h):
    if h <= 75:
        return 'Soft'
    elif h <= 150:
        return 'Moderate'
    elif h <= 200:
        return 'Hard'
    return 'Very Hard'

class EF_BER_Estimator:
    def __init__(self, k_clusters=4, k_neighbors=3):
        self.k_clusters = k_clusters
        self.k_neighbors = k_neighbors
        self.knn_model = None
        self.clusters = []

    def kmeans_bisection(self, data):
        clusters = [data]
        while len(clusters) < self.k_clusters:
            big_idx = np.argmax([len(c) for c in clusters])
            to_split = clusters.pop(big_idx)
            if len(to_split) < 2:
                clusters.append(to_split)
                continue
            i1, i2 = np.random.choice(len(to_split), 2, replace=False)
            c1, c2 = to_split[i1], to_split[i2]
            for _ in range(50):
                cl1, cl2 = [], []
                for x in to_split:
                    if abs(x - c1) < abs(x - c2):
                        cl1.append(x)
                    else:
                        cl2.append(x)
                if not cl1 or not cl2:
                    continue
                new_c1, new_c2 = np.mean(cl1), np.mean(cl2)
                if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
                    break
                c1, c2 = new_c1, new_c2
            clusters.append(np.array(cl1))
            clusters.append(np.array(cl2))
        self.clusters = clusters

    def train(self, data):
        start = time.time()
        self.kmeans_bisection(data)
        X, y = [], []
        noisy = 0
        for c in self.clusters:
            labels = [assign_label(x) for x in c]
            counts = {l: labels.count(l) for l in set(labels)}
            top = max(counts, key=counts.get)
            if counts[top]/len(c) >= 0.9:
                for x in c:
                    X.append([x])
                    y.append(assign_label(x))
            else:
                if self.knn_model:
                    for x in c:
                        if self.knn_model.predict([[x]])[0] != assign_label(x):
                            noisy += 1
        if not X:
            X = [[x] for x in data]
            y = [assign_label(x) for x in data]
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn_model.fit(X, y)
        total = len(X) + noisy
        ber = noisy / total if total else 0
        print("Trained in %.2fs" % (time.time() - start))
        print(f"BER: {ber:.4f} ({noisy}/{total})")

    def predict(self, X):
        if self.knn_model is None:
            raise ValueError("Need to train first")
        return self.knn_model.predict(X.reshape(-1, 1))

    def visualize(self, data):
        if not self.clusters:
            print("No clusters")
            return
        plt.figure(figsize=(10, 5))
        for i, c in enumerate(self.clusters):
            y = np.random.normal(i+1, 0.1, size=len(c))
            plt.scatter(c, y, label=f'C{i+1}')
        plt.axvline(75, color='g', ls='--')
        plt.axvline(150, color='y', ls='--')
        plt.axvline(200, color='r', ls='--')
        plt.legend()
        plt.show()

# for demo
if __name__ == '__main__':
    def gen_data(n):
        a = np.random.normal(50, 10, int(n*0.3))
        b = np.random.normal(110, 15, int(n*0.4))
        c = np.random.normal(170, 10, int(n*0.2))
        d = np.random.normal(230, 10, int(n*0.1))
        return np.concatenate([a, b, c, d])

    d = gen_data(200)
    model = EF_BER_Estimator()
    model.train(d)
    model.visualize(d)
    preds = model.predict(np.array([40, 120, 190, 260]))
    print(preds)
