import numpy as np
import matplotlib.pyplot as plt
# Generate data
np.random.seed(0)
data = np.random.rand(100)
train, test = data[:50], data[50:]
train_labels = ["Class1" if x <= 0.5 else "Class2" for x in train]
true_labels = ["Class1" if x <= 0.5 else "Class2" for x in test]
# Custom majority vote function
def majority_vote(labels):
    count1 = sum(1 for l in labels if l == "Class1")
    count2 = len(labels) - count1
    return "Class1" if count1 >= count2 else "Class2"
# k-NN function
def knn(train, labels, point, k):
    distances = [(abs(point - train[i]), labels[i]) for i in range(len(train))]
    distances.sort(key=lambda x: x[0])
    k_labels = [label for _, label in distances[:k]]
    return majority_vote(k_labels)
# Classify and show accuracy for each k
k_values = [1, 20,21,22,30]
accuracies = []

for k in k_values:
    preds = [knn(train, train_labels, x, k) for x in test]
    acc = np.mean([p == t for p, t in zip(preds, true_labels)]) * 100
    accuracies.append(acc)

    print(f"\nk = {k} → Accuracy: {acc:.2f}%")
    for i, (x, pred) in enumerate(zip(test, preds), 51):
        print(f"x{i} ({x:.2f}) → {pred}")
# Plot accuracy graph at the end
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='purple')
plt.grid(True)
plt.xticks(k_values)
plt.show()
