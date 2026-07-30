import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


def make_dataset(n=2000, size=16):
    """Synthetic 'faces' (fixed eyes/mouth triplet) vs. 'objects' (randomly placed blob triplets)."""
    X = np.zeros((n, 1, size, size), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        img = np.zeros((size, size), dtype=np.float32)
        if i % 2 == 0:
            for (r, c) in [(4, 5), (4, 10), (10, 7)]:
                img[r - 1:r + 2, c - 1:c + 2] = 1.0
            y[i] = 1
        else:
            for _ in range(3):
                r, c = np.random.randint(2, size - 2, 2)
                img[r - 1:r + 2, c - 1:c + 2] = 1.0
            y[i] = 0
        img += np.random.normal(0, 0.05, img.shape).astype(np.float32)
        X[i, 0] = img
    return torch.tensor(X), torch.tensor(y)


class TinyFaceNet(nn.Module):
    def __init__(self, n_units=8):
        super().__init__()
        self.conv = nn.Conv2d(1, n_units, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_units, 2)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        pooled = self.pool(h).flatten(1)
        return self.fc(pooled), pooled


def evaluate(net, X, y):
    with torch.no_grad():
        logits, _ = net(X)
        preds = logits.argmax(1)
    return (preds == y).float().mean().item()


X_train, y_train = make_dataset(2000)
X_test, y_test = make_dataset(400)

net = TinyFaceNet()
opt = optim.Adam(net.parameters(), lr=1e-2)

for epoch in range(30):
    opt.zero_grad()
    logits, _ = net(X_train)
    loss = F.cross_entropy(logits, y_train)
    loss.backward()
    opt.step()

face_mask = y_test == 1
obj_mask = y_test == 0

with torch.no_grad():
    _, acts = net(X_test)

face_act = acts[face_mask].mean(0)
obj_act = acts[obj_mask].mean(0)
selectivity = (face_act - obj_act) / (face_act + obj_act + 1e-6)
ffa_unit = int(selectivity.argmax().item())

print("Per-unit selectivity (face vs. object):")
print(np.round(selectivity.numpy(), 2))
print(f"Most face-selective unit: {ffa_unit} (selectivity = {selectivity[ffa_unit]:.2f})")

acc_faces_before = evaluate(net, X_test[face_mask], y_test[face_mask])
acc_objs_before = evaluate(net, X_test[obj_mask], y_test[obj_mask])
original_state = {k: v.clone() for k, v in net.state_dict().items()}

with torch.no_grad():
    net.conv.weight[ffa_unit] = 0
    net.conv.bias[ffa_unit] = 0

acc_faces_after = evaluate(net, X_test[face_mask], y_test[face_mask])
acc_objs_after = evaluate(net, X_test[obj_mask], y_test[obj_mask])

print(f"\nLesioning the face-selective unit ({ffa_unit}):")
print(f"  Face accuracy:   {acc_faces_before:.2f} -> {acc_faces_after:.2f}")
print(f"  Object accuracy: {acc_objs_before:.2f} -> {acc_objs_after:.2f}")

# Control: restore weights, then lesion the least face-selective unit instead
net.load_state_dict(original_state)
control_unit = int(selectivity.argmin().item())

with torch.no_grad():
    net.conv.weight[control_unit] = 0
    net.conv.bias[control_unit] = 0

acc_faces_control = evaluate(net, X_test[face_mask], y_test[face_mask])
acc_objs_control = evaluate(net, X_test[obj_mask], y_test[obj_mask])

print(f"\nControl: lesioning the least face-selective unit ({control_unit}):")
print(f"  Face accuracy:   {acc_faces_before:.2f} -> {acc_faces_control:.2f}")
print(f"  Object accuracy: {acc_objs_before:.2f} -> {acc_objs_control:.2f}")
