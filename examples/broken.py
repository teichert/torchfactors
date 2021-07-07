import torch
import torchfactors as tx

t = torch.tensor([
    [-7.021953582763672, 3.808997869491577],
    [4.331912040710449, -17.0089054107666]
])
# t = torch.tensor([
#     [0.5, 1],
#     [1, 0.5]
# ])
print(t)
# a b c
# x y z
# out_free = torch.einsum('ax,by,ab,cz,bc->', t, t, t, t, t)

out_is_True = torch.tensor([[False, True], [False, True]])
ax = t.masked_fill(out_is_True, float('-inf'))
by = t.masked_fill(out_is_True, float('-inf'))
cz = t.masked_fill(out_is_True.logical_not(), float('-inf'))

out_free = tx.log_dot(
    [
        (t, tx.ids('ax')),
        (t, tx.ids('bx')),
        (t, tx.ids('cz')),
        (t, tx.ids('ab')),
        (t, tx.ids('bc'))],
    [[]])

out_clamped = tx.log_dot(
    [
        (ax, tx.ids('ax')),
        (by, tx.ids('bx')),
        (cz, tx.ids('cz')),
        (t, tx.ids('ab')),
        (t, tx.ids('bc'))],
    [[]])

a, b, c = [tx.TensorVar(tx.Range(2), tensor=torch.tensor(0), usage=tx.LATENT) for _ in range(3)]
x, y, z = [tx.TensorVar(tx.Range(2), tensor=torch.tensor(0), usage=tx.ANNOTATED) for _ in range(3)]
z.tensor[(...,)] = 1
variables = [a, b, c, x, y, z]
f_ax = tx.TensorFactor(a, x, tensor=t)
f_by = tx.TensorFactor(b, y, tensor=t)
f_cz = tx.TensorFactor(c, z, tensor=t)
f_ab = tx.TensorFactor(a, b, tensor=t)
f_bc = tx.TensorFactor(b, c, tensor=t)
factors = [
    f_ax,
    f_by,
    f_cz,
    f_ab,
    f_bc,
]
bp = tx.BP()
for v in variables:
    v.clamp_annotated()
print([f.dense for f in factors])
logz_clamped = bp.product_marginal(factors)

for v in variables:
    v.unclamp_annotated()
print([f.dense for f in factors])
logz_free = bp.product_marginal(factors)


print(out_free, out_clamped)
print(logz_free, logz_clamped)
