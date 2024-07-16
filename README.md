Disclaimer:

- MEA-GNN: https://github.com/TrustworthyGNN/MEA-GNN
- link_stealing_attack: https://github.com/xinleihe/link_stealing_attack

# How to add another repo:

I plan to use subtree instead of submodule since subtree gives me a better way to modify their code.

```bash
git subtree add --prefix=subtree_directory <repository-url> <branch> --squash
cd subtree_directory
(Make modifications to their source code, and this step is optional).
git add .
git commit -am "commit message"
git push
```

```
No module named 'scipy.sparse.linalg.eigen.arpack'; 'scipy.sparse.linalg.eigen' is not a package

solution:
pip install --upgrade scipy
```
