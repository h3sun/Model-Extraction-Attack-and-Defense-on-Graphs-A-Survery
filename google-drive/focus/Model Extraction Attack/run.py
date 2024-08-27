# @Author  : Edlison
# @Date    : 6/13/24 15:33

from attacks.attack_0 import attack0
from attacks.attack_0 import load_data
from attacks.attack_1 import attack1
from attacks.attack_2 import attack2
from attacks.attack_3 import attack3
from attacks.attack_4 import attack4
from attacks.attack_5 import attack5
from attacks.attack_6 import attack6
import numpy as np

dataset_name = 'pubmed'
attack_node_arg = 0.25
cuda = None

g, features, labels, train_mask, test_mask = load_data(dataset_name)
# attack0(dataset_name, attack_node_arg, cuda)  # Acc: 0.784 Fidelity: 0.859
attack1(dataset_name, attack_node_arg, cuda)  # Acc: 0.161 Fidelity: 0.333
# attack2(dataset_name, attack_node_arg, cuda)  # Acc: 0.748 Fidelity: 0.776
# attack3(dataset_name, attack_node_arg, cuda)  # Acc: 0.667 Fidelity: 0.263
# attack4(dataset_name, attack_node_arg, cuda)  # Acc: 0.155 Fidelity: 0.432
# attack5(dataset_name, attack_node_arg, cuda)  # Acc: 0.814 Fidelity: 0.105
# attack6(dataset_name, attack_node_arg, cuda)  # Miss author generated file
