
## gate: "root/singlets/tcells"

def find_parent_gate(gate: str) -> str:
    return "/".join(gate.split("/")[:-1])

def find_grandparent_gate(gate: str) -> str:
    return find_parent_gate(find_parent_gate(gate))

def find_parents_recursively(gate: str):
    parent = find_parent_gate(gate)
    if parent != "root":
        return find_parents_recursively(parent)

def find_parents_recursively(gate: str, parent_list = None):
    if parent_list is None:
        parent_list = []
    parent = find_parent_gate(gate)
    parent_list.append(parent)
    if parent != "root":
        return find_parents_recursively(parent, parent_list)
    return parent_list
    