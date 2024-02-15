from pandas import DatetimeIndex
from anndata import AnnData

DISALLOWED_CHARACTERS_MAP = {
    "/": "___GATESEPARATOR___"
}

DISALLOWED_CHARACTER_INDICATORS = {value: key for key, value in DISALLOWED_CHARACTERS_MAP.items()}

def _make_obsm_valid(adata: AnnData) -> None:
    slot_keys = list(adata.obsm.keys())
    for slot_key in slot_keys:
        for disallowed_character in DISALLOWED_CHARACTERS_MAP:
            if disallowed_character in slot_key:
                adata.obsm[slot_key.replace(disallowed_character, DISALLOWED_CHARACTERS_MAP[disallowed_character])] = adata.obsm.pop(slot_key)
    return

def _make_varm_valid(adata: AnnData) -> None:
    slot_keys = list(adata.varm.keys())
    for slot_key in slot_keys:
        for disallowed_character in DISALLOWED_CHARACTERS_MAP:
            if disallowed_character in slot_key:
                adata.varm[slot_key.replace(disallowed_character, DISALLOWED_CHARACTERS_MAP[disallowed_character])] = adata.varm.pop(slot_key)
    return

def _make_obsp_valid(adata: AnnData) -> None:
    slot_keys = list(adata.obsp.keys())
    for slot_key in slot_keys:
        for disallowed_character in DISALLOWED_CHARACTERS_MAP:
            if disallowed_character in slot_key:
                adata.obsp[slot_key.replace(disallowed_character, DISALLOWED_CHARACTERS_MAP[disallowed_character])] = adata.obsp.pop(slot_key)
    return

def _make_varp_valid(adata: AnnData) -> None:
    slot_keys = list(adata.varp.keys())
    for slot_key in slot_keys:
        for disallowed_character in DISALLOWED_CHARACTERS_MAP:
            if disallowed_character in slot_key:
                adata.varp[slot_key.replace(disallowed_character, DISALLOWED_CHARACTERS_MAP[disallowed_character])] = adata.varp.pop(slot_key)
    return

def _make_layers_valid(adata: AnnData) -> None:
    slot_keys = list(adata.layers.keys())
    for slot_key in slot_keys:
        for disallowed_character in DISALLOWED_CHARACTERS_MAP:
            if disallowed_character in slot_key:
                adata.layers[slot_key.replace(disallowed_character, DISALLOWED_CHARACTERS_MAP[disallowed_character])] = adata.layers.pop(slot_key)
    return


def _make_var_valid(adata: AnnData) -> None:
    for col in adata.var.columns:
        if adata.var[col].dtype != "category":
            adata.var[col] = adata.var[col].astype("str")
            continue
        if isinstance(adata.var[col].cat.categories, DatetimeIndex):
            adata.var[col] = adata.var[col].astype("str").astype("category")
            ### add warning!
    return

def _make_obs_valid(adata: AnnData) -> None:
    for col in adata.obs.columns:
        if adata.obs[col].dtype != "category":
            continue
        if isinstance(adata.obs[col].cat.categories, DatetimeIndex):
            adata.obs[col] = adata.obs[col].astype("str").astype("category")
            ### add warning!
    return

def _restore_obsm_keys(adata: AnnData):
    slot_keys = list(adata.obsm.keys())
    for slot_key in slot_keys:
        for character_pointer in DISALLOWED_CHARACTER_INDICATORS:
            if character_pointer in slot_key:
                adata.obsm[slot_key.replace(character_pointer, DISALLOWED_CHARACTER_INDICATORS[character_pointer])] = adata.obsm.pop(slot_key)
    return

def _restore_varm_keys(adata: AnnData):
    slot_keys = list(adata.varm.keys())
    for slot_key in slot_keys:
        for character_pointer in DISALLOWED_CHARACTER_INDICATORS:
            if character_pointer in slot_key:
                adata.varm[slot_key.replace(character_pointer, DISALLOWED_CHARACTER_INDICATORS[character_pointer])] = adata.varm.pop(slot_key)
    return

def _restore_varp_keys(adata: AnnData):
    slot_keys = list(adata.varp.keys())
    for slot_key in slot_keys:
        for character_pointer in DISALLOWED_CHARACTER_INDICATORS:
            if character_pointer in slot_key:
                adata.varp[slot_key.replace(character_pointer, DISALLOWED_CHARACTER_INDICATORS[character_pointer])] = adata.varp.pop(slot_key)
    return

def _restore_obsp_keys(adata: AnnData):
    slot_keys = list(adata.obsp.keys())
    for slot_key in slot_keys:
        for character_pointer in DISALLOWED_CHARACTER_INDICATORS:
            if character_pointer in slot_key:
                adata.obsp[slot_key.replace(character_pointer, DISALLOWED_CHARACTER_INDICATORS[character_pointer])] = adata.obsp.pop(slot_key)
    return

def _restore_layers_keys(adata: AnnData):
    slot_keys = list(adata.layers.keys())
    for slot_key in slot_keys:
        for character_pointer in DISALLOWED_CHARACTER_INDICATORS:
            if character_pointer in slot_key:
                adata.layers[slot_key.replace(character_pointer, DISALLOWED_CHARACTER_INDICATORS[character_pointer])] = adata.layers.pop(slot_key)
    return
