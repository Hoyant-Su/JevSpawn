from jev_spawn.runtime.reference_transport import pack_references, unpack_references


def relocate(packed, transport, shared_keys):
    original = unpack_references(packed, transport)
    shared = {key: original['input'][key] for key in shared_keys if key in original['input']}
    local = {**original, 'input': {key: value for key, value in original['input'].items() if key not in shared}}
    segments = {'shared': pack_references(shared, transport), 'local': pack_references(local, transport)}
    return segments


def restore(segments, transport):
    decoded_shared = unpack_references(segments['shared'], transport)
    decoded_local = unpack_references(segments['local'], transport)
    assert not set(decoded_shared).intersection(decoded_local['input'])
    decoded = {**decoded_local, 'input': {**decoded_shared, **decoded_local['input']}}
    return decoded

