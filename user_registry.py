import json
import os
import re
import tempfile


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
USERS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "users")
REGISTRY_FILE = os.path.join(USERS_ROOT, "users.json")
USER_ID_PATTERN = re.compile(r"^User_(\d+)$")


class UserRegistryError(RuntimeError):
    pass


def _validate_registry(data):
    if not isinstance(data, dict):
        raise UserRegistryError("users.json must contain a JSON object.")

    validated = {}
    for name, user_id in data.items():
        if not isinstance(name, str) or not name.strip():
            raise UserRegistryError("users.json contains an invalid user name.")
        if not isinstance(user_id, str) or USER_ID_PATTERN.fullmatch(user_id) is None:
            raise UserRegistryError(
                f"users.json contains invalid user ID for {name!r}: {user_id!r}"
            )
        validated[name] = user_id

    if len(set(validated.values())) != len(validated):
        raise UserRegistryError("users.json maps multiple names to the same user ID.")

    return validated


def _write_registry(registry):
    os.makedirs(USERS_ROOT, exist_ok=True)
    fd, temporary_file = tempfile.mkstemp(
        prefix="users_",
        suffix=".json",
        dir=USERS_ROOT
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(registry, file, indent=4, ensure_ascii=True)
            file.write("\n")
        os.replace(temporary_file, REGISTRY_FILE)
    except Exception:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)
        raise


def _bootstrap_registry():
    registry = {}
    existing_user_folder = os.path.join(USERS_ROOT, "User_001")
    existing_face_embedding = os.path.join(
        PROJECT_ROOT,
        "data",
        "Jithendar V",
        "embedding.npy"
    )
    existing_behaviour_embedding = os.path.join(
        existing_user_folder,
        "behaviour_embedding.npy"
    )
    existing_profile = os.path.join(
        existing_user_folder,
        "stable_behaviour_profile.npz"
    )

    if (
        os.path.isfile(existing_face_embedding)
        and os.path.isdir(existing_user_folder)
        and os.path.isfile(existing_behaviour_embedding)
        and os.path.isfile(existing_profile)
    ):
        registry["Jithendar V"] = "User_001"
        _write_registry(registry)
        print(f"Created user registry: {REGISTRY_FILE}")

    return registry


def load_user_registry():
    if not os.path.exists(REGISTRY_FILE):
        return _bootstrap_registry()

    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise UserRegistryError(
            f"Could not parse {REGISTRY_FILE}: {error}"
        ) from error
    except OSError as error:
        raise UserRegistryError(
            f"Could not read {REGISTRY_FILE}: {error}"
        ) from error

    return _validate_registry(data)


def find_user_name(registry, name):
    normalized_name = name.strip().casefold()
    for registered_name in registry:
        if registered_name.casefold() == normalized_name:
            return registered_name
    return None


def next_user_id(registry):
    highest_id = 0
    for user_id in registry.values():
        match = USER_ID_PATTERN.fullmatch(user_id)
        if match is not None:
            highest_id = max(highest_id, int(match.group(1)))

    for folder_name in os.listdir(USERS_ROOT) if os.path.isdir(USERS_ROOT) else []:
        match = USER_ID_PATTERN.fullmatch(folder_name)
        if match is not None:
            highest_id = max(highest_id, int(match.group(1)))

    return f"User_{highest_id + 1:03d}"


def register_user(name, user_id, directory_already_created=False):
    registry = load_user_registry()
    existing_name = find_user_name(registry, name)
    if existing_name is not None:
        raise UserRegistryError(
            f"User name already exists: {existing_name} -> {registry[existing_name]}"
        )

    if USER_ID_PATTERN.fullmatch(user_id) is None:
        raise UserRegistryError(f"Invalid user ID: {user_id}")

    if user_id in registry.values():
        raise UserRegistryError(f"User ID already exists in users.json: {user_id}")

    user_folder = os.path.join(USERS_ROOT, user_id)
    if os.path.exists(user_folder) and not directory_already_created:
        raise UserRegistryError(
            f"User directory already exists and will not be overwritten: {user_folder}"
        )

    registry[name.strip()] = user_id
    _write_registry(registry)
    return registry
