import hashlib
import hmac
import secrets


DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


class AuthError(Exception):
    pass


def hash_password(password, salt=None):
    if not password:
        raise AuthError("Password cannot be empty.")
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    )
    return digest.hex(), salt


def verify_password(password, password_hash, salt):
    digest, _ = hash_password(password, salt)
    return hmac.compare_digest(digest, password_hash)


class AuthService:
    """Authentication and role-based permission service."""

    def __init__(self, database):
        self.db = database

    def ensure_default_admin(self):
        if not self.db.get_user_by_username(DEFAULT_ADMIN_USERNAME):
            password_hash, salt = hash_password(DEFAULT_ADMIN_PASSWORD)
            user_id = self.db.create_user(
                DEFAULT_ADMIN_USERNAME, password_hash, salt, role="admin"
            )
            self.db.log_operation(
                user_id, "init_admin", "Created default admin account."
            )

    def register(self, username, password, role="user", operator=None):
        username = (username or "").strip()
        if not username:
            raise AuthError("Username cannot be empty.")
        if role not in {"admin", "user"}:
            raise AuthError("Invalid role.")
        if role == "admin" and not self.is_admin(operator):
            raise AuthError("Only admins can create admin accounts.")
        if self.db.get_user_by_username(username):
            raise AuthError("Username already exists.")

        password_hash, salt = hash_password(password)
        user_id = self.db.create_user(username, password_hash, salt, role=role)
        self.db.log_operation(
            operator["id"] if operator else user_id,
            "register",
            f"Created user {username} as {role}.",
        )
        return self.db.get_user_by_id(user_id)

    def login(self, username, password):
        user = self.db.get_user_by_username((username or "").strip())
        if not user or not verify_password(password, user["password_hash"], user["salt"]):
            raise AuthError("Invalid username or password.")
        self.db.log_operation(user["id"], "login", "User signed in.")
        return user

    def change_password(self, username, old_password, new_password, operator=None):
        user = self.db.get_user_by_username(username)
        if not user:
            raise AuthError("User does not exist.")

        admin_override = self.is_admin(operator) and operator["username"] != username
        if not admin_override and not verify_password(
            old_password, user["password_hash"], user["salt"]
        ):
            raise AuthError("Old password is incorrect.")

        password_hash, salt = hash_password(new_password)
        self.db.update_user_password(username, password_hash, salt)
        self.db.log_operation(
            operator["id"] if operator else user["id"],
            "change_password",
            f"Changed password for {username}.",
        )

    def reset_password(self, target_username, new_password, operator):
        if not self.is_admin(operator):
            raise AuthError("Admin permission required.")
        user = self.db.get_user_by_username(target_username)
        if not user:
            raise AuthError("User does not exist.")
        password_hash, salt = hash_password(new_password)
        self.db.update_user_password(target_username, password_hash, salt)
        self.db.log_operation(operator["id"], "reset_password", f"Reset password for {target_username}")

    def change_role(self, target_username, new_role, operator):
        if not self.is_admin(operator):
            raise AuthError("Admin permission required.")
        if target_username == DEFAULT_ADMIN_USERNAME and new_role != "admin":
            raise AuthError("Cannot demote default admin.")
        user = self.db.get_user_by_username(target_username)
        if not user:
            raise AuthError("User does not exist.")
        self.db.update_user_role(target_username, new_role)
        self.db.log_operation(operator["id"], "change_role", f"Changed {target_username} role to {new_role}")

    def list_users(self, operator):
        if not self.is_admin(operator):
            raise AuthError("Admin permission is required.")
        return self.db.list_users()

    def delete_user(self, username, operator):
        if not self.is_admin(operator):
            raise AuthError("Admin permission is required.")
        if username == DEFAULT_ADMIN_USERNAME:
            raise AuthError("Default admin cannot be deleted.")
        affected = self.db.delete_user(username)
        self.db.log_operation(operator["id"], "delete_user", f"Deleted {username}.")
        return affected

    @staticmethod
    def is_admin(user):
        return bool(user and user.get("role") == "admin")
