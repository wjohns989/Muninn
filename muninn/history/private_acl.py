"""Fail-closed filesystem boundary for the separate credential vault.

The directory is secured before any credential metadata or ciphertext is written.
An existing directory is never silently re-permissioned by opening a vault.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class VaultPermissionError(PermissionError):
    """The credential vault path is not private to the current user."""


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _windows_identity():
    try:
        import win32api
        import win32con
        import win32security

        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        try:
            user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        finally:
            token.Close()
        return win32security, user
    except (ImportError, OSError) as exc:
        raise VaultPermissionError("Windows vault ACL verification is unavailable") from exc


def _protect_new_windows(path: Path) -> None:
    import ntsecuritycon
    import win32con

    security, user = _windows_identity()
    acl = security.ACL()
    flags = (win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE) if path.is_dir() else 0
    acl.AddAccessAllowedAceEx(security.ACL_REVISION, flags, ntsecuritycon.FILE_ALL_ACCESS, user)
    security.SetNamedSecurityInfo(
        str(path), security.SE_FILE_OBJECT,
        security.DACL_SECURITY_INFORMATION | security.PROTECTED_DACL_SECURITY_INFORMATION,
        None, None, acl, None,
    )


def verify_private(path: Path) -> None:
    path = Path(path)
    if _is_link(path) or not path.exists():
        raise VaultPermissionError("Credential vault path is missing or linked")
    if os.name == "nt":
        security, user = _windows_identity()
        try:
            descriptor = security.GetNamedSecurityInfo(
                str(path), security.SE_FILE_OBJECT,
                security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION,
            )
            owner = descriptor.GetSecurityDescriptorOwner()
            acl = descriptor.GetSecurityDescriptorDacl()
            control, _revision = descriptor.GetSecurityDescriptorControl()
            if (security.ConvertSidToStringSid(owner) != security.ConvertSidToStringSid(user)
                    or acl is None or acl.GetAceCount() != 1):
                raise VaultPermissionError("Credential vault ACL is not owner-only")
            if not control & security.SE_DACL_PROTECTED:
                raise VaultPermissionError("Credential vault ACL still inherits permissions")
            (ace_type, ace_flags), _mask, ace_sid = acl.GetAce(0)
            if ace_type != security.ACCESS_ALLOWED_ACE_TYPE or ace_flags & security.INHERITED_ACE:
                raise VaultPermissionError("Credential vault ACL has inherited or non-allow entries")
            if security.ConvertSidToStringSid(ace_sid) != security.ConvertSidToStringSid(user):
                raise VaultPermissionError("Credential vault ACL grants another identity")
        except VaultPermissionError:
            raise
        except (OSError, AttributeError, ValueError) as exc:
            raise VaultPermissionError("Credential vault ACL verification failed") from exc
    else:
        details = path.stat()
        if details.st_uid != os.getuid() or details.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise VaultPermissionError("Credential vault permissions are not owner-only")


def create_private_directory(path: Path) -> None:
    """Create a new owner-only directory; never alter an existing directory."""
    path = Path(path)
    if path.exists() or _is_link(path):
        raise VaultPermissionError("Credential vault directory already exists")
    try:
        path.mkdir(mode=0o700, parents=False)
        if os.name == "nt":
            _protect_new_windows(path)
        else:
            path.chmod(0o700)
        verify_private(path)
    except BaseException:
        # The empty path is left for inspection/recovery. Do not recursively delete it.
        raise


def create_private_file(path: Path) -> None:
    """Reserve an empty, owner-only file before its caller writes private content."""
    path = Path(path)
    verify_private(path.parent)
    try:
        with path.open("xb"):
            pass
        if os.name == "nt":
            _protect_new_windows(path)
        else:
            path.chmod(0o600)
        verify_private(path)
    except BaseException:
        # Leave any created empty file for inspection; never overwrite an existing one.
        raise


def seal_owner_only_staging_file(path: Path) -> None:
    """Seal a disposable SQLite staging file that inherited only this user's ACE.

    This is deliberately not a repair operation for vault records or arbitrary
    files. Callers must first constrain the path to their own staging namespace.
    """
    path = Path(path)
    details = path.lstat()
    if (_is_link(path) or not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1):
        raise VaultPermissionError("History staging path is not a regular private file")
    verify_private(path.parent)
    if os.name != "nt":
        verify_private(path)
        return
    if details.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise VaultPermissionError("History staging path is a reparse point")
    security, user = _windows_identity()
    try:
        descriptor = security.GetNamedSecurityInfo(
            str(path), security.SE_FILE_OBJECT,
            security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION,
        )
        owner = descriptor.GetSecurityDescriptorOwner()
        acl = descriptor.GetSecurityDescriptorDacl()
        if (security.ConvertSidToStringSid(owner) != security.ConvertSidToStringSid(user)
                or acl is None or acl.GetAceCount() != 1):
            raise VaultPermissionError("History staging ACL is not owner-only")
        (ace_type, _flags), _mask, ace_sid = acl.GetAce(0)
        if (ace_type != security.ACCESS_ALLOWED_ACE_TYPE
                or security.ConvertSidToStringSid(ace_sid) != security.ConvertSidToStringSid(user)):
            raise VaultPermissionError("History staging ACL grants another identity")
        control, _revision = descriptor.GetSecurityDescriptorControl()
        if not control & security.SE_DACL_PROTECTED:
            _protect_new_windows(path)
        verify_private(path)
    except VaultPermissionError:
        raise
    except (OSError, AttributeError, ValueError) as exc:
        raise VaultPermissionError("History staging ACL sealing failed") from exc
