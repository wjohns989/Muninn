"""The credential store must fail closed on permissive local ACLs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from muninn.history.private_acl import (
    VaultPermissionError,
    create_private_directory,
    create_private_file,
    verify_private,
)


def test_new_directory_and_file_are_owner_only(tmp_path: Path) -> None:
    root = tmp_path / "credential-vault"
    create_private_directory(root)
    create_private_file(root / "header.json")
    verify_private(root)
    verify_private(root / "header.json")


def test_existing_inherited_or_permissive_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "credential-vault"
    root.mkdir()
    if os.name != "nt":
        root.chmod(0o755)
    with pytest.raises(VaultPermissionError):
        verify_private(root)
    with pytest.raises(VaultPermissionError):
        create_private_directory(root)


def test_nonowner_acl_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "credential-vault"
    create_private_directory(root)
    if os.name == "nt":
        import ntsecuritycon
        import win32security

        everyone = win32security.CreateWellKnownSid(win32security.WinWorldSid)
        acl = win32security.ACL()
        acl.AddAccessAllowedAce(win32security.ACL_REVISION, ntsecuritycon.FILE_READ_DATA, everyone)
        win32security.SetNamedSecurityInfo(
            str(root), win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            None, None, acl, None,
        )
    else:
        root.chmod(0o755)
    with pytest.raises(VaultPermissionError):
        verify_private(root)
