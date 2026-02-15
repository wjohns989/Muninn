import asyncio

import pytest

from muninn.core.memory import MuninnMemory


class _Metadata:
    def __init__(self):
        self._profiles = {}

    def set_user_profile(self, *, user_id, profile, source):
        self._profiles[user_id] = {
            "user_id": user_id,
            "profile": profile,
            "source": source,
            "updated_at": 123.0,
        }

    def get_user_profile(self, *, user_id):
        return self._profiles.get(user_id)


def test_get_user_profile_returns_empty_when_unset():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = _Metadata()

    result = asyncio.run(memory.get_user_profile(user_id="global_user"))

    assert result["event"] == "USER_PROFILE_EMPTY"
    assert result["profile"] == {}
    assert result["updated_at"] is None


def test_set_user_profile_merge_patch_updates_nested_fields():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = _Metadata()

    asyncio.run(
        memory.set_user_profile(
            user_id="global_user",
            profile={
                "preferences": {"tone": "direct"},
                "paths": {"workspace": "/repo"},
            },
            merge=False,
            source="test_initial",
        )
    )

    result = asyncio.run(
        memory.set_user_profile(
            user_id="global_user",
            profile={
                "preferences": {"tone": "precise", "review_mode": True},
                "hardware": {"gpu_vram_gb": 16},
            },
            merge=True,
            source="test_merge",
        )
    )

    assert result["event"] == "USER_PROFILE_UPDATED"
    assert result["merge"] is True
    assert result["profile"]["preferences"]["tone"] == "precise"
    assert result["profile"]["preferences"]["review_mode"] is True
    assert result["profile"]["paths"]["workspace"] == "/repo"
    assert result["profile"]["hardware"]["gpu_vram_gb"] == 16
    assert result["source"] == "test_merge"


def test_set_user_profile_rejects_non_object_profile():
    memory = MuninnMemory()
    memory._initialized = True
    memory._metadata = _Metadata()

    with pytest.raises(ValueError, match="profile must be a JSON object"):
        asyncio.run(
            memory.set_user_profile(
                user_id="global_user",
                profile="invalid",  # type: ignore[arg-type]
            )
        )
