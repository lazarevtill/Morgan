from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import CommunicationPrefs, UserModel
from morgan_brain.modules.personalization.passthrough import PassthroughPersonalizer


async def test_passthrough_reflects_comm_prefs_in_fragment():
    p = PassthroughPersonalizer()
    um = UserModel(user_id="u1", comm_prefs=CommunicationPrefs(tone="warm", length="terse"))
    ctx = await p.build(user_model=um, perception=FusedPerception(text="hi"))
    assert "terse" in ctx.system_fragment
    assert ctx.tone == "warm"


async def test_passthrough_empty_for_blank_user_model():
    p = PassthroughPersonalizer()
    ctx = await p.build(user_model=UserModel(user_id="u1"), perception=FusedPerception(text="hi"))
    assert ctx.system_fragment == ""
