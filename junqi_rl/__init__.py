"""Open-information Junqi platform APIs."""

from .legality import LegalityResult, check_move
from .recording import GameRecord
from .rule_profile import RuleProfile, default_rule_profile, load_rule_profile

__all__ = ["GameRecord", "LegalityResult", "RuleProfile", "check_move",
           "default_rule_profile", "load_rule_profile"]
