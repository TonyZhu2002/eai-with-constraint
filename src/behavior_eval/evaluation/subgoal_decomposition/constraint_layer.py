from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from behavior_eval.evaluation.subgoal_decomposition.checkers import Vocab


class ConstraintErrorType(str, Enum):
    VALIDITY = "validity"
    PREREQUISITE = "prerequisite"
    HALLUCINATION = "hallucination"


@dataclass
class ConstraintIssue:
    error_type: ConstraintErrorType
    message: str
    subgoal: str
    action: Optional[str] = None
    objects: Optional[List[str]] = None


@dataclass
class ConstraintResult:
    corrected_subgoals: List[str]
    issues: List[ConstraintIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)


class ConstraintGenerator:
    """Interface for generating constraint-aware subgoal corrections."""

    def generate(
        self, subgoals: Sequence[str], env_summary: Dict[str, Any]
    ) -> ConstraintResult:  # pragma: no cover - interface
        raise NotImplementedError


class VLMConstraintGenerator(ConstraintGenerator):
    """Constraint generator that delegates constraint discovery to a VLM.

    The generator builds a lightweight prompt from the environment summary and
    the subgoal list. A VLM callable can be injected for real inference; when
    absent, the generator falls back to deterministic heuristics derived from
    the vocabulary and environment summary.
    """

    def __init__(
        self,
        vocab: Vocab,
        vlm_callable: Optional[Callable[[str], Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.vocab = vocab
        self.vlm_callable = vlm_callable
        self.system_prompt = system_prompt or "You are an affordance validator that returns JSON only."

    def _build_prompt(
        self,
        subgoals: Sequence[str],
        env_summary: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        task = env_summary.get("task", "unknown task")
        relevant_objects = ", ".join(sorted(env_summary.get("relevant_objects", []))) or "<none>"
        goal_str = " \n".join(f"- {sg}" for sg in subgoals)
        base_prompt = (
            f"{self.system_prompt}\n"
            "Given the environment summary and the proposed subgoals, list three fields in JSON:\n"
            "1) allowed_objects: array of objects that exist and can be referenced;\n"
            "2) allowed_verbs: verbs that are executable in this scene;\n"
            "3) required_prerequisites: mapping from verb to a list of prerequisite verbs that must appear before it.\n"
            "Environment summary: "
            f"task={task}; objects=[{relevant_objects}]\n"
            f"Subgoals:\n{goal_str}\n"
        )
        if feedback:
            issue_lines = []
            for idx, issue in enumerate(feedback.get("issues", []), start=1):
                issue_lines.append(
                    f"{idx}) category={issue.get('type')}, subgoal={issue.get('subgoal')}, "
                    f"action={issue.get('action')}, objects={issue.get('objects')}, message={issue.get('message')}"
                )
            issue_block = "\n".join(issue_lines) or "<none>"
            base_prompt += (
                "A previous domain grounding or TAMP validation failed. Review the issues and revise the "
                "allowed objects, verbs, or prerequisites to repair them.\n"
                "Issues to fix:\n"
                f"{issue_block}\n"
                "Provide updated JSON only."
            )
        else:
            base_prompt += "Respond with a JSON object only."
        return base_prompt

    def _default_constraints(self, env_summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "allowed_objects": env_summary.get("relevant_objects", []),
            "allowed_verbs": sorted(set(self.vocab.action_list + self.vocab.predicate_list)),
            "required_prerequisites": {
                "open": ["navigate_to"],
                "pickup": ["open", "clear"],
                "pick_up": ["open", "clear"],
                "pickplace": ["pickup"],
                "place": ["pickup"],
                "toggleon": ["open"],
                "putin": ["open", "clear"],
            },
        }

    def _normalize_constraints(self, constraints: Any, env_summary: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(constraints, dict):
            return self._default_constraints(env_summary)
        normalized: Dict[str, Any] = self._default_constraints(env_summary)
        if isinstance(constraints.get("allowed_objects"), list):
            normalized["allowed_objects"] = constraints["allowed_objects"]
        if isinstance(constraints.get("allowed_verbs"), list):
            normalized["allowed_verbs"] = constraints["allowed_verbs"]
        if isinstance(constraints.get("required_prerequisites"), dict):
            normalized["required_prerequisites"].update(constraints["required_prerequisites"])
        return normalized

    def generate(
        self,
        subgoals: Sequence[str],
        env_summary: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ConstraintResult:
        prompt = self._build_prompt(subgoals, env_summary, feedback)
        raw_constraints: Any = None
        if self.vlm_callable:
            try:
                raw_constraints = self.vlm_callable(prompt)
            except Exception as exc:  # pragma: no cover - safety net for downstream callers
                raw_constraints = {"error": str(exc)}
        constraints = self._normalize_constraints(raw_constraints, env_summary)
        metadata = {
            "prompt": prompt,
            "raw_vlm_response": raw_constraints,
            "constraints": constraints,
            "feedback": feedback,
        }
        return ConstraintResult(corrected_subgoals=list(subgoals), issues=[], metadata=metadata)


class RuleBasedConstraintGenerator(ConstraintGenerator):
    """Lightweight constraint generator tying issues to vocab and affordances."""

    def __init__(self, vocab: Vocab, affordance_rules: Optional[Dict[str, Iterable[str]]] = None):
        self.vocab = vocab
        self.affordance_rules = affordance_rules or {}

    def generate(self, subgoals: Sequence[str], env_summary: Dict[str, Any]) -> ConstraintResult:
        issues: List[ConstraintIssue] = []
        corrected: List[str] = []
        scene_objects = set(env_summary.get("relevant_objects", []))
        trace: List[Dict[str, Any]] = []

        for subgoal in subgoals:
            name, args = self.parse_subgoal(subgoal)
            expected_args = self._expected_arg_count(name)
            decision_log = {"subgoal": subgoal, "action": name, "applied_repairs": [], "notes": []}

            if expected_args is None:
                issues.append(
                    ConstraintIssue(
                        error_type=ConstraintErrorType.HALLUCINATION,
                        message=f"Unknown predicate or action '{name}' in vocabulary.",
                        subgoal=subgoal,
                        action=name,
                    )
                )
                decision_log["notes"].append("unknown predicate/action")
                continue

            if expected_args != len(args):
                issues.append(
                    ConstraintIssue(
                        error_type=ConstraintErrorType.VALIDITY,
                        message=f"Expected {expected_args} arguments for '{name}' but found {len(args)}.",
                        subgoal=subgoal,
                        action=name,
                        objects=args,
                    )
                )
                # trim or pad to expected length for downstream modules
                if expected_args < len(args):
                    args = args[:expected_args]
                    decision_log["applied_repairs"].append("trim_extra_args")
                else:
                    args = args + ["<unk>"] * (expected_args - len(args))
                    decision_log["applied_repairs"].append("pad_missing_args")

            if name in self.affordance_rules:
                missing = [obj for obj in args if obj not in scene_objects]
                if missing:
                    issues.append(
                        ConstraintIssue(
                            error_type=ConstraintErrorType.PREREQUISITE,
                            message=f"Affordance prerequisite violation for '{name}': missing {missing} in scene.",
                        subgoal=subgoal,
                        action=name,
                        objects=missing,
                    )
                )
                    decision_log["notes"].append(f"missing objects: {missing}")

            corrected.append(self._rebuild_subgoal(name, args))
            trace.append(decision_log)

        metadata = {"scene_objects": sorted(scene_objects)}
        return ConstraintResult(corrected_subgoals=corrected, issues=issues, metadata=metadata, trace=trace)

    @staticmethod
    def parse_subgoal(subgoal: str) -> tuple[str, List[str]]:
        match = re.match(r"(?P<name>[^()\s]+)\((?P<args>.*)\)", subgoal)
        if not match:
            return subgoal.strip(), []
        args = [arg.strip() for arg in match.group("args").split(";") for arg in arg.split(",")]
        args = [arg for arg in args if arg]
        return match.group("name").strip(), args

    def _expected_arg_count(self, name: str) -> Optional[int]:
        if self.vocab.check_action_in_vocab(name):
            return self.vocab.action_param_dict.get(name)
        if self.vocab.check_state_in_vocab(name):
            return self.vocab.state_param_dict.get(name)
        return None

    def _rebuild_subgoal(self, name: str, args: List[str]) -> str:
        if not args:
            return name
        return f"{name}({', '.join(args)})"


class ConstraintFilter:
    """Post-process constraint outputs to repair/clean subgoals."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def filter(self, constraint_result: ConstraintResult) -> ConstraintResult:
        filtered: List[str] = []
        issues = list(constraint_result.issues)
        trace = list(constraint_result.trace)
        for subgoal in constraint_result.corrected_subgoals:
            name, _ = RuleBasedConstraintGenerator.parse_subgoal(subgoal)
            if not (self.vocab.check_action_in_vocab(name) or self.vocab.check_state_in_vocab(name)):
                issues.append(
                    ConstraintIssue(
                        error_type=ConstraintErrorType.HALLUCINATION,
                        message=f"Filtered out unknown token '{name}' after correction stage.",
                        subgoal=subgoal,
                        action=name,
                    )
                )
                continue
            filtered.append(subgoal)
        return ConstraintResult(corrected_subgoals=filtered, issues=issues, metadata=constraint_result.metadata, trace=trace)


class AffordanceValidator:
    """Validate subgoals against affordances, verbs, and prerequisites."""

    def __init__(
        self,
        vocab: Vocab,
        required_prerequisites: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.vocab = vocab
        self.required_prerequisites = required_prerequisites or {
            "open": ["navigate_to"],
            "pickup": ["open", "clear"],
            "pick_up": ["open", "clear"],
            "pickplace": ["pickup"],
            "place": ["pickup"],
            "putin": ["open", "clear"],
            "toggleon": ["open"],
        }

    def _normalized_action(self, action: str) -> str:
        normalized = action.lower().strip()
        return self.vocab.alias_dict.get(normalized, normalized)

    @staticmethod
    def _format_subgoal(action: str, args: List[str]) -> str:
        if not args:
            return action
        return f"{action}({', '.join(args)})"

    def _legal_action(self, action: str, allowed_verbs: Set[str]) -> Tuple[bool, Optional[str]]:
        normalized = self._normalized_action(action)
        if normalized in allowed_verbs:
            return True, normalized
        if normalized in self.vocab.alias_dict.values() and normalized in allowed_verbs:
            return True, normalized
        for alias, canonical in self.vocab.alias_dict.items():
            if canonical == normalized and alias in allowed_verbs:
                return True, alias
        return False, None

    def _inject_prerequisites(
        self,
        action: str,
        args: List[str],
        corrected: List[str],
        seen_actions: Set[str],
        constraints: Dict[str, Any],
        issues: List[ConstraintIssue],
        decision_log: Dict[str, Any],
    ) -> None:
        normalized_action = self._normalized_action(action)
        prereq_map = {}
        prereq_map.update(self.required_prerequisites)
        prereq_map.update(constraints.get("required_prerequisites", {}))
        prerequisites = prereq_map.get(normalized_action, [])
        for prereq in prerequisites:
            if prereq in seen_actions:
                continue
            prereq_goal = self._format_subgoal(prereq, args)
            issues.append(
                ConstraintIssue(
                    error_type=ConstraintErrorType.PREREQUISITE,
                    message=f"Inserted missing prerequisite '{prereq}' for action '{action}'.",
                    subgoal=prereq_goal,
                    action=prereq,
                    objects=args,
                )
            )
            decision_log.setdefault("applied_repairs", []).append(f"add_prerequisite:{prereq}")
            corrected.append(prereq_goal)
            seen_actions.add(self._normalized_action(prereq))

    def validate(
        self,
        subgoals: Sequence[str],
        env_summary: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        prior_issues: Optional[List[ConstraintIssue]] = None,
        prior_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> ConstraintResult:
        issues = list(prior_issues or [])
        trace = list(prior_trace or [])
        constraints = constraints or {}

        allowed_objects = set(env_summary.get("relevant_objects", []))
        allowed_objects.update(constraints.get("allowed_objects", []))

        allowed_verbs = set(self.vocab.action_list + self.vocab.predicate_list)
        extra_verbs = constraints.get("allowed_verbs", [])
        if extra_verbs:
            allowed_verbs = allowed_verbs.intersection(set(extra_verbs)) if allowed_verbs else set(extra_verbs)

        corrected: List[str] = []
        seen_actions: Set[str] = set()

        for subgoal in subgoals:
            action, args = RuleBasedConstraintGenerator.parse_subgoal(subgoal)
            decision_log = {"subgoal": subgoal, "action": action, "applied_repairs": [], "notes": []}

            legal, replacement = self._legal_action(action, allowed_verbs)
            if not legal:
                issues.append(
                    ConstraintIssue(
                        error_type=ConstraintErrorType.VALIDITY,
                        message=f"Action '{action}' is not permitted by allowed verbs.",
                        subgoal=subgoal,
                        action=action,
                    )
                )
                decision_log["applied_repairs"].append("drop_illegal_action")
                trace.append(decision_log)
                continue

            if replacement and replacement != action:
                decision_log["applied_repairs"].append(f"replace_action:{replacement}")
                action = replacement

            missing_objs = [obj for obj in args if obj and obj not in allowed_objects]
            if missing_objs:
                issues.append(
                    ConstraintIssue(
                        error_type=ConstraintErrorType.HALLUCINATION,
                        message=f"Objects {missing_objs} are not present in the environment summary.",
                        subgoal=subgoal,
                        action=action,
                        objects=missing_objs,
                    )
                )
                decision_log["applied_repairs"].append("drop_hallucinated_object")
                trace.append(decision_log)
                continue

            self._inject_prerequisites(action, args, corrected, seen_actions, constraints, issues, decision_log)

            rebuilt = self._format_subgoal(action, args)
            corrected.append(rebuilt)
            seen_actions.add(self._normalized_action(action))
            trace.append(decision_log)

        metadata = {
            "allowed_objects": sorted(allowed_objects),
            "allowed_verbs": sorted(allowed_verbs),
            "required_prerequisites": constraints.get("required_prerequisites", {}),
            "repairs_applied": sum(len(item.get("applied_repairs", [])) for item in trace),
        }
        return ConstraintResult(corrected_subgoals=corrected, issues=issues, metadata=metadata, trace=trace)


class ConstraintPipeline:
    """Pipeline wrapper for constraint generation and filtering."""

    def __init__(
        self,
        generator: ConstraintGenerator,
        filterer: ConstraintFilter,
        validator: Optional[AffordanceValidator] = None,
        max_reprompts: int = 2,
    ):
        self.generator = generator
        self.filterer = filterer
        self.validator = validator
        self.max_reprompts = max_reprompts

    def _should_reprompt(self, result: ConstraintResult, attempt: int) -> bool:
        return (
            isinstance(self.generator, VLMConstraintGenerator)
            and self.generator.vlm_callable is not None
            and bool(result.issues)
            and attempt < self.max_reprompts
        )

    @staticmethod
    def _build_feedback(issues: Sequence[ConstraintIssue], env_summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "issues": [
                {
                    "type": issue.error_type.value,
                    "subgoal": issue.subgoal,
                    "action": issue.action,
                    "objects": issue.objects,
                    "message": issue.message,
                }
                for issue in issues
            ],
            "env_summary": env_summary,
        }

    def run(self, subgoals: Sequence[str], env_summary: Dict[str, Any]) -> ConstraintResult:
        reprompt_history: List[Dict[str, Any]] = []
        feedback: Optional[Dict[str, Any]] = None
        attempt = 0
        current_result: ConstraintResult

        while True:
            if isinstance(self.generator, VLMConstraintGenerator):
                generated = self.generator.generate(subgoals, env_summary, feedback=feedback)
            else:
                generated = self.generator.generate(subgoals, env_summary)
            validated = (
                self.validator.validate(
                    generated.corrected_subgoals,
                    env_summary,
                    constraints=generated.metadata.get("constraints"),
                    prior_issues=generated.issues,
                    prior_trace=generated.trace,
                )
                if self.validator
                else generated
            )
            filtered = self.filterer.filter(validated)

            issue_categories = sorted({issue.error_type.value for issue in filtered.issues})
            reprompt_history.append(
                {
                    "attempt": attempt,
                    "issue_count": len(filtered.issues),
                    "error_categories": issue_categories,
                    "applied_feedback": feedback is not None,
                }
            )

            if not self._should_reprompt(filtered, attempt):
                current_result = filtered
                break

            feedback = self._build_feedback(filtered.issues, env_summary)
            attempt += 1

        reprompt_effects: List[Dict[str, Any]] = []
        for previous, current in zip(reprompt_history, reprompt_history[1:]):
            reprompt_effects.append(
                {
                    "attempt": current["attempt"],
                    "issues_before": previous["issue_count"],
                    "issues_after": current["issue_count"],
                    "improvement": previous["issue_count"] - current["issue_count"],
                }
            )

        current_result.metadata["reprompt_history"] = reprompt_history
        current_result.metadata["reprompt_count"] = max(0, len(reprompt_history) - 1)
        current_result.metadata["reprompt_effects"] = reprompt_effects
        return current_result
