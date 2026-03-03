
import logging
from typing import Dict, List

from reflexion.memory import VectorEpisodicMemory
from reflexion.agents.base import ReflexionAgent

logger = logging.getLogger(__name__)


class VectorReflexionAgent(ReflexionAgent):
    """
    Reflexion agent with semantic episodic memory (VectorEpisodicMemory).

    The base ReflexionAgent already:
      - Creates VectorEpisodicMemory when memory_mode='vector'
      - Retrieves semantically similar memories per task prompt
      - Injects those memories into the LLM prompt
      - Stores failure reflections back into memory

    This subclass extends that with:
      - TOP_K=5 instead of base k=3 (more context for complex tasks)
      - Richer failure reflections (task_id + error + actionable hint)
      - memories_used + memory_pool in result dict (for benchmark metrics)
      - Explicit memory type verification on init (fail fast if misconfigured)
      - Reasoning quality tracking (constraint awareness, memory utilization)
    """

    TOP_K_MEMORIES = 5   # base uses k=3; we retrieve more context

    def __init__(self, llm, max_trials: int = 3) -> None:
        # Base class creates VectorEpisodicMemory via memory_mode='vector'
        super().__init__(llm, memory_mode="vector", max_trials=max_trials)

        # Verify memory type — fail fast rather than silently use wrong memory
        if not isinstance(self.memory, VectorEpisodicMemory):
            raise TypeError(
                f"Expected VectorEpisodicMemory, got {type(self.memory).__name__}. "
                f"Check ReflexionAgent.__init__() memory_mode branching."
            )

        logger.info(
            "VectorReflexionAgent ready | memory=%s | top_k=%d",
            type(self.memory).__name__,
            self.TOP_K_MEMORIES,
        )

    # ------------------------------------------------------------------ #
    #  Override solve_task to use TOP_K=5 and richer reflections          #
    # ------------------------------------------------------------------ #

    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        """
        Solve task using VectorEpisodicMemory with TOP_K=5 retrieval.

        We re-implement the core loop (rather than calling super()) because:
          - We need k=5 not k=3 in get_relevant_memories()
          - We need richer reflection strings for better future retrieval
          - We need memories_used in the result dict for benchmark metrics
          - We track reasoning quality metrics

        The logic mirrors ReflexionAgent.solve_task() exactly — only the
        retrieval k, reflection format, and result keys differ.
        """
        task_id = task["task_id"]
        
        # Track reasoning quality across trials
        reasoning_traces = []

        for trial in range(self.max_trials):

            # ── Retrieve top-5 (not base k=3) semantic memories ───────
            memories: List[str] = self.memory.get_relevant_memories(
                task["prompt"], k=self.TOP_K_MEMORIES
            )
            mem_ctx = "\n".join(f"- {m}" for m in memories) if memories else "None"

            # ── Prompt with lightweight reasoning scaffold ─────────────
            prompt = f"""You are an expert Python programmer. Complete this function:

{task['prompt']}

Past reflections (TOP-{self.TOP_K_MEMORIES} semantically similar — learn from these):
{mem_ctx}

Before coding, briefly note:
- Key constraints/edge cases to handle
- Your approach in 1-2 sentences

Then provide the complete implementation.

Requirements:
1. Complete the function implementation
2. Handle all edge cases
3. Make sure all test cases pass
4. Output ONLY the Python code, no markdown, no explanations

Your code:"""

            try:
                logger.info("🔄 Trial %d/%d — %s | memories=%d",
                            trial + 1, self.max_trials, task_id, len(memories))

                # ── Generate ───────────────────────────────────────────
                response = self.llm.call_llm(prompt, max_tokens=2048)

                if isinstance(response, list):
                    response = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in response
                    )

                # ── Evaluate reasoning quality before code extraction ──
                reasoning_quality = self._evaluate_reasoning_trace(
                    response, task, memories
                )
                reasoning_traces.append(reasoning_quality)

                # ── Clean markdown ─────────────────────────────────────
                code = response
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()

                # ── Evaluate ───────────────────────────────────────────
                results = self.evaluator.evaluate(
                    code, task["entry_point"], task["test"]
                )

                if results["passed"]:
                    logger.info(
                        "✅ %s solved in %d trials | memories=%d | reasoning=%.2f",
                        task_id, trial + 1, len(memories),
                        reasoning_quality["completeness_score"]
                    )
                    return {
                        "task_id":       task_id,
                        "success":       True,
                        "trials":        trial + 1,
                        "code":          code,
                        "agent_type":    "VectorReflexion",
                        "memories_used": len(memories),
                        "memory_pool":   len(self.memory),
                        # NEW: Reasoning quality metrics
                        "reasoning_quality": reasoning_quality,
                        "reasoning_traces": reasoning_traces,
                    }

                # ── Richer failure reflection ──────────────────────────
                # Base stores: "Trial N failed: <error>"
                # We store:    task context + error + actionable hint
                # → future similar tasks get more useful retrieved context
                reflection = self._build_failure_reflection(
                    task_id, trial, results["error"], task
                )
                self.memory.add_reflection(reflection)
                
                logger.warning(
                    "❌ %s trial %d failed | pool=%d | reasoning=%.2f",
                    task_id, trial + 1, len(self.memory),
                    reasoning_quality["completeness_score"]
                )

            except KeyboardInterrupt:
                logger.error("⚠️  Interrupted")
                raise
            except Exception as exc:
                reflection = (
                    f"Task '{task_id}' trial {trial+1} exception: {str(exc)[:120]}. "
                    f"Check input types and function signature."
                )
                self.memory.add_reflection(reflection)
                logger.error("❌ Exception in %s: %s", task_id, exc)

        # Failed after all trials
        logger.warning("❌ %s failed after %d trials | pool=%d",
                       task_id, self.max_trials, len(self.memory))
        
        # Return best reasoning quality from all attempts
        best_reasoning = max(
            reasoning_traces,
            key=lambda r: r["completeness_score"]
        ) if reasoning_traces else self._empty_reasoning_quality()
        
        return {
            "task_id":       task_id,
            "success":       False,
            "trials":        self.max_trials,
            "agent_type":    "VectorReflexion",
            "memories_used": 0,
            "memory_pool":   len(self.memory),
            "reasoning_quality": best_reasoning,
            "reasoning_traces": reasoning_traces,
        }

    # ================================================================
    # REASONING QUALITY EVALUATION (NEW)
    # ================================================================

    def _evaluate_reasoning_trace(
        self,
        response: str,
        task: Dict,
        memories: List[str]
    ) -> Dict:
        """
        Evaluate reasoning quality from the LLM response.
        
        Scores (lightweight version for HumanEval):
        - constraint_awareness: Did response mention edge cases?
        - logical_structure: Is there a clear approach stated?
        - memory_utilization: Did response reference past failures?
        - explanation_depth: How much reasoning was shown?
        """
        
        response_lower = response.lower()
        
        # 1. Constraint awareness
        constraint_keywords = [
            'edge case', 'boundary', 'empty', 'none', 'zero',
            'negative', 'invalid', 'null', 'error', 'exception'
        ]
        constraint_mentions = sum(
            1 for kw in constraint_keywords if kw in response_lower
        )
        constraint_awareness = min(1.0, constraint_mentions / 3)
        
        # 2. Logical structure (approach statement)
        structure_keywords = [
            'approach:', 'strategy:', 'first', 'then', 'finally',
            'algorithm:', 'method:', 'step'
        ]
        has_structure = any(kw in response_lower for kw in structure_keywords)
        logical_structure = 1.0 if has_structure else 0.3
        
        # 3. Memory utilization
        memory_keywords = [
            'reflection', 'previous', 'learned', 'similar',
            'recall', 'pattern', 'mistake'
        ]
        memory_mentions = sum(
            1 for kw in memory_keywords if kw in response_lower
        )
        memory_utilization = min(1.0, memory_mentions / 2) if memories else 0.0
        
        # 4. Explanation depth
        # Count non-code lines that look like reasoning
        lines = response.split('\n')
        reasoning_lines = [
            line for line in lines
            if line.strip()
            and not line.strip().startswith('#')
            and 'def ' not in line
            and 'return' not in line
            and '```' not in line
        ]
        explanation_depth = min(1.0, len(reasoning_lines) / 5)
        
        # 5. Completeness (weighted average)
        completeness = (
            constraint_awareness * 0.35 +
            logical_structure * 0.25 +
            memory_utilization * 0.25 +
            explanation_depth * 0.15
        )
        
        return {
            "constraint_awareness": constraint_awareness,
            "logical_structure": logical_structure,
            "memory_utilization": memory_utilization,
            "explanation_depth": explanation_depth,
            "completeness_score": completeness,
            "reasoning_line_count": len(reasoning_lines),
            "memory_references": memory_mentions,
        }

    def _empty_reasoning_quality(self) -> Dict:
        """Return zero-scored reasoning quality for failed cases."""
        return {
            "constraint_awareness": 0.0,
            "logical_structure": 0.0,
            "memory_utilization": 0.0,
            "explanation_depth": 0.0,
            "completeness_score": 0.0,
            "reasoning_line_count": 0,
            "memory_references": 0,
        }

    def _build_failure_reflection(
        self,
        task_id: str,
        trial: int,
        error: str,
        task: Dict
    ) -> str:
        """
        Build rich failure reflection for future retrieval.
        
        Includes:
        - Task identifier
        - Specific error
        - Actionable hints derived from error type
        """
        
        # Derive hints from error patterns
        error_lower = error.lower() if error else ""
        hints = []
        
        if "index" in error_lower or "out of range" in error_lower:
            hints.append("Check list bounds and empty input handling")
        if "key" in error_lower or "dict" in error_lower:
            hints.append("Verify dictionary key existence")
        if "none" in error_lower or "attribute" in error_lower:
            hints.append("Add None checks before method calls")
        if "type" in error_lower:
            hints.append("Validate input types match expected")
        if "division" in error_lower or "zero" in error_lower:
            hints.append("Handle zero division edge case")
        
        if not hints:
            hints.append("Review edge cases, type handling, boundary conditions")
        
        hint_text = "; ".join(hints[:2])  # Max 2 hints
        
        return (
            f"Task '{task_id}' (trial {trial+1}) failed: {error[:150]}. "
            f"Hint: {hint_text}."
        )