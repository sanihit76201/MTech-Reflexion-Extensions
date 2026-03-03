"""
Extension 4: LunarLanderReflexion → HumanEval Coding (Verbal RL for Code)
✅ FIXED: Generates Python code, NOT rocket actions
✅ FULLY COMPATIBLE with run_comparison.py HumanEval benchmark
"""

import numpy as np
from typing import Dict, Any
import logging
from .base import ReflexionAgent
from ..llm import BaseLLMModel
from ..memory import TemporalMemory

logger = logging.getLogger(__name__)

class LunarLanderReflexion(ReflexionAgent):
    """
    Extension 4: Verbal Reflexion for CODE GENERATION (LunarLander-style RL)
    Applies continuous control reflexion principles to discrete coding tasks.
    """
    
    def __init__(self, llm: BaseLLMModel, max_trials: int = 3):
        super().__init__(llm, memory_mode='temporal', max_trials=max_trials)
    
    def solve_task(self, task: Dict, verbose: bool = False) -> Dict[str, Any]:
        """
        ✅ SOLVES HUMAN EVAL CODING TASKS using LunarLander reflexion loop
        task["prompt"] → Python code via verbal reinforcement learning
        """
        task_id = task['task_id']
        
        if verbose:
            logger.info(f"\n🚀 LunarLander-Style Reflexion: {task_id}")
        
        total_reward = 0
        reflections = []
        
        for trial in range(self.max_trials):
            # 1. OBSERVATION: Task state → text
            state_desc = f"Task: {task_id}\nPrompt: {task['prompt'][:100]}..."
            
            # 2. MEMORY: Past coding "trajectories"
            mem_context = "\n".join(self.memory.get_relevant_memories(task['prompt'], k=3))
            
            # 3. POLICY: LLM generates code (like thruster action)
            code = self.get_code_action(state_desc, mem_context, trial+1)
            
            # 4. EXECUTE + REWARD: Test code → pass/fail
            result = self.evaluator.evaluate(code, task['entry_point'], task['test'])
            reward = 100 if result['passed'] else -50  # RL reward shaping
            total_reward += reward
            
            if verbose:
                logger.info(f"🔄 Trial {trial+1}: Reward={reward}, Passed={result['passed']}")
            
            if result['passed']:
                logger.info(f"✅ {task_id} solved in {trial+1} trials (LunarLander RL)")
                return {
                    'task_id': task_id,
                    'success': True,
                    'trials': trial+1,
                    'total_reward': total_reward,
                    'code': code,
                    'agent_type': 'LunarLanderReflexion'
                }
            
            # 5. REFLEXION: Verbal feedback (LunarLander-style)
            reflection = self.generate_reflection(result['error'], code, reward)
            reflections.append(reflection)
            self.memory.add_reflection(reflection)
        
        # Episode failed
        final_reflection = f"LunarLander RL failed {task_id}: reward={total_reward}"
        self.memory.add_reflection(final_reflection)
        
        return {
            'task_id': task_id,
            'success': False,
            'trials': self.max_trials,
            'total_reward': total_reward,
            'agent_type': 'LunarLanderReflexion',
            'final_reflection': final_reflection
        }
    
    def get_code_action(self, state_desc: str, memory_context: str, trial: int) -> str:
        """LLM policy: State + reflections → Python code (discrete action)."""
        prompt = f"""LunarLander-Style Verbal Reflexion for Coding (Trial {trial})

Past Reflections (like flight trajectory memory):
{memory_context}

Current State: {state_desc}

POLICY: Generate COMPLETE Python function. Use reflections to avoid past failures.

Output ONLY valid Python code:"""
        
        return self.llm.call_llm(prompt, max_tokens=2048)
    
    def generate_reflection(self, error: str, code: str, reward: float) -> str:
        """Verbal RL feedback (altitude/speed → error/pass)."""
        if reward >= 100:
            return f"SUCCESS! Reward +100. Strategy worked perfectly."
        elif reward >= 0:
            return f"Improved (reward {reward}). Error: {error[:100]}"
        else:
            return f"CRASH (reward {reward}). Fix: {error[:100]}. State crash analysis needed."

# Test compatibility
if __name__ == "__main__":
    from ..llm import BaseLLMModel  # Your LLM
    
    llm = BaseLLMModel(...)  # Your config
    agent = LunarLanderReflexion(llm)
    
    dummy_task = {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
        "entry_point": "has_close_elements",
        "test": "assert has_close_elements([1.0, 2.8, 3.3, 4.4, 5.0, 100.0], 0.8) == True"
    }
    
    result = agent.solve_task(dummy_task, verbose=True)
    print(result)
