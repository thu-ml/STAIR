from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Type

@dataclass
class MCTSBaseConfig:

    actor_model_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to actor model dir"}
    )
    mode: str = field(
        default="outcome", metadata={"help": "Select from outcome, safe-constraint."}
    )
    k1: float = field(
        default=1, metadata={"help": "Should >= 1, only activate when safe-constraint"}
    )
    k2: float = field(
        default=1, metadata={"help": "Should > 0 and <= 1, only activate when safe-constraint"}
    )
    generate_mode: str = field(
        default="local", metadata={"help": "local or vllm."}
    )
    worker_num: int = field(
        default=1, metadata={"help": "Should be 1 when local, >=1 when vllm"}
    )
    server_url: Optional[str] = field(
        default=None, metadata={"help": "Actor server url when in vllm mode"}
    )
    worker_prompt_num: int = field(
        default=10, metadata={"help": "Prompt number for each worker"}
    )

    temperature: float = field(
        default=1.2, metadata={"help": "Control diversity. Higher for more diversity."}
    )
    top_p: float = field(
        default=0.9, metadata={"help": "Control diversity. Higher for more diversity."}
    )
    top_k: int = field(
        default=50, metadata={"help": "Control diversity. Higher for more diversity."}
    )
    max_tokens: int = field(
        default=1024, metadata={"help": "Max tokens for each step."}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed"}
    )

    stop_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "Stop tokens for a step"}
    )
    end_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "End tokens for complete response"}
    )

    train_prompt_path: Optional[str] = field(
        default=None, metadata={"help": "Path to training prompt file"}
    )
    output_path: Optional[str] = field(
        default=None, metadata={"help": "Path to output mct result dir"}
    )

    c: float = field(
        default=1.5, metadata={"help": "Hyperparam for c of PUCB or UCB"}
    )
    max_depth: int = field(
        default=7, metadata={"help": "Hyperparam for max_depth of MCT"}
    )
    iterations: int = field(
        default=200, metadata={"help": "MCTS max iterations for one prompt"}
    )
    generate_samples_number: int = field(
        default=4, metadata={"help": "Numbers of expanded children nodes in expand phase"}
    )

    visit_all_node: bool = field(
        default=True, metadata={"help": "Visit all unvisited nodes after all iterations"}
    )
    p_average_strategy: str = field(
        default="uniform", metadata={"help": "Strategy to calculate p in PUCB, select from uniform, arithmetic and geometric. Only uniform available when vllm mode."}
    )
    able_to_reselected: bool = field(
        default=True, metadata={"help": "Whether a node can be selected many times in MCTS as final leaf node"}
    )
    score_type: str = field(
        default="UCB", metadata={"help": "Strategy to calculate node score, select from PUCB and UCB"}
    )

    use_cache: bool = field(
        default=False, metadata={"help": "Whether to use a cache when generating MCT"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to cache dir"}
    )
    log_file: Optional[str] = field(
        default=None, metadata={"help": "Path to output log"}
    )

    