from __future__ import annotations


class Qwen2Config:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *, revision: str,
                        local_files_only: bool, trust_remote_code: bool) -> Qwen2Config: ...
