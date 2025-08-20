# tslib_wrapper_with_args.py
import argparse
from typing import Callable, Optional, Type, Tuple, Any, Dict
from utils.TRACK_Dataset_trllm_cont import TRLLMContDataset, collate_unsuperv_mask_cont
import types
import os

# --- wandb-safe logger (复制你之前给出的版本) ---
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

def _safe_wandb_log(key: str, value):
    try:
        if _HAS_WANDB and wandb.run is not None:
            wandb.log({key: value})
            try:
                if isinstance(value, (int, float)):
                    wandb.run.summary[key] = float(value)
                else:
                    wandb.run.summary[f"{key}_last"] = str(value)[:4096]
            except Exception:
                pass
        else:
            print(f"[WANDB LOG] {key}: {value}")
    except Exception:
        print(f"[WANDB LOG ERROR] {key}: {value}")

def _winfo(msg: str):
    _safe_wandb_log("dataset/info", msg)

def _wwarn(msg: str):
    _safe_wandb_log("dataset/warn", msg)

# --- TSLibDatasetWrapper 支持 args 或 dict ---
class TSLibDatasetWrapper:
    """
    简化 wrapper：接收 config（dict 或 argparse.Namespace）以及原始 dataset 类（orig_dataset_class）。
    用法：
        wrapper = TSLibDatasetWrapper(config_or_args, orig_dataset_class=TRLDataset, collate_fn=my_collate)
        train_loader, eval_loader, test_loader = wrapper.get_data()
        feat = wrapper.get_data_feature()
    注意：
      - 本 wrapper 会尝试以单个 dict 参数调用 orig_dataset_class(config_dict)
      - 若 orig_dataset_class 的构造器不是这种签名，需要调用方在外部先实例化，然后直接传实例（目前 wrapper 仅实现类传入方式）
    """

    def __init__(
        self,
        config: Dict | argparse.Namespace,
        orig_dataset_class: Type = TRLLMContDataset,
        collate_fn: Optional[Callable] = collate_unsuperv_mask_cont,
    ):
        self._raw_config = config
        self.config = self._normalize_config(config)
        self.orig_dataset_class = orig_dataset_class
        self.collate_fn = collate_fn
        self.orig = None

        _winfo(f"Initializing TSLibDatasetWrapper for {self.orig_dataset_class.__name__}")
        self._instantiate_orig()

    def _normalize_config(self, config):
        """
        支持：
          - dict -> 直接使用
          - argparse.Namespace -> vars(namespace)
          - object with __dict__ -> dict(obj.__dict__)
        返回 dict
        """
        if isinstance(config, dict):
            return config
        # argparse.Namespace
        if hasattr(config, "__dict__") and not isinstance(config, dict):
            try:
                # argparse.Namespace -> vars(namespace)
                cfg = vars(config)  # works for Namespace
                # remove None-valued keys? (保留为原样)
                return cfg
            except Exception:
                # fallback to __dict__
                return dict(config.__dict__)
        raise TypeError("config must be a dict or argparse.Namespace-like object")

    def _instantiate_orig(self):
        """
        使用单个 dict 参数尝试实例化原始 dataset。
        """
        try:
            self.orig = self.orig_dataset_class(self.config)
            _winfo(f"Instantiated original dataset: {self.orig_dataset_class.__name__}")
        except Exception as e:
            _wwarn(f"Failed to instantiate {self.orig_dataset_class.__name__} with config dict. Error: {e}")
            # 抛出异常，提示用户用其它方式构造原始 dataset
            raise

        # 注入 collate_fn（如果提供）
        if self.collate_fn is not None:
            try:
                setattr(self.orig, "collate_fn", self.collate_fn)
                _winfo("Injected collate_fn into original dataset instance.")
            except Exception as e:
                _wwarn(f"Failed to inject collate_fn: {e}")

    def get_data(self) -> Tuple[Any, Any, Any]:
        if self.orig is None:
            raise RuntimeError("Original dataset not instantiated.")
        if not hasattr(self.orig, "get_data"):
            msg = f"{self.orig_dataset_class.__name__} has no method get_data()."
            _wwarn(msg)
            raise AttributeError(msg)

        _winfo("Calling original.get_data() ...")
        res = self.orig.get_data()
        _winfo("original.get_data() finished.")
        return res

    def get_data_feature(self) -> Dict:
        if self.orig is None:
            raise RuntimeError("Original dataset not instantiated.")
        if not hasattr(self.orig, "get_data_feature"):
            msg = f"{self.orig_dataset_class.__name__} has no method get_data_feature()."
            _wwarn(msg)
            raise AttributeError(msg)

        _winfo("Calling original.get_data_feature() ...")
        feat = self.orig.get_data_feature()
        _winfo("original.get_data_feature() finished.")
        return feat

    @property
    def orig_instance(self):
        return self.orig