from opencompass.models import HuggingFaceBaseModel
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TruthfulQADataset, TruthfulQAEvaluator

truthfulqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='reference',
    train_split='train',
    test_split='train')

truthfulqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{question}')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

truthfulqa_eval_cfg = dict(
    evaluator=dict(
        type=TruthfulQAEvaluator, metrics=('bleu',), key='ENV'))

datasets = [dict(
    abbr='truthful_qa',
    type=TruthfulQADataset,
    path='parquet',
    data_files='/home/tianyu/opencompass/data/truthful_qa/generation/validation-00000-of-00001.parquet',
    name='generation',
    reader_cfg=truthfulqa_reader_cfg,
    infer_cfg=truthfulqa_infer_cfg,
    eval_cfg=truthfulqa_eval_cfg)]

models = [
    # -- LLaMA family ----------------------------------------------------------
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-7b',
        path='/home/tianyu/models/llama/llama-7b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(use_fast=False, trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-2-7b',
        path='/home/tianyu/models/llama/Llama-2-7b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(use_fast=False, trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-3-8b',
        path='/home/tianyu/models/llama/Llama-3-8b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(use_fast=False, trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-3.1-8b',
        path='/home/tianyu/models/llama/Llama-3.1-8b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(use_fast=False, trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    # -- Qwen family -----------------------------------------------------------
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen-7b',
        path='/home/tianyu/models/Qwen/qwen-7b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen1.5-7b',
        path='/home/tianyu/models/Qwen/qwen1.5-7b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2-7b',
        path='/home/tianyu/models/Qwen/qwen2-7b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2.5-7b',
        path='/home/tianyu/models/Qwen/qwen2.5-7b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen3-8b',
        path='/home/tianyu/models/Qwen/qwen3-8b',
        max_out_len=256,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(trust_remote_code=True),
        model_kwargs=dict(trust_remote_code=True),
    ),
]