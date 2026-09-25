import json, glob
MODELS = ['llama-3-8b','llama-7b','llama-2-7b','llama-3.1-8b','qwen1.5-7b','qwen2-7b',
          'qwen2.5-7b','qwen3-8b','qwen-7b','bert-base','roberta-base',
          'spanbert-base-cased','xlm-roberta-base']
m = [w['word'] for w in json.load(open('data/raw/semcor_words/_manifest.json'))['words']]
tot = 0
for mod in MODELS:
    n = sum(1 for w in m if glob.glob(f'results/words/{w}/*/{mod}/metrics/*separation*'))
    tot += n
    print(f'{mod:22s} {n:4d}/1000  {"done" if n >= 999 else ""}')
print(f'\noverall: {tot}/{len(MODELS)*1000}  ({100*tot/(len(MODELS)*1000):.1f}%)')
