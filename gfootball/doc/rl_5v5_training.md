# Treinamento 5v5 com PPO distribuido

Este guia resume a implementacao adicionada para treinar e executar agentes de RL no cenario 5 vs 5 do Google Research Football.

## Arquivos principais
- `rl_agents/train_5v5_ppo.py`: loop de treino PPO com GAE, suporte a `torchrun` multi-no/GPU, curriculo de dificuldade e salvamento periodico.
- `rl_agents/networks.py`: politica ator-critico estilo IMPALA (conv para SMM/pixels, MLP para vetores) compativel com espacos `Discrete` e `MultiDiscrete`.
- `rl_agents/run_inference.py`: rollout rapido para checar checkpoints.
- `agent.py`: implementacao de `Player` que carrega automaticamente `policy/best_model.pt` (ou `FOOTBALL_CHECKPOINT`) para uso em avaliacao.
- `run.sh`: atalho para treino (`./run.sh train`) ou inferenca (`./run.sh infer`).
- `policy/`: pasta de checkpoints (`best_model.pt`, `last.pt`).

## Como treinar
Requer Python dentro do container base do repositorio (GPU com CUDA). O script usa `torchrun` para paralelizar por GPU/no.

### Single node
```bash
./run.sh train \
  TOTAL_TIMESTEPS=8000000 \
  NUM_ENVS=12 \
  NUM_STEPS=256 \
  LR=0.00008
```

### Multi-no (ex.: 3 maquinas com 1 GPU cada)
Em cada no (ja conectado via VPN/SSH) execute:
```bash
MASTER_ADDR=10.100.0.225 MASTER_PORT=29500 \
NNODES=3 NODE_RANK=<0|1|2> NPROC_PER_NODE=1 \
NUM_ENVS=8 NUM_STEPS=256 TOTAL_TIMESTEPS=30000000 \
./run.sh train
```
`torchrun` sincroniza gradientes via `nccl`; ajuste `NUM_ENVS` por GPU conforme memoria (4090: 12-16 envs com SMM empilhado geralmente cabem).

### Hiperparametros e recursos
- Recompensas padrao: `scoring,checkpoints,shaping` (shaping denso: toque inicial, progresso da bola rumo ao gol, passe recebido/errado, chute certo, penalidade por bola fora; gol continua valendo 1, recompensas clipadas em [-1,1]).
- Observacao: representacao `extracted` com `--stacked` (16 canais) para maior estabilidade.
- Curriculo: `--curriculum` aumenta `right_team_difficulty` de 0.05->0.8 conforme o progresso; desative com `--no-curriculum`.
- Mixed precision: habilitada em CUDA (`--no-amp` para desativar).
- Metricas: `runs/ppo_5v5/train_metrics.jsonl` traz KL/clip_frac/dificuldade e métricas de jogo: `episodic_return_mean/std`, `success_rate`, `goals_per_episode`, `shots_on_target_per_episode`, `ball_possession_time_mean`, `ball_distance_from_goal_mean`, `win_rate`.
- Checkpoints: `policy/last.pt` salvo a cada `--save-interval`; `policy/best_model.pt` apos avaliacoes (`--eval-interval`).

## Como inferir/avaliar
- Via script: `./run.sh infer CHECKPOINT=policy/best_model.pt EPISODES=5`
- Via classe `Player` (para avaliacao automatica): defina `FOOTBALL_CHECKPOINT=/caminho/para/best_model.pt` e inicialize o ambiente com `players=['agent']`.
- Gravacao de video: `RECORD_VIDEO=1 ./run.sh infer` (saida em `./videos`).

## Docker
- Build: `docker build -t gfootball-rl .`
- Treino no container: `docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace gfootball-rl ./run.sh train`
- Inferencia no container: `docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace gfootball-rl ./run.sh infer`

## Notas de arquitetura
- Politica conv do tipo IMPALA com blocos residuais, pooling progressivo e projecao para 256 unidades.
- GAE + PPO com normalizacao de vantagens e clipping, entropia para incentivar exploracao.
- Compativel com `MultiDiscrete` para controlar multiplos jogadores simultaneamente (`--controlled-left`/`--controlled-right`).
- Se `policy/best_model.pt` nao existir, `agent.py` faz fallback para uma politica nao treinada (util para smoke tests).
