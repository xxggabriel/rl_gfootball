# Relatório do Agente de Futebol (5v5)

Documento final do projeto da disciplina: desenvolver, treinar e avaliar um agente capaz de jogar futebol no ambiente Google Research Football (GRF) no cenário 5 vs 5, integrando conceitos de RL, deep learning e exploração–explotação.

## Entregáveis e estado atual
- `policy/best_model.pt` (checkpoint final) e `policy/last.pt`. É possível gerar `policy/best_model.zip` se o formato for exigido.
- `agent.py` ajustado herdando `PlayerBase`, carregando o checkpoint via `FOOTBALL_CHECKPOINT` (padrão `policy/best_model.pt`).
- `run.sh` orquestra treino (`train`) e inferência (`infer`), compatível com Docker/local e GPU (mixed precision).
- Vídeos exemplo gravados em `videos/` (`episode_done_*.webm`, `lost_score_*.webm`).
- Este relatório cobre modelagem, arquitetura, treinamento, resultados, dificuldades e próximos passos.

## Como executar
- Treino (torchrun, PPO distribuído, 32 envs padrão):  
  ```bash
  bash run.sh train
  ```
  Variáveis úteis: `NUM_ENVS`, `NUM_STEPS`, `TOTAL_TIMESTEPS`, `LR`, `ENV_NAME`, `REWARDS`, `OPP_DIFFICULTY_END`, `LOGDIR`, `CHECKPOINT_DIR`, `NPROC_PER_NODE`.
- Inferência / vídeo (usa `policy/best_model.pt` por padrão):  
  ```bash
  bash run.sh infer CHECKPOINT=policy/best_model.pt RENDER=1 RECORD_VIDEO=1 EPISODES=2
  ```
  Ajuste `REPRESENTATION`, `STACKED`, `DIFFICULTY`, `LEFT_PLAYERS`, `RIGHT_PLAYERS`, `VIDEO_FORMAT`.

## Estratégia de modelagem
- **Estado/observação:** representação do GRF; padrão `extracted` (features estruturadas) com empilhamento de 4 quadros quando aplicável. Para imagens (`pixels`/`pixels_gray`), entrada HWC; para vetores (`simple115`, etc.), entrada plana.
- **Ações:** conjunto `full` (padrão no `run.sh`) para permitir troca manual de jogador; `MultiDiscrete` quando há controle de múltiplos atletas (`--controlled-left/right`), caso contrário `Discrete`.
- **Recompensas (combinadas `scoring,checkpoints,shaping` em `gfootball/env/wrappers.py`):**
  - `scoring`: +1 gol a favor, -1 gol sofrido.
  - `checkpoints`: 10 marcos até o gol adversário; posse em cada marco vale +0.1; gol concede bônus dos marcos não coletados.
  - `shaping`: sinal denso clipado em [-2, 1]; inclui primeiro toque (+0.01), passe completo (+0.04), perda de posse (-0.05), falta/bola parada sem gol (-0.02), progresso em x (± até 0.02), chute em zona x>0.9/|y|<=0.12 (+0.05), aproximação ofensiva (+ até 0.03), estagnação defensiva (-0.002/step), salvamento/roubo de bola (+0.05).
  - Penalizações extra: recuos frequentes para o goleiro (player 0) a partir do 2º recuo (-0.05 cada) e gol contra com último toque nosso (-0.7).

## Agente e arquitetura
- **Backbone:** CNN estilo IMPALA para entradas HWC com 4 blocos (32→64→96→128 canais), pooling adaptativo para 6x8 e MLP final; sai vetor latente de 256 unidades. Para observações vetoriais, MLP de duas camadas (ReLU).
- **Cabeças:** ator e crítico lineares sobre as features compartilhadas; suporta `Discrete` e `MultiDiscrete` (via utilitário `MultiCategorical`).
- **Hiperparâmetros principais (defaults `rl_agents/train_5v5_ppo.py`):**
  - `gamma=0.993`, `gae_lambda=0.95`
  - `num_envs=32`, `num_steps=2048` (buffer de rollouts 2048×32)
  - `ppo_epochs=3`, `minibatch_size=1024`
  - `clip_range=0.2`, `entropy_coef=0.01`, `value_coef=0.5`
  - `lr=1e-5` (Adam, `eps=1e-5`), `max_grad_norm=0.5`
  - `stacked=True` (se representação permite), `controlled_left=1`, `controlled_right=0`
- **Técnicas:** mixed precision (AMP) habilitada em GPU; suporte a `torchrun`/DDP; seed global derivada de `--seed` somada ao `rank`.

## Abordagem de treinamento
- **Coleta:** `gym.vector.AsyncVectorEnv` com ambientes reconstruídos conforme dificuldade muda; ações em lote; vantagens normalizadas por batch.
- **Currículo de dificuldade:** inicia em `opponent_difficulty_start=0.01`, alvo `0.3`. Só sobe após `curriculum_warmup_updates=300`, reconstruindo envs quando a variação passa de 0.02. Avaliação respeita o mínimo entre dificuldade atual e `eval_difficulty`.
- **Avaliação:** a cada `eval_interval=10` updates, 8 episódios independentes; `best_model.pt` salvo se `mean_return` (avaliação) melhora.
- **Logs/checkpoints:** métricas em JSONL em `runs/ppo_5v5/train_metrics.jsonl`; `last.pt` salvo a cada `save_interval=10`.

## Resultados obtidos (execução registrada)
- Execução atual: 540 updates ≈ 35.389.440 steps agregados (32 envs × 2048 steps × 540, 1 processo). O currículo não disparou (warmup é 300), então a dificuldade ficou em 0.01.
- Tendências dos logs (`runs/ppo_5v5/train_metrics.jsonl`):
  - Entropia caiu de 3.462 para 3.219, indicando política mais concentrada.
  - KL subiu de 0.0063 para 0.0173; clip_frac final ~0.26 (maior pressão de clipping perto do fim).
  - `reward_mean` ficou próximo de zero (mín -0.00049, máx 0.00022; último 1.2e-05) com `reward_std` mediano 0.01695 — coerente com recompensas densas pequenas.
  - `advantage_std` reduziu de 0.134 para 0.046; `policy_loss` médio -0.0161 (±0.0016), `value_loss` médio 0.00175 (±0.0009).
  - Gradiente (norma) chegou a 8.6 na última iteração; clipping ativo em 0.5.
  - Episódios no batch (`episodes_in_batch`) ficaram 0 porque o wrapper do GRF não retornou `episode_metrics`; é preciso habilitar essa coleta para ver retornos, vitórias e gols.
- Checkpoints: `policy/last.pt` (step ~12.8M) e `policy/best_model.pt` (melhor avaliação registrada). Vídeos de inferência em `videos/episode_done_*.webm` e `videos/lost_score_*.webm`.