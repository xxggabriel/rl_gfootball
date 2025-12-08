# Relatório de Estratégia de Treinamento

Este documento resume a estratégia usada para treinar o agente de futebol no Google Research Football (GRF) implementada neste repositório.

## Visão Geral
- Algoritmo: PPO distribuído com coleta vetorizada e mixed precision opcional (`rl_agents/train_5v5_ppo.py`).
- Cenário principal: `5_vs_5` com recompensa composta `scoring,checkpoints,shaping`.
- Observações: representações do GRF (`extracted`, `pixels`, etc.). Para imagens, 4 frames são empilhados por padrão.
- Controle: normalmente 1 jogador no time da esquerda (`--controlled-left 1`), contra agentes embutidos com dificuldade progressiva.

## Ambiente e Recompensas
- Criação de ambientes via `gfootball.env.create_environment`, com vetorização assíncrona (`gym.vector.AsyncVectorEnv`) para paralelizar a coleta.
- A dificuldade do oponente parte de `opponent_difficulty_start` (0.01) e cresce até `opponent_difficulty_end` (0.3) quando o currículo está ativo, reconstruindo os ambientes ao longo do treinamento.
- Recompensas ativas (`scoring,checkpoints,shaping` em `gfootball/env/wrappers.py`):
  - `scoring`: +1 gol a favor, -1 gol sofrido (GRF base).
  - `checkpoints`: 10 marcos até o gol adversário; cada marco com posse vale +0.1; gol soma bônus pelos marcos não coletados.
  - `shaping`: sinal denso clipado em [-2, 1] após somar tudo; inclui primeiro toque (+0.01), passe completo (+0.04), perda de posse (-0.05), falta/bola parada sem gol (-0.02), progresso em x (± até 0.02), chute na zona x>0.9/|y|<=0.12 (+0.05), aproximação do gol com posse (+ até 0.03), estagnação na própria metade (-0.002/step), “salvamento” defensivo (+0.05), roubo de bola do adversário (+0.05).
  - Penalidades adicionais: recuos para o goleiro (player 0) contam; a partir do 2º recuo no episódio, -0.05 cada; gol contra com último toque do nosso time recebe -0.7 extra.

## Espaço de Observação e Ação
- Observações: tensores HWC para representações baseadas em imagem; para outras (`simple115`, etc.), são vetores/grades numéricas. Quando `--stacked` está ativo, 4 quadros consecutivos são concatenados no canal.
- Ações: espaço `Discrete` ou `MultiDiscrete` (para vários jogadores). A utilidade `MultiCategorical` trata `log_prob`, `entropy` e `sample` consistentes com ações multi-discretas.

## Arquitetura da Política (`rl_agents/networks.py`)
- Encoder convolucional estilo IMPALA para observações HWC: 4 blocos `ImpalaBlock` (32→64→96→128 canais), pooling adaptativo para 6x8 e MLP final, totalizando 256 unidades de estado latente (padrão).
- Encoder MLP para observações vetoriais: flatten + duas camadas totalmente conectadas com ReLU.
- Cabeças de ator e crítico lineares sobre as features compartilhadas (ator gera logits, crítico valor escalar).
- Suporta amostragem e avaliação diferenciáveis para `Discrete` ou `MultiDiscrete`.

## Coleta e GAE
- Buffer de rollouts armazena `num_steps` × `num_envs` (padrão 2048 × 32).
- Retornos calculados com V-trace estilo GAE clássico: `gamma=0.993`, `gae_lambda=0.95`, normalizando vantagens por batch.

## Atualização PPO
- Épocas: 3 por atualização; minibatch de 1024 amostras baralhadas.
- Clipping de razão: `clip_range=0.2`; perda de valor com coeficiente 0.5; entropia com coeficiente 0.01.
- Otimizador: Adam (`lr=8e-5`, `eps=1e-5`); clipping de gradiente em 0.5.
- Mixed precision: `torch.cuda.amp` e `GradScaler` habilitados por padrão em GPU (`--use-amp/--no-amp`).

## Distribuição e Seeds
- Suporte a `torchrun`/`DistributedDataParallel`; `init_distributed` ajusta backend (NCCL ou Gloo) e dispositivo local.
- Semente global derivada de `--seed` offset por `rank` e índice de ambiente para reprodutibilidade com múltiplos processos.

## Currículo de Dificuldade
- Ativado por padrão (`--curriculum`). Após `curriculum_warmup_updates` (padrão 300), a dificuldade sobe linearmente até o alvo, reconstruindo ambientes quando a variação excede 0.02 para evitar overhead excessivo.
- A dificuldade usada na avaliação (`--eval-difficulty`) é limitada pelo valor atual para medir progresso realista.

## Avaliação, Métricas e Checkpoints
- Avaliação a cada `eval_interval` updates (padrão 10) em episódios independentes; acompanha `mean_return`.
- Checkpoints: `last.pt` salvo a cada `save_interval` updates; `best_model.pt` atualizado quando a média de retorno de avaliação melhora.
- Métricas de treino gravadas em JSONL (`runs/ppo_5v5/train_metrics.jsonl` por padrão) incluindo perdas, KL, grad_norm, estatísticas de vantagens e recompensas.
- `EpisodeMetricsCollector` agrega métricas de episódio emitidas pelo ambiente (retorno, sucesso, vitórias, gols, chutes no alvo, posse, progresso da bola) quando disponíveis.

## Execução
- Treinamento padrão (single node):  
  ```bash
  bash run.sh train
  ```
  Variáveis úteis: `TOTAL_TIMESTEPS`, `NUM_ENVS`, `NUM_STEPS`, `LR`, `ENV_NAME`, `REWARDS`, `OPP_DIFFICULTY_END`, `CHECKPOINT_DIR`, `LOGDIR`, `NPROC_PER_NODE`.
- Inferência/avaliação rápida do modelo salvo:  
  ```bash
  python3 -m rl_agents.run_inference --checkpoint policy/best_model.pt --episodes 5
  ```
  Flags: `--representation`, `--stacked/--no-stacked`, `--difficulty`, `--left-players`, `--right-players`, `--record` (salva vídeos em `videos/`).

## Considerações Práticas
- Ativar currículo acelera aprendizagem inicial e evita estagnação em níveis fáceis.
- Usar `pixels`/`pixels_gray` eleva custo computacional; `extracted` oferece bom compromisso sinal/velocidade.
- Mixed precision e vários ambientes aumentam throughput; monitore `grad_norm` e `kl` para detectar instabilidade.
- Ajustar `minibatch-size` e `num-steps` pode ser necessário se `NUM_ENVS` ou memória GPU mudarem.
