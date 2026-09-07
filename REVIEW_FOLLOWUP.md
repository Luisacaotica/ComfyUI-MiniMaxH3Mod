# RefMod — implementação e validação

Atualizado em 07/09/2026. Relatório de implementação e validação da v0.2.0.

## Correções concluídas

- Salvamento de mods e presets respeita a primeira raiz registrada em `folder_paths`, incluindo YAML externo; apenas a ausência de raízes usa `models/refmods`. Selecionar a pasta não cria diretórios.
- Removidos imports e aviso de dependência do pack ComfyUI-MiniMaxH3. O socket legado `av_encoder` usa seu caminho de vídeo com o VAE nativo; Apply continua aceitando conditioning nativo e objetos antigos sem importar o pack.

- Loaders normal e Axis aceitam inputs conectados ainda não resolvidos na fila. Nomes fixos inválidos continuam sendo rejeitados; arquivos conectados inexistentes falham na execução.
- Descoberta recursiva, barras normalizadas, filtro de metadados e exclusão de graph_presets. Raízes registradas e caminhos legados têm precedência consistente entre listagem e carga.
- Step Curve processa o payload atual sem guardar tensores entre chamadas. Cobre refs visuais/áudio, payload vazio, CFG e curvas encadeadas. Schedule isolado por execução.
- Bridge usa hooks públicos no MODEL clonado, sem lista global ou monkeypatch do Continuum. Restaura o conditioning mesmo se o sampler falhar. Disarm permanece por compatibilidade; desligue a bridge no ramo MODEL correspondente.
- Máscaras seguem o crop da imagem. Vídeos sem contagem confiável têm amostragem limitada em memória e fechamento do decoder em finally.
- Cache invalida por fingerprint de arquivo/sidecar; loaders expõem IS_CHANGED. Teto de 24 arquivos e 256 MiB de tensores.
- Copies e valores numéricos são verificados na execução. Orçamento opcional soma todas as cópias; hard cap insuficiente para um frame gera erro explícito.
- Salvamento atômico e gravação de config deduplicada por caminho.

## Manutenção e recursos entregues

- **Extract H3 RefMod Master** reúne imagem/vídeo e áudio opcional, usa os extratores compartilhados em sequência e retorna um bundle único. Salva arquivos separados com sufixos `_visual` e `_audio`; inclui limite total de tokens e diagnóstico. Não representa treinamento conjunto de personagem.

- Blur centralizado; resize e preparo causal compartilhados entre CLI e nodes. Removidos loops e estado redundantes. Harness importa código de produção.
- **Extract H3 Audio RefMod**: VAE H3, mono/estéreo, resample para 32 kHz, encode em blocos de 10 segundos, limite de duração e orçamento por erro ou prefixo contíguo.
- Formato v4 suporta áudio [1,32,2,T], preservando leitura dos visuais antigos e audio_refmod_meta do H3AudioMod. Áudio e visual coexistem no bundle em arquivos independentes.
- Loaders, Apply, Config, Step Curve e bridge reconhecem áudio; models/audio_refmods é consultado.
- **Inspect H3 RefMod**: tipo, caminho, dimensões, descrição, config e custo total. Preview opcional da primeira imagem ou primeiros 2 segundos de áudio, com comparação de força e VAE correspondente.
- Biblioteca nos loaders: busca por nome/pasta/conceito/descrição, filtro de tipo, seleção de slot e Refresh. Endpoint lê apenas metadados. Slots conectados são protegidos contra seleção pelo widget.
- Subpasta opcional no Extract visual/áudio e CLI. Scramble separa embaralhamento de seleção de subconjunto; workflows antigos preservam o comportamento sem o novo parâmetro.
- Presets de identidade, estilo experimental e sequência de movimento. São configurações convenientes, sem alegação de melhoria de qualidade ou separação semântica.

## Validação

**39 testes aprovados na suíte completa antes da publicação da v0.2.0**, com Python embarcado e módulos reais do ComfyUI:

```powershell
../../../python_embeded/python.exe -m unittest discover -s tests -v
```

Cobertura: fila real com upstream não executado, ambos os loaders, caminhos, cache, limites, máscara, vídeo, bridge, curvas, áudio, escrita atômica, inspector, endpoint da biblioteca sem tensores e paridade contra a perda MSE completa com geometrias diferentes.

O módulo real do diálogo foi verificado no navegador em fixture local: layout visível, pesquisa, seleção e Escape com restauração de foco. Ainda é necessário experimentar a extensão no ComfyUI completo após reiniciar.

### VAE real de áudio

Checkpoint: E:\pinokio\api\comfy.git\app\models\vae\minimax_h3_audio_vae_fp32.safetensors. GPU: RTX 4060 Ti.

| Entrada sintética | Latente | Tokens | Decode estéreo a 32 kHz | Pico CUDA |
| --- | --- | --- | --- | --- |
| 0,5 s | [1,32,2,20] | 40 | 16.000 amostras | 668 MiB |
| 10,125 s | [1,32,2,405] | 810 | 324.000 amostras | 1.371 MiB |

Ambos passaram: layout nativo, valores finitos e roundtrip safetensors exato. Resultados: tests/audio_smoke_result.json e tests/audio_chunk_result.json. Reprodução: tests/audio_smoke.py.

### Otimização multi-ref

Três implementações reais, mesma GPU/processo, 4 targets sintéticos de 8x64x64 e 20 passos:

| Estratégia | Tempo | Pico CUDA | Erro máximo contra resident |
| --- | --- | --- | --- |
| Targets residentes, backward sequencial | 0,0538 s | 21,05 MiB | 0 |
| Transferência por target | 0,0934 s | 12,05 MiB | 8,05e-7 |
| Média ponderada por geometria | 0,0264 s | 12,04 MiB | 2,38e-6 |

A terceira é o padrão: preserva o gradiente da média MSE e evita reconstruções repetidas. Paridade passou com rtol=1e-4, atol=1e-5. Resultados em tests/gauntlet_result.json. Medição somente do refinamento sintético: não representa encode, geração completa ou ganho contra todo o pipeline anterior.

## Limites e próximo teste prático

Reinicie o ComfyUI para carregar os novos nodes e a biblioteca. Compare uma voz conhecida com/sem RefMod usando prompts e seeds fixos. Não foi executada geração completa H3/Continuum nesta tarefa: identidade vocal, música, qualidade visual, sincronização audiovisual e VRAM do pipeline inteiro ainda não foram medidos.

Estilo/conceito continuam baseados em reconstrução de latentes; metadados não criam um extrator semântico independente. Presets experimentais precisam de avaliação visual. Disparo por <Nome> permanece fora do escopo solicitado.
