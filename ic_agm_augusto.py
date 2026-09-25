# -*- coding: utf-8 -*-
# %% [markdown]
# # IC — Carteiras de FIIs com a Árvore Geradora Mínima (AGM) — versão corrigida
# **Aluno:** Augusto Carneiro da Silva
# **Orientador:** Prof. Dr. João Roberto Bertini Junior
#
# Esta é a versão corrigida do notebook `IC_AGM_Augusto`. O método é o mesmo
# (distância de Mantegna, AGM por Kruskal, escore composto de grau,
# intermediação e proximidade, walk-forward 360/22). O que mudou está marcado
# no código como `CORREÇÃO (n)`:
#
# | # | Antes | Agora | Por quê |
# |---|---|---|---|
# | 1 | `TAXA_IR = 0.15`, **nunca aplicada** | 20% sobre o ganho de **preço** realizado | Lei 11.033/2004: FII paga 20% no ganho de capital; o dividendo é isento |
# | 2 | custo = 0,3% × \|retorno do dia\|, todo dia | custo = 0,3% × **giro**, no rebalanceamento | custo de transação só existe quando se negocia |
# | 3 | universo filtrado com `datetime.now()` | universo congelado e determinístico | filtrar a janela 2022–25 com dados de hoje é *look-ahead* e muda a cada execução |
# | 4 | carteira = média dos retornos **log** | média dos retornos **simples** | o retorno de uma carteira é a média ponderada dos retornos simples |
# | 5 | laço `range(360, n-22, 22)` | `while t + 22 <= n` | o `range` descartava a última janela completa |
# | 6 | retorno anual = média × 252 | retorno composto | a média aritmética superestima o que o investidor recebe |
# | 7 | só Central-5, Central-10, Periférica-5 | grade do paper (k = 10, 15, 20, com Híbrida) + seu k = 5 | para comparar com o paper |
#
# Os dados vêm congelados do repositório do projeto, então qualquer pessoa que
# rodar este notebook obtém exatamente os mesmos números — em qualquer dia.

# %% CÉLULA 0 — CONFIGURAÇÃO E DADOS CONGELADOS
import os
import subprocess

import networkx as nx
import numpy as np
import pandas as pd

JANELA_TREINO = 360
JANELA_TESTE = 22
CUSTO_OPER = 0.003          # 0,3% sobre o giro
TAXA_IR = 0.20              # CORREÇÃO (1): 20%, não 15% (e agora é aplicada)
TAMANHOS = (10, 15, 20)     # CORREÇÃO (7): grade do paper
TAMANHOS_EXTRA = (5,)       # sua carteira original; não entra no paper
N_BOOT = 5000
SEMENTE = 42

# CORREÇÃO (3): dados congelados em vez de download com datetime.now().
# Os dados vêm do repositório público do projeto. Para usar uma cópia já
# clonada, defina a variável de ambiente IC_DADOS com o caminho de data/raw.
DADOS = os.environ.get("IC_DADOS")
if not DADOS:
    if not os.path.exists("IC_Grafos_FIIs"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/yoonsungj04/IC_Grafos_FIIs.git"], check=True)
    DADOS = os.path.join("IC_Grafos_FIIs", "data", "raw")


def _ler(nome):
    return pd.read_csv(os.path.join(DADOS, nome), index_col=0, parse_dates=True).sort_index()


precos = _ler("precos_ajustados.csv")    # fechamento ajustado por proventos
brutos = _ler("precos_brutos.csv")       # fechamento bruto (base do IR)
divs = _ler("dividendos.csv")
cdi = _ler("cdi.csv").iloc[:, 0]         # CDI diário em decimal
bench = _ler("benchmark.csv").iloc[:, 0]  # XFIX11, ETF que replica o IFIX


def selecionar_universo(precos, min_hist=360, max_falt=0.05):
    """Fundos com histórico e cobertura suficientes NA PRÓPRIA JANELA."""
    total = len(precos)
    aprovados = []
    for tk in precos.columns:
        s = precos[tk]
        n = s.notna().sum()
        if n < min_hist:
            continue
        if s.loc[s.first_valid_index():].isna().mean() > max_falt:
            continue
        if n < total * (1 - max_falt):
            continue
        aprovados.append(tk)
    return sorted(aprovados)


FIIS = selecionar_universo(precos)
P = precos[FIIS].ffill().dropna()
PB = brutos[FIIS].ffill().dropna()
print(f"Universo: {len(FIIS)} FIIs | {P.index[0].date()} a {P.index[-1].date()} "
      f"| CDI médio {cdi.mean() * 252 * 100:.1f}% a.a.")

# %% CÉLULA 1 — RETORNOS, CORRELAÇÃO E DISTÂNCIA
# Retorno total sobre o preço AJUSTADO: Rt = ln(Pt / Pt-1). O ajuste já
# reinveste o provento, então NÃO se soma o dividendo de novo.
df_retornos = np.log(P / P.shift(1)).dropna(how="all")

# Retorno só de PREÇO (fechamento bruto): é a base tributável do IR.
ret_preco = np.log(PB / PB.shift(1)).dropna(how="all").reindex(df_retornos.index)

# A sua fórmula original, Rt = ln((Pt + Dt) / Pt-1) sobre o preço bruto, fica
# guardada para comparação na Célula 8.
D = divs.reindex(columns=FIIS).reindex(PB.index).fillna(0.0)
ret_p_mais_d = np.log((PB + D) / PB.shift(1)).dropna(how="all").reindex(df_retornos.index)


def matriz_distancia(ret):
    rho = ret.corr()
    # a correlação de um ativo consigo mesmo pode sair 1.0000000000000002 e
    # gerar sqrt(negativo) = NaN; o clip e a diagonal zerada evitam arestas
    # espúrias. Trabalha num array próprio porque, no pandas 3, `.values` de
    # um DataFrame é somente leitura.
    d = np.sqrt(np.clip(2 * (1 - rho.to_numpy()), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=rho.index, columns=rho.columns)


# %% CÉLULA 2 — AGM E ESCORE COMPOSTO

def construir_agm(dist):
    G = nx.from_pandas_adjacency(dist)
    return nx.minimum_spanning_tree(G, algorithm="kruskal")


def calcular_score_alinhado(agm):
    df = pd.DataFrame({
        "G": nx.degree_centrality(agm),
        "I": nx.betweenness_centrality(agm, weight="weight"),
        "P": nx.closeness_centrality(agm, distance="weight"),
    })
    df = df.reindex(sorted(df.index))
    df_n = (df - df.min()) / (df.max() - df.min())
    return df_n.mean(axis=1)


agm = construir_agm(matriz_distancia(df_retornos))
score = calcular_score_alinhado(agm).sort_values(ascending=False)
print(f"AGM na janela completa: {agm.number_of_nodes()} nós, {agm.number_of_edges()} arestas")
print("Mais centrais: ", ", ".join(f"{t} ({v:.2f})" for t, v in score.head(5).items()))
print("Mais periféricos:", ", ".join(f"{t} ({v:.2f})" for t, v in score.tail(5).items()))

# %% CÉLULA 3 — FORMAÇÃO DAS CARTEIRAS

def selecionar(score, tipo, k):
    nomes = list(score.sort_values(ascending=False).index)
    if tipo == "central":
        escolhidos = nomes[:k]
    elif tipo == "periferica":
        escolhidos = nomes[-k:]
    else:  # híbrida: metade de cada ponta (7 + 8 quando k = 15)
        m = k // 2
        escolhidos = nomes[:m] + nomes[-(k - m):]
    return {t: 1.0 / len(escolhidos) for t in escolhidos}


def retorno_carteira(ret_log, pesos):
    # CORREÇÃO (4): média ponderada dos retornos SIMPLES dos ativos
    w = pd.Series(pesos)
    return np.expm1(ret_log[w.index]).mul(w, axis=1).sum(axis=1)


def giro(pesos_ant, pesos_novos):
    """Metade da soma das variações absolutas de peso entre carteiras-alvo."""
    ativos = set(pesos_ant) | set(pesos_novos)
    return 0.5 * sum(abs(pesos_novos.get(a, 0.0) - pesos_ant.get(a, 0.0)) for a in ativos)


ROTULOS = ([f"{tp}_{k}" for tp in ("central", "periferica", "hibrida") for k in TAMANHOS]
           + [f"{tp}_{k}" for tp in ("central", "periferica") for k in TAMANHOS_EXTRA])

# %% CÉLULA 4 — BACKTEST WALK-FORWARD

def walk_forward(ret_total, ret_preco):
    oos = {r: [] for r in ROTULOS}
    oos_preco = {r: [] for r in ROTULOS}
    giros = {r: {} for r in ROTULOS}
    anteriores = {r: {} for r in ROTULOS}

    t = JANELA_TREINO
    while t + JANELA_TESTE <= len(ret_total):   # CORREÇÃO (5)
        treino = ret_total.iloc[t - JANELA_TREINO:t]
        teste = ret_total.iloc[t:t + JANELA_TESTE]
        teste_preco = ret_preco.iloc[t:t + JANELA_TESTE]
        data = teste.index[0]

        score_j = calcular_score_alinhado(construir_agm(matriz_distancia(treino)))
        for r in ROTULOS:
            tipo, k = r.rsplit("_", 1)
            pesos = selecionar(score_j, tipo, int(k))
            giros[r][data] = giro(anteriores[r], pesos)
            oos[r].append(retorno_carteira(teste, pesos))
            oos_preco[r].append(retorno_carteira(teste_preco, pesos))
            anteriores[r] = pesos
        t += JANELA_TESTE

    return (pd.DataFrame({r: pd.concat(v) for r, v in oos.items()}),
            pd.DataFrame({r: pd.concat(v) for r, v in oos_preco.items()}),
            pd.DataFrame(giros))


df_bruto, df_preco, df_giro = walk_forward(df_retornos, ret_preco)
print(f"Fora da amostra: {df_bruto.index[0].date()} a {df_bruto.index[-1].date()} "
      f"({len(df_bruto)} pregões, {len(df_giro)} rebalanceamentos)")

# %% CÉLULA 5 — CUSTOS DE TRANSAÇÃO E IMPOSTO

def aplicar_custos(ret_bruto, ret_preco, giros_serie):
    """
    CORREÇÃO (2): custo = 0,3% x giro, debitado no 1º dia de cada janela.
    CORREÇÃO (1): IR = 20% x giro x max(ganho de PREÇO da janela, 0),
    debitado no último dia. O dividendo fica fora da base: é isento.
    """
    janela = np.searchsorted(giros_serie.index, ret_bruto.index, side="right") - 1
    partes = []
    for w in range(len(giros_serie)):
        r = ret_bruto[janela == w].copy()
        if r.empty:
            continue
        to = float(giros_serie.iloc[w])
        r.iloc[0] -= CUSTO_OPER * to
        fator = (1 + r).prod()
        ganho_preco = (1 + ret_preco.reindex(r.index).fillna(0.0)).prod() - 1
        tributo = TAXA_IR * to * max(ganho_preco, 0.0)
        if tributo > 0 and fator > 0:
            r.iloc[-1] = (1 + r.iloc[-1]) * (fator - tributo) / fator - 1
        partes.append(r)
    return pd.concat(partes).sort_index()


df_liquido = pd.DataFrame({r: aplicar_custos(df_bruto[r], df_preco[r], df_giro[r])
                           for r in ROTULOS})

# %% CÉLULA 6 — MÉTRICAS (EXCESSO SOBRE O CDI)
rf = cdi.reindex(df_liquido.index).ffill().bfill()
ret_bench = (bench / bench.shift(1) - 1).reindex(df_retornos.index).dropna().reindex(
    df_liquido.index).dropna()


def get_metrics_alinhadas(r):
    ret_anual = (1 + r).prod() ** (252 / len(r)) - 1   # CORREÇÃO (6): composto
    vol = r.std(ddof=1) * np.sqrt(252)
    excesso = r - rf.reindex(r.index)
    sharpe = excesso.mean() / excesso.std(ddof=1) * np.sqrt(252)
    curva = (1 + r).cumprod()
    return {"Retorno a.a.": ret_anual, "Vol a.a.": vol, "Sharpe": sharpe,
            "Max queda": float((curva / curva.cummax() - 1).min())}


tabela = pd.DataFrame({r: {"Giro médio": df_giro[r].mean(),
                           **get_metrics_alinhadas(df_liquido[r])} for r in ROTULOS}).T
tabela.loc["IFIX (XFIX11)"] = {"Giro médio": np.nan, **get_metrics_alinhadas(ret_bench)}
fmt = tabela.copy()
for c in ("Retorno a.a.", "Vol a.a.", "Max queda"):
    fmt[c] = (fmt[c] * 100).map(lambda x: f"{x:+.1f}%")
fmt["Sharpe"] = fmt["Sharpe"].map(lambda x: f"{x:.2f}")
fmt["Giro médio"] = fmt["Giro médio"].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")
print("\n--- AGM, líquido de custo e IR, fora da amostra ---")
print(fmt.to_string())

# %% CÉLULA 7 — SIGNIFICÂNCIA (PERIFÉRICA-10 MENOS CENTRAL-10)

def _sharpe_aa(x):
    s = x.std(ddof=1)
    return x.mean() / s * np.sqrt(252) if s > 0 else np.nan


def bootstrap_diff_sharpe(a, b):
    """Bootstrap de blocos circulares de 22 dias (preserva a autocorrelação)."""
    df = pd.concat([a, b], axis=1).dropna()
    ea = (df.iloc[:, 0] - rf.reindex(df.index)).to_numpy()
    eb = (df.iloc[:, 1] - rf.reindex(df.index)).to_numpy()
    n = len(ea)
    obs = _sharpe_aa(ea) - _sharpe_aa(eb)
    rng = np.random.default_rng(SEMENTE)
    nb = int(np.ceil(n / JANELA_TESTE))
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ini = rng.integers(0, n, size=nb)
        idx = (ini[:, None] + np.arange(JANELA_TESTE)[None, :]).ravel()[:n] % n
        diffs[i] = _sharpe_aa(ea[idx]) - _sharpe_aa(eb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)
    return obs, lo, hi, p


def jobson_korkie(a, b):
    """Jobson-Korkie com a correção de Memmel (2003)."""
    from scipy import stats
    df = pd.concat([a, b], axis=1).dropna()
    x = df.iloc[:, 0] - rf.reindex(df.index)
    y = df.iloc[:, 1] - rf.reindex(df.index)
    n = len(x)
    sa, sb, rho = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1), x.corr(y)
    theta = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / n
    return 2 * (1 - stats.norm.cdf(abs((sa - sb) / np.sqrt(theta))))


d, lo, hi, p_boot = bootstrap_diff_sharpe(df_liquido["periferica_10"], df_liquido["central_10"])
p_jk = jobson_korkie(df_liquido["periferica_10"], df_liquido["central_10"])
print(f"\nPeriférica-10 - Central-10: dSharpe = {d:+.2f}  IC95 = [{lo:+.2f}, {hi:+.2f}]  "
      f"p(bootstrap) = {p_boot:.2f}  p(JK) = {p_jk:.2f}")

# %% CÉLULA 8 — O QUE CADA ERRO DO NOTEBOOK ORIGINAL CAUSAVA
# Refaz as suas carteiras do jeito original, no MESMO universo e nas MESMAS
# janelas, para isolar o efeito dos erros de método.

def metodo_original(r_log_carteira):
    """Seu cálculo antigo: custo de 0,3% x |retorno do dia|, sem IR,
    retorno anual = média aritmética x 252."""
    liq = r_log_carteira - r_log_carteira.abs() * CUSTO_OPER
    simples = np.expm1(liq)
    exc = simples - rf.reindex(simples.index)
    return simples.mean() * 252, exc.mean() / exc.std(ddof=1) * np.sqrt(252)


comparacao = []
for r in ("central_10", "periferica_10", "hibrida_10", "periferica_5"):
    # carteira antiga: média dos retornos LOG, mesmas composições
    r_log = np.log1p(df_bruto[r])
    ret_ant, sh_ant = metodo_original(r_log)
    m = get_metrics_alinhadas(df_liquido[r])
    comparacao.append({"Carteira": r,
                       "Retorno (original)": f"{ret_ant * 100:+.1f}%",
                       "Retorno (corrigido)": f"{m['Retorno a.a.'] * 100:+.1f}%",
                       "Sharpe (original)": f"{sh_ant:.2f}",
                       "Sharpe (corrigido)": f"{m['Sharpe']:.2f}"})
print("\n--- Método original x corrigido (mesmas carteiras) ---")
print(pd.DataFrame(comparacao).set_index("Carteira").to_string())

# A sua fórmula de retorno, ln((P + D) / P-1), comparada à do preço ajustado.
bruto_pd, preco_pd, giro_pd = walk_forward(ret_p_mais_d, ret_preco)
sh_pd = {r: get_metrics_alinhadas(aplicar_custos(bruto_pd[r], preco_pd[r], giro_pd[r]))["Sharpe"]
         for r in ("central_10", "periferica_10", "hibrida_10")}
print("\n--- Retorno ln((P+D)/P-1) x preço ajustado (Sharpe líquido, k=10) ---")
for r, v in sh_pd.items():
    print(f"  {r:<14} (P+D): {v:.2f}   ajustado: {tabela.loc[r, 'Sharpe']:.2f}")

# %% CÉLULA 9 — CONFERÊNCIA COM O ARTIGO
# Valores publicados para a AGM (Tabelas 3, 4, 5 e 7 do manuscrito).
ARTIGO_SHARPE = {"central_10": -1.15, "central_15": -1.49, "central_20": -1.41,
                 "periferica_10": -1.25, "periferica_15": -1.22, "periferica_20": -1.12,
                 "hibrida_10": -1.49, "hibrida_15": -1.41, "hibrida_20": -1.30}
ARTIGO_GIRO = {"central_10": 0.28, "central_15": 0.27, "central_20": 0.25,
               "periferica_10": 0.44, "periferica_15": 0.35, "periferica_20": 0.30,
               "hibrida_10": 0.44, "hibrida_15": 0.40, "hibrida_20": 0.35}

falhas = 0
print("\n--- Conferência com o artigo ---")
for r in ARTIGO_SHARPE:
    sh, gi = round(tabela.loc[r, "Sharpe"], 2), round(tabela.loc[r, "Giro médio"], 2)
    ok = sh == ARTIGO_SHARPE[r] and gi == ARTIGO_GIRO[r]
    falhas += not ok
    print(f"  {'OK ' if ok else 'ERRO'} {r:<14} Sharpe {sh:+.2f} (artigo {ARTIGO_SHARPE[r]:+.2f})  "
          f"giro {gi:.2f} (artigo {ARTIGO_GIRO[r]:.2f})")

extras = [
    ("Retorno líq. Periférica-10 (Tab. 5)", round(tabela.loc["periferica_10", "Retorno a.a."] * 100, 1), 2.2),
    ("dSharpe Perif-Central (Tab. 7)", round(d, 2), -0.10),
    ("IC95 inferior (Tab. 7)", round(lo, 2), -1.13),
    ("IC95 superior (Tab. 7)", round(hi, 2), 1.07),
    ("p bootstrap (Tab. 7)", round(p_boot, 2), 0.87),
    ("p Jobson-Korkie (Tab. 7)", round(p_jk, 2), 0.86),
]
for nome, obtido, esperado in extras:
    ok = obtido == esperado
    falhas += not ok
    print(f"  {'OK ' if ok else 'ERRO'} {nome:<38} {obtido:+.2f} (artigo {esperado:+.2f})")

print(f"\n{'Tudo confere com o artigo.' if falhas == 0 else f'{falhas} divergência(s).'}")
