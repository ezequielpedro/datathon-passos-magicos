from pathlib import Path
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Passos Mágicos | Risco Educacional",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def localizar_arquivo(*candidatos):
    for caminho in candidatos:
        caminho = Path(caminho)
        if caminho.exists():
            return caminho
    return None


MODEL_PATH = localizar_arquivo(
    PROJECT_ROOT / "models" / "modelo_risco.joblib",
    APP_DIR / "modelo_risco.joblib",
    Path("models/modelo_risco.joblib"),
    Path("modelo_risco.joblib"),
)

METADATA_PATH = localizar_arquivo(
    PROJECT_ROOT / "models" / "modelo_risco_metadata.json",
    APP_DIR / "modelo_risco_metadata.json",
    Path("models/modelo_risco_metadata.json"),
    Path("modelo_risco_metadata.json"),
)

IMPORTANCE_PATH = localizar_arquivo(
    PROJECT_ROOT / "data" / "processed" / "importancia_features.csv",
    APP_DIR / "importancia_features.csv",
    Path("data/processed/importancia_features.csv"),
    Path("importancia_features.csv"),
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        .pm-hero {
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 18px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
            background: linear-gradient(
                135deg,
                rgba(127, 127, 127, 0.08),
                rgba(127, 127, 127, 0.02)
            );
        }

        .pm-kicker {
            display: inline-block;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .04em;
            text-transform: uppercase;
            opacity: .72;
            margin-bottom: .35rem;
        }

        .pm-title {
            font-size: clamp(1.85rem, 3vw, 2.65rem);
            line-height: 1.08;
            font-weight: 800;
            margin: 0 0 .45rem 0;
        }

        .pm-subtitle {
            font-size: 1rem;
            opacity: .74;
            max-width: 920px;
            line-height: 1.55;
        }

        .section-title {
            font-size: 1.22rem;
            font-weight: 750;
            margin-bottom: .65rem;
        }

        .risk-card {
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            border: 1px solid rgba(128, 128, 128, .28);
            margin: .6rem 0 1rem 0;
        }

        .risk-card.high { border-left: 6px solid #ff4b4b; }
        .risk-card.attention { border-left: 6px solid #f4b942; }
        .risk-card.low { border-left: 6px solid #36b37e; }

        .risk-label {
            font-size: .84rem;
            opacity: .68;
            margin-bottom: .25rem;
        }

        .risk-value {
            font-size: clamp(2.4rem, 7vw, 4.5rem);
            font-weight: 850;
            line-height: 1;
            margin-bottom: .5rem;
        }

        .risk-status {
            font-size: 1.2rem;
            font-weight: 760;
            margin-bottom: .25rem;
        }

        .risk-desc {
            opacity: .75;
            line-height: 1.5;
        }

        .info-strip {
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, .24);
            padding: .8rem .95rem;
            margin: .35rem 0 1rem 0;
            opacity: .90;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 14px;
            padding: .85rem 1rem;
            background: rgba(127, 127, 127, 0.035);
        }

        [data-testid="stForm"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 16px;
            padding: 1.1rem 1.15rem 1.25rem 1.15rem;
        }

        div[data-baseweb="tab-list"] { gap: .35rem; }
        div[data-baseweb="tab"] { padding-left: .7rem; padding-right: .7rem; }

        .footer-note {
            font-size: .8rem;
            opacity: .58;
            text-align: center;
            padding-top: .3rem;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .pm-hero { padding: 1.1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_LABELS = {
    "fase_ordem_2023": "Fase",
    "idade_2023": "Idade",
    "genero_2023": "Gênero",
    "ano_ingresso_2023": "Ano de ingresso",
    "instituicao_ensino_2023": "Instituição de ensino",
    "inde_2023": "INDE",
    "ian_2023": "IAN",
    "ida_2023": "IDA",
    "ieg_2023": "IEG",
    "iaa_2023": "IAA",
    "ips_2023": "IPS",
    "ipp_2023": "IPP",
    "ipv_2023": "IPV",
    "nota_matematica_2023": "Nota de Matemática",
    "nota_portugues_2023": "Nota de Português",
    "nota_ingles_2023": "Nota de Inglês",
    "defasagem_2023": "Defasagem atual",
    "tempo_pm_2023": "Tempo na Passos Mágicos",
    "media_notas_2023": "Média das notas",
    "media_indicadores_2023": "Média dos indicadores",
    "min_indicador_2023": "Menor indicador",
}


@st.cache_resource
def carregar_modelo(caminho):
    bundle = joblib.load(caminho)
    if not isinstance(bundle, dict):
        raise ValueError("O arquivo do modelo não possui o formato esperado.")
    obrigatorios = {"pipeline", "threshold", "features"}
    if not obrigatorios.issubset(bundle.keys()):
        raise ValueError("O bundle do modelo não contém pipeline, threshold e features.")
    return bundle


@st.cache_data
def carregar_metadata(caminho):
    if caminho is None:
        return {}
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


@st.cache_data
def carregar_importancias(caminho):
    if caminho is None:
        return pd.DataFrame()
    return pd.read_csv(caminho)


if MODEL_PATH is None:
    st.error("Não encontrei `modelo_risco.joblib`. Coloque o arquivo na pasta `models/` do projeto.")
    st.stop()

try:
    bundle = carregar_modelo(MODEL_PATH)
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    features_modelo = list(bundle["features"])
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()

metadata = carregar_metadata(METADATA_PATH)
importancias = carregar_importancias(IMPORTANCE_PATH)

st.markdown(
    """
    <div class="pm-hero">
        <div class="pm-kicker">Datathon Pós-Tech • Associação Passos Mágicos</div>
        <div class="pm-title">Predição de Risco Educacional</div>
        <div class="pm-subtitle">
            Ferramenta de apoio para estimar a probabilidade de um aluno,
            atualmente sem defasagem, entrar em situação de defasagem no próximo ciclo.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predicao, tab_modelo = st.tabs(["🔎 Predição individual", "📊 Sobre o modelo"])

with tab_predicao:
    st.markdown(
        """
        <div class="info-strip">
            O modelo foi treinado para alunos que estavam <b>sem defasagem em 2023</b>.
            A sinalização deve ser usada como apoio à priorização de acompanhamento,
            e não como decisão automática.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("form_predicao"):
        st.markdown('<div class="section-title">1. Perfil do aluno</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fase_ordem_2023 = st.number_input(
                "Fase / ordem", min_value=0, max_value=8, value=3, step=1,
                help="Faixa observada na população utilizada no treinamento."
            )
        with c2:
            idade_2023 = st.number_input("Idade", min_value=7, max_value=26, value=12, step=1)
        with c3:
            genero_2023 = st.selectbox("Gênero", ["Feminino", "Masculino"])

        c4, c5 = st.columns(2)
        with c4:
            ano_ingresso_2023 = st.number_input(
                "Ano de ingresso na Passos Mágicos", min_value=2016, max_value=2023, value=2022, step=1
            )
        with c5:
            instituicao_ensino_2023 = st.selectbox(
                "Instituição de ensino",
                [
                    "Pública",
                    "Privada - Programa de Apadrinhamento",
                    "Privada *Parcerias com Bolsa 100%",
                    "Privada",
                    "Privada - Pagamento por *Empresa Parceira",
                    "Concluiu o 3º EM",
                ],
            )

        st.divider()
        st.markdown('<div class="section-title">2. Indicadores educacionais</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            inde_2023 = st.number_input("INDE", min_value=0.0, max_value=10.0, value=7.85, step=0.1, format="%.2f")
            ian_2023 = st.number_input(
                "IAN", min_value=0.0, max_value=10.0, value=10.0, step=0.1, format="%.2f",
                help="Na população elegível usada no treinamento, o IAN observado foi 10."
            )
        with c2:
            ida_2023 = st.number_input("IDA", min_value=0.0, max_value=10.0, value=7.05, step=0.1, format="%.2f")
            ieg_2023 = st.number_input("IEG", min_value=0.0, max_value=10.0, value=9.30, step=0.1, format="%.2f")
        with c3:
            iaa_2023 = st.number_input("IAA", min_value=0.0, max_value=10.0, value=8.50, step=0.1, format="%.2f")
            ips_2023 = st.number_input("IPS", min_value=0.0, max_value=10.0, value=5.00, step=0.1, format="%.2f")
        with c4:
            ipp_2023 = st.number_input("IPP", min_value=0.0, max_value=10.0, value=7.81, step=0.1, format="%.2f")
            ipv_2023 = st.number_input("IPV", min_value=0.0, max_value=10.5, value=8.29, step=0.1, format="%.2f")

        st.divider()
        st.markdown('<div class="section-title">3. Notas e adequação</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            nota_matematica_2023 = st.number_input("Matemática", min_value=0.0, max_value=10.0, value=6.80, step=0.1, format="%.2f")
        with c2:
            nota_portugues_2023 = st.number_input("Português", min_value=0.0, max_value=10.0, value=7.25, step=0.1, format="%.2f")
        with c3:
            nota_ingles_2023 = st.number_input("Inglês", min_value=0.0, max_value=10.0, value=6.80, step=0.1, format="%.2f")
        with c4:
            defasagem_2023 = st.number_input(
                "Defasagem atual", value=0.0, disabled=True,
                help="O modelo foi construído para alunos que ainda não estavam em defasagem no período de origem."
            )

        submitted = st.form_submit_button("Analisar risco", use_container_width=True, type="primary")

    if submitted:
        tempo_pm_2023 = max(2023 - ano_ingresso_2023, 0)
        media_notas_2023 = np.mean([nota_matematica_2023, nota_portugues_2023, nota_ingles_2023])
        lista_indicadores = [ian_2023, ida_2023, ieg_2023, iaa_2023, ips_2023, ipp_2023, ipv_2023]
        media_indicadores_2023 = np.mean(lista_indicadores)
        min_indicador_2023 = np.min(lista_indicadores)

        registro = {
            "fase_ordem_2023": fase_ordem_2023,
            "idade_2023": idade_2023,
            "genero_2023": genero_2023,
            "ano_ingresso_2023": ano_ingresso_2023,
            "instituicao_ensino_2023": instituicao_ensino_2023,
            "inde_2023": inde_2023,
            "ian_2023": ian_2023,
            "ida_2023": ida_2023,
            "ieg_2023": ieg_2023,
            "iaa_2023": iaa_2023,
            "ips_2023": ips_2023,
            "ipp_2023": ipp_2023,
            "ipv_2023": ipv_2023,
            "nota_matematica_2023": nota_matematica_2023,
            "nota_portugues_2023": nota_portugues_2023,
            "nota_ingles_2023": nota_ingles_2023,
            "defasagem_2023": defasagem_2023,
            "tempo_pm_2023": tempo_pm_2023,
            "media_notas_2023": media_notas_2023,
            "media_indicadores_2023": media_indicadores_2023,
            "min_indicador_2023": min_indicador_2023,
        }

        entrada = pd.DataFrame([registro])
        faltantes = [coluna for coluna in features_modelo if coluna not in entrada.columns]
        if faltantes:
            st.error("As seguintes variáveis exigidas pelo modelo não foram criadas: " + ", ".join(faltantes))
            st.stop()

        entrada = entrada[features_modelo]
        probabilidade = float(pipeline.predict_proba(entrada)[0, 1])
        classificacao = int(probabilidade >= threshold)
        faixa_atencao = threshold * 0.70

        if classificacao == 1:
            css_classe = "high"
            status = "Prioridade de revisão"
            descricao = (
                "A probabilidade estimada ficou acima do limiar definido pelo modelo. "
                "O caso deve ser priorizado para avaliação da equipe."
            )
        elif probabilidade >= faixa_atencao:
            css_classe = "attention"
            status = "Atenção"
            descricao = (
                "A probabilidade permanece abaixo do limiar oficial, "
                "mas está próxima o suficiente para justificar acompanhamento."
            )
        else:
            css_classe = "low"
            status = "Acompanhamento de rotina"
            descricao = "A probabilidade estimada ficou abaixo do limiar de sinalização do modelo."

        st.divider()
        st.markdown('<div class="section-title">Resultado da análise</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="risk-card {css_classe}">
                <div class="risk-label">Probabilidade estimada de entrada em defasagem</div>
                <div class="risk-value">{probabilidade:.1%}</div>
                <div class="risk-status">{status}</div>
                <div class="risk-desc">{descricao}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidade estimada", f"{probabilidade:.1%}")
        c2.metric("Limiar do modelo", f"{threshold:.1%}")
        c3.metric("Distância do limiar", f"{(probabilidade - threshold):+.1%}")
        st.progress(min(max(int(round(probabilidade * 100)), 0), 100))

        with st.expander("Ver informações derivadas utilizadas pelo modelo"):
            derivados = pd.DataFrame(
                {
                    "Variável": [
                        "Tempo na Passos Mágicos",
                        "Média das notas",
                        "Média dos indicadores",
                        "Menor indicador",
                    ],
                    "Valor": [
                        tempo_pm_2023,
                        media_notas_2023,
                        media_indicadores_2023,
                        min_indicador_2023,
                    ],
                }
            )
            st.dataframe(derivados.style.format({"Valor": "{:.2f}"}), use_container_width=True, hide_index=True)

        with st.expander("Como interpretar este resultado"):
            st.markdown(
                """
                - A probabilidade representa uma **estimativa estatística**, não uma certeza.
                - A classificação depende do **limiar definido durante o treinamento**.
                - Um resultado acima do limiar serve como **sinal para priorização de análise humana**.
                - O modelo não substitui avaliações pedagógicas, psicopedagógicas ou psicossociais.
                """
            )

with tab_modelo:
    st.markdown('<div class="section-title">Desempenho do modelo</div>', unsafe_allow_html=True)

    modelo_nome = metadata.get("modelo", "Random Forest")
    n_amostras = metadata.get("n_amostras", 370)
    taxa_positiva = metadata.get("taxa_classe_positiva", 0.227)
    metricas = metadata.get("metricas_teste", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo", modelo_nome)
    c2.metric("Amostra", f"{n_amostras} alunos")
    c3.metric("Recall", f"{metricas.get('recall', 0.9048):.1%}")
    c4.metric("ROC-AUC", f"{metricas.get('roc_auc', 0.9530):.1%}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metricas.get('precision', 0.6552):.1%}")
    c2.metric("F2", f"{metricas.get('f2', 0.8407):.1%}")
    c3.metric("PR-AUC", f"{metricas.get('pr_auc', 0.8577):.1%}")
    c4.metric("Taxa positiva", f"{taxa_positiva:.1%}")

    st.markdown(
        f"""
        <div class="info-strip">
            O modelo utiliza informações de <b>2023</b> para estimar o risco de um aluno,
            ainda sem defasagem naquele período, entrar em defasagem em 2024.
            O limiar de decisão é <b>{threshold:.0%}</b>, escolhido nos dados de treino
            com foco no F2.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Principais variáveis preditivas</div>', unsafe_allow_html=True)

    if not importancias.empty:
        grafico = importancias.copy().sort_values("Importancia_media", ascending=False).head(10)
        grafico["Variável"] = grafico["Feature"].map(FEATURE_LABELS).fillna(grafico["Feature"])

        chart = (
            alt.Chart(grafico)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("Variável:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230)),
                x=alt.X("Importancia_media:Q", title="Importância preditiva", axis=alt.Axis(format=".3f")),
                tooltip=[
                    alt.Tooltip("Variável:N"),
                    alt.Tooltip("Importancia_media:Q", title="Importância", format=".4f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Permutation importance calculada no conjunto de teste. "
            "Importância preditiva não implica causalidade."
        )
    else:
        st.info(
            "Arquivo `importancia_features.csv` não encontrado. "
            "A aplicação continua funcionando normalmente."
        )

    st.markdown('<div class="section-title">Como interpretar as métricas</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Recall**  
            Capacidade de identificar alunos que realmente entraram em defasagem.

            **Precision**  
            Proporção dos alertas emitidos que efetivamente se confirmaram.
            """
        )
    with c2:
        st.markdown(
            """
            **ROC-AUC**  
            Capacidade geral de separar alunos de maior e menor risco.

            **F2**  
            Combina precision e recall, atribuindo maior peso ao recall.
            """
        )

    st.markdown('<div class="section-title">Limitações</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - A amostra temporal utilizada no problema preditivo é relativamente pequena.
        - A validação utiliza dados históricos de uma única transição anual.
        - Alguns indicadores originalmente possuíam valores ausentes e foram tratados pelo pipeline.
        - A probabilidade produzida pelo modelo é uma estimativa estatística, não uma certeza individual.
        - O resultado não deve ser utilizado para tomada de decisão automática sobre o aluno.
        """
    )

st.divider()
st.markdown(
    """
    <div class="footer-note">
        Datathon Pós-Tech • Associação Passos Mágicos •
        Modelo preditivo para apoio à identificação preventiva de risco educacional
    </div>
    """,
    unsafe_allow_html=True,
)
