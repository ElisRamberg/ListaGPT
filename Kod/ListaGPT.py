import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta
import io

# Initiera variabler
next_call_time = datetime.now()
rate_limit = 0.15
lock = asyncio.Lock()

async def fetch_openai_response(session, message, model, api_key, temperature, top_p):
    global next_call_time
    async with lock:
        await asyncio.sleep(max((next_call_time - datetime.now()).total_seconds(), 0))
        next_call_time = datetime.now() + timedelta(seconds=rate_limit)
    try:
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        body = {
            "model": model,
            "messages": message,
            "temperature": temperature,
            "top_p": top_p
        }

        async with session.post(url, json=body, headers=headers) as response:
            response_data = await response.json()
            if response.status != 200:
                error_message = response_data.get("error", {}).get("message", "Okänt fel")
                return f"API-fel: {error_message}"
            if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                return "Fel: Ogiltigt svarsdatalformat eller tomma val."
    except Exception as e:
        return f"Ett fel inträffade: {e}"

async def process_dataframe_row(session, row, index, column_name, prompt, model, api_key, temperature, top_p):
    # Vi behöver inte längre uppdatera status_text här
    text = row[column_name]

    if pd.isna(text) or str(text).strip() == '':
        return (None, index)
    text = str(text).strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]

    response = await fetch_openai_response(session, messages, model, api_key, temperature, top_p)
    if "Fel:" in response or "Ett fel inträffade:" in response or "API-fel:" in response:
        return (response, index)  # Returnera felmeddelandet för att inkludera det i resultatet
    else:
        return (response, index)

async def process_comments(df, column_name, prompt, model, api_key, num_rows, temperature, top_p, max_concurrent_requests, timeout_value):
    st.write("Startar bearbetning...")
    start_time = datetime.now()
    if num_rows is None:
        num_rows = len(df)

    # Initiera progressbar och status-text
    progress_bar = st.progress(0)
    status_text = st.empty()  # Skapa en tom plats för statusuppdateringar
    total_tasks = min(num_rows, len(df))

    timeout = aiohttp.ClientTimeout(total=timeout_value)
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            process_dataframe_row(session, row, index, column_name, prompt, model, api_key, temperature, top_p)
            for index, row in df.head(num_rows).iterrows()
        ]
        results = []
        tasks_completed = 0
        for future in asyncio.as_completed(tasks):
            try:
                result, index = await future
                results.append((result, index))
            except Exception as e:
                results.append((f"Ett fel inträffade: {e}", index))
            tasks_completed += 1
            progress_bar.progress(tasks_completed / total_tasks)
            status_text.text(f"Bearbetar rad {tasks_completed} av {total_tasks}")

    # Uppdatera DataFrame med resultaten
    for result, index in results:
        df.at[index, 'Resultat'] = result if result is not None else "Timeout"

    end_time = datetime.now()
    st.success(f"Bearbetning slutförd på: {end_time - start_time}")
    return df

def main():
    st.set_page_config(page_title="Textanalys med OpenAI API", layout="wide")
    st.title("📊 Textanalys med OpenAI API")

    # Instruktioner
    with st.expander("💡 Instruktioner för hur man använder verktyget"):
        st.markdown("""
        **Så här använder du verktyget:**

        1. **API-konfiguration:**
            - Ange din OpenAI API-nyckel i sidofältet.
            - Välj önskad OpenAI-modell.

        2. **Ladda upp din Excel-fil:**
            - Ladda upp en Excel-fil med data du vill analysera.
            - Efter uppladdning visas en förhandsvisning av dina data.

        3. **Välj kolumn för analys:**
            - Välj kolumnen som innehåller den textdata du vill analysera.

        4. **Välj typ av analys:**
            - Välj typ av analys: Sentimentanalys, Summering eller Kategoriindelning.

        5. **Granska och redigera prompt:**
            - Expandera promptredigeraren för att granska och eventuellt redigera prompten som kommer att skickas till OpenAI API.

        6. **Avancerade inställningar (Valfritt):**
            - Ange antalet rader som ska bearbetas.
            - Ange filnamn för utdatafilen.
            - Justera modellens inställningar som temperature och top_p.
            - Ändra API-anrop inställningar som max antal samtidiga förfrågningar och timeout.

        7. **Starta analys:**
            - Klicka på "Starta bearbetning" för att börja analysen.
            - En förloppsindikator och statusmeddelanden håller dig informerad.

        8. **Ladda ner bearbetad data:**
            - Efter bearbetning kan du förhandsgranska resultaten.
            - Klicka på "Ladda ner bearbetad data" för att ladda ner Excel-filen med analysresultaten.

        **Noteringar:**

        - Se till att din Excel-fil är korrekt formaterad och att textkolumnen innehåller data du vill analysera.
        - Verktyget bearbetar data asynkront för att förbättra prestanda.
        - Din API-nyckel hålls konfidentiell; den matas in säkert och lagras inte.
        """)

    # Sidofält för API-nyckel och modellval
    with st.sidebar:
        st.header("🔑 API-konfiguration")
        api_key = st.text_input("Ange din OpenAI API-nyckel", type="password", help="Din API-nyckel hålls konfidentiell.")
        model = st.selectbox("Välj OpenAI-modell (Standard: gpt-4o-mini)", ["gpt-4o-mini","gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"], help="Välj modell för textanalys.")

    # Sektion för filuppladdning
    st.header("📂 Ladda upp din Excel-fil")
    uploaded_file = st.file_uploader("Ladda upp en Excel-fil", type=["xlsx"], help="Se till att din Excel-fil är korrekt formaterad.")

    if uploaded_file is not None and api_key:
        df = pd.read_excel(uploaded_file)
        st.success("Filen har laddats upp framgångsrikt!")
        st.write("Här är en förhandsvisning av dina data:")
        st.dataframe(df.head())

        # Val av kolumn
        st.header("📝 Välj kolumn för analys")
        column_name = st.selectbox("Välj kolumn att analysera", df.columns, help="Välj kolumnen som innehåller textdata.")

        # Val av analys typ
        st.header("🔍 Välj typ av analys")
        analysis_type = st.selectbox("Välj typ av analys", ["Sentimentanalys", "Summering", "Kategoriindelning"], index=0)

        # Definiera prompts för olika analys typer
        prompts = {
            "Sentimentanalys": """
Bedöm sentimentet i följande text. Svara ENDAST med en av följande:
"Positiv"
"Negativ"
"Neutral"
""",
            "Summering": """
Sammanfatta följande text i en mening.
""",
            "Kategoriindelning": """
Bestäm vilken kategori följande text tillhör. Välj mellan:
"Nyheter"
"Sport"
"Underhållning"
"Teknik"
"""
        }

        # Hämta vald prompt
        prompt = prompts[analysis_type]

        # Visa och tillåt redigering av prompten
        st.header("✏️ Granska och redigera prompt")
        with st.expander("Klicka för att visa/redigera prompten"):
            prompt = st.text_area("Prompt för analysen:", value=prompt, height=150)

        # Avancerade inställningar
        st.header("⚙️ Avancerade inställningar")
        with st.expander("Valfria inställningar"):
            num_rows = st.number_input(
                "Antal rader att bearbeta",
                min_value=1,
                max_value=len(df),
                value=len(df),
                help="Ange hur många rader som ska bearbetas."
            )
            filename = st.text_input("Filnamn för utdatafilen", value="bearbetad_data.xlsx", help="Namn på den nedladdade filen.")

            # Modellens inställningar
            st.subheader("Modellens inställningar")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01, help="Ange temperatur för modellen.")
            top_p = st.slider("Top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.01, help="Ange top_p för modellen.")

            # API-anrop inställningar
            st.subheader("API-anrop inställningar")
            max_concurrent_requests = st.number_input("Max antal samtidiga förfrågningar", min_value=1, max_value=100, value=20, help="Ange max antal samtidiga API-förfrågningar.")
            timeout_value = st.number_input("Timeout (sekunder)", min_value=10, max_value=600, value=60, help="Ange timeout för API-förfrågningar i sekunder.")

        # Starta bearbetning
        st.header("🚀 Starta analys")
        if st.button("Starta bearbetning"):
            # Kör bearbetningen
            with st.spinner("Bearbetar... Vänligen vänta."):
                df_processed = asyncio.run(
                    process_comments(df, column_name, prompt, model, api_key, num_rows, temperature, top_p, max_concurrent_requests, timeout_value)
                )
                # Lagra den bearbetade DataFrame i session_state
                st.session_state['df_processed'] = df_processed
                # Lagra filnamnet i session_state
                st.session_state['filename'] = filename

        # Kontrollera om bearbetad data finns i session_state
        if 'df_processed' in st.session_state:
            df_processed = st.session_state['df_processed']
            filename = st.session_state.get('filename', 'bearbetad_data.xlsx')
            # Visa exempel på bearbetade data
            st.header("📄 Förhandsvisning av bearbetad data")
            st.write("Här är en förhandsvisning av de bearbetad data:")
            st.dataframe(df_processed.head())

            # Förbered DataFrame för nedladdning
            towrite = io.BytesIO()
            df_processed.to_excel(towrite, index=False)
            towrite.seek(0)
            st.download_button(
                label="📥 Ladda ner bearbetad data",
                data=towrite,
                file_name=filename,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    elif not api_key:
        st.warning("Vänligen ange din OpenAI API-nyckel i sidofältet för att fortsätta.")
    else:
        st.info("Väntar på filuppladdning...")

if __name__ == '__main__':
    main()